//===- SCFFloatStrengthReduce.cpp -----------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of `mtrt-scf-float-strength-reduce` pass.
///
/// This pass identifies scf.while loops with floating-point loop-carried
/// variables that are updated with constant additive steps, and transforms
/// them to use integer counters with scale factors. When possible, it uplifts
/// the transformed loop to scf.for.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "mtrt-scf-float-strength-reduce"

namespace mtrt {
#define GEN_PASS_DEF_SCFFLOATSTRENGTHREDUCEPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mtrt

using namespace mlir;
using namespace mlir::scf;

//===----------------------------------------------------------------------===//
// Float Loop Pattern Analysis
//===----------------------------------------------------------------------===//

namespace {

/// Describes a detected float loop pattern that can be strength-reduced.
struct FloatLoopPattern {
  /// The index of the float argument in the while loop's before region
  /// (also corresponds to inits and yield operands).
  unsigned beforeArgIndex;

  /// The index of the float argument in the while loop's after region
  /// (corresponds to condition args and while op results).
  unsigned afterArgIndex;

  /// The float type of the loop-carried variable.
  FloatType floatType;

  /// The initial value (must be a constant).
  APFloat initValue;

  /// The step value (must be a constant, from addf/subf).
  APFloat stepValue;

  /// The limit value for the loop condition (must be a constant).
  APFloat limitValue;

  /// The comparison predicate used in the condition.
  arith::CmpFPredicate predicate;

  /// The computed scale factor (absolute value of step).
  APFloat scaleFactor;

  /// The integer equivalents after scaling.
  int64_t intInit;
  int64_t intStep;
  int64_t intLimit;

  /// The arith.addf or arith.subf operation that updates the float value.
  Operation *updateOp;

  /// The arith.cmpf operation used in the condition.
  Operation *cmpOp;

  /// Constructor with explicit initialization.
  FloatLoopPattern(FloatType ft)
      : beforeArgIndex(0), afterArgIndex(0), floatType(ft),
        initValue(ft.getFloatSemantics()), stepValue(ft.getFloatSemantics()),
        limitValue(ft.getFloatSemantics()),
        predicate(arith::CmpFPredicate::OEQ),
        scaleFactor(ft.getFloatSemantics()), intInit(0), intStep(0),
        intLimit(0), updateOp(nullptr), cmpOp(nullptr) {}
};

/// Check if an APFloat value divided by scale gives an exact integer.
/// Returns true if successful and stores the result in `result`.
static bool isExactInteger(const APFloat &value, const APFloat &scale,
                           int64_t &result) {
  APFloat quotient = value;
  quotient.divide(scale, APFloat::rmNearestTiesToEven);

  // Check if the quotient is an integer
  bool isExact;
  APSInt intVal(64, /*isUnsigned=*/false);
  APFloat::opStatus status =
      quotient.convertToInteger(intVal, APFloat::rmNearestTiesToEven, &isExact);

  if (status != APFloat::opOK || !isExact)
    return false;

  result = intVal.getExtValue();
  return true;
}

/// Compute the integer limit value, adjusting conservatively based on the
/// comparison predicate.
static bool computeIntLimit(const APFloat &limit, const APFloat &scale,
                            arith::CmpFPredicate pred, int64_t &intLimit) {
  APFloat quotient = limit;
  quotient.divide(scale, APFloat::rmNearestTiesToEven);

  // For predicates like >= or >, we might need to round appropriately
  // to maintain correctness.
  APFloat::roundingMode rm;
  switch (pred) {
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    // For "greater than" comparisons, round up to be conservative
    rm = APFloat::rmTowardPositive;
    break;
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    // For "less than" comparisons, round down to be conservative
    rm = APFloat::rmTowardNegative;
    break;
  default:
    // For equality, we need exact conversion
    rm = APFloat::rmNearestTiesToEven;
    break;
  }

  bool isExact;
  APSInt intVal(64, /*isUnsigned=*/false);
  APFloat::opStatus status = quotient.convertToInteger(intVal, rm, &isExact);

  // For non-equality predicates, we allow non-exact conversion
  // as long as we're being conservative.
  if (status == APFloat::opInvalidOp)
    return false;

  intLimit = intVal.getExtValue();
  return true;
}

/// Convert an arith::CmpFPredicate to arith::CmpIPredicate.
/// Returns std::nullopt if the predicate cannot be converted.
static std::optional<arith::CmpIPredicate>
convertToCmpIPredicate(arith::CmpFPredicate pred) {
  switch (pred) {
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    return arith::CmpIPredicate::eq;
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    return arith::CmpIPredicate::ne;
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    return arith::CmpIPredicate::slt;
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    return arith::CmpIPredicate::sle;
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    return arith::CmpIPredicate::sgt;
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    return arith::CmpIPredicate::sge;
  default:
    return std::nullopt;
  }
}

/// Analyze a scf.while loop to detect a float loop pattern.
/// Returns std::nullopt if no suitable pattern is found.
static std::optional<FloatLoopPattern> analyzeWhileLoop(scf::WhileOp whileOp,
                                                        unsigned beforeArgIdx) {
  // Get the before and after regions
  assert(whileOp.getBefore().hasOneBlock() &&
         whileOp.getAfter().hasOneBlock() &&
         "While loop must have exactly one block in each region");

  Block &beforeBlock = *whileOp.getBeforeBody();
  Block &afterBlock = *whileOp.getAfterBody();

  // Check if the argument is a float type
  assert(beforeArgIdx < beforeBlock.getNumArguments() &&
         "Invalid argument index");

  BlockArgument beforeArg = beforeBlock.getArgument(beforeArgIdx);
  auto floatType = dyn_cast<FloatType>(beforeArg.getType());
  if (!floatType)
    return std::nullopt;

  // Get the initial value - must be a constant
  Value initValue = whileOp.getInits()[beforeArgIdx];
  APFloat initFloat(floatType.getFloatSemantics());
  if (!matchPattern(initValue, m_ConstantFloat(&initFloat)))
    return std::nullopt;

  // Find the condition op
  auto conditionOp = whileOp.getConditionOp();

  // Check if the before argument is passed to the after region via the
  // condition op. The condition op's args are forwarded to the after region.
  OperandRange condArgs = conditionOp.getArgs();
  std::optional<unsigned> afterArgIdx;
  for (unsigned i = 0; i < condArgs.size(); ++i) {
    if (condArgs[i] == beforeArg) {
      afterArgIdx = i;
      break;
    }
  }

  // If the before argument is not passed to the after region, we cannot
  // transform this loop.
  if (!afterArgIdx)
    return std::nullopt;

  // Find the cmpf operation used for the condition
  Value condValue = conditionOp.getCondition();
  auto cmpOp = condValue.getDefiningOp<arith::CmpFOp>();
  if (!cmpOp)
    return std::nullopt;

  // Check if the comparison involves our float argument
  Value cmpLhs = cmpOp.getLhs();
  Value cmpRhs = cmpOp.getRhs();

  arith::CmpFPredicate pred = cmpOp.getPredicate();

  const bool argIsLhs = (cmpLhs == beforeArg);
  const bool argIsRhs = (cmpRhs == beforeArg);

  if (!argIsLhs && !argIsRhs)
    return std::nullopt;

  // Get the limit value
  Value limitValue = argIsLhs ? cmpRhs : cmpLhs;
  APFloat limitFloat(floatType.getFloatSemantics());
  if (!matchPattern(limitValue, m_ConstantFloat(&limitFloat)))
    return std::nullopt;

  // If the argument is on the RHS, we need to swap the predicate
  if (argIsRhs) {
    switch (pred) {
    case arith::CmpFPredicate::OLT:
      pred = arith::CmpFPredicate::OGT;
      break;
    case arith::CmpFPredicate::OLE:
      pred = arith::CmpFPredicate::OGE;
      break;
    case arith::CmpFPredicate::OGT:
      pred = arith::CmpFPredicate::OLT;
      break;
    case arith::CmpFPredicate::OGE:
      pred = arith::CmpFPredicate::OLE;
      break;
    case arith::CmpFPredicate::ULT:
      pred = arith::CmpFPredicate::UGT;
      break;
    case arith::CmpFPredicate::ULE:
      pred = arith::CmpFPredicate::UGE;
      break;
    case arith::CmpFPredicate::UGT:
      pred = arith::CmpFPredicate::ULT;
      break;
    case arith::CmpFPredicate::UGE:
      pred = arith::CmpFPredicate::ULE;
      break;
    default:
      break; // Keep symmetric predicates as-is
    }
  }

  // Analyze the after region to find the update pattern
  BlockArgument afterArg = afterBlock.getArgument(*afterArgIdx);

  // Find the yield op
  auto yieldOp = whileOp.getYieldOp();

  // The yield operands correspond to the before region args (not after)
  Value yieldedValue = yieldOp.getResults()[beforeArgIdx];

  // Check if the yielded value is computed by addf or subf
  Operation *updateOp = yieldedValue.getDefiningOp();
  if (!updateOp)
    return std::nullopt;

  const bool isAdd = isa<arith::AddFOp>(updateOp);
  const bool isSub = isa<arith::SubFOp>(updateOp);

  if (!isAdd && !isSub)
    return std::nullopt;

  // Get the operands
  Value updateLhs = updateOp->getOperand(0);
  Value updateRhs = updateOp->getOperand(1);

  // One operand must be the after block argument, the other must be a constant
  Value stepValue;
  if (updateLhs == afterArg) {
    stepValue = updateRhs;
  } else if (updateRhs == afterArg && isAdd) {
    // Addition is commutative
    stepValue = updateLhs;
  } else {
    return std::nullopt;
  }

  APFloat stepFloat(floatType.getFloatSemantics());
  if (!matchPattern(stepValue, m_ConstantFloat(&stepFloat)))
    return std::nullopt;

  // For subtraction, negate the step
  if (isSub)
    stepFloat.changeSign();

  // Compute the scale factor (absolute value of step)
  APFloat scaleFactor = stepFloat;
  if (scaleFactor.isNegative())
    scaleFactor.changeSign();

  // Check that the scale factor is non-zero
  if (scaleFactor.isZero())
    return std::nullopt;

  // Compute integer equivalents
  int64_t intInit, intStep, intLimit;

  if (!isExactInteger(initFloat, scaleFactor, intInit))
    return std::nullopt;

  if (!isExactInteger(stepFloat, scaleFactor, intStep))
    return std::nullopt;

  if (!computeIntLimit(limitFloat, scaleFactor, pred, intLimit))
    return std::nullopt;

  // Create and return the pattern
  FloatLoopPattern pattern(floatType);
  pattern.beforeArgIndex = beforeArgIdx;
  pattern.afterArgIndex = *afterArgIdx;
  pattern.initValue = initFloat;
  pattern.stepValue = stepFloat;
  pattern.limitValue = limitFloat;
  pattern.predicate = pred;
  pattern.scaleFactor = scaleFactor;
  pattern.intInit = intInit;
  pattern.intStep = intStep;
  pattern.intLimit = intLimit;
  pattern.updateOp = updateOp;
  pattern.cmpOp = cmpOp.getOperation();

  return pattern;
}

//===----------------------------------------------------------------------===//
// Transformation Pattern
//===----------------------------------------------------------------------===//

/// Pattern to transform float loop-carried variables to integer counters.
struct FloatToIntStrengthReducePattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    // Try to analyze each float argument
    for (unsigned i = 0; i < whileOp.getInits().size(); ++i) {
      auto pattern = analyzeWhileLoop(whileOp, i);
      if (!pattern)
        continue;

      LLVM_DEBUG(llvm::dbgs()
                 << "Found float loop pattern at arg " << i << ":\n"
                 << "  init: " << pattern->intInit
                 << ", step: " << pattern->intStep
                 << ", limit: " << pattern->intLimit << "\n");

      // Try to uplift to scf.for first
      if (succeeded(tryUpliftToFor(whileOp, *pattern, rewriter)))
        return success();

      // Otherwise, transform the while loop in-place
      if (succeeded(transformWhileLoop(whileOp, *pattern, rewriter)))
        return success();
    }

    return failure();
  }

private:
  /// Try to uplift the while loop to a for loop.
  LogicalResult tryUpliftToFor(scf::WhileOp whileOp,
                               const FloatLoopPattern &pattern,
                               PatternRewriter &rewriter) const {
    // For uplift to for loop, we need:
    // 1. Step to be +1 or -1
    // 2. A suitable comparison predicate (slt or sgt)

    if (pattern.intStep != 1 && pattern.intStep != -1)
      return failure();

    // Determine the for loop bounds based on the predicate and step
    int64_t lb, ub, step;

    if (pattern.intStep > 0) {
      // Counting up: need predicate like < or <=
      switch (pattern.predicate) {
      case arith::CmpFPredicate::OLT:
      case arith::CmpFPredicate::ULT:
        lb = pattern.intInit;
        ub = pattern.intLimit;
        step = pattern.intStep;
        break;
      case arith::CmpFPredicate::OLE:
      case arith::CmpFPredicate::ULE:
        lb = pattern.intInit;
        ub = pattern.intLimit + 1;
        step = pattern.intStep;
        break;
      default:
        return failure();
      }
    } else {
      // Counting down: need predicate like > or >=
      // We'll transform to a count-up loop
      switch (pattern.predicate) {
      case arith::CmpFPredicate::OGT:
      case arith::CmpFPredicate::UGT:
        // f > limit means i > intLimit, so count from intInit down to intLimit
        // + 1 Transform: for i in (intLimit+1, intInit+1) step 1, use (intInit
        // - i + intLimit + 1)
        lb = pattern.intLimit + 1;
        ub = pattern.intInit + 1;
        step = 1;
        break;
      case arith::CmpFPredicate::OGE:
      case arith::CmpFPredicate::UGE:
        // f >= limit means i >= intLimit
        lb = pattern.intLimit;
        ub = pattern.intInit + 1;
        step = 1;
        break;
      default:
        return failure();
      }
    }

    // Check that the before region only has the cmpf and condition
    Block &beforeBlock = whileOp.getBefore().front();
    if (!llvm::hasSingleElement(beforeBlock.without_terminator()))
      return failure();

    // Verify the before block only contains the compare operation
    if (&beforeBlock.front() != pattern.cmpOp)
      return failure();

    Location loc = whileOp.getLoc();

    // Create constants for loop bounds
    Value lbVal = rewriter.create<arith::ConstantIndexOp>(loc, lb);
    Value ubVal = rewriter.create<arith::ConstantIndexOp>(loc, ub);
    Value stepVal = rewriter.create<arith::ConstantIndexOp>(loc, step);

    // Prepare init values for the for loop (excluding the float counter)
    SmallVector<Value> forInits;
    for (unsigned i = 0; i < whileOp.getInits().size(); ++i) {
      if (i != pattern.beforeArgIndex)
        forInits.push_back(whileOp.getInits()[i]);
    }

    // Create the scale factor constant
    Value scaleConst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(pattern.floatType, pattern.scaleFactor));

    // Track whether we're counting down and need to reverse
    bool isCountingDown = (pattern.intStep < 0);

    // Create the for loop
    auto forOp = rewriter.create<scf::ForOp>(
        loc, lbVal, ubVal, stepVal, forInits,
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
          // Compute the actual integer counter value
          Value intCounter;
          if (isCountingDown) {
            // For counting down: actual_i = intInit - (iv - lb)
            // Simplifies to: actual_i = intInit - iv + lb
            Value lbIndex =
                builder.create<arith::ConstantIndexOp>(loc, pattern.intLimit);
            Value initIndex =
                builder.create<arith::ConstantIndexOp>(loc, pattern.intInit);
            Value diff = builder.create<arith::SubIOp>(loc, iv, lbIndex);
            intCounter = builder.create<arith::SubIOp>(loc, initIndex, diff);
          } else {
            intCounter = iv;
          }

          // Convert to the appropriate integer type and then to float
          Value intVal = builder.create<arith::IndexCastOp>(
              loc, builder.getI64Type(), intCounter);
          Value floatVal =
              builder.create<arith::SIToFPOp>(loc, pattern.floatType, intVal);
          Value scaledFloat =
              builder.create<arith::MulFOp>(loc, floatVal, scaleConst);

          // Create a mapping from old after block args to new values
          IRMapping mapping;

          // Map the float argument to the scaled float value
          Block &afterBlock = whileOp.getAfter().front();
          mapping.map(afterBlock.getArgument(pattern.afterArgIndex),
                      scaledFloat);

          // Map other arguments to the iter args
          unsigned iterArgIdx = 0;
          for (unsigned i = 0; i < afterBlock.getNumArguments(); ++i) {
            if (i != pattern.afterArgIndex) {
              mapping.map(afterBlock.getArgument(i), iterArgs[iterArgIdx++]);
            }
          }

          // Clone the after block contents (except the update and yield)
          SmallVector<Value> yieldValues;
          for (Operation &op : afterBlock.without_terminator()) {
            // Skip the float update operation
            if (&op == pattern.updateOp)
              continue;
            builder.clone(op, mapping);
          }

          // Collect yield values (excluding the float counter)
          // Yield operands correspond to before region args
          auto yieldOp = cast<scf::YieldOp>(afterBlock.getTerminator());
          for (unsigned i = 0; i < yieldOp.getResults().size(); ++i) {
            if (i != pattern.beforeArgIndex) {
              Value yielded = yieldOp.getResults()[i];
              if (mapping.contains(yielded))
                yieldValues.push_back(mapping.lookup(yielded));
              else
                yieldValues.push_back(yielded);
            }
          }

          builder.create<scf::YieldOp>(loc, yieldValues);
        });

    // Compute the final float value after the loop
    // For a completed loop, the value would be the limit (or just past it)
    Value finalIntVal;
    if (isCountingDown) {
      // After counting down, the final value is at intLimit
      finalIntVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(pattern.intLimit));
    } else {
      // After counting up, the final value is at ub (or just before based on
      // predicate)
      finalIntVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(ub));
    }
    Value finalFloat =
        rewriter.create<arith::SIToFPOp>(loc, pattern.floatType, finalIntVal);
    Value finalScaledFloat =
        rewriter.create<arith::MulFOp>(loc, finalFloat, scaleConst);

    // Build the replacement values
    // Results correspond to condition args, so use afterArgIndex
    SmallVector<Value> replacements;
    unsigned forResultIdx = 0;
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
      if (i == pattern.afterArgIndex) {
        replacements.push_back(finalScaledFloat);
      } else {
        replacements.push_back(forOp.getResult(forResultIdx++));
      }
    }

    rewriter.replaceOp(whileOp, replacements);
    return success();
  }

  /// Transform the while loop in-place to use an integer counter.
  LogicalResult transformWhileLoop(scf::WhileOp whileOp,
                                   const FloatLoopPattern &pattern,
                                   PatternRewriter &rewriter) const {
    Location loc = whileOp.getLoc();

    // Create the integer type for the counter
    IntegerType intType = rewriter.getI64Type();

    // Prepare new init values
    SmallVector<Value> newInits;
    SmallVector<Type> newResultTypes;

    for (unsigned i = 0; i < whileOp.getInits().size(); ++i) {
      if (i == pattern.beforeArgIndex) {
        // Replace float init with integer init
        Value intInit = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI64IntegerAttr(pattern.intInit));
        newInits.push_back(intInit);
        newResultTypes.push_back(intType);
      } else {
        newInits.push_back(whileOp.getInits()[i]);
        newResultTypes.push_back(whileOp.getResultTypes()[i]);
      }
    }

    // Create scale factor constant
    Value scaleConst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(pattern.floatType, pattern.scaleFactor));

    // Create integer limit constant
    Value intLimitConst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(pattern.intLimit));

    // Create integer step constant
    Value intStepConst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(pattern.intStep));

    // Get the integer predicate
    auto intPred = convertToCmpIPredicate(pattern.predicate);
    if (!intPred)
      return failure();

    // Create the new while loop
    auto newWhileOp = rewriter.create<scf::WhileOp>(
        loc, newResultTypes, newInits,
        /*beforeBuilder=*/
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          // Create integer comparison (no float conversion needed in before
          // region). Args are before region args, so use beforeArgIndex.
          Value intCounter = args[pattern.beforeArgIndex];
          Value cond = builder.create<arith::CmpIOp>(loc, *intPred, intCounter,
                                                     intLimitConst);

          // Build the condition operands
          SmallVector<Value> condArgs;
          for (unsigned i = 0; i < args.size(); ++i) {
            condArgs.push_back(args[i]);
          }

          builder.create<scf::ConditionOp>(loc, cond, condArgs);
        },
        /*afterBuilder=*/
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          // Compute the float value from the integer counter.
          // Args are after region args, so use afterArgIndex.
          Value intCounter = args[pattern.afterArgIndex];
          Value floatVal = builder.create<arith::SIToFPOp>(
              loc, pattern.floatType, intCounter);
          Value scaledFloat =
              builder.create<arith::MulFOp>(loc, floatVal, scaleConst);

          // Create mapping for cloning
          IRMapping mapping;
          Block &oldAfterBlock = whileOp.getAfter().front();

          // Map arguments (after region args use afterArgIndex)
          for (unsigned i = 0; i < args.size(); ++i) {
            if (i == pattern.afterArgIndex) {
              mapping.map(oldAfterBlock.getArgument(i), scaledFloat);
            } else {
              mapping.map(oldAfterBlock.getArgument(i), args[i]);
            }
          }

          // Clone operations except the update op
          for (Operation &op : oldAfterBlock.without_terminator()) {
            if (&op == pattern.updateOp)
              continue;
            builder.clone(op, mapping);
          }

          // Create the new yield with updated integer counter
          Value newIntCounter =
              builder.create<arith::AddIOp>(loc, intCounter, intStepConst);

          // Yield operands correspond to before region args, so use
          // beforeArgIndex
          SmallVector<Value> yieldValues;
          auto oldYieldOp = cast<scf::YieldOp>(oldAfterBlock.getTerminator());
          for (unsigned i = 0; i < oldYieldOp.getResults().size(); ++i) {
            if (i == pattern.beforeArgIndex) {
              yieldValues.push_back(newIntCounter);
            } else {
              Value yielded = oldYieldOp.getResults()[i];
              if (mapping.contains(yielded))
                yieldValues.push_back(mapping.lookup(yielded));
              else
                yieldValues.push_back(yielded);
            }
          }

          builder.create<scf::YieldOp>(loc, yieldValues);
        });

    // Create final float value from the integer result.
    // Results correspond to condition args, so use afterArgIndex.
    Value finalIntCounter = newWhileOp.getResult(pattern.afterArgIndex);
    Value finalFloatVal = rewriter.create<arith::SIToFPOp>(
        loc, pattern.floatType, finalIntCounter);
    Value finalScaledFloat =
        rewriter.create<arith::MulFOp>(loc, finalFloatVal, scaleConst);

    // Replace uses (results use afterArgIndex)
    SmallVector<Value> replacements;
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
      if (i == pattern.afterArgIndex) {
        replacements.push_back(finalScaledFloat);
      } else {
        replacements.push_back(newWhileOp.getResult(i));
      }
    }

    rewriter.replaceOp(whileOp, replacements);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

class SCFFloatStrengthReducePass
    : public mtrt::impl::SCFFloatStrengthReducePassBase<
          SCFFloatStrengthReducePass> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<FloatToIntStrengthReducePattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "Failed to run float to int strength reduction patterns";
      return signalPassFailure();
    }
  }
};

} // namespace
