//===- BoundsAnalysis.cpp -------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of Plan dialect dataflow analyses for shape/value bounds.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/Analysis/BoundsAnalysis.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/IR/PlanInterfaces.h"
#include "mlir-tensorrt/Interfaces/InferTensorValueRangeInterface.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/Support/Debug.h"
#include <limits>

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::plan;
using namespace mtrt::compiler;

#define DEBUG_TYPE "plan-bounds-analysis"
#define DBGS(x) llvm::dbgs() << " [" DEBUG_TYPE "][" x "] "

template <typename T>
std::optional<T> maybeGetFunctionArgBound(Value value, StringRef attrName) {
  BlockArgument source = dyn_cast<BlockArgument>(value);
  if (!source)
    return {};
  func::FuncOp func = dyn_cast<func::FuncOp>(source.getOwner()->getParentOp());
  if (!func)
    return {};
  auto shapeProfile = func.getArgAttrOfType<T>(source.getArgNumber(), attrName);
  if (!shapeProfile)
    return {};
  return shapeProfile;
}

static bool hasShapeFuncMarker(Value value, StringRef attrName) {
  BlockArgument source = dyn_cast<BlockArgument>(value);
  if (!source)
    return false;
  func::FuncOp func = dyn_cast<func::FuncOp>(source.getOwner()->getParentOp());
  if (!func)
    return false;
  Attribute attr = func->getAttr(attrName);
  if (attr)
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static FailureOr<SmallVector<ConstantIntRanges>>
intersectDimBoundsWithScalarBounds(
    ArrayRef<const IntegerValueRangeLattice *> scalarBounds,
    const ShapeBoundsLattice *dimBounds) {
  SmallVector<ConstantIntRanges> ranges;
  ranges.reserve(scalarBounds.size());

  for (unsigned i = 0, e = scalarBounds.size(); i < e; i++) {
    const IntegerValueRangeLattice *scalar = scalarBounds[i];

    bool scalarIsInvalidOrMaxRange =
        !scalar || scalar->getValue().isUninitialized();

    if (dimBounds && !dimBounds->getValue().isUninitialized()) {
      if (!scalarIsInvalidOrMaxRange) {
        ranges.push_back(scalar->getValue().getValue().intersection(
            dimBounds->getValue().getValue()[i]));
        continue;
      }
      ranges.push_back(dimBounds->getValue().getValue()[i]);
      continue;
    }

    if (!scalarIsInvalidOrMaxRange) {
      // If we only have the scalar to go off of, then we intersect the scalar
      // with positive range. This is only necessary due to issues in upstream
      // op's shape materialization functions. It is necessary
      // because the bounds calculation may be operating on values that can only
      // give course bounds. For, example, function argument host tensor inputs
      // for a slice "offset" and "limit" are given with individual ranges. the
      // calculation "limit - offset" to get the size of the slice can have a
      // negative lower bound. If the lower bound is negative, this just means
      // that it is possible for the func caller to give values within the input
      // limits where a zero-size slice is expected.
      unsigned numBits = scalar->getValue().getValue().smin().getBitWidth();
      ranges.push_back(scalar->getValue().getValue().intersection(
          ConstantIntRanges::fromSigned(APInt::getZero(numBits),
                                        APInt::getSignedMaxValue(numBits))));
      continue;
    }

    return failure();
  }
  return ranges;
}

//===----------------------------------------------------------------------===//
// ShapeBoundsForwardAnalysis
//===----------------------------------------------------------------------===//

void ShapeBoundsForwardAnalysis::setToEntryState(ShapeBoundsLattice *lattice) {
  ShapedType shapedType = dyn_cast<ShapedType>(lattice->getAnchor().getType());
  if (!shapedType)
    return propagateIfChanged(lattice, lattice->join(BoundsArray()));

  if (shapedType.hasStaticShape())
    return propagateIfChanged(
        lattice, lattice->join(BoundsArray::getMaxRangeForShapeBounds(
                     lattice->getAnchor())));

  std::optional<BoundsAttrInterface> shapeProfile =
      maybeGetFunctionArgBound<BoundsAttrInterface>(
          lattice->getAnchor(), plan::PlanDialect::getShapeBoundsAttrName());

  if (!shapeProfile)
    return propagateIfChanged(lattice, lattice->join(BoundsArray()));
  SmallVector<int64_t> minBound, maxBound;
  if (failed(shapeProfile->getShapeRange(minBound, maxBound)))
    return;
  return propagateIfChanged(
      lattice, lattice->join(BoundsArray::fromShapeBounds(minBound, maxBound)));
}

LogicalResult ShapeBoundsForwardAnalysis::visitOperation(
    Operation *op, ArrayRef<const ShapeBoundsLattice *> operands,
    ArrayRef<ShapeBoundsLattice *> results) {

  LLVM_DEBUG(DBGS("ShapeBoundsForwardAnalysis") << "visiting " << *op << "\n");

  auto joinCallback = [&](Value v, ArrayRef<ConstantIntRanges> attrs) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));
    ShapeBoundsLattice *lattice = results[result.getResultNumber()];
    const BoundsArray &oldRanges = lattice->getValue();
    BoundsArray newRange{llvm::to_vector(attrs)};

    LLVM_DEBUG(DBGS("ShapeBoundsForwardAnalysis")
               << "inferred " << newRange << " for\n\t" << v << "\n");

    ChangeResult changed = lattice->join(newRange);

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldRanges.isUninitialized() &&
        !(lattice->getValue() == oldRanges)) {
      LLVM_DEBUG(DBGS("ShapeBoundsForwardAnalysis")
                 << "Loop variant loop result detected\n");
      changed |= lattice->join(BoundsArray::getMaxRangeForShapeBounds(v));
    }
    propagateIfChanged(lattice, changed);
  };

  for (ShapeBoundsLattice *lattice : results) {
    if (auto rtt = dyn_cast<RankedTensorType>(lattice->getAnchor().getType())) {
      if (rtt.hasStaticShape())
        propagateIfChanged(lattice,
                           lattice->join(BoundsArray::getMaxRangeForShapeBounds(
                               lattice->getAnchor())));
    }
  }

  auto getScalarLatticeValues = [&](ValueRange scalars) {
    SmallVector<const IntegerValueRangeLattice *> result;
    result.reserve(scalars.size());
    for (Value v : scalars)
      result.emplace_back(this->getOrCreateFor<IntegerValueRangeLattice>(
          getProgramPointAfter(op), v));
    return result;
  };

  if (auto withOp = dyn_cast<plan::WithShapeOp>(op)) {
    FailureOr<SmallVector<ConstantIntRanges>> ranges =
        intersectDimBoundsWithScalarBounds(
            getScalarLatticeValues(withOp.getShape()), operands[0]);
    if (failed(ranges))
      return success();
    joinCallback(withOp.getResult(), *ranges);
    return success();
  }
  return success();
}

void ShapeBoundsForwardAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<ShapeBoundsLattice *> argLattices, unsigned firstIndex) {
  LLVM_DEBUG(DBGS("ShapeBoundsForwardAnalysis")
                 << "visiting non-control-flow arguments for ";
             op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
             llvm::dbgs() << "\n");
}

//===----------------------------------------------------------------------===//
// ShapeBoundsBackwardsAnalysis
//===----------------------------------------------------------------------===//

void ShapeBoundsBackwardsAnalysis::setToExitState(ShapeBoundsLattice *lattice) {
  LLVM_DEBUG(
      DBGS("ShapeBoundsBackwardsAnalysis") << "setting to exit state: ";
      lattice->getAnchor().print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n");

  auto shapedType = dyn_cast<ShapedType>(lattice->getAnchor().getType());
  if (!shapedType)
    return propagateIfChanged(lattice, lattice->join(BoundsArray()));

  if (shapedType.hasStaticShape())
    return propagateIfChanged(
        lattice, lattice->meet(BoundsArray::getMaxRangeForShapeBounds(
                     lattice->getAnchor())));

  return propagateIfChanged(lattice, lattice->join(BoundsArray()));
}

LogicalResult ShapeBoundsBackwardsAnalysis::visitOperation(
    Operation *op, ArrayRef<ShapeBoundsLattice *> operands,
    ArrayRef<const ShapeBoundsLattice *> results) {
  LLVM_DEBUG({
    DBGS("ShapeBoundsBackwardsAnalysis") << "visiting ";
    op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
    llvm::dbgs() << "\n";
  });

  auto getScalarLatticeValues = [&](ValueRange scalars) {
    SmallVector<const IntegerValueRangeLattice *> result;
    result.reserve(scalars.size());
    for (Value v : scalars)
      result.emplace_back(this->getOrCreateFor<IntegerValueRangeLattice>(
          getProgramPointAfter(op), v));
    return result;
  };

  if (auto withOp = dyn_cast<plan::WithShapeOp>(op)) {
    BoundsArray oldRanges = operands[0]->getValue();

    FailureOr<SmallVector<ConstantIntRanges>> ranges =
        intersectDimBoundsWithScalarBounds(
            getScalarLatticeValues(withOp.getShape()), results[0]);
    if (failed(ranges)) {
      propagateIfChanged(operands[0], operands[0]->meet(BoundsArray()));
      return success();
    }
    LLVM_DEBUG(DBGS("ShapeBoundsBackwardsAnalysis")
               << "inferred " << BoundsArray(*ranges) << " for\n\t" << withOp
               << "\n");

    ChangeResult changed = operands[0]->meet(BoundsArray(std::move(*ranges)));

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(withOp->getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>() &&
             isa<LoopLikeOpInterface>(op->getParentOp());
    });
    if (isYieldedResult && !operands[0]->getValue().isUninitialized() &&
        !(operands[0]->getValue() == oldRanges)) {
      LLVM_DEBUG(DBGS("ShapeBoundsBackwardsAnalysis")
                 << "Loop variant loop result detected\n");
      changed |= operands[0]->join(
          BoundsArray::getMaxRangeForShapeBounds(withOp.getOperand()));
    }
    propagateIfChanged(operands[0], changed);
  }
  return success();
}

void ShapeBoundsBackwardsAnalysis::visitBranchOperand(OpOperand &operand) {
  LLVM_DEBUG(DBGS("ShapeBoundsBackwardsAnalysis")
                 << "visiting non-forwarded branch operand ";
             operand.get().print(llvm::dbgs(), OpPrintingFlags().skipRegions());
             llvm::dbgs() << "\n");
}
void ShapeBoundsBackwardsAnalysis::visitCallOperand(OpOperand &operand) {
  LLVM_DEBUG(DBGS("ShapeBoundsBackwardsAnalysis")
                 << "visiting non-forwarded call operand ";
             operand.get().print(llvm::dbgs(), OpPrintingFlags().skipRegions());
             llvm::dbgs() << "\n");
}

//===----------------------------------------------------------------------===//
// ShapeIntegerRangeAnalysis
//===----------------------------------------------------------------------===//

static std::optional<ConstantIntRanges>
maybeGetValueBounds(Value value, ArrayRef<uint64_t> coordinate = {}) {
  RankedTensorType rtt = cast<RankedTensorType>(value.getType());
  Type elType = rtt.getElementType();
  assert(elType.isSignlessIntOrIndex() && "expected integer or index type");
  std::optional<BoundsAttrInterface> bound =
      maybeGetFunctionArgBound<BoundsAttrInterface>(
          value, plan::PlanDialect::getValueBoundsAttrName());
  if (!bound)
    return {};

  if (!coordinate.empty())
    return bound->getIntegerValueRangeAtCoordinate(coordinate);
  if (rtt.getNumElements() == 1)
    return bound->getIntegerValueRangeAtCoordinate(
        SmallVector<uint64_t>(rtt.getRank(), 0));

  std::optional<SmallVector<ConstantIntRanges>> ranges =
      bound->getIntegerValueRanges();
  if (!ranges || ranges->empty())
    return {};
  ConstantIntRanges r = ranges->front();
  for (const auto &it : *ranges)
    r = r.rangeUnion(it);
  return r;
}

static ConstantIntRanges
getMaxDimRange(unsigned bitWidth = IndexType::kInternalStorageBitWidth) {
  return ConstantIntRanges::fromSigned(
      APInt(bitWidth, 0), APInt(bitWidth, std::numeric_limits<int32_t>::max()));
}
static ConstantIntRanges getMaxValueRange(unsigned bitWidth) {
  return ConstantIntRanges::fromSigned(APInt::getSignedMinValue(bitWidth),
                                       APInt::getSignedMaxValue(bitWidth));
}

static ConstantIntRanges truncateToNonNegative(const ConstantIntRanges &lhs) {
  APInt zero(lhs.smin().getBitWidth(), 0);
  const APInt &smin = lhs.smin().sgt(zero) ? lhs.smin() : zero;
  return ConstantIntRanges::fromSigned(smin, lhs.smax());
}

/// If the `extractOp` indices can be trivially folded to constants, then
/// return the equivalent static linear index.
static std::optional<SmallVector<uint64_t>>
getStaticCoordinates(tensor::ExtractOp extractOp) {
  if (!extractOp.getTensor().getType().hasStaticShape())
    return {};

  SmallVector<uint64_t> indices;
  for (Value v : extractOp.getIndices()) {
    APInt index;
    if (!matchPattern(v, m_ConstantInt(&index)))
      return {};
    indices.push_back(index.getZExtValue());
  }
  return indices;
}

/// Infer the integer range bounds for the result of a tensor.dim operation.
/// This function analyzes the dimension being queried and returns appropriate
/// bounds based on the tensor's shape information.
///
/// The analysis considers several cases:
/// 1. If the dimension index is not constant and the tensor doesn't have a
///    static shape, returns maximum possible dimension range.
/// 2. If the dimension index is not constant but the tensor has a static shape,
///    returns the range [min_dim, max_dim] across all dimensions.
/// 3. If the dimension index is constant and refers to a static dimension,
///    returns the exact static size as a singleton range.
/// 4. If the dimension index is constant and refers to a dynamic dimension,
///    attempts to use shape bounds from function argument attributes if
///    available.
///
/// \param dimOp The tensor.dim operation to analyze
/// \param ranges Input ranges for the operands (unused in this implementation)
/// \param setResultRanges Callback to set the computed range for the result
static void inferResultRanges(tensor::DimOp dimOp,
                              ArrayRef<ConstantIntRanges> ranges,
                              SetIntRangeFn setResultRanges) {
  TensorType tensorType = dimOp.getSource().getType();

  std::optional<int64_t> staticDimNum =
      getConstantIntValue(dimOp.getDimension());
  // Case 1: We cannot determine which dimension is being queried.
  if (!staticDimNum) {
    // Case 1.1: The tensor has a dynamic shape, no bounds can be inferred.
    if (!tensorType.hasStaticShape())
      return setResultRanges(dimOp.getResult(), getMaxDimRange());
    // Case 1.2: The tensor has a static shape, we can infer a bound by reducing
    // across all dimensions.
    auto shape = tensorType.getShape();
    int64_t minVal = *std::min_element(shape.begin(), shape.end());
    int64_t maxVal = *std::max_element(shape.begin(), shape.end());
    return setResultRanges(
        dimOp.getResult(),
        ConstantIntRanges::fromSigned(
            APInt(IndexType::kInternalStorageBitWidth, minVal),
            APInt(IndexType::kInternalStorageBitWidth, maxVal)));
  }

  // Case 2: The dimension queried is statically known.
  if (!tensorType.isDynamicDim(*staticDimNum)) {
    APInt intStatic(IndexType::kInternalStorageBitWidth,
                    tensorType.getDimSize(*staticDimNum));
    return setResultRanges(dimOp.getResult(), ConstantIntRanges::fromUnsigned(
                                                  intStatic, intStatic));
  }

  // Case 3: The dimension queried is dynamic, we can use shape bounds from
  // function argument attributes if available, otherwise we cannot infer any
  // bounds.
  std::optional<BoundsAttrInterface> shapeProfile =
      maybeGetFunctionArgBound<BoundsAttrInterface>(
          dimOp.getSource(), plan::PlanDialect::getShapeBoundsAttrName());
  if (!shapeProfile)
    return setResultRanges(dimOp.getResult(), getMaxDimRange());

  SmallVector<int64_t> minBound, maxBound;
  if (failed(shapeProfile->getShapeRange(minBound, maxBound)))
    return setResultRanges(dimOp.getResult(), getMaxDimRange());

  setResultRanges(
      dimOp.getResult(),
      ConstantIntRanges::fromSigned(
          APInt(IndexType::kInternalStorageBitWidth, minBound[*staticDimNum]),
          APInt(IndexType::kInternalStorageBitWidth, maxBound[*staticDimNum])));
}

static void inferResultRanges(tensor::ExtractOp extractOp,
                              ArrayRef<ConstantIntRanges> ranges,
                              SetIntRangeFn setResultRanges) {

  assert(extractOp.getResult().getType().isIntOrIndex() &&
         "expected index or integer type result");
  TensorType operandType = extractOp.getTensor().getType();
  Type elType = extractOp.getType();
  if (!elType.isSignlessIntOrIndex())
    return;
  unsigned bitWidth = elType.isIndex() ? IndexType::kInternalStorageBitWidth
                                       : elType.getIntOrFloatBitWidth();
  if (!operandType.hasStaticShape())
    return setResultRanges(extractOp.getResult(), getMaxValueRange(bitWidth));

  std::optional<SmallVector<uint64_t>> coordinates =
      getStaticCoordinates(extractOp);
  std::optional<ConstantIntRanges> intRange = maybeGetValueBounds(
      extractOp.getTensor(), coordinates ? *coordinates : ArrayRef<uint64_t>{});
  if (!intRange)
    return setResultRanges(extractOp.getResult(), getMaxValueRange(bitWidth));

  return setResultRanges(extractOp.getResult(), *intRange);
}

/// At an entry point, we cannot reason about integer value ranges except if
/// we are in a function where the bounds are encoded into the arguments.
void ShapeIntegerRangeAnalysis::setToEntryState(
    IntegerValueRangeLattice *lattice) {
  bool isShapeFunc = hasShapeFuncMarker(lattice->getAnchor(),
                                        PlanDialect::kShapeFuncMarkerAttrName);
  // No need to perform integer range analysis for shape func
  if (isShapeFunc) {
    LLVM_DEBUG(DBGS(
        "Skipping shape integer analysis for a shape function. "
        "We can not reason about integer value range for a shape function"));
    return propagateIfChanged(
        lattice,
        lattice->join(IntegerValueRange::getMaxRange(lattice->getAnchor())));
  }

  bool hasShapeUser =
      llvm::any_of(lattice->getAnchor().getUsers(), [&](Operation *user) {
        return isa<plan::WithShapeOp>(user);
      });

  if (!lattice->getAnchor().getType().isIntOrIndex())
    return propagateIfChanged(lattice, lattice->join(IntegerValueRange()));

  std::optional<BoundsAttrInterface> shapeProfile =
      maybeGetFunctionArgBound<BoundsAttrInterface>(
          lattice->getAnchor(), plan::PlanDialect::getValueBoundsAttrName());
  std::optional<ConstantIntRanges> intRange =
      shapeProfile ? shapeProfile->getIntegerValueRange() : std::nullopt;
  if (!intRange) {
    IntegerValueRange range =
        IntegerValueRange::getMaxRange(lattice->getAnchor());
    if (hasShapeUser)
      range = IntegerValueRange(truncateToNonNegative(range.getValue()));
    return propagateIfChanged(lattice, lattice->join(range));
  }
  return propagateIfChanged(lattice,
                            lattice->join(IntegerValueRange(*intRange)));
}

/// Visit an operation. Invoke the transfer function on each operation that
/// implements `InferIntRangeInterface`.
LogicalResult ShapeIntegerRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerValueRangeLattice *> operands,
    ArrayRef<IntegerValueRangeLattice *> results) {
  for (const IntegerValueRangeLattice *lattice : operands) {
    if (lattice->getAnchor().getType().isIntOrIndex() &&
        lattice->getValue().isUninitialized())
      return success();
  }

  SmallVector<ConstantIntRanges> argRanges(
      llvm::map_range(operands, [](const IntegerValueRangeLattice *val) {
        if (!val->getAnchor().getType().isIntOrIndex())
          return ConstantIntRanges::maxRange(64);
        return val->getValue().getValue();
      }));

  /// The join callback, reproduced from
  /// third_party/llvm-project/mlir/lib/Analysis/DataFlow/IntegerRangeAnalysis.cpp.
  auto setResultRanges = [&](Value v, const ConstantIntRanges &range) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    IntegerValueRangeLattice *lattice = results[result.getResultNumber()];
    IntegerValueRange oldRange = lattice->getValue();
    ConstantIntRanges baseRange = range;
    if (llvm::any_of(v.getUsers(), [](Operation *user) {
          return isa<plan::WithShapeOp>(user);
        }))
      baseRange = truncateToNonNegative(baseRange);

    LLVM_DEBUG(DBGS("ShapeIntegerRange")
               << " inferred range " << baseRange << " for:\n\t" << v << "\n");

    ChangeResult changed = lattice->join(IntegerValueRange{baseRange});

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldRange.isUninitialized() &&
        !(lattice->getValue() == oldRange)) {
      LLVM_DEBUG(DBGS("ShapeIntegerRange")
                 << "Loop variant loop result detected\n");
      changed |= lattice->join(IntegerValueRange::getMaxRange(v));
    }
    propagateIfChanged(lattice, changed);
  };

  if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
    if (!extractOp.getType().isIntOrIndex()) {
      propagateIfChanged(results[0], results[0]->join(IntegerValueRange()));
      return success();
    }
    inferResultRanges(extractOp, argRanges, setResultRanges);
    return success();
  }
  if (auto dimOp = dyn_cast<tensor::DimOp>(op)) {
    inferResultRanges(dimOp, argRanges, setResultRanges);
    return success();
  }

  return IntegerRangeAnalysis::visitOperation(op, operands, results);
}

//===----------------------------------------------------------------------===//
// TensorValueBoundsAnalysis
//===----------------------------------------------------------------------===//

bool TensorValueBoundsAnalysis::shouldAnalyzeValueBounds(Type type) {
  return BoundsArray::shouldAnalyzeValueBounds(type);
}

bool TensorValueBoundsAnalysis::shouldAnalyzeValueBounds(Value value) {
  return BoundsArray::shouldAnalyzeValueBounds(value);
}

void TensorValueBoundsAnalysis::setToEntryState(
    TensorValueBoundsLattice *lattice) {
  Value point = lattice->getAnchor();
  if (!shouldAnalyzeValueBounds(point))
    return propagateIfChanged(lattice, lattice->join(BoundsArray()));

  std::optional<BoundsAttrInterface> shapeProfile =
      maybeGetFunctionArgBound<BoundsAttrInterface>(
          point, plan::PlanDialect::getValueBoundsAttrName());
  if (!shapeProfile)
    return propagateIfChanged(lattice, lattice->join(BoundsArray()));

  std::optional<SmallVector<ConstantIntRanges>> ranges =
      shapeProfile->getIntegerValueRanges();
  if (!ranges || ranges->empty())
    return propagateIfChanged(lattice, lattice->join(BoundsArray()));

  return propagateIfChanged(lattice, lattice->join(BoundsArray(*ranges)));
}

LogicalResult TensorValueBoundsAnalysis::visitOperation(
    Operation *op, ArrayRef<const TensorValueBoundsLattice *> operands,
    ArrayRef<TensorValueBoundsLattice *> results) {

  if (!isa<InferTensorValueRangeInterface>(op) &&
      !op->hasTrait<OpTrait::ConstantLike>()) {
    setAllToEntryStates(results);
    return success();
  }

  LLVM_DEBUG(DBGS("TensorValueBoundsAnalysis") << "visiting " << *op << "\n");

  SmallVector<IntOrTensorValueRange> argRanges;
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (isa<RankedTensorType>(operand.getType()) &&
        shouldAnalyzeValueBounds(operand)) {
      if (operands[idx])
        argRanges.emplace_back(&operands[idx]->getValue());
      else
        argRanges.emplace_back(nullptr);
      continue;
    }

    if (isa<IntegerType, IndexType>(operand.getType())) {
      const auto *scalarLattice =
          this->getOrCreateFor<IntegerValueRangeLattice>(
              getProgramPointAfter(op), operand);
      if (scalarLattice) {
        argRanges.emplace_back(&scalarLattice->getValue());
      } else {
        argRanges.emplace_back(nullptr);
      }
      continue;
    }

    setAllToEntryStates(results);
    return success();
  }

  auto joinCallback = [&](Value v, BoundsArray newRange) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    TensorValueBoundsLattice *lattice = results[result.getResultNumber()];
    const BoundsArray &oldRanges = lattice->getValue();

    LLVM_DEBUG(DBGS("TensorValueBoundsAnalysis")
               << "inferred " << newRange << " for\n\t" << v << "\n");

    ChangeResult changed = lattice->join(newRange);

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldRanges.isUninitialized() &&
        !(lattice->getValue() == oldRanges)) {
      LLVM_DEBUG(DBGS("TensorValueBoundsAnalysis")
                 << "Loop variant loop result detected\n");
      changed |= lattice->join(BoundsArray::getMaxRangeForValueBounds(v));
    }
    propagateIfChanged(lattice, changed);
  };

  if (op->hasTrait<OpTrait::ConstantLike>() && op->getNumResults() == 1) {
    // If the value is produced by constant op, populate ranges appropriately.
    // NOTE: we should instead use the mechanism from ConstantIntRanges lattice?
    Value point = results[0]->getAnchor();
    if (!shouldAnalyzeValueBounds(point)) {
      setAllToEntryStates(results);
      return success();
    }
    DenseIntElementsAttr attr;
    if (!matchPattern(point, m_Constant(&attr))) {
      setAllToEntryStates(results);
      return success();
    }
    joinCallback(point, BoundsArray::getFromConstantValue(attr));
    return success();
  }

  auto inferrable = dyn_cast<InferTensorValueRangeInterface>(op);
  if (!inferrable) {
    setAllToEntryStates(results);
    return success();
  }

  inferrable.inferResultRangesFromOptional(argRanges, joinCallback);
  return success();
}

void TensorValueBoundsAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<TensorValueBoundsLattice *> argLattices, unsigned firstIndex) {}
