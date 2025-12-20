//===- BufferResultsToOutParams.cpp ---------------------------------------===//
//
// Modified from upstream 'BufferResultsToOutParams.cpp', part of the LLVM
// Project, under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Changes: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Implementation of the Plan buffer results to out params pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

namespace mlir::plan {
#define GEN_PASS_DEF_PLANBUFFERRESULTSTOOUTPARAMSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::bufferization;

/// Visit all the return ops in the function. If the visitTerminator returns
/// failure, then the traversal is interrupted.
static LogicalResult
visitReturnOps(func::FuncOp func,
               function_ref<LogicalResult(func::ReturnOp)> visitTerminator) {
  for (Block &block : func.getBody()) {
    if (auto terminator = dyn_cast<func::ReturnOp>(block.getTerminator())) {
      if (failed(visitTerminator(terminator)))
        return failure();
    }
  }
  return success();
}

/// Return a vector which maps results to block argument indices if, for each
/// returned value index, all `return` ops in the func uniformly return the same
/// block argument.
static SmallVector<std::optional<unsigned>>
getReturnedBlockArgs(func::FuncOp func) {
  SmallVector<std::optional<unsigned>> returnsBlockArgs(func.getNumResults(),
                                                        std::nullopt);
  for (unsigned i = 0, e = func.getNumResults(); i < e; ++i) {
    if (failed(visitReturnOps(func, [&](func::ReturnOp op) -> LogicalResult {
          auto blockArg = dyn_cast<BlockArgument>(op.getOperand(i));
          if (!blockArg || blockArg.getOwner()->getParentOp() != func)
            return success();
          std::optional<unsigned> &blockArgIdx = returnsBlockArgs[i];
          if (!blockArgIdx) {
            blockArgIdx = blockArg.getArgNumber();
            return success();
          }
          if (*blockArgIdx != blockArg.getArgNumber())
            return failure();
          return success();
        }))) {
      returnsBlockArgs[i] = std::nullopt;
      continue;
    }
  }
  return returnsBlockArgs;
}

/// Given a memref value, return the "base" value by skipping over all
/// ViewLikeOpInterface ops (if any) in the reverse use-def chain.
static Value getViewBase(Value value) {
  while (auto viewLikeOp = value.getDefiningOp<ViewLikeOpInterface>())
    value = viewLikeOp.getViewSource();
  return value;
}

/// Return "true" if the given values are guaranteed to be different (and
/// non-aliasing) allocations based on the fact that one value is the result
/// of an allocation and the other value is a block argument of a parent block.
/// Note: This is a best-effort analysis that will eventually be replaced by a
/// proper "is same allocation" analysis. This function may return "false" even
/// though the two values are distinct allocations.
static bool distinctAllocAndBlockArgument(Value v1, Value v2) {
  Value v1Base = getViewBase(v1);
  Value v2Base = getViewBase(v2);
  auto areDistinct = [](Value v1, Value v2) {
    if (Operation *op = v1.getDefiningOp())
      if (hasEffect<MemoryEffects::Allocate>(op, v1))
        if (auto bbArg = dyn_cast<BlockArgument>(v2))
          if (bbArg.getOwner()->findAncestorOpInBlock(*op))
            return true;
    return false;
  };
  return areDistinct(v1Base, v2Base) || areDistinct(v2Base, v1Base);
}

/// Checks if `memref` may potentially alias a MemRef in `otherList`. It is
/// often a requirement of optimization patterns that there cannot be any
/// aliasing memref in order to perform the desired simplification.
static bool potentiallyAliasesMemref(BufferOriginAnalysis &analysis,
                                     ValueRange otherList, Value memref) {
  for (auto other : otherList) {
    if (!isa<BaseMemRefType>(other.getType()))
      continue;
    if (distinctAllocAndBlockArgument(other, memref))
      continue;
    std::optional<bool> analysisResult =
        analysis.isSameAllocation(other, memref);
    if (!analysisResult.has_value() || analysisResult == true)
      return true;
  }
  return false;
}

/// Check whether a MemRefValue is produced by a set of "hoistable" operations.
/// The operations are hoistable if they are composed of a closed (less function
/// arguments) set of operations involving an allocation and zero or more pure
/// operations.
static FailureOr<llvm::SetVector<Operation *>>
getHoistableOperations(Value value, func::FuncOp func) {
  llvm::SetVector<Operation *> slice;
  BackwardSliceOptions sliceOptions{};
  sliceOptions.omitBlockArguments = false;
  sliceOptions.omitUsesFromAbove = false;
  sliceOptions.inclusive = true;
  sliceOptions.filter = [](Operation *op) {
    return isa<AllocationOpInterface>(op) || mlir::isPure(op);
  };
  if (failed(mlir::getBackwardSlice(value, &slice, sliceOptions)))
    return failure();

  // Slice must be closed.
  bool hasAllocation = false;
  for (Operation *op : slice) {
    if (isa<AllocationOpInterface>(op))
      hasAllocation = true;
    for (Value operand : op->getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (blockArg.getOwner()->getParentOp() == func)
          continue;
      }
      if (!slice.contains(operand.getDefiningOp()))
        return failure();
    }
  }

  if (slice.empty() || !hasAllocation)
    return failure();

  // Now check that we can replace all uses outside the set with a single block
  // argument (the value currently being returned).
  auto isInCluster = [&](Operation *op) {
    if (slice.contains(op))
      return true;
    while (Operation *parent = op->getParentOp()) {
      if (parent == func)
        return false;
      if (slice.contains(parent))
        return true;
    }
    return false;
  };
  for (Operation *op : slice) {
    for (OpOperand &use : op->getUses()) {
      if (use.get() != value && !isInCluster(use.getOwner()))
        return failure();
    }
  }

  return slice;
}

/// Return true if the result at `resultIdx` can be promoted to an argument. A
/// result can be promoted to an argument if:
/// - It is a MemRefType
/// - For all return ops in the function, it does not potentially alias any
///   other returned operand.
///
/// The specific method in which the result is promoted to an argument (e.g.
/// via hoisting allocation or via copy insertion) is not determined here.
static bool checkAliasingPrecondition(func::FuncOp func, unsigned resultIdx,
                                      BufferOriginAnalysis &analysis) {
  MemRefType resultType =
      dyn_cast<MemRefType>(func.getResultTypes()[resultIdx]);
  if (!resultType)
    return false;

  return succeeded(visitReturnOps(func, [&](func::ReturnOp term) {
    SmallVector<Value> otherOperands(term->getOperands());
    otherOperands.erase(otherOperands.begin() + resultIdx);
    return success(!potentiallyAliasesMemref(analysis, otherOperands,
                                             term.getOperand(resultIdx)));
  }));
}

namespace {
struct ResultPromotionPlan {

  /// Specifies which return values can be dropped and replaced with a block
  /// argument without inserting any copies by inserting a single
  /// statically-sized allocation at each call site.
  BitVector simplyHoistableAllocations;
  /// Specifies which return values can be dropped and replaced with a block
  /// argument by cloning a tree of hoistable operations at each call site.
  /// This requires that each returned operand at the corresponding position in
  /// all return ops is the same value.
  BitVector hoistableAllocations;

  /// Specifies which return values can be dropped by inserting a new block
  /// argument + a copy at each function return.
  BitVector promotableToCopyOut;

  /// Specifies which return values are already block arguments.
  SmallVector<std::optional<unsigned>> returnsExistingBlockArg;

  /// The union of `simplyHoistableAllocations`, `hoistableAllocations`, and
  /// `returnsExistingBlockArg`.
  BitVector resultsToDrop;

  /// Specifies the tree of operations to hoist for each return value that is
  /// set true in `hoistableAllocations`.
  SmallVector<SetVector<Operation *>> operationsToHoist;
};
} // namespace

/// Returns true if a returned value is "simply hoistable", meaning it is
/// directly produced by an allocation op and has static shape and identity
/// layout.
static bool isSimplyHoistableAllocation(Value value) {
  return isa_and_nonnull<bufferization::AllocationOpInterface>(
             value.getDefiningOp()) &&
         cast<MemRefType>(value.getType()).hasStaticShape() &&
         cast<MemRefType>(value.getType()).getLayout().isIdentity();
}

/// Constructs a "ResultPromotionPlan" by identifying all results which can be
/// dropped and passed as MemRef arguments instead. It updates the func op type
/// and entry block arguments.
static FailureOr<ResultPromotionPlan>
updateFuncOp(RewriterBase &rewriter, func::FuncOp func,
             SmallVectorImpl<BlockArgument> &appendedEntryArgs,
             BufferOriginAnalysis &analysis) {
  auto functionType = func.getFunctionType();

  ResultPromotionPlan plan{/*simplyHoistableAllocations=*/
                           BitVector(functionType.getNumResults(), false),
                           /*hoistableAllocations=*/
                           BitVector(functionType.getNumResults(), false),
                           /*promotableToCopyOut=*/
                           BitVector(functionType.getNumResults(), false),
                           /*returnsExistingBlockArg=*/
                           getReturnedBlockArgs(func),
                           /*resultsToPromote=*/
                           BitVector(functionType.getNumResults(), false),
                           SmallVector<SetVector<Operation *>>()};

  // Collect information about the results will become appended arguments.
  SmallVector<Type> newBlockArgTypes;
  for (auto [idx, resultType] : llvm::enumerate(functionType.getResults())) {

    auto memrefType = dyn_cast<MemRefType>(resultType);
    if (!memrefType)
      continue;

    if (plan.returnsExistingBlockArg[idx]) {
      plan.resultsToDrop.set(idx);
      continue;
    }

    if (!checkAliasingPrecondition(func, idx, analysis)) {
      continue;
    }

    std::optional<SetVector<Operation *>> hoistableOperations{};
    bool isSimplyHoistable = true;
    if (failed(visitReturnOps(
            func,
            [&, idx = idx](func::ReturnOp term) {
              isSimplyHoistable &=
                  isSimplyHoistableAllocation(term.getOperand(idx));
              if (isSimplyHoistable)
                return success();
              FailureOr<SetVector<Operation *>> tmp =
                  getHoistableOperations(term.getOperand(idx), func);
              if (failed(tmp))
                return failure();
              if (!hoistableOperations) {
                hoistableOperations = std::move(*tmp);
                return success();
              }
              return success(*hoistableOperations == *tmp);
            })) ||
        (!isSimplyHoistable && !hoistableOperations)) {
      if (memrefType.hasStaticShape() && memrefType.getLayout().isIdentity()) {
        plan.promotableToCopyOut.set(idx);
        plan.resultsToDrop.set(idx);
        newBlockArgTypes.push_back(resultType);
      }
      continue;
    }

    if (isSimplyHoistable) {
      plan.simplyHoistableAllocations.set(idx);
      plan.resultsToDrop.set(idx);
      newBlockArgTypes.push_back(resultType);
      continue;
    }

    plan.operationsToHoist.emplace_back(std::move(*hoistableOperations));
    plan.hoistableAllocations.set(idx);
    plan.resultsToDrop.set(idx);
    newBlockArgTypes.push_back(resultType);
  }

  // Add the new arguments to the function type.
  auto newArgTypes = llvm::to_vector(
      llvm::concat<const Type>(functionType.getInputs(), newBlockArgTypes));
  auto newFunctionType = FunctionType::get(func.getContext(), newArgTypes,
                                           functionType.getResults());

  rewriter.modifyOpInPlace(func, [&]() { func.setType(newFunctionType); });

  // Transfer the result attributes to arg attributes.
  unsigned newArgIdx = functionType.getNumInputs();
  for (auto [idx, erasedResultIdx] :
       llvm::enumerate(plan.resultsToDrop.set_bits())) {
    if (plan.returnsExistingBlockArg[erasedResultIdx])
      continue;
    func.setArgAttrs(newArgIdx, func.getResultAttrs(erasedResultIdx));
    newArgIdx++;
  }

  // Erase the results. This takes care of updating the result attributes array.
  LogicalResult result = success();
  rewriter.modifyOpInPlace(
      func, [&]() { result = func.eraseResults(plan.resultsToDrop); });
  if (failed(result))
    return failure();

  // Add the new arguments to the entry block if the function is not external.
  if (func.isExternal())
    return plan;

  Location loc = func.getLoc();
  rewriter.modifyOpInPlace(func, [&]() {
    for (Type type : newBlockArgTypes)
      appendedEntryArgs.push_back(func.front().addArgument(type, loc));
  });

  return plan;
}

/// Updates all ReturnOps in the scope of the given func::FuncOp by either
/// keeping them as return values or dropping the return value, replacing uses
/// and inserting copies as required.
static LogicalResult
updateReturnOps(RewriterBase &rewriter, func::FuncOp func,
                ArrayRef<BlockArgument> appendedEntryArgs,
                ResultPromotionPlan &plan,
                bufferization::BufferResultsToOutParamsOpts &options) {
  OpBuilder::InsertionGuard g(rewriter);

  return visitReturnOps(func, [&](func::ReturnOp term) -> LogicalResult {
    rewriter.setInsertionPoint(term);
    SmallVector<Value> keepAsReturnOperands;
    llvm::SmallDenseMap<Value, BlockArgument> copyIntoOutParams;
    llvm::SmallDenseMap<Value, BlockArgument> valuesToHoist;
    unsigned appendedEntryArgIdx = 0;

    for (auto [idx, operand] : llvm::enumerate(term.getOperands())) {
      if (plan.resultsToDrop.test(idx)) {
        if (plan.hoistableAllocations.test(idx) ||
            plan.simplyHoistableAllocations.test(idx)) {
          valuesToHoist[operand] = appendedEntryArgs[appendedEntryArgIdx++];
        } else if (plan.promotableToCopyOut.test(idx)) {
          copyIntoOutParams[operand] = appendedEntryArgs[appendedEntryArgIdx++];
        }
        continue;
      }
      keepAsReturnOperands.push_back(operand);
    }

    for (auto [hoistable, appendedEntryArg] : valuesToHoist)
      rewriter.replaceAllUsesWith(hoistable, appendedEntryArg);

    for (auto [orig, arg] : copyIntoOutParams) {
      if (failed(options.memCpyFn(rewriter, term.getLoc(), orig, arg)))
        return failure();
    }

    rewriter.modifyOpInPlace(term, [&]() {
      term.getOperandsMutable().assign(keepAsReturnOperands);
    });

    return success();
  });
}

/// Updates all CallOps in the scope of the given ModuleOp by allocating
/// temporary buffers for newly introduced out params or cloning the required
/// operations to produce the new output buffer.
static LogicalResult
updateCalls(RewriterBase &rewriter, func::FuncOp func,
            const ResultPromotionPlan &plan, const SymbolUserMap &symbolUserMap,
            const bufferization::BufferResultsToOutParamsOpts &options) {
  OpBuilder::InsertionGuard g(rewriter);
  for (auto symbolUser : symbolUserMap.getUsers(func)) {
    auto call = dyn_cast<func::CallOp>(symbolUser);
    if (!call)
      continue;
    SmallVector<Type> newResultTypes;
    SmallVector<Value> newOperands(call.getOperands());
    SmallVector<Value> replaceWithNewCallResults;
    rewriter.setInsertionPoint(call);
    auto hoistableOpsIt = plan.operationsToHoist.begin();
    for (auto [idx, result] : llvm::enumerate(call.getResults())) {
      if (plan.resultsToDrop.test(idx)) {
        auto memrefType = cast<MemRefType>(result.getType());
        if (plan.promotableToCopyOut.test(idx) ||
            plan.simplyHoistableAllocations.test(idx)) {
          FailureOr<Value> maybeOutParam = options.allocationFn(
              rewriter, call.getLoc(), memrefType, ValueRange{});
          if (failed(maybeOutParam))
            return call.emitError()
                   << "failed to create allocation when promoting "
                      "a buffer result to an output parameter";
          rewriter.replaceAllUsesWith(result, *maybeOutParam);
          newOperands.push_back(*maybeOutParam);
          continue;
        }
        if (plan.hoistableAllocations.test(idx)) {
          const SetVector<Operation *> &hoistableOps = *hoistableOpsIt++;
          assert(!hoistableOps.empty() && "hoistableOps is empty");
          IRMapping mapping;
          mapping.map(func.getArguments(), call.getOperands());
          for (Operation *op : hoistableOps)
            rewriter.clone(*op, mapping);
          Value operandReplacement =
              mapping.lookup(hoistableOps.back()->getResult(0));
          rewriter.replaceAllUsesWith(result, operandReplacement);
          newOperands.push_back(operandReplacement);
          continue;
        }
        if (std::optional<unsigned> existingBlockArg =
                plan.returnsExistingBlockArg[idx]) {
          rewriter.replaceAllUsesWith(result,
                                      call.getOperand(*existingBlockArg));
          continue;
        }
        llvm_unreachable("unhandled case");
      }
      newResultTypes.push_back(result.getType());
      replaceWithNewCallResults.push_back(result);
    }

    auto newCall = rewriter.create<func::CallOp>(
        call.getLoc(), call.getCalleeAttr(), newResultTypes, newOperands);
    for (auto [valueToReplace, replacement] :
         llvm::zip_equal(replaceWithNewCallResults, newCall.getResults()))
      rewriter.replaceAllUsesWith(valueToReplace, replacement);
    rewriter.eraseOp(call);
  }

  return success();
}

namespace {
struct PlanBufferResultsToOutParamsPass
    : public plan::impl::PlanBufferResultsToOutParamsPassBase<
          PlanBufferResultsToOutParamsPass> {
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override {
    options.allocationFn = [](OpBuilder &builder, Location loc, MemRefType type,
                              ValueRange dynShape) -> FailureOr<Value> {
      return builder.create<memref::AllocOp>(loc, type, dynShape).getResult();
    };

    options.memCpyFn = [](OpBuilder &builder, Location loc, Value src,
                          Value dst) -> LogicalResult {
      builder.create<memref::CopyOp>(loc, src, dst);
      return success();
    };
    options.filterFn = [&](func::FuncOp *op) {
      if (ignorePublicFunctions && op->isPublic())
        return false;
      if (executor::abi::isABIWrapperFunction(*op))
        return false;
      return !op->isDeclaration();
    };

    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTableCollection symbolTables;
    SymbolUserMap symbolUserMap(symbolTables, module);
    for (auto func : module.getOps<func::FuncOp>()) {
      if (!options.filterFn(&func))
        continue;

      BufferOriginAnalysis analysis(func);

      IRRewriter rewriter(func);
      SmallVector<BlockArgument, 6> appendedEntryArgs;

      FailureOr<ResultPromotionPlan> updatePlan =
          updateFuncOp(rewriter, func, appendedEntryArgs, analysis);
      if (failed(updatePlan))
        return signalPassFailure();

      if (func.isExternal())
        continue;

      if (failed(updateReturnOps(rewriter, func, appendedEntryArgs, *updatePlan,
                                 options)))
        return signalPassFailure();

      if (failed(
              updateCalls(rewriter, func, *updatePlan, symbolUserMap, options)))
        return signalPassFailure();
    }
  }

private:
  bufferization::BufferResultsToOutParamsOpts options;
};
} // namespace
