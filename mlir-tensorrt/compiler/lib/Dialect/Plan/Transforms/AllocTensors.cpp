//===- AllocTensors.cpp  --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
///  Implementation of the `plan-alloc-tensors` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/ModuleBufferization/ModuleBufferization.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Utils/ModuleUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "plan-alloc-tensors"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "
#define DBGF(fmt, ...)                                                         \
  LLVM_DEBUG(                                                                  \
      llvm::dbgs() << llvm::formatv(                                           \
          stderr, "{0}:{1}:{2}(): ", "AllocTensors.cpp", __LINE__, __func__);  \
      llvm::dbgs() << llvm::formatv(fmt, __VA_ARGS__));

namespace mlir::plan {
#define GEN_PASS_DEF_PLANALLOCTENSORSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;
using bufferization::OneShotAnalysisState;
using bufferization::func_ext::FuncAnalysisState;
using bufferization::func_ext::FuncOpAnalysisState;

/// Creates a DPS argument of type `argType` in the first block of `func` by
/// appending to the end of current arguments. It then updates the function
/// type, adds a `executor.result_arg` argument attribute to the new arg, and
/// returns the new block argument.
static FailureOr<BlockArgument>
updateFunctionWithNewDpsArg(func::FuncOp func, Location loc, Type argType,
                            unsigned tiedResult) {
  MLIRContext *ctx = func->getContext();
  auto argAttrs = DictionaryAttr::get(
      ctx,
      {NamedAttribute(StringAttr::get(ctx, PlanDialect::kResultArgAttrName),
                      UnitAttr::get(ctx))});
  func.insertArgument(func.getNumArguments(), argType, argAttrs, loc);

  if (auto boundsAttr = func.getResultAttr(
          tiedResult, plan::PlanDialect::kShapeBoundsAttrName))
    func.setArgAttr(func.getNumArguments() - 1,
                    plan::PlanDialect::kShapeBoundsAttrName, boundsAttr);
  if (auto boundsAttr = func.getResultAttr(
          tiedResult, plan::PlanDialect::kValueBoundsAttrName))
    func.setArgAttr(func.getNumArguments() - 1,
                    plan::PlanDialect::kValueBoundsAttrName, boundsAttr);

  return func.getArguments().back();
}

/// Type for function that takes a type and an operand of a block terminator and
/// retrieves or creates a corresponding destination block arg to achieve
/// destination passing style.
using GetOrCreateBlockArgFunc = llvm::function_ref<FailureOr<BlockArgument>(
    Type argType, OpOperand &terminatorOperand)>;

/// Return the state (phase) of analysis of the FuncOp.
static FuncOpAnalysisState
getFuncOpAnalysisState(const OneShotAnalysisState &state, func::FuncOp funcOp) {
  if (!isa<OneShotAnalysisState>(state))
    return FuncOpAnalysisState::NotAnalyzed;
  auto *funcState = static_cast<const OneShotAnalysisState &>(state)
                        .getExtension<FuncAnalysisState>();
  if (!funcState)
    return FuncOpAnalysisState::NotAnalyzed;
  const auto &analyzedFuncOps = funcState->analyzedFuncOps;
  auto it = analyzedFuncOps.find(funcOp);
  if (it == analyzedFuncOps.end())
    return FuncOpAnalysisState::NotAnalyzed;
  return it->second;
}
/// Get FuncAnalysisState.
static const FuncAnalysisState &
getFuncAnalysisState(const OneShotAnalysisState &state) {
  assert(isa<OneShotAnalysisState>(state) && "expected OneShotAnalysisState");
  auto *result = static_cast<const OneShotAnalysisState &>(state)
                     .getExtension<FuncAnalysisState>();
  assert(result && "FuncAnalysisState does not exist");
  return *result;
}

/// Return the index of the bbArg in the given FuncOp that is equivalent to the
/// specified return value (if any).
static std::optional<int64_t>
getEquivalentFuncArgIdx(func::FuncOp funcOp, const FuncAnalysisState &state,
                        int64_t returnValIdx) {
  auto funcOpIt = state.equivalentFuncArgs.find(funcOp);
  if (funcOpIt == state.equivalentFuncArgs.end())
    // No equivalence info stores for funcOp.
    return std::nullopt;

  auto retValIt = funcOpIt->getSecond().find(returnValIdx);
  if (retValIt == funcOpIt->getSecond().end())
    // Return value has no equivalent bbArg.
    return std::nullopt;

  return retValIt->getSecond();
}

/// Returns a vector which maps each yielded value of the block to a
/// BlockArgument (which may be nullptr if no match can be found). This
/// procedure just tries to heuristically align as many yielded values as
/// possible to BlockArguments of the same type, assuming that BlockArguments
/// and yielded values that can be matched up will appear in the same order (it
/// doesn't consider permutations). This is just meant to be robust to simple
/// situations, like when the yielded values of a while op's "before" region is
/// just a subset of the regions's arguments.
static llvm::SmallVector<BlockArgument>
getYieldedValueToBlockArgMap(Block *block, ValueRange yieldedValues) {
  Block::BlockArgListType blockArgs = block->getArguments();
  unsigned yieldedArgIdx = 0, blockArgIdx = 0;
  SmallVector<BlockArgument> result(yieldedValues.size(), nullptr);
  // Subroutine that searches forward in the block argument list to find an
  // argument matching type of currently considered yielded arg.
  auto searchForward = [&]() -> bool {
    Type yieldedArgType = yieldedValues[yieldedArgIdx].getType();
    for (unsigned i = blockArgIdx, e = blockArgs.size(); i < e; ++i) {
      if (blockArgs[i].getType() == yieldedArgType) {
        result[yieldedArgIdx++] = blockArgs[i];
        blockArgIdx = i + 1;
        return true;
      }
    }
    return false;
  };

  while (yieldedArgIdx < yieldedValues.size() &&
         blockArgIdx < blockArgs.size()) {
    Value yielded = yieldedValues[yieldedArgIdx];
    // If this is a BlockArgument of the current block, then we use its index
    // align the current blockArgIdx for the remaining arguments.
    if (auto yieldedBlockArg = dyn_cast<BlockArgument>(yielded)) {
      if (yieldedBlockArg.getOwner() == block) {
        result[yieldedArgIdx++] = yieldedBlockArg;
        blockArgIdx = yieldedBlockArg.getArgNumber() + 1;
        continue;
      }
    }
    if (searchForward())
      continue;
    // No matching type found, map to nothing.
    result[yieldedArgIdx++] = nullptr;
  }
  return result;
}

/// Rewrite a single function to destination passing style. Update callers
/// appropriately.
static LogicalResult rewriteBlockToDestinationStyle(
    RewriterBase &rewriter, Block *block,
    MutableOperandRange yieldedTerminatorOperands,
    Block::BlockArgListType carriedBlockArgs,
    const bufferization::OneShotAnalysisState &state) {

  SmallVector<BlockArgument> yieldedValueToBlockArg =
      getYieldedValueToBlockArgMap(
          block, yieldedTerminatorOperands.getAsOperandRange());

  for (auto [idx, v] : llvm::enumerate(yieldedTerminatorOperands)) {
    if (!isa<TensorType>(v.get().getType()))
      continue;
    if (!yieldedValueToBlockArg[idx]) {
      LLVM_DEBUG(
          DBGS() << llvm::formatv(
              "yielded value #{0} could not be aligned to a block argument\n",
              idx));
      continue;
    }

    // Find equivalent 'tensor.empty. operation.
    bufferization::TraversalConfig config;
    config.followEquivalentOnly = true;
    config.alwaysIncludeLeaves = false;
    SetVector<Value> equivalentValues = state.findValueInReverseUseDefChain(
        &v, /*condition=*/
        [&, v = v.get()](Value val) {
          auto emptyOp = val.getDefiningOp<tensor::EmptyOp>();
          return emptyOp && emptyOp.getType() == v.getType();
        },
        config);

    LLVM_DEBUG({
      DBGS() << llvm::formatv("equivalent values for yielded value #{0}:\n",
                              idx);
      llvm::interleave(equivalentValues, llvm::dbgs(), "\n - ");
      llvm::dbgs() << "\n";
    });

    if (equivalentValues.size() != 1)
      continue;

    // Replace only uses inside the loop block.
    rewriter.replaceOpUsesWithIf(
        equivalentValues.front().getDefiningOp(), yieldedValueToBlockArg[idx],
        [&](OpOperand &use) { return use.getOwner()->getBlock() == block; });
  }
  return success();
}

/// Traverse a loop operation's regions and try to establish DPS connectivity by
/// connecting yielded values to block arguments.
static void visitLoopOp(RewriterBase &rewriter, Operation *loopOp,
                        const OneShotAnalysisState &state) {
  DBGF("visiting loop {0}", *loopOp);

  llvm::TypeSwitch<Operation *>(loopOp)
      .Case([&](scf::WhileOp whileOp) {
        if (failed(rewriteBlockToDestinationStyle(
                rewriter, whileOp.getBeforeBody(),
                cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator())
                    .getArgsMutable(),
                whileOp.getBeforeBody()->getArguments(), state)))
          return;
        if (failed(rewriteBlockToDestinationStyle(
                rewriter, whileOp.getAfterBody(),
                cast<scf::YieldOp>(whileOp.getAfterBody()->getTerminator())
                    .getResultsMutable(),
                whileOp.getAfterBody()->getArguments(), state)))
          return;
      })
      .Case([&](scf::ForOp forOp) {
        if (failed(rewriteBlockToDestinationStyle(
                rewriter, forOp.getBody(),
                cast<scf::YieldOp>(forOp.getBody()->getTerminator())
                    .getResultsMutable(),
                forOp.getRegionIterArgs(), state)))
          return;
      })
      .Default(
          [](Operation *loopOp) { DBGF("unhandled loop type: ", *loopOp); });
}

/// Visit all loop-like operations and try to establish DPS connectivity where
/// it is not present.
static LogicalResult rewriteLoopBlocksToDestinationStyle(RewriterBase &rewriter,
                                                         ModuleOp op) {
  bufferization::OneShotBufferizationOptions options;
  options.allowReturnAllocsFromLoops = true;
  options.bufferizeFunctionBoundaries = true;
  OneShotAnalysisState state(op, options);
  if (failed(plan::analyzeOneModuleOp(ModuleLikeOp(op), state, nullptr)))
    return failure();

  op->walk([&](Operation *nested) {
    if (nested->hasTrait<OpTrait::SymbolTable>())
      return nested == op ? WalkResult::advance() : WalkResult::skip();
    if (isa<LoopLikeOpInterface>(nested))
      visitLoopOp(rewriter, nested, state);
    return WalkResult::advance();
  });

  return success();
}

static FailureOr<Value> getShape(RewriterBase &rewriter, Location loc,
                                 TypedValue<RankedTensorType> v) {
  RankedTensorType type = v.getType();
  auto shapeType = RankedTensorType::get(
      static_cast<int64_t>(type.getShape().size()), rewriter.getIndexType(),
      plan::MemorySpaceAttr::get(type.getContext(), plan::MemorySpace::host));
  if (type.hasStaticShape())
    return rewriter
        .create<arith::ConstantOp>(
            loc, DenseElementsAttr::get(shapeType, type.getShape()))
        .getResult();

  assert(v.getDefiningOp() && "expected a defining op");
  ReifiedRankedShapedTypeDims shape;
  if (failed(reifyResultShapes(rewriter, v.getDefiningOp(), shape)))
    return failure();
  return rewriter
      .create<tensor::FromElementsOp>(
          loc, shapeType,
          getValueOrCreateConstantIndexOp(rewriter, loc, shape.front()))
      .getResult();
}

static FailureOr<TypedValue<RankedTensorType>>
maybeReshapeOrCast(RewriterBase &rewriter, Location loc,
                   TypedValue<RankedTensorType> v,
                   TypedValue<RankedTensorType> toReplace) {
  RankedTensorType type = toReplace.getType();
  if (v.getType() == type)
    return v;

  // Check if we can satisfy the type difference using `tensor.cast`.
  if (tensor::CastOp::areCastCompatible(v.getType(), type) &&
      v.getType().getEncoding() == type.getEncoding())
    return cast<TypedValue<RankedTensorType>>(
        rewriter.create<tensor::CastOp>(loc, type, v).getResult());

  if (v.getType().getElementType() != type.getElementType()) {
    // Check if we can use `tensor.bitcast`. It only supports integer and float
    // element types, or it will crash.
    /// TODO: Fix this upstream.
    if (isa<IntegerType, FloatType>(v.getType().getElementType()) &&
        isa<IntegerType, FloatType>(type.getElementType()) &&
        tensor::BitcastOp::areCastCompatible(v.getType(), type))
      return cast<TypedValue<RankedTensorType>>(
          rewriter.create<tensor::BitcastOp>(loc, type, v).getResult());
    // If we can't use `tensor.bitcast`, then there's no other op to use
    // currently.
    return failure();
  }

  // We may need to reshape `v` to the same shape as `toReplace`. We do this
  // through `tensor.expand_shape` and `tensor.collapse_shape` operations if
  // possible, otherwise fallback to `tensor.reshape`.
  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(v.getType(), type);
  if (!reassociation) {
    FailureOr<Value> shape = getShape(rewriter, loc, toReplace);
    if (failed(shape))
      return failure();

    return cast<TypedValue<RankedTensorType>>(
        rewriter.create<tensor::ReshapeOp>(loc, type, v, *shape).getResult());
  }

  if (v.getType().getRank() > type.getRank())
    return cast<TypedValue<RankedTensorType>>(
        rewriter.create<tensor::CollapseShapeOp>(loc, type, v, *reassociation)
            .getResult());

  ReifiedRankedShapedTypeDims shape;
  if (failed(reifyResultShapes(rewriter, toReplace.getDefiningOp(), shape)))
    return failure();
  return cast<TypedValue<RankedTensorType>>(
      rewriter
          .create<tensor::ExpandShapeOp>(loc, type, v, *reassociation,
                                         shape.front())
          .getResult());
}

/// Return true if the given function argument is considered "writable". It is
/// writable if it has attribute 'bufferization.writable' set to true or has
/// unit attribute 'plan.result_arg' set.
static bool isArgumentWritable(func::FuncOp func, int64_t idx) {
  if (func.getArgAttr(idx, PlanDialect::kResultArgAttrName))
    return true;
  if (auto writableAttr = func.getArgAttrOfType<BoolAttr>(
          idx, bufferization::BufferizationDialect::kWritableAttrName))
    return writableAttr.getValue();
  return false;
}

/// This function integrates a destination-passing style `argument` into the
/// function body. This `argument` could be newly created to enable an in-place
/// update, or it could be an existing argument that was identified as donating
/// its buffer for `termOperand`.
///
/// The goal is to ensure that `termOperand`, which represents a function
/// result, will use the same buffer as that of `argument`, if possible OR its
/// value will be available in `argument`. This is achieved in one of two ways:
///
/// 1.  Replace an Equivalent Allocation: Find a value (lets call it
///     `toReplace` value) inside the function (e.g., from a `tensor.empty`)
///     that bufferizes to the same buffer as `termOPerand`. If such value
///     exists, that value is replaced with the new `argument` everywhere.
///
/// 2.  Copy buffer: If no such value exists, this function insert a
///     `linalg::CopyOp` operation so that buffer associated with
///     `termOperand` is copied into `argument`.
///
/// For 1, we use reverse def chain analysis to find all values equivalent to
/// `termOperand`.
static LogicalResult accommodateDestinationStyleArgument(
    RewriterBase &rewriter, TypedValue<RankedTensorType> argument,
    func::ReturnOp term, OpOperand &termOperand,
    bufferization::OneShotAnalysisState &state) {

  // Find equivalent values.
  bufferization::TraversalConfig config;
  config.followEquivalentOnly = true;
  config.followInPlaceOnly = true;
  config.alwaysIncludeLeaves = true;
  SetVector<Value> equivalentValues = state.findValueInReverseUseDefChain(
      &termOperand, /*condition=*/
      [](Value val) { return false; }, config);

  LLVM_DEBUG({
    DBGS() << llvm::formatv("equivalent values for return value #{0}:\n",
                            termOperand.getOperandNumber());
    llvm::interleave(equivalentValues, llvm::dbgs(), "\n - ");
    llvm::dbgs() << "\n";
  });

  //  We're looking for a single `tensor.empty` or `bufferization.alloc_tensor`
  //  operation that we can replace.
  if (equivalentValues.size() != 1 ||
      !isa_and_present<tensor::EmptyOp, bufferization::AllocTensorOp>(
          equivalentValues.front().getDefiningOp())) {
    // In this case we can't find a single `tensor.empty` or
    // `bufferization.alloc_tensor` operation that we can replace, we
    // insert a `linalg::CopyOp` operation so that buffer associated with
    // `termOperand` is copied into `argument`.
    rewriter.setInsertionPoint(term);

    RankedTensorType targetType =
        cast<RankedTensorType>(termOperand.get().getType());
    auto copyOp = rewriter.create<linalg::CopyOp>(
        termOperand.get().getLoc(), targetType, termOperand.get(), argument);
    term.getOperandsMutable()[termOperand.getOperandNumber()].assign(
        copyOp.getResult(0));
    return success();
  }

  // Our action now depends on what kind of equivalent value we found.
  Operation *equivalentOp = equivalentValues.front().getDefiningOp();

  // Check if the user of `toReplace` is a `tensor.reshape` operation. Op
  // `tensor.reshape` bufferizes to equivalent of `toReplace` since it doesn't
  // allocate new memory but instead simply provides new view. Thus, we can just
  // try to replace the reshape instead.
  if (equivalentOp->hasOneUse()) {
    if (auto reshapeOp =
            dyn_cast<tensor::ReshapeOp>(*equivalentOp->user_begin())) {
      if (reshapeOp.getSource() == equivalentOp->getResult(0))
        equivalentOp = reshapeOp;
    }
  }

  // A reshape or cast may be required if the equivalent value has a different
  // type than the new function argument.
  rewriter.setInsertionPointAfter(equivalentOp);
  FailureOr<TypedValue<RankedTensorType>> reshaped = maybeReshapeOrCast(
      rewriter, equivalentOp->getLoc(), argument,
      cast<TypedValue<RankedTensorType>>(equivalentOp->getResult(0)));
  if (failed(reshaped))
    return failure();
  argument = *reshaped;

  // If the equivalent value is a `tensor.empty`, we can replace its uses
  // with the new function argument. A reshape or cast may be required.
  if (auto emptyOp = dyn_cast<tensor::EmptyOp>(equivalentOp)) {
    rewriter.replaceAllOpUsesWith(equivalentOp, argument);
    return success();
  }
  if (auto allocOp = dyn_cast<bufferization::AllocTensorOp>(equivalentOp)) {
    if (!allocOp.getCopy()) {
      rewriter.replaceAllOpUsesWith(equivalentOp, argument);
      return success();
    }
    rewriter.replaceOpWithNewOp<bufferization::MaterializeInDestinationOp>(
        allocOp, allocOp.getCopy(), argument);
    return success();
  }
  if (auto constOp = dyn_cast<arith::ConstantOp>(equivalentOp)) {
    auto matOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        constOp.getLoc(), constOp, argument);
    rewriter.replaceAllUsesExcept(constOp, matOp.getResult(), matOp);
    return success();
  }
  if (auto reshapeOp = dyn_cast<tensor::ReshapeOp>(equivalentOp)) {
    rewriter.replaceAllOpUsesWith(equivalentOp, argument);
    return success();
  }
  llvm_unreachable("unexpected leaf operation kind");
}

/// Rewrite a single function to destination passing style. Update callers
/// appropriately.
static LogicalResult rewriteFuncToDestinationPassingStyle(
    RewriterBase &rewriter, func::FuncOp func, SymbolUserMap &callerMap,
    bufferization::OneShotAnalysisState &state) {
  assert(!func.isDeclaration());

  // We don't know how to handle callers other than 'func.call'.
  // Note that we store references to the "shape function" of public
  // entrypoints using the symbol name in a function's attributes, so also
  // allow that.
  if (!llvm::all_of(callerMap.getUsers(func),
                    llvm::IsaPred<func::CallOp, func::FuncOp>))
    return failure();

  if (getFuncOpAnalysisState(state, func) != FuncOpAnalysisState::Analyzed) {
    LLVM_DEBUG(DBGS() << "function was not analyzed\n");
    return failure();
  }

  auto term = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  const FuncAnalysisState &funcState = getFuncAnalysisState(state);

  // Build a map of result index -> argument index for donated arguments.
  llvm::DenseMap<int64_t, unsigned> donatedArgMap;
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    if (auto donationAttr = func.getArgAttrOfType<IntegerAttr>(
            i, plan::PlanDialect::kDonationArgAttrName)) {
      // Check result index donated by donation arguments are within bound.
      if (donationAttr.getInt() >= func.getFunctionType().getNumResults())
        return failure();
      donatedArgMap[donationAttr.getInt()] = i;
    }
  }

  for (auto [idx, v] : llvm::enumerate(term->getOpOperands())) {
    if (!isa<RankedTensorType>(v.get().getType()))
      continue;

    // Check if there is already an equivalent function argument.
    if (std::optional<int64_t> equivalent =
            getEquivalentFuncArgIdx(func, funcState, idx);
        equivalent && isArgumentWritable(func, *equivalent)) {
      LLVM_DEBUG(
          DBGS() << llvm::formatv("for return value #{0} found existing "
                                  "equivalent result arg -- argument #{1}\n",
                                  idx, *equivalent));
      continue;
    }

    // Check if some argument is donated for this result value. Shape and
    // element type of donated argument must match with result.
    auto donatedIt = donatedArgMap.find(idx);
    if (donatedIt != donatedArgMap.end()) {
      auto donatedArg = cast<TypedValue<RankedTensorType>>(
          func.getArgument(donatedIt->second));
      if (donatedArg.getType() != v.get().getType()) {
        return func.emitError("donation argument is found but its type (")
               << donatedArg.getType()
               << ") doesn't match with corresponding result type ("
               << v.get().getType() << ")";
      }
      if (failed(accommodateDestinationStyleArgument(rewriter, donatedArg, term,
                                                     v, state)))
        return failure();
      continue;
    }

    // There is no existing equivalent or valid donation function argument, so
    // we must create a new one.
    if (failed(updateFunctionWithNewDpsArg(func, v.get().getLoc(),
                                           v.get().getType(), idx)))
      return failure();
    auto replacement =
        cast<TypedValue<RankedTensorType>>(func.getArguments().back());

    if (failed(accommodateDestinationStyleArgument(rewriter, replacement, term,
                                                   v, state)))
      return failure();
  }
  return success();
}

/// Create a `bufferization.alloc_tensor` operation that clones the given
/// tensor value.
static Value createTensorClone(RewriterBase &rewriter, Location loc, Value v) {
  RankedTensorType type = cast<RankedTensorType>(v.getType());
  return rewriter.create<bufferization::AllocTensorOp>(loc, type, ValueRange{},
                                                       /*copy=*/v);
}

/// If the pass options specify that all entrypoint functions should allocate
/// their results, then we must ensure that a copy is made if the current
/// returned value is equivalent to a function argument. Otherwise, the end
/// result will appear be missing a result value.
static LogicalResult
enforceResultAllocationPolicy(RewriterBase &rewriter, func::FuncOp func,
                              SymbolUserMap &callerMap,
                              bufferization::OneShotAnalysisState &state) {
  assert(!func.isDeclaration() && "expected a function with a body");

  if (getFuncOpAnalysisState(state, func) != FuncOpAnalysisState::Analyzed) {
    LLVM_DEBUG(DBGS() << "function was not analyzed\n");
    return failure();
  }

  OpBuilder::InsertionGuard g(rewriter);
  auto term = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  rewriter.setInsertionPoint(term);
  const FuncAnalysisState &funcState = getFuncAnalysisState(state);

  for (auto [idx, v] : llvm::enumerate(term->getOpOperands())) {
    if (!isa<RankedTensorType>(v.get().getType()))
      continue;

    // Check if there is already an equivalent function argument.
    if (std::optional<int64_t> equivalent =
            getEquivalentFuncArgIdx(func, funcState, idx)) {
      LLVM_DEBUG(
          DBGS() << llvm::formatv("for return value #{0} found "
                                  "equivalent result arg -- argument #{1}\n",
                                  idx, *equivalent));
      Value cloned = createTensorClone(rewriter, term.getLoc(), v.get());
      rewriter.modifyOpInPlace(term,
                               [v = &v, &cloned]() { v->assign(cloned); });
      continue;
    }
  }

  return success();
}

/// Updates entrypoint functions to destination-passing style or to require
/// returning allocations.
static LogicalResult enforceFunctionCallingStylePolicy(
    RewriterBase &rewriter, SymbolTableCollection &symbolTables, ModuleOp op,
    bool forceEntrypointsReturnAllocs) {

  SymbolUserMap userMap(symbolTables, op);
  bufferization::OneShotBufferizationOptions options;
  options.allowReturnAllocsFromLoops = true;
  options.bufferizeFunctionBoundaries = true;
  OneShotAnalysisState state(op, options);
  if (failed(plan::analyzeOneModuleOp(ModuleLikeOp(op), state, nullptr)))
    return failure();

  // Locate entrypoint functions.
  SmallVector<func::FuncOp> orderedFuncOps, remainingFuncOps;
  if (failed(mlir::getFuncOpsOrderedByCalls(
          mlir::ModuleLikeOp(op), orderedFuncOps, remainingFuncOps,
          [&](func::FuncOp func) -> bool {
            return func.isPublic() && !func.isDeclaration() &&
                   func->getParentWithTrait<OpTrait::SymbolTable>() == op &&
                   llvm::none_of(userMap.getUsers(func),
                                 llvm::IsaPred<CallOpInterface>) &&
                   (llvm::any_of(func.getArgumentTypes(),
                                 llvm::IsaPred<TensorType>) ||
                    llvm::any_of(func.getResultTypes(),
                                 llvm::IsaPred<TensorType>));
          })))
    return failure();

  for (func::FuncOp func : orderedFuncOps) {
    LLVM_DEBUG(DBGS() << "encountered func " << func.getName() << "\n");
    if (func.isDeclaration())
      continue;

    // All functions should be single-block at this point.
    if (func.getBlocks().size() != 1)
      return failure();

    LLVM_DEBUG(DBGS() << "considering func " << func.getName() << "\n");
    if (!forceEntrypointsReturnAllocs &&
        failed(rewriteFuncToDestinationPassingStyle(rewriter, func, userMap,
                                                    state)))
      return failure();

    if (forceEntrypointsReturnAllocs &&
        failed(enforceResultAllocationPolicy(rewriter, func, userMap, state)))
      return failure();
  }
  return success();
}

/// Our algorithm for converting to DPS requires that each 'tensor.empty' has a
/// unique user.
static void uniqueEmptyTensorUses(RewriterBase &rewriter, ModuleLikeOp op) {
  op->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
    if (ModuleLikeOp(nestedOp) && nestedOp != op)
      return WalkResult::skip();
    auto emptyOp = dyn_cast<tensor::EmptyOp>(nestedOp);
    if (!emptyOp)
      return WalkResult::advance();
    if (nestedOp->hasOneUse())
      return WalkResult::advance();
    unsigned firstUse = true;
    for (OpOperand &use : llvm::make_early_inc_range(emptyOp->getUses())) {
      if (firstUse) {
        firstUse = false;
        continue;
      }
      rewriter.setInsertionPoint(use.getOwner());
      auto clonedOp = cast<tensor::EmptyOp>(rewriter.clone(*emptyOp));
      use.assign(clonedOp);
    }
    return WalkResult::advance();
  });
}

namespace {
class AllocTensorsPass
    : public plan::impl::PlanAllocTensorsPassBase<AllocTensorsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp op = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    /// Some 'tensor.empty' can have multiple uses. Duplicate the 'tensor.empty'
    /// in these cases so that each tensor.empty has a single use. This helps
    /// with establishing optimal DPS connectivity (e.g. in cases where a single
    /// tensor.empty is used in multiple linalg 'outs' operands, we don't want
    /// to assign each 'outs' operand to the same DPS output arg).
    uniqueEmptyTensorUses(rewriter, ModuleLikeOp(op));

    /// Establish DPS connectivity in loop regions by establishing DPS
    /// correspondence between loop yields and block arguments wherever
    /// possible.
    if (failed(rewriteLoopBlocksToDestinationStyle(rewriter, op))) {
      op->emitError("failed to establish DPS connectivity in loop-like "
                    "operation regions");
      return signalPassFailure();
    }

    // Depending on the options (DPS vs. force return allocations), we may need
    // to update the entrypoint function(s) signanatures or internals to reflect
    // the desired calling style.
    SymbolTableCollection symbolTables;
    if (failed(enforceFunctionCallingStylePolicy(
            rewriter, symbolTables, op, forceEntrypointsReturnAllocs))) {
      emitError(op.getLoc(),
                "failed to establish DPS connectivity in public functions");
      return signalPassFailure();
    }

    // Remove leftover empty tensors.
    op->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (ModuleLikeOp(nestedOp) && nestedOp != op)
        return WalkResult::skip();
      auto emptyOp = dyn_cast<tensor::EmptyOp>(nestedOp);
      if (!emptyOp || !emptyOp.use_empty())
        return WalkResult::advance();
      rewriter.eraseOp(emptyOp);
      return WalkResult::skip();
    });
  }
};
} // namespace
