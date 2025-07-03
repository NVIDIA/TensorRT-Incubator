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
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/ModuleBufferization/ModuleBufferization.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Utils/DataFlowUtils.h"
#include "mlir-tensorrt/Utils/ModuleUtils.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
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

static constexpr int64_t kHostConstantToFromElementsNumElementsLimit = 16;

/// Create a tensor using a sequence of chained tensor.insert into a
/// `bufferization.alloc_tensor` in the specified memory space (which must
/// either be 'host' or 'host_pinned'). The provided `elements` should be given
/// in the canonical row-major order.
static Value createTensorFromElements(RewriterBase &rewriter, Location loc,
                                      RankedTensorType type,
                                      ValueRange elements,
                                      MemorySpace memorySpace) {
  assert(memorySpace == MemorySpace::host ||
         memorySpace == MemorySpace::host_pinned &&
             "tensor.from_elements must be lowered into an allocation in "
             "a host-visible space");
  assert(llvm::all_equal(llvm::concat<const Type>(
             elements.getTypes(), TypeRange{type.getElementType()})) &&
         "expected all scalars to have the same type");

  // Create the allocation.
  RankedTensorType tensorType = RankedTensorType::Builder(type).setEncoding(
      MemorySpaceAttr::get(rewriter.getContext(), memorySpace));
  auto allocOp = rewriter.create<bufferization::AllocTensorOp>(loc, tensorType,
                                                               ValueRange());
  allocOp.setMemorySpaceAttr(tensorType.getEncoding());

  // Handle the rank 0 case and early exit.
  if (tensorType.getRank() == 0) {
    Value replacement = rewriter.create<tensor::InsertOp>(
        loc, elements.front(), allocOp.getResult(), ValueRange{});
    return replacement;
  }

  // Create the chain of `tensor.insert` operations.
  SmallVector<int64_t> basis =
      mlir::computeSuffixProduct(tensorType.getShape());
  Value result = allocOp.getResult();
  for (auto [i, element] : llvm::enumerate(elements)) {
    SmallVector<Value> coords = llvm::map_to_vector(
        mlir::delinearize(i, basis), [&](int64_t dim) -> Value {
          return rewriter.create<arith::ConstantIndexOp>(loc, dim);
        });
    result = rewriter.create<tensor::InsertOp>(loc, element, result, coords);
  }
  return result;
}

/// Find a `bufferization.alloc_tensor` user of `v` that copies `v` to the
/// specified space or create a new `bufferization.alloc_tensor` operation to
/// accomplish the copy.
static Value findOrCreateCopyToSpace(RewriterBase &rewriter, Value v,
                                     plan::MemorySpace memSpace) {
  for (Operation *user : v.getUsers()) {
    auto allocOp = dyn_cast<bufferization::AllocTensorOp>(user);
    if (!allocOp)
      continue;
    auto space = llvm::dyn_cast_or_null<plan::MemorySpaceAttr>(
        allocOp.getMemorySpaceAttr());
    if (!space)
      continue;
    if (allocOp.getCopy() == v && space.getValue() == memSpace)
      return allocOp.getResult();
  }

  return rewriter.create<bufferization::AllocTensorOp>(
      v.getLoc(), v.getType(), /*dynamic_sizes=*/ValueRange{},
      /*copy=*/v,
      /*size_hint=*/Value{},
      plan::MemorySpaceAttr::get(v.getContext(), memSpace));
}

/// Iterate over the uses of `v` and find uses that are statically known to be
/// on host or device.
static llvm::SmallPtrSet<OpOperand *, 4> getUsesFromSpace(Value v,
                                                          TensorKind useKind) {
  assert((useKind == TensorKind::Device || useKind == TensorKind::Host) &&
         "useKind should be either Device or Host");
  llvm::SmallPtrSet<OpOperand *, 4> uses;
  for (OpOperand &operand : v.getUses()) {
    TensorKind kind = TensorKindAnalysis::getStaticOperandTensorKind(operand);
    if (kind == useKind)
      uses.insert(&operand);
  }
  return uses;
}

/// Given value `v` that has mixed host/device uses (`TensorKind::Both`), find
/// or create a `bufferization.alloc_tensor` operation that copies `v` into a
/// host tensor and use that to replace all uses that are statically known to
/// require host access.
static LogicalResult replaceHostUsesWithHostAlloc(RewriterBase &rewriter,
                                                  Value v) {
  OpBuilder::InsertionGuard g(rewriter);
  if (auto blockArg = dyn_cast<BlockArgument>(v))
    rewriter.setInsertionPointToStart(blockArg.getOwner());
  else
    rewriter.setInsertionPointAfter(v.getDefiningOp());

  llvm::SmallPtrSet<OpOperand *, 4> hostUses =
      getUsesFromSpace(v, TensorKind::Host);
  if (hostUses.empty())
    return failure();

  // Create the host tensor. Note that dynamic sizes are not needed to be
  // pased since we pass the `copy` argument.
  Value alloc =
      findOrCreateCopyToSpace(rewriter, v, plan::MemorySpace::host_pinned);
  rewriter.replaceUsesWithIf(
      v, alloc, [&](OpOperand &use) { return hostUses.contains(&use); });
  return success();
}

namespace {

/// Rewrite `tensor.from_elements` to be in destination-passing-style. It
/// creates a tensor with `bufferization.alloc_tensor` in the host_pinned space,
/// populates the elements, and either replaces all uses with the result (if
/// the TensorKindAnalysis knows that all sues are host uses), or it also
/// creates a host-to-device copy and replaces only uses statically known to be
/// on host with the host buffer and the rest of the uses with the device copy.
struct RewriteFromElements : public OpRewritePattern<tensor::FromElementsOp> {
  DataFlowSolver &solver;

  RewriteFromElements(MLIRContext *ctx, DataFlowSolver &solver,
                      PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), solver(solver) {}

  LogicalResult matchAndRewrite(tensor::FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType originalType = op.getType();
    Location loc = op.getLoc();
    const TensorKindLattice *lattice =
        solver.lookupState<TensorKindLattice>(op.getResult());
    assert(lattice && !lattice->getValue().isUninitialized());
    TensorKindInfo placementInfo = lattice->getValue();

    std::optional<MemorySpace> originalMemorySpace{};
    if (auto constraint =
            dyn_cast_or_null<MemorySpaceAttr>(op.getType().getEncoding())) {
      if (constraint.isHostVisible())
        return failure();
      originalMemorySpace = constraint.getValue();
    }

    // Create a host allocation and insert the elements.
    MemorySpace memorySpace = MemorySpace::host_pinned;
    Value hostReplacement = createTensorFromElements(
        rewriter, op.getLoc(), op.getType(), op.getElements(), memorySpace);
    Value hostReplacementCasted =
        rewriter.create<tensor::CastOp>(loc, originalType, hostReplacement);
    bool canOptimizeHostReplacement =
        !originalMemorySpace || (*originalMemorySpace == memorySpace);
    if (placementInfo.isHostOnly() && canOptimizeHostReplacement) {
      rewriter.replaceOp(op, hostReplacementCasted);
      return success();
    }

    // Now insert another `bufferization.alloc_tensor` to force copying to the
    // device --- only if we know that the user can interpret this correctly.
    RankedTensorType destType = RankedTensorType::get(
        originalType.getShape(), originalType.getElementType(),
        MemorySpaceAttr::get(originalType.getContext(), MemorySpace::device));
    auto allocOp = rewriter.create<bufferization::AllocTensorOp>(loc, destType,
                                                                 ValueRange{});
    allocOp.setMemorySpaceAttr(destType.getEncoding());
    Value devReplacement =
        rewriter
            .create<bufferization::MaterializeInDestinationOp>(
                loc, destType, hostReplacement, allocOp.getResult())
            .getResult();
    devReplacement =
        rewriter.create<tensor::CastOp>(loc, originalType, devReplacement);

    if (canOptimizeHostReplacement)
      rewriter.replaceOpUsesWithIf(
          op, hostReplacementCasted, [&](OpOperand &use) {
            return TensorKindAnalysis::getStaticOperandTensorKind(use) ==
                   TensorKind::Host;
          });
    rewriter.replaceOpUsesWithIf(op, devReplacement, [&](OpOperand &use) {
      return !canOptimizeHostReplacement ||
             TensorKindAnalysis::getStaticOperandTensorKind(use) !=
                 TensorKind::Host;
    });
    return success();
  }
};

/// Simplify a func.return operand produced by
/// `materialize_in_dest(cast(materialize_in_dest(..., %alloc)), %out_arg)` so
/// that only the single `materialize_in_dest` is used directly into the block
/// argument.
struct RemoveRedundantMaterializeInDestPattern
    : OpRewritePattern<bufferization::MaterializeInDestinationOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(bufferization::MaterializeInDestinationOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse() || !isa<func::ReturnOp>(*op->user_begin()))
      return failure();

    auto dest = dyn_cast<BlockArgument>(op.getDest());
    auto castOp = op.getSource().getDefiningOp<tensor::CastOp>();
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!castOp || !dest || !funcOp ||
        dest.getOwner() != &funcOp.getBody().front())
      return failure();

    auto producer =
        castOp.getSource()
            .getDefiningOp<bufferization::MaterializeInDestinationOp>();
    if (!producer || !producer->hasOneUse() ||
        !producer.getDest().hasOneUse() ||
        !producer.getDest().getDefiningOp<bufferization::AllocTensorOp>())
      return failure();

    // Replace the returned value with the result of the cast.
    Location loc = op->getLoc();
    rewriter.replaceOp(op, castOp);

    // Create a new cast on the block arg to the type of the producer alloc
    // result.
    rewriter.setInsertionPoint(producer);
    auto blockArgCast = rewriter.create<tensor::CastOp>(
        loc, producer.getDest().getType(), dest);
    // Update the producer materialization to materialize into the block arg.
    rewriter.replaceOp(producer.getDest().getDefiningOp(), blockArgCast);
    return success();
  }
};

/// Rewrite `memref.load` that acts on device memory to first copy the buffer to
/// the host and load from the host buffer.
struct TensorDeviceExtractRewriter
    : public OpRewritePattern<tensor::ExtractOp> {
  DataFlowSolver &solver;

  TensorDeviceExtractRewriter(MLIRContext *ctx, DataFlowSolver &solver,
                              PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), solver(solver) {}

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {

    // Return early if there are no device placement requirements.
    const TensorKindLattice *lattice =
        solver.lookupState<TensorKindLattice>(op.getTensor());
    if (!lattice || lattice->getValue().isUninitialized())
      return rewriter.notifyMatchFailure(op, "lattice value is uninitialized");
    if (lattice->getValue().isHostOnly())
      return rewriter.notifyMatchFailure(op, "lattice value is host-only");

    if (auto constraint = dyn_cast_if_present<MemorySpaceAttr>(
            op.getTensor().getType().getEncoding())) {
      if (constraint.isHostVisible())
        return rewriter.notifyMatchFailure(
            op, "source tensor already is in a host-visible space");
    }

    if (failed(replaceHostUsesWithHostAlloc(rewriter, op.getTensor())))
      return failure();

    return success();
  }
};

/// Rewrite `arith.constant` host tensors to use an explicit local buffer using
/// `bufferization.alloc_tensors`. Otherwise, the constant will be turned into a
/// memref.global during bufferization.
/// TODO: This pattern is required because currently we may not have memory
/// space annotations on the constant tensors. It could be removed if we revise
/// the strategy to populate memory space annotations on all tensors.
struct HostShapeConstantsToAllocTensorPattern
    : public OpRewritePattern<arith::ConstantOp> {
  HostShapeConstantsToAllocTensorPattern(MLIRContext *ctx,
                                         DataFlowSolver &solver)
      : OpRewritePattern(ctx), solver(solver) {}
  DataFlowSolver &solver;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto elementsAttr = dyn_cast<ElementsAttr>(op.getValue());
    if (!elementsAttr ||
        // Only tensor-typed elements are supported.
        !isa<RankedTensorType>(elementsAttr.getType())
        // Complex element types are not supported.
        || !elementsAttr.getElementType().isIntOrIndexOrFloat())
      return failure();
    if (elementsAttr.getNumElements() >
        kHostConstantToFromElementsNumElementsLimit)
      return failure();
    const TensorKindLattice *lattice =
        solver.lookupState<TensorKindLattice>(op.getResult());
    if (!lattice || lattice->getValue().isUninitialized() ||
        lattice->getValue().isDeviceOnly())
      return failure();

    // Enumerate the uses which require host-side access. We only use this set
    // if the value kind is 'both host and device', but we want to bail out
    // early if we can't find any host uses. That indicates that the pattern
    // already ran (but the analysis might now be out-of-date).
    llvm::SmallPtrSet<OpOperand *, 4> hostUses =
        getUsesFromSpace(op.getResult(), TensorKind::Host);
    if (hostUses.empty())
      return failure();

    SmallVector<Value> elements;
    for (Attribute attr : elementsAttr.getValues<Attribute>())
      elements.push_back(rewriter.create<arith::ConstantOp>(
          op.getLoc(), cast<TypedAttr>(attr)));

    Value result = createTensorFromElements(
        rewriter, op.getLoc(), cast<RankedTensorType>(elementsAttr.getType()),
        elements, plan::MemorySpace::host);

    if (lattice->getValue().isHostOnly()) {
      rewriter.replaceOp(op, result);
      return success();
    }

    assert(lattice->getValue().isBothHostAndDevice() &&
           "expected value to be used on host");
    rewriter.replaceUsesWithIf(op.getResult(), result, [&](OpOperand &use) {
      return hostUses.contains(&use);
    });
    return success();
  }
};

/// Rewrite `arith.constant` host tensors that are used on the host.
/// This pattern differs from the above in that it is used only for constants
/// larger than the limit `kHostConstantToFromElementsNumElementsLimit`.
struct LargeHostConstantsAllocTensorPattern
    : public OpRewritePattern<arith::ConstantOp> {
  LargeHostConstantsAllocTensorPattern(MLIRContext *ctx, DataFlowSolver &solver)
      : OpRewritePattern(ctx), solver(solver) {}
  DataFlowSolver &solver;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto elementsAttr = dyn_cast<ElementsAttr>(op.getValue());
    if (!elementsAttr || !isa<RankedTensorType>(elementsAttr.getType()))
      return failure();
    if (elementsAttr.getNumElements() <=
        kHostConstantToFromElementsNumElementsLimit)
      return failure();
    const TensorKindLattice *lattice =
        solver.lookupState<TensorKindLattice>(op.getResult());
    if (!lattice || lattice->getValue().isUninitialized() ||
        lattice->getValue().isDeviceOnly())
      return failure();

    if (failed(replaceHostUsesWithHostAlloc(rewriter, op.getResult())))
      return failure();
    return success();
  }
};

/// Rewrite `tensor.empty` to `bufferization.alloc_tensor` in the `device`
/// memory space.
struct RewriteEmptyTensor : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(
        op, op.getType(), op.getDynamicSizes(),
        /*copy=*/Value{}, /*size_hint=*/Value{},
        plan::MemorySpaceAttr::get(op.getContext(), plan::MemorySpace::device));
    return success();
  }
};

/// Drop `bufferization.alloc_tensor` operations that do not have uses.
struct CleanupAllocTensorOps
    : public OpRewritePattern<bufferization::AllocTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(bufferization::AllocTensorOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->use_empty())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

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
          tiedResult, plan::PlanDialect::getShapeBoundsAttrName()))
    func.setArgAttr(func.getNumArguments() - 1,
                    plan::PlanDialect::getShapeBoundsAttrName(), boundsAttr);
  if (auto boundsAttr = func.getResultAttr(
          tiedResult, plan::PlanDialect::getValueBoundsAttrName()))
    func.setArgAttr(func.getNumArguments() - 1,
                    plan::PlanDialect::getValueBoundsAttrName(), boundsAttr);

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
  if (tensor::CastOp::areCastCompatible(v.getType(), type))
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

  for (auto [idx, v] : llvm::enumerate(term->getOpOperands())) {
    if (!isa<RankedTensorType>(v.get().getType()))
      continue;

    // Check if there is already an equivalent function argument.
    if (std::optional<int64_t> equivalent =
            getEquivalentFuncArgIdx(func, funcState, idx);
        equivalent &&
        func.getArgAttr(*equivalent, PlanDialect::kResultArgAttrName)) {
      LLVM_DEBUG(
          DBGS() << llvm::formatv("for return value #{0} found existing "
                                  "equivalent result arg -- argument #{1}\n",
                                  idx, *equivalent));
      continue;
    }

    // There is no existing equivalent function argument, so we must create a
    // new one.
    if (failed(updateFunctionWithNewDpsArg(func, v.get().getLoc(),
                                           v.get().getType(), idx)))
      return failure();
    auto replacement =
        cast<TypedValue<RankedTensorType>>(func.getArguments().back());

    // The function argument must replace some value or force the returned value
    // to bufferize into a copy into the new function argument. To do this, we
    // use the bufferization analysis to find a value derived from e.g.
    // `tensor.empty` that bufferizes into the same  buffer as the returned
    // value.
    bufferization::TraversalConfig config;
    config.followEquivalentOnly = true;
    config.followInPlaceOnly = true;
    config.alwaysIncludeLeaves = true;
    SetVector<Value> equivalentValues = state.findValueInReverseUseDefChain(
        &v, /*condition=*/
        [](Value val) { return false; }, config);

    LLVM_DEBUG({
      DBGS() << llvm::formatv("equivalent values for return value #{0}:\n",
                              idx);
      llvm::interleave(equivalentValues, llvm::dbgs(), "\n - ");
      llvm::dbgs() << "\n";
    });

    // We're looking for a single `tensor.empty` or `bufferization.alloc_tensor`
    // operation that we can replace.
    if (equivalentValues.size() != 1 ||
        !isa_and_present<tensor::EmptyOp, bufferization::AllocTensorOp,
                         arith::ConstantOp>(
            equivalentValues.front().getDefiningOp())) {
      // If we can't find a single `tensor.empty` or
      // `bufferization.alloc_tensor` operation that we can replace, we must
      // insert a `bufferization.materialize_in_destination` operation to force
      // the returned value to bufferize into the new function argument.
      rewriter.setInsertionPoint(term);
      replacement = cast<TypedValue<RankedTensorType>>(
          rewriter
              .create<bufferization::MaterializeInDestinationOp>(
                  v.get().getLoc(), v.get().getType(), v.get(), replacement)
              .getResult());
      term.getOperandsMutable()[idx].assign(replacement);
      continue;
    }

    // Our action now depends on what kind of equivalent value we found.
    Operation *equivalentOp = equivalentValues.front().getDefiningOp();

    // Check if the user of `toReplace` is a `tensor.reshape` operation.
    // Since `toReplace` is bufferizes to the equivalent of `tensor.reshape`,
    // we can just try to replace the reshape instead.
    if (equivalentOp->hasOneUse()) {
      if (auto reshapeOp =
              dyn_cast<tensor::ReshapeOp>(*equivalentOp->user_begin())) {
        if (reshapeOp.getSource() == equivalentOp->getResult(0)) {
          equivalentOp = reshapeOp;
        }
      }
    }

    // A reshape or cast may be required if the equivalent value has a different
    // type than the new function argument.
    rewriter.setInsertionPointAfter(equivalentOp);
    FailureOr<TypedValue<RankedTensorType>> reshaped = maybeReshapeOrCast(
        rewriter, equivalentOp->getLoc(), replacement,
        cast<TypedValue<RankedTensorType>>(equivalentOp->getResult(0)));
    if (failed(reshaped))
      return failure();
    replacement = *reshaped;

    // If the equivalent value is a `tensor.empty`, we can replace its uses
    // with the new function argument. A reshape or cast may be required.
    if (auto emptyOp = dyn_cast<tensor::EmptyOp>(equivalentOp)) {
      rewriter.replaceAllOpUsesWith(equivalentOp, replacement);
      continue;
    }
    if (auto allocOp = dyn_cast<bufferization::AllocTensorOp>(equivalentOp)) {
      if (!allocOp.getCopy()) {
        rewriter.replaceAllOpUsesWith(equivalentOp, replacement);
        continue;
      }
      rewriter.replaceOpWithNewOp<bufferization::MaterializeInDestinationOp>(
          allocOp, allocOp.getCopy(), replacement);
      continue;
    }
    if (auto constOp = dyn_cast<arith::ConstantOp>(equivalentOp)) {
      auto matOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          constOp.getLoc(), constOp, replacement);
      rewriter.replaceAllUsesExcept(constOp, matOp.getResult(), matOp);
      continue;
    }
    if (auto reshapeOp = dyn_cast<tensor::ReshapeOp>(equivalentOp)) {
      rewriter.replaceAllOpUsesWith(equivalentOp, replacement);
      continue;
    }
    llvm_unreachable("unexpected leaf operation kind");
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
    for (OpOperand &use : llvm::make_early_inc_range(emptyOp->getUses())) {
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

    SymbolTableCollection symbolTables;
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TensorKindAnalysis>(symbolTables);

    if (failed(solver.initializeAndRun(op))) {
      op.emitError() << "failed to run TensorKindAnalysis";
      return signalPassFailure();
    }

    SolverStateListener<TensorKindLattice,
                        dataflow::Lattice<dataflow::ConstantValue>,
                        dataflow::Executable>
        solverAwareListener(solver);
    GreedyRewriteConfig config;
    config.listener = &solverAwareListener;
    FrozenRewritePatternSet patterns = [&]() {
      RewritePatternSet patterns_(ctx);
      patterns_.insert<HostShapeConstantsToAllocTensorPattern,
                       RewriteFromElements, TensorDeviceExtractRewriter,
                       LargeHostConstantsAllocTensorPattern>(ctx, solver);
      return patterns_;
    }();
    for (FunctionOpInterface func : op.getOps<FunctionOpInterface>()) {
      if (failed(applyPatternsGreedily(func, patterns))) {
        op->emitError() << "failed to run " << getArgument()
                        << " patterns for rewriting host constants";
        return signalPassFailure();
      }
    }

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
    if (failed(enforceFunctionCallingStylePolicy(
            rewriter, symbolTables, op, forceEntrypointsReturnAllocs))) {
      emitError(op.getLoc(),
                "failed to establish DPS connectivity in public functions");
      return signalPassFailure();
    }

    // Eliminate any straggling `tensor.empty` operations. Only run this on
    // functions in the host module.
    {
      FrozenRewritePatternSet patterns = [&]() {
        RewritePatternSet patterns_(ctx);
        patterns_.insert<RewriteEmptyTensor, CleanupAllocTensorOps,
                         RemoveRedundantMaterializeInDestPattern>(ctx);
        return patterns_;
      }();
      for (FunctionOpInterface func : op.getOps<FunctionOpInterface>()) {
        if (failed(applyPatternsGreedily(func, patterns))) {
          op->emitError() << "failed to run " << getArgument() << " patterns";
          return signalPassFailure();
        }
      }
    }
  }
};
} // namespace
