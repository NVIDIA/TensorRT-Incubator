//===- AllocTensors.cpp  --------------------------------------------------===//
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
///  Implementation of the `plan-alloc-tensors` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace plan {
#define GEN_PASS_DEF_PLANALLOCTENSORSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace plan
} // namespace mlir

using namespace mlir;
using namespace mlir::plan;

static constexpr int64_t kHostConstantToFromElementsNumElementsLimit = 16;

namespace {

/// Remap relevant analysis state of type T from `original` to `replacement`.
template <typename T>
static void remapLatticeState(DataFlowSolver &solver, Value original,
                              Value replacement) {
  if constexpr (!std::is_same_v<T, dataflow::Executable>) {
    if (const T *lattice = solver.lookupState<T>(original)) {
      T *latticeReplacement = solver.getOrCreateState<T>(replacement);
      latticeReplacement->getValue() = lattice->getValue();
    }
  } else {
    // do nothing for liveness analysis for the moment except create the state
    if (const auto *oldState =
            solver.lookupState<dataflow::Executable>(original)) {
      dataflow::Executable *newState = solver.getOrCreateState<T>(replacement);
      // Set to live if old state is live. We ignore change status.
      if (oldState->isLive())
        (void)newState->setToLive();
    }
  }
}

/// A rewrite listener that transfers replacements to updates to the solver
/// state.
class SolverStateListener : public RewriterBase::Listener {
public:
  SolverStateListener(DataFlowSolver &solver)
      : RewriterBase::Listener(), solver(solver) {}

private:
  void notifyOperationReplaced(Operation *op,
                               ValueRange replacements) override {
    for (auto [original, replacement] :
         llvm::zip_equal(op->getResults(), replacements)) {
      remapLatticeState<TensorKindLattice>(solver, original, replacement);
      remapLatticeState<dataflow::Lattice<dataflow::ConstantValue>>(
          solver, original, replacement);
      remapLatticeState<dataflow::Executable>(solver, original, replacement);
    }
    solver.eraseState(solver.getProgramPointAfter(op));
  }
  void notifyOperationReplaced(Operation *op, Operation *replacement) override {
    notifyOperationReplaced(op, replacement->getResults());
  }

  void notifyOperationErased(Operation *op) override {
    solver.eraseState(solver.getProgramPointAfter(op));
    for (Value res : op->getResults())
      solver.eraseState(res);
  }

  DataFlowSolver &solver;
};

} // namespace

/// Lower tensor.from_elements to a sequence of chained tensor.insert into a
/// `bufferization.alloc_tensor` in the specified memory space (which must
/// either be 'host' or 'host_pinned').
static Value lowerFromElementsOp(RewriterBase &rewriter,
                                 tensor::FromElementsOp fromElementsOp,
                                 MemorySpace memorySpace) {
  assert(memorySpace == MemorySpace::host ||
         memorySpace == MemorySpace::host_pinned &&
             "tensor.from_elements must be lowered into an allocation in "
             "a host-visible space");
  Location loc = fromElementsOp.getLoc();

  // Create the allocation.
  RankedTensorType tensorType =
      RankedTensorType::Builder(fromElementsOp.getType())
          .setEncoding(
              MemorySpaceAttr::get(rewriter.getContext(), memorySpace));
  auto allocOp = rewriter.create<bufferization::AllocTensorOp>(loc, tensorType,
                                                               ValueRange());
  allocOp.setMemorySpaceAttr(tensorType.getEncoding());

  // Handle the rank 0 case and early exit.
  if (tensorType.getRank() == 0) {
    Value replacement = rewriter.create<tensor::InsertOp>(
        loc, fromElementsOp.getElements().front(), allocOp.getResult(),
        ValueRange{});
    return replacement;
  }

  // Create the chain of `tensor.insert` operations.
  SmallVector<int64_t> basis =
      mlir::computeSuffixProduct(tensorType.getShape());
  Value result = allocOp.getResult();
  for (auto [i, element] : llvm::enumerate(fromElementsOp.getElements())) {
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

    MemorySpace originalMemorySpaceConstraint = MemorySpace::host_pinned;
    if (auto constraint =
            dyn_cast_or_null<MemorySpaceAttr>(op.getType().getEncoding())) {
      // A pre-specified 'device' constraint is not allowed.
      if (constraint.getValue() != MemorySpace::host &&
          constraint.getValue() != MemorySpace::host_pinned)
        return failure();
      originalMemorySpaceConstraint = constraint.getValue();
    }

    // Create a host allocation and insert the elements.
    Value hostReplacement =
        lowerFromElementsOp(rewriter, op, originalMemorySpaceConstraint);
    Value hostReplacementCasted =
        rewriter.create<tensor::CastOp>(loc, originalType, hostReplacement);
    if (placementInfo.isHostOnly()) {
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
    rewriter.replaceOpUsesWithIf(
        op, hostReplacementCasted, [&](OpOperand &use) {
          return TensorKindAnalysis::getStaticOperandTensorKind(use) ==
                 TensorKind::Host;
        });
    rewriter.replaceOpUsesWithIf(op, devReplacement, [&](OpOperand &use) {
      return TensorKindAnalysis::getStaticOperandTensorKind(use) !=
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

    Value source = op.getTensor();

    if (failed(replaceHostUsesWithHostAlloc(rewriter, source)))
      return failure();

    return success();
  }
};

/// Rewrite `arith.constant` host tensors that are used purely on the host using
/// `tensor.from_elements`.
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

    auto hostAllocOp = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(), op.getType(), ValueRange{}, Value{}, Value{},
        plan::MemorySpaceAttr::get(rewriter.getContext(),
                                   plan::MemorySpace::host));

    RankedTensorType type = cast<RankedTensorType>(elementsAttr.getType());
    SmallVector<int64_t> basis = mlir::computeSuffixProduct(type.getShape());
    Value result = hostAllocOp.getResult();
    for (unsigned i = 0; i < elements.size(); i++) {
      SmallVector<Value> offsets = llvm::map_to_vector(
          mlir::delinearize(i, basis), [&](int64_t x) -> Value {
            return rewriter.create<arith::ConstantIndexOp>(op.getLoc(), x);
          });
      result = rewriter.create<tensor::InsertOp>(op.getLoc(), elements[i],
                                                 result, offsets);
    }

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

/// Returns true if `op` is a DPS operation or an op that can be considered DPS.
/// Operations such as `scf.for` and `scf.while` are DPS in the tensor form but
/// don't strictly conform to DPS interface in the memref form.
static bool isDpsOrDpsLikeOp(Operation *op) {
  if (!op)
    return false;
  if (isa<DestinationStyleOpInterface>(op))
    return true;
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return TypeRange(whileOp.getInits()) == TypeRange(whileOp.getResults());
  return isa<scf::ForOp>(op);
}

/// Similar to the above, return the tied operand for a DPS operation and the
/// equivalent for scf.for/scf.while.
static OpOperand *getTiedOpOperand(Operation *op, OpResult result) {
  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op))
    return dpsOp.getTiedOpOperand(result);
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return &forOp.getInitArgsMutable()[result.getResultNumber()];
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return &whileOp.getInitsMutable()[result.getResultNumber()];
  llvm_unreachable("unhandled op type");
}

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

/// Walk the IR up from `v` by tracing DPS producers to init args. Returns the
/// final OpOperand reached where the value is either a block arg or produced by
/// a non-DPS operation.
static OpOperand *traverseDpsUseDefChain(OpOperand &v,
                                         Operation *resultDpsProducer) {
  OpOperand *lastTiedDpsOperandToResult = nullptr;
  Value current = v.get();
  while (true) {
    assert(isa<OpResult>(current) && "expected value to be an OpResult");
    lastTiedDpsOperandToResult =
        getTiedOpOperand(resultDpsProducer, cast<OpResult>(current));
    assert(lastTiedDpsOperandToResult && "expected tied operand");
    Operation *lastTiedDpsOperandProducer =
        lastTiedDpsOperandToResult->get().getDefiningOp();

    // If producer is null, then `lastTiedDpsOperand` is a BlockArgument.
    if (!lastTiedDpsOperandProducer)
      return lastTiedDpsOperandToResult;

    // If value is produced by a DPS op. If not, return failure. The caller will
    // handle this case by inserting a copy.
    if (!isDpsOrDpsLikeOp(lastTiedDpsOperandProducer))
      return lastTiedDpsOperandToResult;

    resultDpsProducer = lastTiedDpsOperandProducer;
    current = lastTiedDpsOperandToResult->get();
  }
  llvm_unreachable("expected traversal loop to return");
}

/// Operand `v` is the output of a DPS op `dpsProducer`. Traverse use-def chain
/// of the `dpsProducer` to find the DPS op whose corresponding DPS init
/// argument is produced by `tensor.empty` or is a BlockArgument and return the
/// OpOperand for the init arg.
static LogicalResult
travelUseDefChainAndUpdateUses(RewriterBase &rewriter, OpOperand &v,
                               Operation *resultDpsProducer,
                               GetOrCreateBlockArgFunc getDpsArgument) {
  OpOperand *lastTiedDpsOperand = traverseDpsUseDefChain(v, resultDpsProducer);
  if (&v == lastTiedDpsOperand)
    return failure();

  if (!isa<OpResult>(lastTiedDpsOperand->get()))
    return failure();

  if (auto emptyOp = lastTiedDpsOperand->get().getDefiningOp()) {
    if (!isa<tensor::EmptyOp, bufferization::AllocTensorOp>(emptyOp))
      return failure();

    // Add DPS arg to the function for replacing `tensor.empty()` at this
    // use.
    FailureOr<BlockArgument> destArg =
        getDpsArgument(emptyOp->getResultTypes().front(), v);
    if (failed(destArg))
      return failure();

    // Replace last tied DPS operand with the new destination arg.
    lastTiedDpsOperand->assign(*destArg);
    return success();
  }

  return failure();
}

/// Rewrites a block to conform to DPS style. This is done in two steps: 1) for
/// each terminator returned value `v`, check whether `v` is the output of a DPS
/// style operation and whether tied DPS operand `v_tied` is produced by
/// `tensor.empty` op (recursively, by traveling use-def chain). 2.A) If yes,
/// get/create the DPS block argument and rewrite the relevant DPS op so that
/// the DPS init argument of this op is the newly added function return argument
/// (`arg`) rather than the output of `tensor.empty` (`v_tied` -> `arg`). 2.B)
/// If no, get/create a DPS block argument and create a
/// `bufferization.materialize_in_destination` op where source is the return
/// value of the function (`v`) and destination is newly added DPS argument
/// `arg` and update the terminator operand.
///
/// The `getDpsArgument` callback allows for either a) inserting the new DPS
/// BlockArgument if it does not already exist or b) identifying the existing
/// DPS block argument if the block is already in DPS style.
static LogicalResult
correctDestinationPassingStyleBlock(RewriterBase &rewriter, Block *block,
                                    GetOrCreateBlockArgFunc getDpsArgument) {
  Operation *returnOp = block->getTerminator();
  MutableArrayRef<OpOperand> operands = returnOp->getOpOperands();
  rewriter.setInsertionPoint(returnOp);
  for (OpOperand &v : operands) {

    // If returned type is not tensor, we do nothing.
    if (!llvm::isa<RankedTensorType>(v.get().getType()))
      continue;

    Operation *producer = v.get().getDefiningOp();
    if (isDpsOrDpsLikeOp(producer)) {
      // Producer is a DPS op. Travel use-def chain, add DPS arg ,and update
      // uses. A return value here can be the original value `v` or the output
      // of the `bufferization.materialize_in_destination` op.
      if (succeeded(travelUseDefChainAndUpdateUses(rewriter, v, producer,
                                                   getDpsArgument)))
        continue;
    }
    // Producer of the result is not a DPS op or is a block arg.
    // Add `bufferization.materialize_in_destination` to make the return
    // value output of a DPS op.
    // First add argument to the function for the result `v`
    FailureOr<BlockArgument> destArg = getDpsArgument(v.get().getType(), v);
    if (failed(destArg))
      return failure();
    v.assign(rewriter
                 .create<bufferization::MaterializeInDestinationOp>(
                     v.get().getLoc(),
                     /*source=*/v.get(),
                     /*dest=*/*destArg)
                 .getResult());
  }

  return success();
}

/// Rewrites non-private function/s from the top level module op (ModuleOp) in
/// the destination passing style (DPS). This is done in three steps, 1) Find
/// non-private function/s from the top level module op. 2) For each non-private
/// function `f`, for each return value `v` (iterate over operands of
/// `func::ReturnOp` of `f`), check whether `v` is the output of a DPS style
/// operation and whether tied DPS operand `v_tied` is produced by
/// `tensor.empty` op (recursively, by traveling use-def chain). 3.A) If yes,
/// create a new DPS init argument `arg` in the function and rewrite the
/// relevant DPS op so that the DPS init argument of this op is the newly added
/// function return argument (`arg`) rather than the output of `tensor.empty`
/// (`v_tied` -> `arg`). 3.B) If no, create a DPS init argument in the function
/// `arg` and create a `bufferization.materialize_in_destination` op where
/// source is the return value of the function (`v`) and destination is newly
/// added DPS argument `arg`. Finally, update `func::ReturnOp` is updated so
/// that result of this newly added `bufferization.materialize_in_destination`
/// op is returned. A special case of 3.A is handled is when same SSA value is
/// returned multiple times. This case is handled by adding explicit copy using
/// `bufferization.materialize_in_destination` where destination is newly added
/// argument and source is duplicated SSA value.
static LogicalResult rewriteNotPrivateFuncsToDPS(RewriterBase &rewriter,
                                                 ModuleOp op) {
  // Find non-private functions
  SmallVector<func::FuncOp> nonPrivateFunctions;
  auto moduleFunctions = op.getOps<func::FuncOp>();
  for (auto f : moduleFunctions) {
    // Update all functions except functions explicitly marked private
    if (!f.isPrivate())
      nonPrivateFunctions.push_back(f);
  }
  if (nonPrivateFunctions.empty())
    return success();

  for (func::FuncOp nonPrivateFunction : nonPrivateFunctions) {
    // All functions should be single-block at this point.
    if (nonPrivateFunction.getBlocks().size() != 1)
      return failure();

    // Check if the function is already in DPS style.
    Operation *term = nonPrivateFunction.getBody().front().getTerminator();
    if (llvm::all_of(term->getOpOperands(), [&](OpOperand &operand) {
          BlockArgument arg =
              isDpsOrDpsLikeOp(operand.get().getDefiningOp())
                  ? dyn_cast<BlockArgument>(
                        traverseDpsUseDefChain(operand,
                                               operand.get().getDefiningOp())
                            ->get())
                  : dyn_cast<BlockArgument>(operand.get());
          if (!arg)
            return false;
          return nonPrivateFunction.getArgAttr(
                     arg.getArgNumber(), PlanDialect::kResultArgAttrName) !=
                 nullptr;
        }))
      continue;

    if (failed(correctDestinationPassingStyleBlock(
            rewriter, &nonPrivateFunction.getBody().front(),
            /*getDpsArgument=*/
            [&](Type argType,
                OpOperand &returnOperand) -> FailureOr<BlockArgument> {
              return updateFunctionWithNewDpsArg(
                  nonPrivateFunction, returnOperand.get().getLoc(), argType,
                  returnOperand.getOperandNumber());
            })))
      return failure();
  }
  return success();
}

/// For `region` of a `scf.for` or `scf.while` operation, return the
/// block/region arguments that correspond to the loop carried values.
static Block::BlockArgListType getLoopCarriedArguments(Region *region) {
  Operation *op = region->getParentOp();
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return forOp.getRegionIterArgs();
  if (isa<scf::WhileOp>(op))
    return region->getArguments();
  llvm_unreachable("unhandled loop op type");
}

/// For `region` of a `scf.for` or `scf.while` operation, return the operands
/// yielded by the terminator to the successor region.
static MutableOperandRange getLoopRegionYieldedOperands(Region *region) {
  Operation *op = region->getParentOp();
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return MutableOperandRange(forOp.getBody()->getTerminator());
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    if (region == &whileOp.getBefore())
      return whileOp.getConditionOp().getArgsMutable();
    return MutableOperandRange(whileOp.getAfterBody()->getTerminator());
  }
  llvm_unreachable("unhandled loop op type");
}

/// Rewrites regions of `scf.for` and `scf.while` to conform to DPS style, if
/// posible. The loops must have operands/results that make this possible (in
/// the case of `scf.while`). DPS operations internal to the regions will be
/// linked to the region iteration arguments, if possible. This increases the
/// liklihood that loops iteration arguments can be bufferized in-place, which
/// is often critical for performance.
static LogicalResult rewriteLoopBodyRegionsToDPS(RewriterBase &rewriter,
                                                 ModuleOp op) {
  SmallVector<Region *> loopRegions;
  auto moduleFunctions = op.getOps<func::FuncOp>();
  for (auto f : moduleFunctions) {
    f.walk([&](LoopLikeOpInterface loopOp) {
      if (llvm::isa<scf::ForOp, scf::WhileOp>(*loopOp))
        llvm::append_range(loopRegions, loopOp.getLoopRegions());
    });
  }

  if (loopRegions.empty())
    return success();

  for (Region *region : loopRegions) {
    // Skip regions with multiple blocks. At this point, we expect all loop
    // bodies should be single-block.
    if (region->getBlocks().size() != 1)
      continue;
    Block &body = region->front();

    // Ignore regions that don't have number/types of terminator operands
    // match number/types of arguments.
    Block::BlockArgListType iterArgs = getLoopCarriedArguments(region);
    MutableOperandRange yieldedOperands = getLoopRegionYieldedOperands(region);
    if (yieldedOperands.size() != iterArgs.size() ||
        TypeRange(yieldedOperands) != TypeRange(iterArgs))
      continue;

    // Skip if the body is already in DPS style (at least to the best that we
    // can check).
    if (llvm::all_of(yieldedOperands, [&](OpOperand &operand) {
          if (auto dpsProducer =
                  operand.get().getDefiningOp<DestinationStyleOpInterface>())
            return isa<BlockArgument>(
                traverseDpsUseDefChain(operand, dpsProducer)->get());
          return isa<BlockArgument>(operand.get());
        }))
      continue;

    if (failed(correctDestinationPassingStyleBlock(
            rewriter, &body,
            [&](Type argType,
                OpOperand &returnOperand) -> FailureOr<BlockArgument> {
              // Account for potential offset due to ops like scf.condition.
              unsigned yieldedIdx = returnOperand.getOperandNumber() -
                                    yieldedOperands[0].getOperandNumber();
              assert(yieldedIdx < body.getNumArguments() &&
                     "expected yield operand idx to be smaller than number of "
                     "block arguments");
              BlockArgument blockArg = iterArgs[yieldedIdx];
              assert(argType == blockArg.getType() &&
                     "expected yielded operand type to match block arg type");
              return blockArg;
            })))
      return failure();
  }

  return success();
}

namespace {

class AllocTensorsPass
    : public plan::impl::PlanAllocTensorsPassBase<AllocTensorsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp op = getOperation();
    MLIRContext *ctx = &getContext();

    SymbolTableCollection symbolTable;
    DataFlowConfig config;
    config.setInterprocedural(false);
    DataFlowSolver solver(config);
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TensorKindAnalysis>(symbolTable);

    if (failed(solver.initializeAndRun(op))) {
      op.emitError() << "failed to run TensorKindAnalysis";
      return signalPassFailure();
    }

    {
      SolverStateListener solverAwareListener(solver);
      GreedyRewriteConfig config;
      config.listener = &solverAwareListener;
      RewritePatternSet patterns(ctx);
      patterns.insert<HostShapeConstantsToAllocTensorPattern,
                      RewriteFromElements, TensorDeviceExtractRewriter,
                      LargeHostConstantsAllocTensorPattern>(ctx, solver);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        op->emitError() << "failed to run " << getArgument()
                        << " patterns for rewriting host constants";
        return signalPassFailure();
      }
    }

    IRRewriter rewriter(ctx);

    // First rewrite public functions to conform to DPS style.
    if (!forceEntrypointsReturnAllocs &&
        failed(rewriteNotPrivateFuncsToDPS(rewriter, op))) {
      op->emitError("Failed to convert non-private functions to DPS");
      return signalPassFailure();
    }

    // Rewrite SCF for and while loop bodies for better bufferization results,
    // if possible.
    if (failed(rewriteLoopBodyRegionsToDPS(rewriter, op))) {
      op->emitError("failed to convert loop body regions to DPS");
      return signalPassFailure();
    }

    // Eliminate any straggling `tensor.empty` operations.
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<RewriteEmptyTensor, CleanupAllocTensorOps,
                      RemoveRedundantMaterializeInDestPattern>(ctx);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        op->emitError() << "failed to run " << getArgument() << " patterns";
        return signalPassFailure();
      }
    }
  }
};
} // namespace
