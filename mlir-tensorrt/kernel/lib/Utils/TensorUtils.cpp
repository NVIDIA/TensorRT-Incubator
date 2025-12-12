//===- TensorUtils.cpp ----------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-kernel/Utils/TensorUtils.h"
#include "mlir-kernel/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <deque>

#define DEBUG_TYPE "kernel-tensor-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

using namespace mlir;
using namespace mlir::tensor;
using namespace mlir::tensor_ext;

static FailureOr<std::tuple<OpFoldResult, OpFoldResult, OpFoldResult>>
sliceExpandShapeCondition(RewriterBase &rewriter, Location loc,
                          ArrayRef<OpFoldResult> offsetGroup,
                          ArrayRef<OpFoldResult> sizeGroup,
                          ArrayRef<OpFoldResult> stridesGroup,
                          ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {

  OpFoldResult offset = rewriter.getIndexAttr(0);
  OpFoldResult size = rewriter.getIndexAttr(0);

  // We expect the stride to be 1 or equal to the stride of the one dimension
  // that has non-unit size slice.
  int64_t stride = 1;
  unsigned numNonUnitSizeSlicedDims = 0;
  for (auto [o, s, st, dimSize, dimStride, idx] :
       llvm::zip(offsetGroup, sizeGroup, stridesGroup, shape, strides,
                 llvm::seq<unsigned>(0, shape.size()))) {
    std::optional<int64_t> sizeStatic = getConstantIntValue(s);
    if (!sizeStatic)
      return failure();
    bool isSliced = !isConstantIntValue(o, 0) ||
                    !isConstantIntValue(s, dimSize) ||
                    !isConstantIntValue(st, 1);
    bool isNonUnitSize = !isConstantIntValue(s, 1);
    if (isSliced && isNonUnitSize) {
      numNonUnitSizeSlicedDims++;
      if (numNonUnitSizeSlicedDims > 1)
        return failure();
      stride = dimStride;
    }

    // Accumulate the offset.
    auto d0 = rewriter.getAffineSymbolExpr(0);
    auto d1 = rewriter.getAffineSymbolExpr(1);
    if (!isConstantIntValue(o, 0)) {
      offset = affine::makeComposedFoldedAffineApply(
          rewriter, loc, d0 * dimStride + d1,
          ArrayRef<OpFoldResult>{o, offset});
    }

    // Update the size by multiplying.
    auto currSize = getConstantIntValue(size);
    if (currSize && currSize == 0)
      size = s;
    else
      size = affine::makeComposedFoldedAffineApply(
          rewriter, loc, d0 * d1, ArrayRef<OpFoldResult>{s, size});
  }
  return std::make_tuple(offset, size,
                         OpFoldResult(rewriter.getIndexAttr(stride)));
}

FailureOr<tensor::ExtractSliceOp> tensor_ext::materializeSliceOfExpandShape(
    RewriterBase &rewriter, tensor::ExpandShapeOp op,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides) {
  // For now only allow unit-strides.
  if (!llvm::all_of(strides, [](OpFoldResult r) {
        std::optional<int64_t> constVal = getConstantIntValue(r);
        return constVal && *constVal == 1;
      }))
    return failure();

  // Just static shape for now.
  if (!op.getResultType().hasStaticShape())
    return failure();

  ArrayRef<int64_t> shape = op.getResultType().getShape();
  SmallVector<ReassociationIndices> reassociationIndices =
      op.getReassociationIndices();
  SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
  // In order for this to be possible, for each reassociation group of size M,
  // [dimN, dimN+1, .., dimN+M-1], the following cases are allowed:
  // - the last dim (dimN+M-1) is sliced, and optionally one other dimension
  //   may be sliced.
  // We cannot allow multiple dimensions that are not the last to be sliced
  // because this would generate a variable stride.
  for (const auto &re : reassociationIndices) {
    ArrayRef<int64_t> subShape = shape.slice(re.front(), re.size());
    ArrayRef<OpFoldResult> subOffsets = offsets.slice(re.front(), re.size());
    ArrayRef<OpFoldResult> subSizes = sizes.slice(re.front(), re.size());
    ArrayRef<OpFoldResult> subStrides = strides.slice(re.front(), re.size());
    SmallVector<int64_t> subShapeBasis = mlir::computeSuffixProduct(subShape);
    FailureOr<std::tuple<OpFoldResult, OpFoldResult, OpFoldResult>> ofrs =
        sliceExpandShapeCondition(rewriter, op.getLoc(), subOffsets, subSizes,
                                  subStrides, subShape, subShapeBasis);
    if (failed(ofrs))
      return failure();
    auto [newOfft, newSize, newStr] = *ofrs;
    newOffsets.push_back(newOfft);
    newSizes.push_back(newSize);
    newStrides.push_back(newStr);
  }
  return rewriter.create<tensor::ExtractSliceOp>(
      op.getLoc(), op.getSrc(), newOffsets, newSizes, newStrides);
}

/// Replace `sliceOp` with `tileValue`, inserting reshapes to try to workaround
/// shape differences when `sliceOp` is rank reducing.
static LogicalResult replaceSliceWithTile(RewriterBase &rewriter,
                                          tensor::ExtractSliceOp sliceOp,
                                          Value tileValue) {
  RankedTensorType originalType = sliceOp.getResultType();
  if (originalType == tileValue.getType()) {
    rewriter.replaceOp(sliceOp, tileValue);
    return success();
  }

  std::optional<SmallVector<ReassociationIndices>> re =
      getReassociationIndicesForReshape(cast<ShapedType>(sliceOp.getType()),
                                        cast<ShapedType>(tileValue.getType()));
  if (!re)
    return failure();
  int64_t srcRank = originalType.getRank();
  int64_t newRank = cast<ShapedType>(tileValue.getType()).getRank();
  if (srcRank < newRank) {
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        sliceOp, sliceOp.getType(), tileValue, *re);
    return success();
  }
  if (srcRank > newRank) {
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        sliceOp, sliceOp.getType(), tileValue, *re);
    return success();
  }
  return failure();
}

FailureOr<TilingResult> tensor_ext::replaceExtractSliceWithTiledProducer(
    RewriterBase &rewriter, tensor::ExtractSliceOp sliceOp, OpResult producer) {
  // `TilingInterface` currently only supports strides being 1.
  if (llvm::any_of(sliceOp.getMixedStrides(), [](OpFoldResult ofr) {
        return !isConstantIntValue(ofr, 1);
      }))
    return failure();

  if (auto producerOp = dyn_cast<TilingInterface>(producer.getOwner())) {

    // For linalg ops, we can't tile if there are negative coefficients on the
    // indexing maps.
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(producerOp.getOperation())) {
      if (linalg_ext::hasNegativeMultiplicationCoefficients(
              linalgOp.getIndexingMapsArray()))
        return failure();
    }

    FailureOr<TilingResult> tile = producerOp.generateResultTileValue(
        rewriter, producer.getResultNumber(), sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes());
    if (failed(tile))
      return failure();
    if (failed(
            replaceSliceWithTile(rewriter, sliceOp, tile->tiledValues.front())))
      return failure();
    return *tile;
  }

  if (auto expandShape = dyn_cast<ExpandShapeOp>(producer.getOwner())) {
    FailureOr<tensor::ExtractSliceOp> tile = materializeSliceOfExpandShape(
        rewriter, expandShape, sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
    if (failed(tile))
      return failure();
    if (failed(replaceSliceWithTile(rewriter, sliceOp, *tile)))
      return failure();
    return TilingResult{SmallVector<Operation *>{*tile},
                        SmallVector<Value>{*tile}, SmallVector<Operation *>{}};
  }

  return failure();
}

/// Attempt to fuse the producer of a slice of a `scf.forall` `shared_outs`
/// argument (`blockArg`). On success, a pair (original `shared_outs` tied
/// operand, the replacement for `sliceOp`) are returned. This function ensures
/// that the fused producer's dest operands are correctly updated.
FailureOr<std::pair<OpResult, OpResult>>
tensor_ext::tryToFuseThroughSharedOutsBlockArg(RewriterBase &rewriter,
                                               tensor::ExtractSliceOp sliceOp,
                                               scf::ForallOp forallOp,
                                               BlockArgument blockArg) {
  auto tiedOperand =
      dyn_cast_or_null<OpResult>(forallOp.getTiedOpOperand(blockArg)->get());
  if (!tiedOperand)
    return failure();
  auto tilingProducer =
      llvm::dyn_cast_or_null<TilingInterface>(tiedOperand.getDefiningOp());
  if (!tilingProducer)
    return failure();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(sliceOp);

  SmallVector<Value> tilingProducerDestValues;
  if (failed(tensor::getOrCreateDestinations(rewriter, tilingProducer.getLoc(),
                                             tilingProducer,
                                             tilingProducerDestValues)))
    return failure();

  // Clone the producer into the body. This is always legal. However, we also
  // make the correct substitutions to ensure that the dest operands are updated
  // to the appropriate shared_outs block arguments. We then tile this clone
  // instead of the original, since tiling the original would not have correct
  // dest args.
  IRMapping mapping;
  unsigned resultIdx = tiedOperand.getResultNumber();
  mapping.map(tilingProducerDestValues[resultIdx], blockArg);
  Operation *clonedProducer = rewriter.clone(*tilingProducer, mapping);

  // Perform the TilingInterface-based substitution.
  FailureOr<TilingResult> tilingResult =
      tensor_ext::replaceExtractSliceWithTiledProducer(
          rewriter, sliceOp, clonedProducer->getOpResult(resultIdx));
  if (failed(tilingResult))
    return failure();
  assert(!tilingResult->tiledValues.empty() &&
         "expected non-empty tiled values");
  OpResult fusedProducerValue =
      cast<OpResult>(tilingResult->tiledValues.front());
  return std::make_pair(tiedOperand, fusedProducerValue);
}

/// Given the target operation, iterate over its operands to check if
/// their defining ops are ExtractSliceOps. If so, enqueue them into the
/// candidates queue.
static void
enqueueCandidateExtractSliceOps(Operation *target,
                                std::deque<tensor::ExtractSliceOp> &candidates,
                                int64_t searchDepth = 1) {
  struct SearchQueue {
    Operation *target;
    int64_t depth;
  };
  std::deque<SearchQueue> searchQueue;
  llvm::SmallPtrSet<Operation *, 4> visited;
  searchQueue.push_back({target, 1});
  while (!searchQueue.empty()) {
    SearchQueue candidate = searchQueue.front();
    searchQueue.pop_front();
    if (candidate.depth <= searchDepth) {
      for (Value operand : candidate.target->getOperands()) {
        if (auto sliceOp = operand.getDefiningOp<tensor::ExtractSliceOp>()) {
          if (!llvm::is_contained(candidates, sliceOp)) {
            LLVM_DEBUG(DBGS()
                       << "adding candidate slice op: " << sliceOp << "\n");
            candidates.push_back(sliceOp);
            visited.insert(sliceOp);
            continue;
          }
        }
        if (auto producer = operand.getDefiningOp();
            producer && producer->getBlock() == target->getBlock() &&
            !visited.contains(producer)) {
          searchQueue.push_back({producer, candidate.depth + 1});
          visited.insert(producer);
        }
      }
    }
  }
}

static FailureOr<std::pair<OpResult, OpResult>>
getFusableProducer(RewriterBase &rewriter,
                   tensor::ExtractSliceOp candidateSliceOp) {
  Value source = candidateSliceOp.getSource();

  // If the source is a shared_outs block arg, then apply the special fusion
  // logic.
  if (auto blockArg = dyn_cast<BlockArgument>(source)) {
    auto loopOp = dyn_cast<scf::ForallOp>(blockArg.getOwner()->getParentOp());
    if (!loopOp)
      return failure();
    return tryToFuseThroughSharedOutsBlockArg(rewriter, candidateSliceOp,
                                              loopOp, blockArg);
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(candidateSliceOp);
  OpResult producerResult = cast<OpResult>(source);
  FailureOr<TilingResult> tilingResult =
      tensor_ext::replaceExtractSliceWithTiledProducer(
          rewriter, candidateSliceOp, producerResult);
  if (failed(tilingResult))
    return failure();
  assert(!tilingResult->tiledValues.empty() &&
         "expected non-empty tiled values");
  return std::make_pair(producerResult,
                        cast<OpResult>(tilingResult->tiledValues.front()));
}

/// Fuse all fusable producers greedily into the given `target`
/// op. It runs in an iterative manner until nothing can be further fused.
LogicalResult tensor_ext::fuseGreedily(Operation *target,
                                       RewriterBase &rewriter,
                                       bool removeDeadProducer) {
  OpBuilder::InsertionGuard g(rewriter);

  if (target->getNumResults() == 0)
    return failure();

  // Initialize the worklist of candidate ExtractSliceOps
  std::deque<tensor::ExtractSliceOp> candidates;
  enqueueCandidateExtractSliceOps(target, candidates, /*depth=*/2);

  // Fuse the head candidate from the worklist. This step may result in
  // new ExtractSliceOps, so we search for new candidates and add to the
  // worklist
  while (!candidates.empty()) {
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();
    LLVM_DEBUG(DBGS() << "considering candidate: " << candidateSliceOp << "\n");

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(candidateSliceOp);

    // Try to fuse producer.
    FailureOr<std::pair<OpResult, OpResult>> fusionResult =
        getFusableProducer(rewriter, candidateSliceOp);
    if (failed(fusionResult)) {
      LLVM_DEBUG(DBGS() << "failed  to fuse\n");
      continue;
    }

    auto [originalProducerResult, newProducerResult] = *fusionResult;
    Operation *fusedProducer = newProducerResult.getDefiningOp();

    // Search for new candidates from the fused producer
    enqueueCandidateExtractSliceOps(fusedProducer, candidates);

    // Update scf.forall args accordingly
    SmallVector<Operation *> destToUpdate;
    if (auto forallOp = target->getParentOfType<scf::ForallOp>()) {
      destToUpdate.push_back(forallOp);
    }
    for (Operation *destArgUser : destToUpdate) {
      SmallVector<Value> unfusedProducerOpDestValues;
      if (auto unfusedProducerOp = dyn_cast<DestinationStyleOpInterface>(
              originalProducerResult.getOwner())) {
        unfusedProducerOpDestValues = unfusedProducerOp.getDpsInits();
        for (OpOperand &use :
             llvm::make_early_inc_range(originalProducerResult.getUses())) {
          if (use.getOwner() == destArgUser) {
            if (isa<scf::ForallOp>(destArgUser)) {
              unsigned resultNumber = originalProducerResult.getResultNumber();
              unsigned operandNumber = use.getOperandNumber();
              destArgUser->setOperand(
                  operandNumber, unfusedProducerOpDestValues[resultNumber]);
            }
          }
        }
      }
    }

    if (removeDeadProducer &&
        originalProducerResult.getDefiningOp()->use_empty())
      rewriter.eraseOp(originalProducerResult.getDefiningOp());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// For/Forall Interchange
//===----------------------------------------------------------------------===//

/// Collect dependencies recursively by following operands of `op`. For all
/// ops within the `forallOp` body, we put them in `dep`. This is used to
/// collect dependencies rooted at a `tensor.extract_slice` which must remain in
/// the `forallOp` during interchange.
///
/// We also verify dominance conditions required for interchanging `forOp`
/// and `forallOp` while keeping all the collected `dep` operations within
/// `forallOp`. For this to occur, we cannot have a `dep` that dominates
/// `forallOp` but not `forOp` (in that case it would be in the body of the for
/// op but sitting above the forallOp).
static LogicalResult collectDependencies(llvm::SmallDenseSet<Operation *> &dep,
                                         Operation *op, Block *block,
                                         const DominanceInfo &domInfo,
                                         scf::ForOp forOp,
                                         scf::ForallOp forallOp) {
  // Check that `defOp` isn't a block arg of `forOp` and isn't in the body block
  // of `forOp` prior to the `forallOp`
  if (forOp->isProperAncestor(op) && domInfo.properlyDominates(op, forallOp))
    return failure();
  if (op->getBlock() != block)
    return success();
  // Success if already in set.
  if (!dep.insert(op).second)
    return success();
  for (Value operand : op->getOperands()) {
    if (Operation *defOp = operand.getDefiningOp()) {
      if (failed(
              collectDependencies(dep, defOp, block, domInfo, forOp, forallOp)))
        return failure();
      continue;
    }
    if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      Operation *owner = blockArg.getOwner()->getParentOp();
      if (owner == forallOp)
        continue;
      return failure();
    }
  }
  return success();
}
namespace {
/// A utility for discovering a potential for/forall interchange and collecting
/// the required information to perform the transformation.
struct InterchangeForallAndForInfo {

  LogicalResult analyze(scf::ForallOp op) {
    DominanceInfo domInfo(op);

    for (auto [outsIdx, outsOperand] :
         llvm::enumerate(op.getOutputsMutable())) {
      Value v = outsOperand.get();
      auto blockArg = dyn_cast<BlockArgument>(v);
      if (!blockArg)
        return failure();

      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      if (!forOp || op->getBlock() != forOp.getBody())
        return failure();

      if (candidate && candidate != forOp)
        return failure();
      if (!candidate)
        candidate = forOp;

      if (forOp.getRegionIterArgs().size() != op.getOutputs().size() ||
          forOp.getRegionIterArgs()[outsIdx] != outsOperand.get())
        return failure();

      BlockArgument forIterArg = forOp.getRegionIterArgs()[outsIdx];
      if (!forIterArg.hasOneUse())
        return failure();

      auto forYielded =
          dyn_cast<OpResult>(forOp.getTiedLoopYieldedValue(forIterArg)->get());
      if (!forYielded || forYielded.getOwner() != op ||
          forYielded.getResultNumber() != outsIdx)
        return failure();

      // Collect insert slice.
      BlockArgument forallBlockArg = op.getRegionOutArgs()[outsIdx];
      SmallVector<Operation *> combiners = op.getCombiningOps(forallBlockArg);
      if (combiners.size() != 1)
        return failure();

      auto insertSliceOp =
          cast<tensor::ParallelInsertSliceOp>(combiners.front());
      parallelInsertOps.push_back(insertSliceOp);
      tensor::ExtractSliceOp matchingExtract{};
      for (OpOperand &use : forallBlockArg.getUses()) {
        if (use.getOwner() == insertSliceOp)
          continue;
        if (auto extract = dyn_cast<tensor::ExtractSliceOp>(use.getOwner())) {
          if (matchingExtract)
            return failure();
          if (use.get() != extract.getSource())
            return failure();
          if (extract.getOffsets() != insertSliceOp.getOffsets() ||
              extract.getSizes() != insertSliceOp.getSizes() ||
              extract.getStrides() != insertSliceOp.getStrides())
            return failure();
          matchingExtract = extract;
          continue;
        }
        return failure();
      }
      if (!matchingExtract)
        return failure();

      matchingExtracts.insert(matchingExtract);
      if (failed(collectDependencies(parallelInsertDeps.emplace_back(),
                                     matchingExtract, op.getBody(), domInfo,
                                     forOp, op)))
        return failure();
    }

    // No ops between the forall and for terminator.
    if (op->getNextNode() != candidate.getBody()->getTerminator())
      return failure();

    return success();
  }

  /// Candiate 'scf.for' op to interchange.
  scf::ForOp candidate;

  /// For each parallel insert slice, get the index dependencies within 'forall'
  /// op body.
  SmallVector<llvm::SmallDenseSet<Operation *>> parallelInsertDeps;
  SmallVector<tensor::ParallelInsertSliceOp> parallelInsertOps;

  /// Matching "extract slice" for each parallel insert slice source.
  llvm::SmallSetVector<tensor::ExtractSliceOp, 4> matchingExtracts;
};

} // namespace

LogicalResult tensor_ext::interchangeForallAndFor(RewriterBase &rewriter,
                                                  scf::ForallOp op,
                                                  scf::ForOp forOp) {
  InterchangeForallAndForInfo info;
  if (failed(info.analyze(op)))
    return failure();

  if (forOp && forOp != info.candidate)
    return failure();

  forOp = info.candidate;

  // Replace forall results with for results.
  rewriter.replaceAllUsesWith(forOp->getResults(), op->getResults());

  // Perform rewriring of block args and terminators.
  rewriter.startOpModification(op);
  rewriter.startOpModification(forOp);
  rewriter.startOpModification(forOp.getBody()->getTerminator());
  for (auto [idx, forallArg, forArg] :
       llvm::enumerate(op.getRegionOutArgs(), forOp.getRegionIterArgs())) {

    // Each forall block argument should have one use besides terminator,
    // which is the extract slice op.
    // Set extract slice source to be the for iter arg.
    tensor::ExtractSliceOp extractOp = info.matchingExtracts.getArrayRef()[idx];
    tensor::ParallelInsertSliceOp parallelInsertSliceOp =
        info.parallelInsertOps[idx];

    rewriter.modifyOpInPlace(extractOp, [&, forallArg = forallArg]() {
      extractOp.getSourceMutable().assign(forallArg);
    });

    // Set users of extract op. Users should now use the 'for' region iter
    // arg.
    rewriter.replaceAllUsesWith(extractOp, forArg);

    // Rewire the scf.for to yield source of parallel_insert_slice.
    Value insertSource = parallelInsertSliceOp.getSource();
    forOp.getTiedLoopYieldedValue(forArg)->assign(insertSource);

    // Update 'parallel_insert_slice' source to be 'scf.for' result.
    OpResult forResult = forOp.getTiedLoopResult(forArg);
    rewriter.modifyOpInPlace(parallelInsertSliceOp, [&]() {
      parallelInsertSliceOp.getSourceMutable().assign(forResult);
    });

    // Modify scf result and block argument type.
    forResult.setType(insertSource.getType());
    forArg.setType(insertSource.getType());

    // For init becomes extract slice output.
    // Forall init becomes original for init.
    Value originalForInit = forOp.getTiedLoopInit(forArg)->get();
    forOp.getTiedLoopInit(forArg)->assign(extractOp);
    op.getTiedLoopInit(forallArg)->assign(originalForInit);
  }
  rewriter.finalizeOpModification(forOp.getBody()->getTerminator());
  rewriter.finalizeOpModification(forOp);
  rewriter.finalizeOpModification(op);

  // Move 'forall' body ops to 'for' body except for extract slice ops and
  // terminators.
  Block::iterator it = op->getBlock()->begin();
  if (Operation *opPriorToForall = op->getPrevNode())
    it = Block::iterator(opPriorToForall);
  for (Operation &inner :
       llvm::make_early_inc_range(op.getBody()->without_terminator())) {
    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(inner)) {
      if (info.matchingExtracts.contains(extractOp))
        continue;
    }
    if (llvm::any_of(info.parallelInsertDeps,
                     [&](const auto &deps) { return deps.contains(&inner); }))
      continue;
    rewriter.moveOpAfter(&inner, op->getBlock(), it);
    it = Block::iterator(&inner);
  }

  // Move forall to before for.
  rewriter.moveOpBefore(op, forOp);

  // Move `forOp` for within forall just before the terminator.
  rewriter.moveOpBefore(forOp, op.getBody()->getTerminator());

  return success();
}
