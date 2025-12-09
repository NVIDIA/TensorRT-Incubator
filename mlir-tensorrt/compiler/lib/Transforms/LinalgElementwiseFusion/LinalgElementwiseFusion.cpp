//===- LinalgElementwiseFusion.cpp ---------------------------------------===//
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
/// This pass is a pre-processing pass that fuses elementwise operations in the
/// IR. It is used to prepare the IR for the kernel segmentation pass
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mtrt {
#define GEN_PASS_DEF_LINALGELEMENTWISEFUSIONPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mtrt

using namespace mlir;

/// Returns true if the producer is a copy/fill/transpose or projection, i.e. it
/// performs no computation is is just yielding an argument.
static bool isCopyFillTransposeOrProject(Operation *producerOp) {
  Block &body = producerOp->getRegion(0).front();
  if (std::begin(body)->hasTrait<OpTrait::IsTerminator>())
    return true;

  // A 'tensor.extract'-like operation is a gather, which we don't count as pure
  // data-movement since gather operations like im2col have certain significance
  // and shouldn't be combined with the contracting consumer in this pass.
  if (llvm::any_of(body.without_terminator(),
                   [](Operation &op) { return isa<tensor::ExtractOp>(op); }))
    return false;

  if (llvm::all_of(body.getArguments(),
                   [](BlockArgument arg) { return arg.use_empty(); }))
    return true;

  return false;
}

/// This function controls whether the linalg elementwise fusion pattern should
/// fuse the producer of `candidate` operand with the consumer (linalg owner of
/// candidate operand).
static bool shouldFuseProducer(OpOperand *fusedOperand) {
  Operation *producerOp = fusedOperand->get().getDefiningOp();
  Operation *consumerOp = fusedOperand->getOwner();

  auto linalgConsumerOp = dyn_cast<linalg::LinalgOp>(consumerOp);
  if (!linalgConsumerOp)
    return false;

  // Don't fuse if all of the consumer maps aren't projected permutations.
  if (!llvm::all_of(linalgConsumerOp.getIndexingMapsArray(), [](AffineMap map) {
        return map.isProjectedPermutation();
      })) {
    return false;
  }

  // If the generic op pure data movement (e.g. copy/transpose/fill), then
  // always fuse.
  if (isCopyFillTransposeOrProject(producerOp))
    return true;

  // Don't fuse if it would duplicate the computation. This pass is just
  // pre-processing prior to the kernel segmentation pass, which uses more
  // complicated cost model-based heuristics to determine whether to duplicate a
  // producer across consumers.
  if (!producerOp->hasOneUse())
    return false;

  // Always fuse for pure-parallel consumers.
  if (linalgConsumerOp.getNumReductionLoops() == 0)
    return true;

  // Only fuse into reductions if indexing map is permutation (matches rank of
  // iteration space).
  // If the indexing map was a projection, then we would be introducing
  // redundant computation across the reduction dimension. These operations will
  // still be "fused" but the fusion happens during code generation tiling, not
  // via combining of the linalg operations.
  if (!linalgConsumerOp.getMatchingIndexingMap(fusedOperand).isPermutation())
    return false;

  // Don't fuse into contractions (e.g. matmul) or convolutions. This would
  // disrupt our ability to pattern-match these operations without really
  // changing the outcome of code generation.
  if (linalg::isaContractionOpInterface(linalgConsumerOp) ||
      linalg::isaConvolutionOpInterface(linalgConsumerOp))
    return false;

  return true;
}

namespace {

// Linalg elementwise op fusion doesn't handle `tensor.extract` in the body.
// This pattern replaces `tensor.extract` with the producer's body when
// possible.
struct FuseThroughTensorExtractPattern final
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // Check if extractOp is inside a generic op
    auto consumerOp =
        dyn_cast_or_null<linalg::GenericOp>(extractOp->getParentOp());
    if (!consumerOp)
      return rewriter.notifyMatchFailure(
          extractOp, "expected extract op to be inside a generic op");

    auto producerOp = extractOp.getTensor().getDefiningOp<linalg::GenericOp>();
    if (!producerOp)
      return rewriter.notifyMatchFailure(
          consumerOp, "expected extract operand to be a generic op");

    // Check if the producerOp is fusible
    // Note that the `isElementwise` check enforces no linalg.index ops, so we
    // don't worry about those below.
    if (producerOp.getNumResults() != 1 || !isElementwise(producerOp))
      return rewriter.notifyMatchFailure(producerOp,
                                         "producer op is not fusible");

    auto result = cast<OpResult>(extractOp.getTensor());
    AffineMap resultMap = producerOp.getIndexingMapMatchingResult(result);
    AffineMap inverseResultMap = inversePermutation(resultMap);
    SmallVector<Value> extractOps;
    SmallVector<Value> extractIndices = llvm::to_vector(extractOp.getIndices());
    for (OpOperand &operand : producerOp->getOpOperands()) {
      AffineMap inputMap = producerOp.getMatchingIndexingMap(&operand);
      AffineMap composedMap = inputMap.compose(inverseResultMap);
      SmallVector<Value> indices =
          applyPermutationMap<Value>(composedMap, extractIndices);
      auto newExtract = rewriter.create<tensor::ExtractOp>(
          extractOp.getLoc(), operand.get(), indices);
      extractOps.push_back(newExtract);
    }

    rewriter.cloneRegionBefore(producerOp.getRegion(), consumerOp.getRegion(),
                               consumerOp.getRegion().begin());
    Block &clonedBlock = consumerOp.getRegion().front();
    auto producerTermOp = clonedBlock.getTerminator();
    rewriter.inlineBlockBefore(&clonedBlock, extractOp->getNextNode(),
                               extractOps);

    // Replace the the all references to the original extract result with the
    // result from the inlined producerOp.
    rewriter.replaceOp(extractOp, producerTermOp->getOperand(0));
    rewriter.eraseOp(producerTermOp);
    return success();
  }
};

/// This is the function which will be used to control whether the Linalg
/// elementwise fusion pattern should be applied to fuse `candidate` with its
/// consumer (the operand owner).
static bool fusionControlFunction(OpOperand *candidate) {
  Operation *producer = candidate->get().getDefiningOp();
  if (!producer)
    return false;
  return shouldFuseProducer(candidate);
}

class LinalgElementwiseFusionPass
    : public mtrt::impl::LinalgElementwiseFusionPassBase<
          LinalgElementwiseFusionPass> {
  using Base::Base;

  LogicalResult initialize(MLIRContext *ctx) override {
    // Initialize the patterns.
    RewritePatternSet tmpPatterns(ctx);
    linalg::populateElementwiseOpsFusionPatterns(tmpPatterns,
                                                 fusionControlFunction);
    tmpPatterns.add<FuseThroughTensorExtractPattern>(ctx);
    patterns = std::move(tmpPatterns);

    // Set the rewrite config to apply patterns without limit.
    // It is possible to hit the limit when a function has 100s of linalg
    // generics.
    rewriteConfig.maxIterations = GreedyRewriteConfig::kNoLimit;

    return success();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    if (failed(applyPatternsGreedily(op, patterns, rewriteConfig))) {
      emitError(op->getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }

  FrozenRewritePatternSet patterns;
  GreedyRewriteConfig rewriteConfig{};
};
} // namespace
