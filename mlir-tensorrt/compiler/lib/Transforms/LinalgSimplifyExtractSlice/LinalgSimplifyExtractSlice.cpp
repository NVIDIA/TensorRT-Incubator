//===- LinalgSimplifyExtractSlice.cpp -------------------------------------===//
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
/// Runs auxillary patterns to remove `tensor.extract_slice` operations when
/// possible.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mtrt {
#define GEN_PASS_DEF_LINALGSIMPLIFYEXTRACTSLICEPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mtrt

using namespace mlir;

namespace {

/// Simplify `tensor.extract_slice` of a TilingInterface operation if they are
/// in the same block and the TilingInterface operation has only a single use.
struct RewriteExtractSliceOp : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {

    auto producerOp = op.getSource().getDefiningOp<TilingInterface>();
    if (!producerOp)
      return failure();

    // TODO: handle the rank-reduced case.
    if (op.getType().getRank() < op.getSourceType().getRank())
      return failure();

    /// This isn't meant to replace tile-and-fuse patterns, just the case where
    /// they are in the same block.
    if (producerOp->getBlock() != op->getBlock() || !producerOp->hasOneUse())
      return failure();

    OpResult producer = cast<OpResult>(op.getSource());
    FailureOr<TilingResult> tileAndFuseResult =
        tensor::replaceExtractSliceWithTiledProducer(rewriter, op, producer);
    if (failed(tileAndFuseResult))
      return failure();

    assert(llvm::hasSingleElement(tileAndFuseResult->tiledValues) &&
           "expected single tile-and-fuse result");
    Value replacement = tileAndFuseResult->tiledValues.front();
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

class LinalgSimplifyExtractSlicePass
    : public mtrt::impl::LinalgSimplifyExtractSlicePassBase<
          LinalgSimplifyExtractSlicePass> {
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<RewriteExtractSliceOp>(ctx);
    tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    linalg::FillOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateDropRedundantInsertSliceRankExpansionPatterns(patterns);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc())
          << "failed to apply " << getArgument() << " patterns";
      return signalPassFailure();
    }
  }
};
} // namespace
