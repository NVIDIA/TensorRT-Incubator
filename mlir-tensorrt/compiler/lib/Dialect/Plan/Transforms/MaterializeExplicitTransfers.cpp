//===- MaterializeExplicitTransfers.cpp -----------------------------------===//
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
///  Implementation of the `plan-materialize-explicit-transfers` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::plan {
#define GEN_PASS_DEF_PLANMATERIALIZEEXPLICITTRANSFERSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

/// Get the dynamic dimensions for a shaped value.
static FailureOr<SmallVector<Value>> getDynamicDims(OpBuilder &b, Location loc,
                                                    Value tensor) {
  if (!llvm::isa<RankedTensorType>(tensor.getType()))
    return failure();

  RankedTensorType tensorType = llvm::cast<RankedTensorType>(tensor.getType());
  SmallVector<Value> dynamicSizes;
  // Compute the dynamic part of the shape.
  // First try to query the shape via ReifyRankedShapedTypeOpInterface.
  if (llvm::isa<OpResult>(tensor)) {
    ReifiedRankedShapedTypeDims resultDims;
    if (succeeded(reifyResultShapes(b, tensor.getDefiningOp(), resultDims))) {
      const SmallVector<OpFoldResult> &shape =
          resultDims[llvm::cast<OpResult>(tensor).getResultNumber()];
      for (const auto &dim : enumerate(tensorType.getShape()))
        if (ShapedType::isDynamic(dim.value()))
          dynamicSizes.push_back(cast<Value>(shape[dim.index()]));
      return dynamicSizes;
    }
  }

  // If the shape could not be reified, create DimOps.
  bufferization::populateDynamicDimSizes(b, loc, tensor, dynamicSizes);
  return dynamicSizes;
}

namespace {

struct RewriteAllocTensorWithCopyToCastPattern
    : public OpRewritePattern<bufferization::AllocTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(bufferization::AllocTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto copySource = op.getCopy();
    if (!copySource)
      return failure();
    auto copySourceType = dyn_cast<RankedTensorType>(copySource.getType());
    if (!copySourceType)
      return failure();

    auto memorySpace =
        llvm::dyn_cast_or_null<MemorySpaceAttr>(op.getMemorySpaceAttr());
    if (!memorySpace)
      return failure();

    auto newType = op.getType().cloneWithEncoding(memorySpace);

    auto castOp =
        rewriter.create<tensor::CastOp>(op.getLoc(), newType, copySource);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newType, castOp);
    return success();
  }
};

struct TensorCastToAllocAndCopyPattern
    : public OpRewritePattern<tensor::CastOp> {
  using OpRewritePattern<tensor::CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::CastOp op,
                                PatternRewriter &rewriter) const override {

    auto sourceType = cast<RankedTensorType>(op.getOperand().getType());
    auto targetType = cast<RankedTensorType>(op.getType());
    // Source/target could be unranked for `tensor.cast`.
    if (!sourceType || !targetType)
      return failure();

    auto sourceSpace =
        dyn_cast_or_null<MemorySpaceAttr>(sourceType.getEncoding());
    auto targetSpace =
        dyn_cast_or_null<MemorySpaceAttr>(targetType.getEncoding());

    if (!sourceSpace || !targetSpace || sourceSpace == targetSpace)
      return rewriter.notifyMatchFailure(
          op, "skipping no space encoding or same space");

    FailureOr<SmallVector<Value>> dynamicDims =
        getDynamicDims(rewriter, op.getLoc(), op.getOperand());
    if (failed(dynamicDims))
      return failure();

    auto allocOp = rewriter.create<bufferization::AllocTensorOp>(
        op.getLoc(), targetType, *dynamicDims, /*copy=*/Value{},
        /*size_hint=*/Value{},
        /*memory_space=*/targetSpace);

    rewriter.replaceOpWithNewOp<bufferization::MaterializeInDestinationOp>(
        op, targetType, op.getOperand(), allocOp.getResult());

    return success();
  }
};

/// Remove redundant explicit `bufferization.materialize_in_dest` ops.
struct RemoveRedundantMaterializeInDestPattern
    : OpRewritePattern<bufferization::MaterializeInDestinationOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(bufferization::MaterializeInDestinationOp op,
                                PatternRewriter &rewriter) const override {
    auto producer =
        op.getDest().getDefiningOp<bufferization::MaterializeInDestinationOp>();
    if (!producer)
      return failure();

    if (producer.getDest().getType() != op.getDest().getType())
      return failure();

    rewriter.modifyOpInPlace(
        op, [&]() { op.getDestMutable().assign(producer.getDest()); });
    return success();
  }
};

class MaterializeExplicitTransfersPass
    : public plan::impl::PlanMaterializeExplicitTransfersPassBase<
          MaterializeExplicitTransfersPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    // Eliminate `bufferization.alloc_tensor` ops with `copy` argument and
    // simplify `tensor.cast` ops.
    {
      RewritePatternSet patterns(op->getContext());
      patterns.add<RewriteAllocTensorWithCopyToCastPattern>(op->getContext());
      tensor::CastOp::getCanonicalizationPatterns(patterns, op->getContext());
      if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc())
            << "failed to apply patterns in " << getArgument();
        return;
      }
    }

    RewritePatternSet patterns(op->getContext());
    patterns.add<TensorCastToAllocAndCopyPattern,
                 RemoveRedundantMaterializeInDestPattern>(op->getContext());

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc())
          << "failed to apply patterns in " << getArgument();
      return;
    }
  }
};
} // namespace
