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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

    auto allocOp = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), targetType.getShape(), targetType.getElementType(),
        *dynamicDims, targetType.getEncoding());

    rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, targetType, op.getOperand(),
                                                allocOp.getResult());

    return success();
  }
};
} // namespace

template <typename OpTy>
struct RemoveRedundantCopyOpPatternBase : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  virtual Value getDest(OpTy op) const = 0;
  virtual MutableOperandRange getDestMutable(OpTy op) const = 0;
  virtual Value getSource(OpTy op) const = 0;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto producer = getDest(op).template getDefiningOp<OpTy>();
    if (!producer)
      return failure();

    if (getDest(producer).getType() != getDest(op).getType())
      return failure();

    rewriter.modifyOpInPlace(
        op, [&]() { getDestMutable(op).assign(getDest(producer)); });
    return success();
  }
};

namespace {
/// Remove redundant explicit `bufferization.materialize_in_dest` ops.
struct RemoveRedundantMaterializeInDestPattern final
    : RemoveRedundantCopyOpPatternBase<
          bufferization::MaterializeInDestinationOp> {
  using RemoveRedundantCopyOpPatternBase::RemoveRedundantCopyOpPatternBase;
  Value getDest(bufferization::MaterializeInDestinationOp op) const override {
    return op.getDest();
  }
  MutableOperandRange
  getDestMutable(bufferization::MaterializeInDestinationOp op) const override {
    return op.getDestMutable();
  }
  Value getSource(bufferization::MaterializeInDestinationOp op) const override {
    return op.getSource();
  }
};

/// Remove redundant explicit `bufferization.materialize_in_dest` ops if the
/// consumer is materializing into a `tensor.empty` without changing the memory
/// space and the producer has a single use. Then we can just replace with the
/// producer.
struct RemoveRedundantMaterializationsPattern final
    : OpRewritePattern<bufferization::MaterializeInDestinationOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(bufferization::MaterializeInDestinationOp op,
                                PatternRewriter &rewriter) const override {
    auto producer =
        op.getSource()
            .getDefiningOp<bufferization::MaterializeInDestinationOp>();
    if (!producer || !producer.getResult().hasOneUse() ||
        producer.getType() != op.getType())
      return failure();

    rewriter.replaceOp(op, producer);
    return success();
  }
};

/// Remove redundant explicit `linalg.copy` pattern.
struct RemoveRedundantLinalgCopyPattern final
    : RemoveRedundantCopyOpPatternBase<linalg::CopyOp> {
  using RemoveRedundantCopyOpPatternBase::RemoveRedundantCopyOpPatternBase;
  Value getDest(linalg::CopyOp op) const override {
    return op.getOutputs().front();
  }
  MutableOperandRange getDestMutable(linalg::CopyOp op) const override {
    return op.getOutputsMutable();
  }
  Value getSource(linalg::CopyOp op) const override {
    return op.getInputs().front();
  }
};

class MaterializeExplicitTransfersPass
    : public plan::impl::PlanMaterializeExplicitTransfersPassBase<
          MaterializeExplicitTransfersPass> {
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override {

    // Populate patterns that are used to clean up the IR prior to the
    // main pattern application.
    preprocessingPatterns = std::make_shared<FrozenRewritePatternSet>([&] {
      RewritePatternSet patterns(context);
      patterns.add<
          // Currently `bufferization.alloc_tensor` ops with `copy` argument
          // where the `memory_spacee` annotation does not match the type
          // will break bufferization. Rewrite into a `tensor.cast` pattern.
          RewriteAllocTensorWithCopyToCastPattern>(context);

      // Ensure that we fold `tensor.cast` into `tensor.empty`. Otherwise we
      // will create trivially redundant copies.
      tensor::EmptyOp::getCanonicalizationPatterns(patterns, context);

      return patterns;
    }());

    // Populate the main patterns.
    mainPatterns = std::make_shared<FrozenRewritePatternSet>([&] {
      RewritePatternSet patterns(context);
      patterns.add<TensorCastToAllocAndCopyPattern,
                   RemoveRedundantMaterializeInDestPattern,
                   RemoveRedundantMaterializationsPattern,
                   RemoveRedundantLinalgCopyPattern>(context);
      return patterns;
    }());

    return success();
  }

  void runOnOperation() override {
    Operation *op = getOperation();

    if (failed(applyPatternsGreedily(op, *preprocessingPatterns))) {
      emitError(op->getLoc())
          << "failed to apply patterns in " << getArgument();
      return;
    }

    if (failed(applyPatternsGreedily(op, *mainPatterns))) {
      emitError(op->getLoc())
          << "failed to apply patterns in " << getArgument();
      return;
    }
  }

  std::shared_ptr<FrozenRewritePatternSet> preprocessingPatterns;
  std::shared_ptr<FrozenRewritePatternSet> mainPatterns;
};
} // namespace
