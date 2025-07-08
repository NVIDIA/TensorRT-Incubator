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

    rewriter.replaceOpWithNewOp<bufferization::MaterializeInDestinationOp>(
        op, targetType, op.getOperand(), allocOp.getResult());

    return success();
  }
};

/// One pattern that consistently causes failures is if one branch of  an
/// `scf.if` operation returns a `bufferization.materialize_in_destination` op
/// but the other branch returns a constant. This can cause
/// one-shot-bufferization to fail with a "cannot avoid a RaW conflict" error.
/// To solve this, we can create an explicit materialization for the constant as
/// well.
/// TODO: this creates an allocation+copy for the constant, which is not
/// necessarily actually needed if the result of `scf.if` is only read. To fix
/// this, we need to fix the issues with `bufferization.alloc_tensor` or create
/// our own alloc tensor op with a combined allocate+copy semantic until the
/// upstream one can be fixed can be fixed.
struct HandleIfOpPattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0)
      return rewriter.notifyMatchFailure(op, "op has no results");

    Region &thenRegion = op.getThenRegion();
    Region &elseRegion = op.getElseRegion();
    if (!thenRegion.hasOneBlock() || !elseRegion.hasOneBlock())
      return rewriter.notifyMatchFailure(op, "expected single block regions");

    auto thenYield = cast<scf::YieldOp>(thenRegion.front().getTerminator());
    auto elseYield = cast<scf::YieldOp>(elseRegion.front().getTerminator());
    auto thenOperands = thenYield.getResultsMutable();
    auto elseOperands = elseYield.getResultsMutable();
    Location loc = op.getLoc();
    SmallVector<Value> newThenYieldOperands(thenYield->getOperands());
    SmallVector<Value> newElseYieldOperands(elseYield->getOperands());
    bool thenChanged = false, elseChanged = false;

    auto createExplictMaterialize = [&](scf::YieldOp yield, Value oldOperand,
                                        RankedTensorType tensorType) {
      rewriter.setInsertionPoint(yield);
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, tensorType.getShape(), tensorType.getElementType(), ValueRange{},
          tensorType.getEncoding());
      return rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, tensorType, oldOperand, emptyTensor);
    };

    for (auto [idx, thenOperand, elseOperand, resultType] :
         llvm::enumerate(thenOperands, elseOperands, op->getResultTypes())) {
      auto tensorType = dyn_cast<RankedTensorType>(resultType);
      if (!tensorType || !tensorType.hasStaticShape())
        continue;

      auto thenDef = thenOperand.get().getDefiningOp();
      auto elseDef = elseOperand.get().getDefiningOp();
      if (!thenDef || !elseDef)
        continue;

      if (thenDef->hasTrait<OpTrait::ConstantLike>() &&
          isa<bufferization::MaterializeInDestinationOp>(elseDef)) {
        auto materializeOp =
            createExplictMaterialize(thenYield, thenOperand.get(), tensorType);
        newThenYieldOperands[idx] = materializeOp.getResult();
        thenChanged = true;
        continue;
      }

      if (elseDef->hasTrait<OpTrait::ConstantLike>() &&
          isa<bufferization::MaterializeInDestinationOp>(thenDef)) {
        auto materializeOp =
            createExplictMaterialize(elseYield, elseOperand.get(), tensorType);
        newElseYieldOperands[idx] = materializeOp.getResult();
        elseChanged = true;
        continue;
      }
    }

    if (thenChanged)
      rewriter.modifyOpInPlace(thenYield, [&]() {
        thenYield.getResultsMutable().assign(newThenYieldOperands);
      });

    if (elseChanged)
      rewriter.modifyOpInPlace(elseYield, [&]() {
        elseYield.getResultsMutable().assign(newElseYieldOperands);
      });
    return success(thenChanged || elseChanged);
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

/// Remove redundant explicit `bufferization.materialize_in_dest` ops if the
/// consumer is materializing into a `tensor.empty` without changing the memory
/// space and the producer has a single use. Then we can just replace with the
/// producer.
struct RemoveRedundantMaterializationsPattern
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
                   RemoveRedundantMaterializationsPattern, HandleIfOpPattern>(
          context);
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
