//===- StablehloPrepareScatter.cpp  ---------------------------------------===//
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
/// Prepare `stablehlo.scatter` for conversion to TensorRT dialect. This pass
/// canonicalizes the scatter operations so that they have a form compatible
/// with the "onnx.ScatterNd" semantic, which is the same as the
/// `tensorrt.scatter` operation semantic.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Transforms/StablehloInputPreprocessing/StablehloPrepareScatter.h"
#include "mlir-tensorrt/Dialect/StableHloExt/Utils/GatherScatterUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tensorrt-stablehlo-prepare-scatter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

// Ensure that there are enough "inserted window dimensions" so that the window
// update and the batch dims access disjoint areas of the result index space.
// This ensures we can map to `tensorr.scatter` or `onnx.scatter_nd`. This must
// be called after ensuring that there is a single scatter batch dimension.
static FailureOr<SmallVector<Value>>
stablehloReshapeScatterUpdatesToAddInsertedDims(OpBuilder &b, Location loc,
                                                ValueRange updates,
                                                int64_t indexDepth,
                                                int64_t inputRank) {
  assert(indexDepth >= 1 && "expected non-zero index depth");
  const size_t numScatterBatchDims = 1;
  RankedTensorType updateType =
      updates.front().getType().cast<RankedTensorType>();
  const int64_t currUpdateSliceRank =
      updateType.getRank() - numScatterBatchDims;
  const int64_t expectedUpdateSliceRank = inputRank - indexDepth;
  const int64_t expectedInsertWindowDims = indexDepth;
  assert(expectedInsertWindowDims >= 1 &&
         "expected positive number of window insert dims");

  LLVM_DEBUG(DBGS() << "update slice rank expected = "
                    << expectedUpdateSliceRank << ", current = "
                    << currUpdateSliceRank << " = " << updateType << "\n");

  // No need to do anything.
  if (expectedUpdateSliceRank == currUpdateSliceRank)
    return llvm::to_vector(updates);

  // Otherwise, we need to drop leading dimensions (hopefully). If leading
  // dimensions are not unit dims, then we can't proceed.
  if (currUpdateSliceRank > expectedUpdateSliceRank) {
    const int64_t dimToDrop = currUpdateSliceRank - expectedUpdateSliceRank;
    LLVM_DEBUG(DBGS() << "need to drop " << dimToDrop
                      << " from the updates tensor (after index dim)\n");
    if (!llvm::all_equal(
            updateType.getShape().drop_front(1).take_front(dimToDrop)) ||
        updateType.getDimSize(1) != 1)
      return failure();

    RankedTensorType newShape =
        RankedTensorType::Builder(updateType)
            .setShape(updateType.getShape().drop_front(1 + dimToDrop))
            .insertDim(updateType.getDimSize(0), 0);
    return llvm::to_vector(llvm::map_range(updates, [&](Value update) -> Value {
      return b.create<stablehlo::ReshapeOp>(loc, newShape, update);
    }));
  }

  return failure();
}

bool tensorrt::isCanonicalScatterNd(stablehlo::ScatterOp scatterOp) {
  if (llvm::any_of(scatterOp.getOperandTypes(), [](Type operandType) {
        return !operandType.isa<RankedTensorType>();
      }))
    return false;
  stablehlo::ScatterDimensionNumbersAttr dimsAttrs =
      scatterOp.getScatterDimensionNumbers();
  auto indicesType =
      scatterOp.getScatterIndices().getType().cast<RankedTensorType>();
  auto operandType =
      scatterOp.getInputs().front().getType().cast<RankedTensorType>();
  auto updateType =
      scatterOp.getUpdates().front().getType().cast<RankedTensorType>();
  auto isSeq = [](ArrayRef<int64_t> ar, int64_t start, int64_t end) {
    return llvm::equal(ar, llvm::seq<int64_t>(start, end));
  };
  int64_t indexDepth = indicesType.getDimSize(indicesType.getRank() - 1);
  return indicesType.getRank() == 2 && dimsAttrs.getIndexVectorDim() == 1 &&
         isSeq(dimsAttrs.getUpdateWindowDims(), 1, updateType.getRank()) &&
         isSeq(dimsAttrs.getScatterDimsToOperandDims(), 0,
               indicesType.getDimSize(1)) &&
         isSeq(dimsAttrs.getInsertedWindowDims(), 0, indexDepth) &&
         ((operandType.getRank() - indexDepth) + (indicesType.getRank() - 1)) ==
             updateType.getRank();
}

namespace {
/// Simplify `stablehlo.scatter` to conform with `tensorrt.scatter`.
struct StablehloCanonicalizeScatterToTensorRtScatterNdFormat
    : public OpRewritePattern<stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    // If we are already in canonical form, then there is nothing to do.
    if (tensorrt::isCanonicalScatterNd(op))
      return failure();

    // Only proceed if we are in the StableHLO canonical form. This is covered
    // by the "stablehlo-ext-canonicalize-scatter" pass that runs before this
    // pass. All scatter ops should be in stablehlo canonical form at this
    // point.
    if (!stablehlo::isCanonicalScatter(op))
      return failure();

    RankedTensorType canonicalizedInputType =
        op.getInputs().front().getType().cast<RankedTensorType>();
    RankedTensorType canonicalizedIndexType =
        op.getScatterIndices().getType().cast<RankedTensorType>();

    LLVM_DEBUG(DBGS() << "canonicalizing " << op << "\n");

    // Reshape the updates if possible.
    int64_t inputRank = canonicalizedInputType.getRank();
    int64_t indexDepth = canonicalizedIndexType.getDimSize(1);
    FailureOr<SmallVector<Value>> canonicalizedUpdates =
        stablehloReshapeScatterUpdatesToAddInsertedDims(
            rewriter, op.getLoc(), op.getUpdates(), indexDepth, inputRank);
    if (failed(canonicalizedUpdates))
      return rewriter.notifyMatchFailure(op, "failed to canonicalize updates");

    // Create the new scatter op.
    auto canonicalizedUpdatesType =
        canonicalizedUpdates->front().getType().cast<RankedTensorType>();
    assert(((canonicalizedInputType.getRank() - indexDepth) +
            (canonicalizedIndexType.getRank() - 1)) ==
               canonicalizedUpdatesType.getRank() &&
           "expected slice size to equal inputRank - index_depth");
    auto newConfig = stablehlo::ScatterDimensionNumbersAttr::get(
        getContext(),
        /*updateWindowDims=*/
        llvm::to_vector(
            llvm::seq<int64_t>(1, canonicalizedUpdatesType.getRank())),
        /*insertedWindowDims=*/
        llvm::to_vector(llvm::seq<int64_t>(0, indexDepth)),
        /*inputBatchingDims=*/{},
        /*scatterIndicesBatchingDims=*/{},
        /*scatterDimsToOperandDims=*/
        llvm::to_vector(llvm::seq<int64_t>(0, indexDepth)), 1);
    auto scatterOp = rewriter.create<stablehlo::ScatterOp>(
        op.getLoc(), TypeRange(ValueRange(op.getInputs())), op.getInputs(),
        op.getScatterIndices(), *canonicalizedUpdates, newConfig);
    Region &region = scatterOp.getUpdateComputation();
    rewriter.inlineRegionBefore(op.getUpdateComputation(), region,
                                region.end());
    rewriter.replaceOp(op, scatterOp.getResults());

    scatterOp->setAttr("tensorrt.canonicalized_scatter",
                       rewriter.getUnitAttr());

    return success();
  }
};

/// Rewrite `arith.constant` to `stablehlo.constant`. Arith constant can be
/// created by `tensor` dialect canonicalizers. Some `arith` constants may be
/// created by `stablehlo-canonicalize-scatter` pass.
struct RewriteArithConstToStablehlo
    : public OpRewritePattern<arith ::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isa<RankedTensorType>())
      return failure();
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, op.getType(), cast<ElementsAttr>(op.getValueAttr()));
    return success();
  }
};
} // namespace

/// Rewrite `tensor.expand_shape`/`tensor.collapse_shape` into a
/// `stablehlo.reshape` operation.
template <typename OpType = tensor::ExpandShapeOp>
static LogicalResult
stablehloRewriteTensorExpandCollapseShape(OpType op,
                                          PatternRewriter &rewriter) {
  if (!op.getType().hasStaticShape())
    return failure();
  rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(),
                                                    op->getOperand(0));
  return success();
}

void tensorrt::populateCanonicalizeStablehloScatterForTensorRTPatterns(
    RewritePatternSet &patterns) {
  patterns.add(
      stablehloRewriteTensorExpandCollapseShape<tensor::CollapseShapeOp>,
      PatternBenefit(1), {"tensorCollapseShapeToStablehloReshape"});
  patterns.add(stablehloRewriteTensorExpandCollapseShape<tensor::ExpandShapeOp>,
               PatternBenefit(1), {"tensorExpandShapeToStablehloReshape"});
  patterns.insert<StablehloCanonicalizeScatterToTensorRtScatterNdFormat,
                  RewriteArithConstToStablehlo>(patterns.getContext());
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns,
                                                     patterns.getContext());
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns,
                                                       patterns.getContext());
}