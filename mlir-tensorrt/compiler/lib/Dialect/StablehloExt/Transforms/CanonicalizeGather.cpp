//===- CanonicalizeGather.cpp  --------------------------------------------===//
//
// The canonicalize gather pass logic is adapted from the XLA project
// `xla/mlir_hlo/mhlo/transforms/mhlo_canonicalize_gather/mhlo_canonicalize_gather.cc`
// and has the original license: Apache License v2.0. See
// https://github.com/openxla/xla/blob/main/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
// Changes are copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of the `stablehlo-ext-canonicalize-gather` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Patterns.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/GatherScatterUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo_ext {
#define GEN_PASS_DEF_CANONICALIZEGATHERPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace stablehlo_ext
} // namespace mlir

using namespace mlir;
using namespace mlir::stablehlo;

// Given an input tensor, collapse dimensions 1+collapsedSliceDims[...].
static TypedValue<RankedTensorType>
collapseSliceDims(RewriterBase &rewriter, Location loc,
                  TypedValue<RankedTensorType> input,
                  ArrayRef<int64_t> collapsedSliceDims) {
  if (collapsedSliceDims.empty())
    return input;

  SmallVector<int64_t> newShape{input.getType().getShape()};
  // collapsedSliceDims is small in practice.
  for (int64_t dim : llvm::reverse(collapsedSliceDims))
    // Dimension 0 is the collapsed batch dimension.
    newShape.erase(newShape.begin() + 1 + dim);

  std::optional<SmallVector<ReassociationIndices>> reassociations =
      getReassociationIndicesForCollapse(input.getType().getShape(), newShape);
  if (!reassociations)
    return {};

  return cast<TypedValue<RankedTensorType>>(
      stablehlo_ext::createCollapsingReshape(rewriter, loc, input,
                                             *reassociations));
}

// Expands the first dimension of `input` into the shape of `startIndices`,
// removing the index vector dimension.
static TypedValue<RankedTensorType>
expandBatchDimension(RewriterBase &rewriter, Location loc,
                     TypedValue<RankedTensorType> input,
                     GatherOp originalGatherOp) {
  llvm::SmallVector<int64_t> newShape{
      originalGatherOp.getStartIndices().getType().getShape()};
  // Erase the index vector dimension if it wasn't implicit.
  int64_t indexDim = originalGatherOp.getDimensionNumbers().getIndexVectorDim();
  if (indexDim < static_cast<int64_t>(newShape.size()))
    newShape.erase(newShape.begin() + indexDim);

  // `input` has one batch dimension, if we still have one now, there is nothing
  // to do.
  if (newShape.size() == 1)
    return input;

  // Copy the slice dimensions.
  llvm::copy(input.getType().getShape().drop_front(1),
             std::back_inserter(newShape));

  auto newType =
      RankedTensorType::get(newShape, input.getType().getElementType());
  std::optional<SmallVector<ReassociationIndices>> reassociations =
      *getReassociationIndicesForReshape(input.getType(), newType);
  if (!reassociations)
    return {};
  if (static_cast<int64_t>(newShape.size()) > input.getType().getRank())
    return cast<TypedValue<RankedTensorType>>(
        stablehlo_ext::createExpandingReshape(rewriter, loc, newType, input,
                                              *reassociations));

  return cast<TypedValue<RankedTensorType>>(
      stablehlo_ext::createCollapsingReshape(rewriter, loc, input,
                                             *reassociations));
}

static TypedValue<RankedTensorType>
moveOffsetDimensions(RewriterBase &rewriter, Location loc,
                     TypedValue<RankedTensorType> input,
                     GatherOp originalGatherOp) {
  const auto &dims = originalGatherOp.getDimensionNumbers();
  int64_t outputRank = input.getType().getRank();
  int64_t offsetDimIndex = outputRank - dims.getOffsetDims().size();
  int64_t batchDimIndex = 0;
  llvm::SmallVector<int64_t> outputPermutation;
  outputPermutation.reserve(outputRank);
  for (int64_t i = 0; i < outputRank; ++i) {
    if (llvm::is_contained(dims.getOffsetDims(), i)) {
      outputPermutation.push_back(offsetDimIndex++);
    } else {
      outputPermutation.push_back(batchDimIndex++);
    }
  }
  return rewriter.create<TransposeOp>(loc, input, outputPermutation);
}

namespace {
struct CanonicalizeGather : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    if (stablehlo_ext::isCanonicalGather(gatherOp) ||
        stablehlo_ext::isSingleDimSimpleGatherWithExplicitIndexDim(gatherOp) ||
        stablehlo_ext::isSingleDimSimpleGatherWithImplicitIndexDim(gatherOp) ||
        stablehlo_ext::isSimpleLeadingMultiDimGather(gatherOp) ||
        stablehlo_ext::isSimpleLeadingMultiDimGatherWithDegenerateDims(
            gatherOp))
      return failure();
    Location loc = gatherOp.getLoc();

    // If any slice size is 0, the convention followed by upstream
    // canonicalizations is to return a 'tensor.empty'.
    if (llvm::is_contained(gatherOp.getSliceSizes(), 0)) {
      if (!gatherOp.getType().hasStaticShape())
        return failure();
      rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
          gatherOp.getOperation(), gatherOp.getType().getShape(),
          gatherOp.getType().getElementType());
      return success();
    }

    const auto &dims = gatherOp.getDimensionNumbers();

    // This pattern does not support batching dimensions.
    if (!dims.getOperandBatchingDims().empty())
      return failure();

    int64_t operandRank =
        dims.getCollapsedSliceDims().size() + dims.getOffsetDims().size();

    // Make the operand conform to start_index_map.
    auto [operandPermutation, operandPermutationInverse] =
        stablehlo_ext::makeOperandStartIndexPermutations(
            dims.getStartIndexMap(), operandRank);

    Value operand = rewriter.create<TransposeOp>(loc, gatherOp.getOperand(),
                                                 operandPermutation);
    auto startIndices = stablehlo_ext::canonicalizeStartIndices(
        rewriter, loc, gatherOp.getStartIndices(), dims.getIndexVectorDim());

    // Permute the slice sizes according to start_index_map and compute the new
    // output shape for the Gather op.
    auto offsetDims = llvm::to_vector(llvm::seq(int64_t{1}, 1 + operandRank));
    auto startIndexMap = llvm::to_vector(llvm::seq(
        int64_t{0}, static_cast<int64_t>(dims.getStartIndexMap().size())));

    auto newDims = GatherDimensionNumbersAttr::get(
        rewriter.getContext(), offsetDims,
        /*collapsedSliceDims=*/{}, /*operandBatchingDims=*/{},
        /*startIndicesBatchingDims=*/{}, startIndexMap,
        /*indexVectorDim=*/1);
    TypedValue<RankedTensorType> result = rewriter.create<GatherOp>(
        loc, operand, startIndices, newDims,
        mlir::applyPermutation(gatherOp.getSliceSizes(), operandPermutation),
        gatherOp.getIndicesAreSorted());

    // Undo the startIndexMap transpose.
    for (int64_t &dim : operandPermutationInverse)
      ++dim;
    // Add the batch dimension and keep it at the front.
    operandPermutationInverse.insert(operandPermutationInverse.begin(), 0);
    result =
        rewriter.create<TransposeOp>(loc, result, operandPermutationInverse);

    // Collapse the requested dimensions.
    result =
        collapseSliceDims(rewriter, loc, result, dims.getCollapsedSliceDims());
    if (!result)
      return failure();

    // Expand the start index dimensions.
    result = expandBatchDimension(rewriter, loc, result, gatherOp);
    if (!result)
      return failure();

    // Move the offset dims to the final locations.
    result = moveOffsetDimensions(rewriter, loc, result, gatherOp);

    rewriter.replaceOp(gatherOp.getOperation(), {result});
    return success();
  }
};
} // namespace

void stablehlo_ext::populateCanonicalizeStablehloGatherPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CanonicalizeGather>(patterns.getContext());
}

namespace {

struct CanonicalizeGatherPass
    : public stablehlo_ext::impl::CanonicalizeGatherPassBase<
          CanonicalizeGatherPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    stablehlo_ext::populateCanonicalizeStablehloGatherPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
