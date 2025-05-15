//===- StablehloPrepareConvolution.cpp ------------------------------------===//
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
/// Implements patterns that prepare `stablehlo.convolution` for conversion to
/// tensorrt dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_CANONICALIZECONVOLUTIONPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

using namespace mlir;
using namespace mlir::stablehlo;

/// Expand (n, c, h) input to (n, c, 1, h) input.
static Value stablehloExpandSpatialDims(OpBuilder &b, Location loc,
                                        Value input) {
  RankedTensorType rtt = cast<RankedTensorType>(input.getType());
  Type expandedType = RankedTensorType::Builder(rtt).insertDim(1, 2);
  return b.create<stablehlo::ReshapeOp>(loc, expandedType, input);
}

/// Collapse NCHW to NCW where must H be 1.
static Value stablehloCollapseNchwToNch(OpBuilder &b, Location loc,
                                        Value input) {
  auto rtt = cast<RankedTensorType>(input.getType());
  assert(rtt.getRank() == 4 && "expected 4D tensor");
  assert(rtt.getDimSize(2) == 1);
  RankedTensorType newType = RankedTensorType::Builder(rtt).dropDim(2);
  return b.create<stablehlo::ReshapeOp>(loc, newType, input);
}

// Expand a Nx2 dense int64 attribute describing padding to a (N+1)x2 tensor by
// prepending a row of all `val`s and adjusting the type accordingly.
static DenseIntElementsAttr
prependIntElementsAttrs2dRow(DenseIntElementsAttr attr, int64_t val = 0) {
  if (!attr)
    return nullptr;
  auto attrType = cast<RankedTensorType>(attr.getType());
  assert(attrType.getRank() == 2);
  const int64_t numRows = attrType.getDimSize(0);
  const int64_t rowSize = attrType.getDimSize(1);
  SmallVector<int64_t> values(rowSize, val);
  values.reserve(rowSize * (numRows + 1));
  llvm::append_range(values, attr.getValues<int64_t>());
  return DenseIntElementsAttr::get(
      RankedTensorType::get({numRows + 1, rowSize}, attrType.getElementType()),
      values);
}

/// Prepend `val` to the 1D array attr, adjusting type accordingly.
static DenseI64ArrayAttr prependI64ArrayAttr1d(DenseI64ArrayAttr attr,
                                               int64_t val = 0) {
  if (!attr)
    return nullptr;
  auto attrArray = attr.asArrayRef();
  SmallVector<int64_t> values = {val};
  values.reserve(attrArray.size() + 1);
  llvm::append_range(values, attrArray);
  return DenseI64ArrayAttr::get(attr.getContext(), values);
}

/// Prepend `val` to the 1D array attr, adjusting type accordingly.
static DenseBoolArrayAttr prependBoolArrayAttr1d(DenseBoolArrayAttr attr,
                                                 bool val = false) {
  if (!attr)
    return nullptr;
  auto attrArray = attr.asArrayRef();
  SmallVector<bool> values = {val};
  values.reserve(attrArray.size() + 1);
  llvm::append_range(values, attrArray);
  return DenseBoolArrayAttr::get(attr.getContext(), values);
}

static bool isContiguousSequence(ArrayRef<int64_t> sequence) {
  if (sequence.empty())
    return true;
  return llvm::equal(
      sequence,
      llvm::seq<int64_t>(sequence.front(), sequence.front() + sequence.size()));
}

/// Create a permutation that orders dimension numbers `firstDim`, `secondDim`,
/// and `spatialDims` into [`firstDim`, `secondDim`, `spatialDims`...]. If
/// `invert` is true, then return the inverse of that permutation.
static std::optional<SmallVector<int64_t>>
getPermutation(ArrayRef<int64_t> spatialDims, int64_t firstDim,
               int64_t secondDim, bool invert = false) {
  SmallVector<int64_t> permutation = {
      firstDim,
      secondDim,
  };
  llvm::append_range(permutation, spatialDims);
  if (invert)
    permutation = invertPermutationVector(permutation);
  if (llvm::equal(permutation, llvm::seq<int64_t>(0, permutation.size())))
    return std::nullopt;
  return permutation;
}

namespace {
// Ensure that convolutions are NCHW/RSCK and have at least two spatial
// dimensions.
class StablehloRewriteConvolution
    : public OpRewritePattern<stablehlo::ConvolutionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    stablehlo::ConvDimensionNumbersAttr dimNumbers = op.getDimensionNumbers();
    const bool isConv1d = dimNumbers.getInputSpatialDimensions().size() == 1;
    const int64_t numSpatialDims =
        std::max<int64_t>(2, dimNumbers.getInputSpatialDimensions().size());

    if (dimNumbers.getOutputSpatialDimensions().empty() ||
        !isContiguousSequence(dimNumbers.getOutputSpatialDimensions()))
      return failure();

    // Create the LHS transpose permutation.
    SmallVector<int64_t, 4> newSpatialDims =
        llvm::to_vector(llvm::seq<int64_t>(2, 2 + numSpatialDims));

    std::optional<SmallVector<int64_t>> lhsTranspose =
        getPermutation(dimNumbers.getInputSpatialDimensions(),
                       dimNumbers.getInputBatchDimension(),
                       dimNumbers.getInputFeatureDimension());
    std::optional<SmallVector<int64_t>> rhsTranspose =
        getPermutation(dimNumbers.getKernelSpatialDimensions(),
                       dimNumbers.getKernelOutputFeatureDimension(),
                       dimNumbers.getKernelInputFeatureDimension());
    std::optional<SmallVector<int64_t>> outputTranspose =
        getPermutation(dimNumbers.getOutputSpatialDimensions(),
                       dimNumbers.getOutputBatchDimension(),
                       dimNumbers.getOutputFeatureDimension(), /*invert=*/true);
    if (!lhsTranspose && !rhsTranspose && !outputTranspose && !isConv1d)
      return failure();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    SmallVector<int64_t> resultShape = {
        op.getType().getDimSize(dimNumbers.getOutputBatchDimension()),
        op.getType().getDimSize(dimNumbers.getOutputFeatureDimension())};
    llvm::append_range(resultShape,
                       op.getType().getShape().slice(
                           dimNumbers.getOutputSpatialDimensions().front(),
                           dimNumbers.getOutputSpatialDimensions().size()));
    if (isConv1d)
      resultShape.insert(resultShape.begin() + 2, 1);

    if (lhsTranspose)
      lhs = rewriter.create<stablehlo::TransposeOp>(op.getLoc(), lhs,
                                                    *lhsTranspose);
    if (rhsTranspose)
      rhs = rewriter.create<stablehlo::TransposeOp>(op.getLoc(), rhs,
                                                    *rhsTranspose);

    if (isConv1d) {
      lhs = stablehloExpandSpatialDims(rewriter, op.getLoc(), lhs);
      rhs = stablehloExpandSpatialDims(rewriter, op.getLoc(), rhs);
    }

    DenseIntElementsAttr paddingAttr = op.getPaddingAttr();
    auto lhsDilationAttr =
        dyn_cast_or_null<DenseI64ArrayAttr>(op.getLhsDilationAttr());
    auto strides =
        dyn_cast_or_null<DenseI64ArrayAttr>(op.getWindowStridesAttr());
    auto rhsDilationAttr =
        dyn_cast_or_null<DenseI64ArrayAttr>(op.getRhsDilationAttr());
    auto windowReversal =
        dyn_cast_or_null<DenseBoolArrayAttr>(op.getWindowReversalAttr());

    // For conv1d, we updateto conv2d, so we ned to update these parameters as
    // well.
    if (isConv1d) {
      paddingAttr = prependIntElementsAttrs2dRow(paddingAttr);
      lhsDilationAttr = prependI64ArrayAttr1d(lhsDilationAttr, 1);
      rhsDilationAttr = prependI64ArrayAttr1d(rhsDilationAttr, 1);
      strides = prependI64ArrayAttr1d(strides, 1);
      windowReversal = prependBoolArrayAttr1d(windowReversal);
    }

    // Create the new convolution attribute.
    auto newDimNumbers = stablehlo::ConvDimensionNumbersAttr::get(
        op.getContext(),
        /*inputBatchDimension=*/0,
        /*inputFeatureDimension=*/1,
        /*inputSpatialDimensions=*/newSpatialDims,
        /*kernelInputFeatureDimension=*/1,
        /*kernelOutputFeatureDimension=*/0,
        /*kernelSpatialDimensions=*/newSpatialDims,
        /*outputBatchDimension=*/0,
        /*outputFeatureDimension=*/1,
        /*outputSpatialDimensions=*/newSpatialDims);
    auto newConv = rewriter.create<stablehlo::ConvolutionOp>(
        op.getLoc(),
        RankedTensorType::get(resultShape, op.getType().getElementType()),
        SmallVector<Value>{lhs, rhs}, op->getAttrs());
    if (paddingAttr)
      newConv.setPaddingAttr(paddingAttr);
    if (lhsDilationAttr)
      newConv.setLhsDilationAttr(lhsDilationAttr);
    if (rhsDilationAttr)
      newConv.setRhsDilationAttr(rhsDilationAttr);
    if (strides)
      newConv.setWindowStridesAttr(strides);
    if (windowReversal)
      newConv.setWindowReversalAttr(windowReversal);
    newConv.setDimensionNumbersAttr(newDimNumbers);

    Value result = newConv.getResult();
    if (isConv1d)
      result = stablehloCollapseNchwToNch(rewriter, op.getLoc(), result);
    if (outputTranspose)
      result = rewriter.create<stablehlo::TransposeOp>(op.getLoc(), result,
                                                       *outputTranspose);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrite `stablehlo.convolution` to `stablehlo.dot` if the possible.
class StablehloRewriteConvolutionToDot
    : public OpRewritePattern<stablehlo::ConvolutionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    ConvDimensionNumbersAttr dims = op.getDimensionNumbers();
    if (!dims.getInputSpatialDimensions().empty() ||
        !dims.getKernelSpatialDimensions().empty() ||
        !dims.getOutputSpatialDimensions().empty())
      return rewriter.notifyMatchFailure(op, "spatial dims are present");
    if ((op.getLhsDilationAttr() && !op.getLhsDilationAttr().empty()) ||
        (op.getRhsDilationAttr() && !op.getRhsDilationAttr().empty()))
      return rewriter.notifyMatchFailure(op, "non-default dilation");
    if ((op.getWindowStridesAttr() && !op.getWindowStridesAttr().empty()) ||
        (op.getWindowReversalAttr() && !op.getWindowReversalAttr().empty()))
      return rewriter.notifyMatchFailure(op, "non default strides or reversal");
    if (op.getPaddingAttr() && !op.getPaddingAttr().empty())
      return rewriter.notifyMatchFailure(op, "non-default padding");
    if (op.getBatchGroupCount() != 1 || op.getFeatureGroupCount() != 1)
      return rewriter.notifyMatchFailure(op, "non-default group count");

    RankedTensorType inputType = op.getLhs().getType();
    RankedTensorType kernelType = op.getRhs().getType();
    RankedTensorType outputType = op.getType();

    if (inputType.getRank() != 2 || kernelType.getRank() != 2 ||
        outputType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "requries 2D input, kernel, and output");

    int64_t inputFeatureDim = dims.getInputFeatureDimension();
    int64_t kernelInputFeatureDim = dims.getKernelInputFeatureDimension();
    int64_t outputBatchDim = dims.getOutputBatchDimension();
    int64_t outputFeatureDim = dims.getOutputFeatureDimension();

    if (outputBatchDim != 0)
      return rewriter.notifyMatchFailure(op, "output batch dim is not 0");
    if (outputFeatureDim != 1)
      return rewriter.notifyMatchFailure(op, "output feature dim is not 1");

    auto dotDimNumbers = stablehlo::DotDimensionNumbersAttr::get(
        op.getContext(),
        /*lhsBatchingDimensions=*/{},
        /*rhsBatchingDimensions=*/{},
        /*lhsContractingDimensions=*/{inputFeatureDim},
        /*rhsContractingDimensions=*/{kernelInputFeatureDim});

    auto dot = rewriter.create<stablehlo::DotGeneralOp>(
        op.getLoc(), op.getType(), op.getLhs(), op.getRhs(), dotDimNumbers,
        op.getPrecisionConfigAttr(), DotAlgorithmAttr{});
    rewriter.replaceOp(op, dot.getResult());
    return success();
  }
};
} // namespace

void stablehlo_ext::populateCanonicalizeStablehloConvolutionPatterns(
    RewritePatternSet &patterns) {
  patterns
      .insert<StablehloRewriteConvolution, StablehloRewriteConvolutionToDot>(
          patterns.getContext());
}

namespace {
class CanonicalizeConvolutionPass
    : public stablehlo_ext::impl::CanonicalizeConvolutionPassBase<
          CanonicalizeConvolutionPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    stablehlo_ext::populateCanonicalizeStablehloConvolutionPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc())
          << "failed to apply rewrite patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
