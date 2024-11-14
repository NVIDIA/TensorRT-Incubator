//===- StablehloTensorKindOpInterfaceImpl.cpp -----------------------------===//
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
/// Implementation of ReifyRankedShapedTypeOpInterface for specific StableHlo
/// ops.
///
/// Most of the functions in this file are straight-forward dynamic
/// implementations of the static/constexpr inference functions implemented for
/// the ConvolutionOp here:
/// https://github.com/openxla/stablehlo/tree/main/stablehlo/dialect/TypeInference.cpp.
///
/// These functions should ideally be merged upstream in order to reduce
/// overhead, except that upstream would probably require us to use
/// InferShapedTypeOpInterface (tensor-based) instead of
/// ReifyRankedShapedTypeOpInterface here, which uses scalar operations and can
/// leverage affine.apply and affine.max ops.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StableHloExt/IR/StableHloExt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"

using namespace mlir;
using namespace mlir::stablehlo;
using namespace mlir::hlo;

namespace {
// DynamicWindowDimension is just like WindowDimension, but we use a potentially
// dynamic dimension size instead of a purely static dimension size.stat
struct DynamicWindowDimension {
  OpFoldResult size = nullptr;
  int64_t stride = 1;
  int64_t paddingLow = 0;
  int64_t paddingHigh = 0;
  int64_t windowDilation = 1;
  int64_t baseDilation = 1;
  bool windowReversal = false;
};
} // namespace

// This is equivalent to `hlo::dilatedBound`, but it uses affine.max instead of
// static calculation.
static OpFoldResult dilatedBound(OpBuilder &b, Location loc, OpFoldResult bound,
                                 int64_t dilation) {
  // We need to calculate max(dilation*(bound-1)+1,0) in case the
  // dimension is dynamically 0 (and thus escapes the first 'if' above).
  SmallVector<Value> operands;
  if (bound.is<Value>())
    operands.push_back(bound.get<Value>());
  AffineExpr d0 = bound.is<Value>()
                      ? b.getAffineDimExpr(0)
                      : b.getAffineConstantExpr(*getConstantIntValue(bound));
  Value result =
      affine::expandAffineExpr(b, loc, dilation * (d0 - 1) + 1, operands, {});
  return b.createOrFold<arith::MaxSIOp>(
      loc, result, b.create<arith::ConstantIndexOp>(loc, 0));
}

// Documentation from the static HLO version of this function:
// Returns the number of valid positions of a window with the given size and
// stride within an array with the given bound. This is the bound of an output
// array with one element per valid position of the window.
//
// For example, for arguments of (bound=5, window_size=2, stride=2), the
// returned value is 2. There are valid positions at offset 0 and offset 2,
// while offset 4 is not valid since the window's last entry would be at 5,
// which is beyond the bound of 5.
static OpFoldResult stridedBound(OpBuilder &b, Location loc, Value bound,
                                 Value windowSize, int64_t stride) {
  assert(stride > 0 && "expected positive stride");
  auto map = AffineMap::get(
      2, 0,
      {(b.getAffineDimExpr(0) - b.getAffineDimExpr(1)).floorDiv(stride) + 1,
       b.getAffineConstantExpr(0)},
      b.getContext());
  std::optional<SmallVector<Value>> results =
      affine::expandAffineMap(b, loc, map, {bound, windowSize});
  assert(results);
  return b.createOrFold<arith::MaxSIOp>(loc, (*results)[0], (*results)[1]);
}

/// Documentation from teh static HLO version of this function:
/// Infer the shape of the output window.
///  Foreach dimension d,
///    output-window-shape[d] =
///            stridedBound(padding_low + dilatedBound(base_shape[d]) +
///            padding_high,
///                         dilatedBound(window_shape[d]))
///      where (padding_low, padding_high) is the padding-pair for d.
static SmallVector<OpFoldResult>
inferWindowOutputShape(OpBuilder &b, Location loc,
                       ArrayRef<OpFoldResult> baseShape,
                       ArrayRef<DynamicWindowDimension> window) {
  assert(baseShape.size() == window.size() &&
         "Size of window dimensions must match the size of base shape.");
  SmallVector<OpFoldResult> outputDimensions(window.size());
  for (int64_t i = 0; i < static_cast<int64_t>(window.size()); ++i) {
    const DynamicWindowDimension &dim = window[i];
    const OpFoldResult dilatedBase =
        dilatedBound(b, loc, baseShape[i], dim.baseDilation);
    const OpFoldResult paddedDilatedBase =
        affine::makeComposedFoldedAffineApply(
            b, loc,
            AffineMap::get(
                1, 0, dim.paddingLow + b.getAffineDimExpr(0) + dim.paddingHigh),
            {dilatedBase});
    const OpFoldResult dilatedWindow =
        dilatedBound(b, loc, dim.size, dim.windowDilation);
    outputDimensions[i] = stridedBound(
        b, loc, getValueOrCreateConstantIndexOp(b, loc, paddedDilatedBase),
        getValueOrCreateConstantIndexOp(b, loc, dilatedWindow), dim.stride);
  }
  return outputDimensions;
}

/// Pack the convolution dimension information into a set
/// 'DynamicWindowDimension' objects which capture the important information
/// about each dimension.
static FailureOr<SmallVector<DynamicWindowDimension>> getWindowDimensionInfo(
    ArrayRef<OpFoldResult> windowDimensions, ArrayRef<int64_t> windowStrides,
    ArrayRef<std::pair<int64_t, int64_t>> padding,
    ArrayRef<int64_t> lhsDilation, ArrayRef<int64_t> rhsDilation,
    ArrayRef<bool> windowReversal, std::optional<Location> loc) {

  SmallVector<DynamicWindowDimension> window(windowDimensions.size());
  for (size_t i = 0; i < windowDimensions.size(); i++) {
    DynamicWindowDimension &dim = window[i];
    dim.size = windowDimensions[i];

    if (!windowStrides.empty())
      dim.stride = windowStrides[i];

    if (!lhsDilation.empty())
      dim.baseDilation = lhsDilation[i];

    if (!rhsDilation.empty())
      dim.windowDilation = rhsDilation[i];

    if (!padding.empty()) {
      dim.paddingLow = padding[i].first;
      dim.paddingHigh = padding[i].second;
    }
  }

  return window;
}

/// Return the (possibly dynamic, possibly static) tensor dimension extent of
/// `val` at dimension `dim`.
static OpFoldResult getDimExtent(OpBuilder &b, Location loc, Value val,
                                 int64_t dim) {
  assert(dim < cast<RankedTensorType>(val.getType()).getRank());
  return b.createOrFold<tensor::DimOp>(
      loc, val, b.create<arith::ConstantIndexOp>(loc, dim));
}

namespace {
class ConvolutionReifyRankedShapedTypeOpInterfaceImpl
    : public ReifyRankedShapedTypeOpInterface::ExternalModel<
          ConvolutionReifyRankedShapedTypeOpInterfaceImpl,
          stablehlo::ConvolutionOp> {
public:
  LogicalResult
  reifyResultShapes(Operation *op_, OpBuilder &builder,
                    ReifiedRankedShapedTypeDims &reifiedReturnShapes) const {
    auto op = cast<stablehlo::ConvolutionOp>(op_);
    Location loc = op.getLoc();

    ConvDimensionNumbersAttr dimensionNumbers = op.getDimensionNumbers();

    int64_t inputBatchDimension = dimensionNumbers.getInputBatchDimension();
    ArrayRef<int64_t> inputSpatialDimensions =
        dimensionNumbers.getInputSpatialDimensions();

    int64_t kernelOutputFeatureDimension =
        dimensionNumbers.getKernelOutputFeatureDimension();
    ArrayRef<int64_t> kernelSpatialDimensions =
        dimensionNumbers.getKernelSpatialDimensions();

    int64_t outputBatchDimension = dimensionNumbers.getOutputBatchDimension();
    int64_t outputFeatureDimension =
        dimensionNumbers.getOutputFeatureDimension();
    ArrayRef<int64_t> outputSpatialDimensions =
        dimensionNumbers.getOutputSpatialDimensions();

    size_t batchGroupCount = op.getBatchGroupCount();

    FailureOr<SmallVector<std::pair<int64_t, int64_t>>> padding =
        convertPaddingAttribute(op.getPadding(), loc);
    if (failed(padding))
      return failure();

    SmallVector<OpFoldResult> windowDimensions(kernelSpatialDimensions.size());
    assert(kernelSpatialDimensions.size() == windowDimensions.size());
    for (size_t i = 0; i < windowDimensions.size(); i++)
      windowDimensions[i] = builder.createOrFold<tensor::DimOp>(
          loc, op.getRhs(),
          builder.create<arith::ConstantIndexOp>(loc,
                                                 kernelSpatialDimensions[i]));

    FailureOr<SmallVector<DynamicWindowDimension>> windowOrErr =
        getWindowDimensionInfo(
            windowDimensions,
            op.getWindowStrides().value_or(ArrayRef<int64_t>{}), *padding,
            op.getLhsDilation().value_or(ArrayRef<int64_t>{}),
            op.getRhsDilation().value_or(ArrayRef<int64_t>{}),
            op.getWindowReversal().value_or(ArrayRef<bool>{}), loc);
    if (failed(windowOrErr))
      return failure();

    // The resultShape holds one scalar extent `index`-typed value for each
    // dimension of the result.
    SmallVector<OpFoldResult> resultShape(op.getType().getRank(), nullptr);

    // The batch dimension in the result is just mapped to the batch dimension
    // of the input.
    Value inputBatchSize = builder.create<tensor::DimOp>(
        loc, op.getLhs(),
        builder.create<arith::ConstantIndexOp>(loc, inputBatchDimension));

    resultShape[outputBatchDimension] = builder.createOrFold<arith::DivUIOp>(
        loc, inputBatchSize,
        builder.create<arith::ConstantIndexOp>(loc, batchGroupCount));

    unsigned numSpatialDims = inputSpatialDimensions.size();
    SmallVector<OpFoldResult> inputSpatialDimVals(numSpatialDims);
    for (int64_t i = 0; i < static_cast<int64_t>(numSpatialDims); ++i)
      inputSpatialDimVals[i] =
          getDimExtent(builder, loc, op.getLhs(), inputSpatialDimensions[i]);

    SmallVector<OpFoldResult> windowOutputShape =
        inferWindowOutputShape(builder, loc, inputSpatialDimVals, *windowOrErr);

    for (int64_t i = 0; i < static_cast<int64_t>(windowOrErr->size()); ++i)
      resultShape[outputSpatialDimensions[i]] = windowOutputShape[i];

    resultShape[outputFeatureDimension] =
        getDimExtent(builder, loc, op.getRhs(), kernelOutputFeatureDimension);

    // Fixup the result to enforce the required convention for
    // `reifyResultShapes` -- if the dimension is dynamic and we infer a static
    // integer extent, we must still return a Value. Likewise, the above routine
    // may produce a `Value` even though the result type already contains a
    // known fixed extent.
    RankedTensorType resultType = op.getType();
    for (auto [idx, ofr] : llvm::enumerate(resultShape)) {
      assert(ofr && "result shape is missing a value");
      if (resultType.isDynamicDim(idx) && !ofr.is<Value>())
        resultShape[idx] = getValueOrCreateConstantIndexOp(builder, loc, ofr);
      if (!resultType.isDynamicDim(idx) && !ofr.is<Attribute>())
        resultShape[idx] = builder.getIndexAttr(resultType.getDimSize(idx));
    }

    reifiedReturnShapes.emplace_back(std::move(resultShape));
    return success();
  }
};

class ReduceWindowReifyRankedShapedTypeOpInterfaceImpl
    : public ReifyRankedShapedTypeOpInterface::ExternalModel<
          ReduceWindowReifyRankedShapedTypeOpInterfaceImpl,
          stablehlo::ReduceWindowOp> {
public:
  LogicalResult
  reifyResultShapes(Operation *op_, OpBuilder &builder,
                    ReifiedRankedShapedTypeDims &reifiedReturnShapes) const {
    auto op = cast<stablehlo::ReduceWindowOp>(op_);
    Location loc = op.getLoc();

    FailureOr<SmallVector<std::pair<int64_t, int64_t>>> padding =
        convertPaddingAttribute(op.getPadding(), loc);
    if (failed(padding))
      return failure();

    // In ReduceWindowOp, size of window_dim, padding, stride and dilation all
    // equal to input rank. So the output shape is inferred altogether.
    SmallVector<int64_t> windowDims = llvm::to_vector(op.getWindowDimensions());
    SmallVector<OpFoldResult> windowDimensionVals(windowDims.size());
    for (size_t i = 0; i < windowDims.size(); i++)
      windowDimensionVals[i] =
          builder.createOrFold<arith::ConstantIndexOp>(loc, windowDims[i]);

    FailureOr<SmallVector<DynamicWindowDimension>> windowOrErr =
        getWindowDimensionInfo(
            windowDimensionVals,
            op.getWindowStrides().value_or(ArrayRef<int64_t>{}), *padding,
            op.getBaseDilations().value_or(ArrayRef<int64_t>{}),
            op.getWindowDilations().value_or(ArrayRef<int64_t>{}),
            ArrayRef<bool>{}, loc);
    if (failed(windowOrErr))
      return failure();

    int64_t inputRank = static_cast<int64_t>(windowDims.size());
    SmallVector<OpFoldResult> inputDimVals(inputRank);
    for (int64_t i = 0; i < inputRank; ++i)
      inputDimVals[i] = getDimExtent(builder, loc, op.getInputs().front(), i);

    SmallVector<OpFoldResult> resultShape =
        inferWindowOutputShape(builder, loc, inputDimVals, *windowOrErr);

    // Fixup the result to enforce the required convention for
    // `reifyResultShapes` -- if the dimension is dynamic and we infer a static
    // integer extent, we must still return a Value. Likewise, the above routine
    // may produce a `Value` even though the result type already contains a
    // known fixed extent.
    RankedTensorType resultType = cast<RankedTensorType>(op.getType(0));
    for (auto [idx, ofr] : llvm::enumerate(resultShape)) {
      assert(ofr && "result shape is missing a value");
      if (resultType.isDynamicDim(idx) && !ofr.is<Value>())
        resultShape[idx] = getValueOrCreateConstantIndexOp(builder, loc, ofr);
      if (!resultType.isDynamicDim(idx) && !ofr.is<Attribute>())
        resultShape[idx] = builder.getIndexAttr(resultType.getDimSize(idx));
    }

    reifiedReturnShapes.emplace_back(std::move(resultShape));
    return success();
  }
};
} // namespace

void stablehlo::registerTypeInferenceExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, stablehlo::StablehloDialect *dialect) {
        stablehlo::ConvolutionOp::attachInterface<
            ConvolutionReifyRankedShapedTypeOpInterfaceImpl>(*ctx);
      });
  registry.addExtension(
      +[](MLIRContext *ctx, stablehlo::StablehloDialect *dialect) {
        stablehlo::ReduceWindowOp::attachInterface<
            ReduceWindowReifyRankedShapedTypeOpInterfaceImpl>(*ctx);
      });
}
