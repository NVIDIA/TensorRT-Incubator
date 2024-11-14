//===- ShapeUtils.cpp -------------------------------------------*- c++ -*-===//
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
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

using namespace mlir;
using namespace mlir::tensorrt;

/// Tries to broadcast two shapes. Here "broadcastable" means that a) they have
/// equal rank and b) for each dimension i, either lhs_shape[i] == rhs_shape[i]
/// or one of the dimension size values is 1. In the case where both dimensions
/// are dynamic, then it is unknown whether the types are broadcastable and
/// this function will return success. The broadcasted shape is returned.
/// TODO: in the case of two unknown dimensions, we should have a constraint op
/// in place to verify the shapes are broadcastable. This is akin to what HLO
/// does with the shape dialect.
FailureOr<SmallVector<int64_t>>
tensorrt::getBroadcastedShape(RankedTensorType lhs, RankedTensorType rhs) {
  return getBroadcastedShape(lhs.getShape(), rhs.getShape());
}

FailureOr<SmallVector<int64_t>>
tensorrt::getBroadcastedShape(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  SmallVector<int64_t> shape;
  if (lhs.size() != rhs.size())
    return failure();

  for (auto [lhsDim, rhsDim] : llvm::zip(lhs, rhs)) {
    // Dimensions are both unknown. In this case we cannot be sure if the shapes
    // are broadcastable. Don't fail because we can't say for sure it's
    // invalid.
    if (lhsDim == rhsDim && lhsDim == ShapedType::kDynamic) {
      shape.push_back(lhsDim);
      continue;
    }
    // Dimensions are equal.
    if (lhsDim == rhsDim) {
      shape.push_back(lhsDim);
      continue;
    }
    // Rhs dimension is 1, rhs is broadcasted to lhs.
    if (rhsDim == 1) {
      shape.push_back(lhsDim);
      continue;
    }
    // Lhs dimension is 1, lhs is broadcasted to rhs.
    if (lhsDim == 1) {
      shape.push_back(rhsDim);
      continue;
    }
    // If one of the lhs or rhs dimension is known and >1 and another is
    // dynamic. In this case dynamic dimension is broadcasted to the known
    // dimension.
    if (lhsDim != rhsDim &&
        (lhsDim == ShapedType::kDynamic || rhsDim == ShapedType::kDynamic)) {
      shape.push_back(lhsDim == ShapedType::kDynamic ? rhsDim : lhsDim);
      continue;
    }

    return failure();
  }
  return shape;
}

FailureOr<SmallVector<int64_t>>
tensorrt::getBroadcastedShape(ArrayRef<ArrayRef<int64_t>> shapes) {
  if (shapes.empty())
    return failure();
  SmallVector<int64_t> shape;

  // Should be equal ranks.
  const unsigned rank = shapes.front().size();
  if (!llvm::all_of(shapes,
                    [&](ArrayRef<int64_t> s) { return s.size() == rank; }))
    return failure();

  // Lambda to get dim-th number from each shape.
  auto gatherDim = [](ArrayRef<ArrayRef<int64_t>> shapes, unsigned dim) {
    SmallVector<int64_t> result;
    for (auto shape : shapes)
      result.push_back(shape[dim]);
    return result;
  };

  // Resolve the result dim given the collection of dimension sizes from each
  // shape for a single dimension.
  auto resolveResultDim = [](ArrayRef<int64_t> dimSizes) -> FailureOr<int64_t> {
    assert(!dimSizes.empty() && "expected non-empty sizes");
    // All dims are  unknown. In this case we cannot be sure if the
    // shapes are broadcastable. Don't fail because we can't say for sure it's
    // invalid.
    const bool allEqual = llvm::all_equal(dimSizes);
    if (allEqual && dimSizes.front() == ShapedType::kDynamic)
      return ShapedType::kDynamic;

    // Dimensions are all equal to a static size.
    if (allEqual)
      return dimSizes.front();

    // Some dims are '1', all other dims are equal to another fixed number or
    // dynamic.
    std::optional<int64_t> nonUnitSize{};
    for (int64_t dimSize : dimSizes) {
      if (dimSize == 1)
        continue;
      if (ShapedType::isDynamic(dimSize))
        continue;
      if (nonUnitSize && dimSize == *nonUnitSize)
        continue;
      if (nonUnitSize && dimSize != *nonUnitSize)
        return failure();
      nonUnitSize = dimSize;
    }
    if (nonUnitSize)
      return *nonUnitSize;

    // No other case is valid.
    return failure();
  };

  for (auto dim : llvm::seq<unsigned>(0, rank)) {
    SmallVector<int64_t> dimSizes = gatherDim(shapes, dim);
    FailureOr<int64_t> resultDim = resolveResultDim(dimSizes);
    if (failed(resultDim))
      return failure();
    shape.push_back(*resultDim);
  }
  return shape;
}

LogicalResult tensorrt::checkShapesBroadcastable(ArrayRef<int64_t> lhs,
                                                 ArrayRef<int64_t> rhs) {
  for (auto [lhsDim, rhsDim] : llvm::zip(lhs, rhs)) {
    // If either is unknown, we can't verify it's not ok.
    if (lhsDim == ShapedType::kDynamic || rhsDim == ShapedType::kDynamic)
      continue;

    // Dimensions are equal.
    if (lhsDim == rhsDim)
      continue;

    // Rhs is broadcasted to lhs or vice-versa.
    if (rhsDim == 1 || lhsDim == 1)
      continue;

    return failure();
  }
  return success();
}

LogicalResult tensorrt::checkShapesBroadcastable(TensorType lhs,
                                                 TensorType rhs) {
  if (!lhs.hasRank() || !rhs.hasRank())
    return failure();
  if (lhs.getRank() != rhs.getRank())
    return failure();
  return checkShapesBroadcastable(lhs.getShape(), rhs.getShape());
}

LogicalResult tensorrt::checkLhsShapeBroadcastableToRhs(ArrayRef<int64_t> lhs,
                                                        ArrayRef<int64_t> rhs) {
  if (lhs.size() > rhs.size())
    return failure();

  for (auto [lhsDim, rhsDim] : llvm::zip(lhs, rhs)) {
    // If either is unknown, we can't verify it's not ok.
    if (lhsDim == ShapedType::kDynamic || rhsDim == ShapedType::kDynamic)
      continue;

    // Dimensions are equal.
    if (lhsDim == rhsDim)
      continue;

    // lhs dimension is 1, rhs dimension > 1, Then lhs is broadcasted to rhs.
    if (lhsDim == 1)
      continue;

    return failure();
  }
  return success();
}

LogicalResult tensorrt::isUnitDimRankExpanding(TensorType fromTensor,
                                               TensorType toTensor) {
  if (fromTensor.getRank() >= toTensor.getRank())
    return failure();

  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(fromTensor, toTensor);
  if (!reassociation.has_value())
    return failure();

  for (const auto &indexSet : *reassociation) {
    // Check  that there is at most one non-unit dim to the toTensor dims.
    if (llvm::count_if(indexSet, [&](int64_t toIdx) {
          return toTensor.getDimSize(toIdx) != 1;
        }) > 1)
      return failure();
  }
  return success();
}

LogicalResult tensorrt::isUnitDimRankReducing(TensorType fromTensor,
                                              TensorType toTensor) {
  if (fromTensor.getRank() <= toTensor.getRank())
    return failure();
  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(fromTensor, toTensor);
  if (!reassociation.has_value())
    return failure();

  for (const auto &indexSet : *reassociation) {
    // Check  that there is at most one non-unit dim to the toTensor dims.
    if (llvm::count_if(indexSet, [&](int64_t srcIdx) {
          return fromTensor.getDimSize(srcIdx) != 1;
        }) > 1)
      return failure();
  }
  return success();
}

bool tensorrt::areShapesEquivalentUpToDynamicDims(ArrayRef<int64_t> lhs,
                                                  ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [lhsDim, rhsDim] : llvm::zip(lhs, rhs)) {
    if (lhsDim == ShapedType::kDynamic || rhsDim == ShapedType::kDynamic)
      continue;
    if (lhsDim != rhsDim)
      return false;
  }
  return true;
}

bool tensorrt::areShapesEquivalentUpToDynamicDims(ShapedType lhs,
                                                  ShapedType rhs) {
  if (lhs.getElementType() != rhs.getElementType())
    return false;
  return areShapesEquivalentUpToDynamicDims(lhs.getShape(), rhs.getShape());
}

bool tensorrt::isTargetRefinementOfSource(ArrayRef<int64_t> source,
                                          ArrayRef<int64_t> target) {
  if (source.size() != target.size())
    return false;
  for (auto [srcDim, tgtDim] : llvm::zip_equal(source, target)) {
    // If the source dimension is dynamic, then the target dimension can be
    // dynamic or static.
    if (ShapedType::isDynamic(srcDim))
      continue;
    // Static source dim and dynamic result dim -> not a refinement.
    if (ShapedType::isDynamic(tgtDim))
      return false;
    // Static source dim != static result dim -> not a refinement.
    if (srcDim != tgtDim)
      return false;
  }
  return true;
}

bool tensorrt::isPermutationMap(ArrayRef<int64_t> permutation,
                                MLIRContext *context) {
  if (permutation.empty())
    return true;
  SmallVector<AffineExpr, 4> affExprs;
  for (auto index : permutation)
    affExprs.push_back(getAffineDimExpr(index, context));
  const auto *m = std::max_element(permutation.begin(), permutation.end());
  auto permutationMap = AffineMap::get(*m + 1, 0, affExprs, context);
  return permutationMap.isPermutation();
}

AffineMap tensorrt::getAsPermutationMap(MLIRContext *ctx,
                                        ArrayRef<int64_t> perm) {
  if (perm.empty())
    return AffineMap::get(ctx);
  return AffineMap::getPermutationMap(
      llvm::to_vector(llvm::map_range(
          perm, [](int64_t x) { return static_cast<unsigned>(x); })),
      ctx);
}

FailureOr<SmallVector<int64_t>> tensorrt::inferReshapeResultShape(
    RankedTensorType inputShape, ArrayRef<int64_t> reshapeSpec,
    bool zeroIsPlaceholder,
    std::function<LogicalResult(const std::string &msg)> errorCallback) {
  SmallVector<int64_t> inferredShape;
  std::optional<int64_t> negOneIdx = std::nullopt; // Index of -1 in reshape.
  inferredShape.reserve(reshapeSpec.size());

  // For static input shape:
  // We must resolve reshape in two phases - fill placeholders, then solve for
  // -1. Consider:
  //  - input shape = "[1, 2, 3, 4, 5, 6]"
  //  - reshape     = "[0, -1, 0, 30]"
  // We first fill in all ther zeros:
  //  - result  = "[1, -1, 3, 30]""
  // Then solve for the -1:
  //  - result = "[1, 8, 3, 30]"

  // For dynamic input shape:
  // Similar with procedure above - fill placeholders, then replace -1 with ?.
  // Consider:
  //  - input shape = "[2, 2, ?, ?]"
  //  - reshape     = "[0, -1, 1, 0]"
  // We first fill in all ther zeros:
  //  - result  = "[2, -1, 1, ?]""
  // Then solve for the -1:
  //  - result = "[2, ?, 1, ?]"

  for (auto [idx, specVal] : llvm::enumerate(reshapeSpec)) {
    if (specVal == 0) {
      if (zeroIsPlaceholder) {
        // In "zero is placeholder mode", we copy from the input shape.
        if (static_cast<int64_t>(idx) >= inputShape.getRank())
          return errorCallback("invalid reshape specification - 0 placeholder "
                               "maps to out-of-bounds index of input shape");
        inferredShape.push_back(inputShape.getDimSize(idx));
      } else {
        inferredShape.push_back(0);
      }
      continue;
    }

    if (specVal > 0) {
      inferredShape.push_back(specVal);
      continue;
    }

    if (specVal == -1) {
      negOneIdx = idx;
      // Record a 1 so that it's easier to solve in the next step.
      inferredShape.push_back(1);
      continue;
    }

    // Other cases are invalid.
    return errorCallback("invalid shape value");
  }

  if (negOneIdx)
    inferredShape[*negOneIdx] =
        inputShape.hasStaticShape()
            ? inputShape.getNumElements() /
                  std::accumulate(inferredShape.begin(), inferredShape.end(), 1,
                                  std::multiplies<>())
            : ShapedType::kDynamic;

  return inferredShape;
}

FailureOr<ConvDeconvPoolTensorShapeComponents>
ConvDeconvPoolTensorShapeComponents::createFromInputShape(
    ArrayRef<int64_t> inputShape) {
  if (inputShape.size() == 4 || inputShape.size() == 5)
    return ConvDeconvPoolTensorShapeComponents()
        .setShape(inputShape)
        .setSpatialDims(inputShape.drop_front(2))
        .setBatchSize(inputShape[0])
        .setChannels(inputShape[1]);
  return failure();
}

FailureOr<SmallVector<int64_t>>
ConvDeconvPoolTensorShapeComponents::getShape() {
  SmallVector<int64_t> outShape;
  if (spatialDims.size() == 0)
    return failure();
  outShape.reserve(this->spatialDims.size() + 2);
  outShape.push_back(this->batchSize);
  outShape.push_back(this->channels);
  llvm::append_range(outShape, spatialDims);
  return outShape;
}

ConvDeconvPoolTensorShapeComponents &
ConvDeconvPoolTensorShapeComponents::setShape(ArrayRef<int64_t> shape) {
  this->shape = llvm::to_vector(shape);
  return *this;
}

ConvDeconvPoolTensorShapeComponents &
ConvDeconvPoolTensorShapeComponents::setSpatialDims(
    ArrayRef<int64_t> spatialDims) {
  this->spatialDims = llvm::to_vector(spatialDims);
  return *this;
}

ConvDeconvPoolTensorShapeComponents &
ConvDeconvPoolTensorShapeComponents::setBatchSize(int64_t batchSize) {
  this->batchSize = batchSize;
  return *this;
}

ConvDeconvPoolTensorShapeComponents &
ConvDeconvPoolTensorShapeComponents::setChannels(int64_t channels) {
  this->channels = channels;
  return *this;
}

FailureOr<ConvDeconvKernelShapeComponents>
ConvDeconvKernelShapeComponents::createFromKernelShape(
    ArrayRef<int64_t> kernelShape, bool isOpConv, int32_t numGroups) {
  // Kernel weight must be 4D or 5D
  if (kernelShape.size() == 4 or kernelShape.size() == 5) {
    auto kernelShapeComp = ConvDeconvKernelShapeComponents();
    kernelShapeComp.setSpatialDims(kernelShape.drop_front(2));
    kernelShapeComp.setOpType(isOpConv);
    if (isOpConv) {
      kernelShapeComp.setOutChannels(kernelShape[0]);
      kernelShapeComp.setInChannels(
          numGroups == 1 ? kernelShape[1] : kernelShape[1] * numGroups);
    } else {
      kernelShapeComp.setInChannels(kernelShape[0]);
      kernelShapeComp.setOutChannels(
          numGroups == 1 ? kernelShape[1] : kernelShape[1] * numGroups);
    }
    return kernelShapeComp;
  }
  return failure();
}

ConvDeconvKernelShapeComponents &
ConvDeconvKernelShapeComponents::setSpatialDims(ArrayRef<int64_t> spatialDims) {
  this->spatialDims = llvm::to_vector(spatialDims);
  return *this;
}

ConvDeconvKernelShapeComponents &
ConvDeconvKernelShapeComponents::setOutChannels(int64_t outChannels) {
  this->outChannels = outChannels;
  return *this;
}

ConvDeconvKernelShapeComponents &
ConvDeconvKernelShapeComponents::setInChannels(int64_t inChannels) {
  this->inChannels = inChannels;
  return *this;
}

ConvDeconvKernelShapeComponents &
ConvDeconvKernelShapeComponents::setOpType(bool isOpConv) {
  this->isOpConv = isOpConv;
  return *this;
}

ConvDeconvPoolLayerComponents &
ConvDeconvPoolLayerComponents::setStride(ArrayRef<int64_t> stride) {
  this->stride = llvm::to_vector(stride);
  return *this;
}

ConvDeconvPoolLayerComponents &
ConvDeconvPoolLayerComponents::setPrePadding(ArrayRef<int64_t> prePadding) {
  this->prePadding = llvm::to_vector(prePadding);
  return *this;
}

ConvDeconvPoolLayerComponents &
ConvDeconvPoolLayerComponents::setPostPadding(ArrayRef<int64_t> postPadding) {
  this->postPadding = llvm::to_vector(postPadding);
  return *this;
}

ConvDeconvPoolLayerComponents &ConvDeconvPoolLayerComponents::setPoolingWindow(
    ArrayRef<int64_t> poolingWindow) {
  this->poolingWindow = llvm::to_vector(poolingWindow);
  return *this;
}

ConvDeconvPoolLayerComponents &ConvDeconvPoolLayerComponents::setDilation(
    std::optional<ArrayRef<int64_t>> dilation) {
  if (dilation.has_value()) {
    this->dilation = llvm::to_vector(*dilation);
  }
  return *this;
}

ConvDeconvPoolLayerComponents &ConvDeconvPoolLayerComponents::setPaddingMode(
    shape_utils::TensorRTPaddingMode paddingMode) {
  this->paddingMode = paddingMode;
  return *this;
}

ConvDeconvPoolLayerComponents &
ConvDeconvPoolLayerComponents::setNumGroups(int32_t numGroups) {
  this->numGroups = numGroups;
  return *this;
}

FailureOr<ConvDeconvPoolTensorShapeComponents>
tensorrt::getConvDeconvOpOutputShape(
    ConvDeconvPoolTensorShapeComponents &inpShapeComp,
    ConvDeconvKernelShapeComponents &kernelShapeComp,
    ConvDeconvPoolLayerComponents &layerComp, bool isConv) {
  // Supports only TensorRT's default round down padding mode.
  if (layerComp.paddingMode !=
      tensorrt::shape_utils::TensorRTPaddingMode::EXPLICIT_ROUND_DOWN)
    return failure();
  int64_t numSpatialDims = inpShapeComp.spatialDims.size();
  auto outShapeComp = tensorrt::ConvDeconvPoolTensorShapeComponents();
  // Update batch size and output channels in the output.
  outShapeComp.batchSize = inpShapeComp.batchSize;
  outShapeComp.channels = kernelShapeComp.outChannels;
  // Compute spatial dimensions of the output.
  for (int i = 0; i < numSpatialDims; i++) {
    if (ShapedType::isDynamic(inpShapeComp.shape[i + 2])) {
      outShapeComp.spatialDims.push_back(inpShapeComp.shape[i + 2]);
      continue;
    }
    int64_t d = layerComp.dilation.has_value() ? (*layerComp.dilation)[i] : 1;
    if (isConv) {
      int64_t dimOut = static_cast<int64_t>(
          floor(((inpShapeComp.shape[i + 2] + layerComp.prePadding[i] +
                  layerComp.postPadding[i] -
                  d * (kernelShapeComp.spatialDims[i] - 1) - 1) /
                     float(layerComp.stride[i]) +
                 1)));
      outShapeComp.spatialDims.push_back(dimOut);
    } else {
      int64_t dimOut = (inpShapeComp.shape[i + 2] - 1) * layerComp.stride[i] -
                       layerComp.prePadding[i] - layerComp.postPadding[i] +
                       d * (kernelShapeComp.spatialDims[i] - 1) + 1;
      outShapeComp.spatialDims.push_back(dimOut);
    }
  }
  return outShapeComp;
}

FailureOr<ConvDeconvPoolTensorShapeComponents>
tensorrt::getPoolingOpOutputShape(
    ConvDeconvPoolTensorShapeComponents &inpShapeComp,
    ConvDeconvPoolLayerComponents &layerComp) {
  int64_t numSpatialDims = inpShapeComp.spatialDims.size();
  auto outShapeComp = tensorrt::ConvDeconvPoolTensorShapeComponents();
  // Update batch size and output channels in the output.
  outShapeComp.setBatchSize(inpShapeComp.batchSize);
  outShapeComp.setChannels(inpShapeComp.channels);
  // Compute spatial dimensions of the output.
  for (int i = 0; i < numSpatialDims; i++) {
    if (ShapedType::isDynamic(inpShapeComp.shape[i + 2])) {
      outShapeComp.spatialDims.push_back(inpShapeComp.shape[i + 2]);
      continue;
    }
    int64_t m = inpShapeComp.shape[i + 2] + layerComp.prePadding[i] +
                layerComp.postPadding[i];
    if (layerComp.paddingMode ==
        tensorrt::shape_utils::TensorRTPaddingMode::EXPLICIT_ROUND_DOWN) {
      outShapeComp.spatialDims.push_back(
          static_cast<int64_t>(
              floor((m - layerComp.poolingWindow[i]) / layerComp.stride[i])) +
          1);
    } else if (layerComp.paddingMode ==
               tensorrt::shape_utils::TensorRTPaddingMode::EXPLICIT_ROUND_UP) {
      outShapeComp.spatialDims.push_back(
          static_cast<int64_t>(
              ceil((m - layerComp.poolingWindow[i]) / layerComp.stride[i])) +
          1);
    } else {
      return failure();
    }
  }
  return outShapeComp;
}

SmallVector<int64_t> tensorrt::shapeDivide(ArrayRef<int64_t> lhs,
                                           ArrayRef<int64_t> rhs) {
  assert(lhs.size() == rhs.size() && "expected equal ranks");
  SmallVector<int64_t> result(lhs.size());
  for (unsigned i = 0; i < lhs.size(); i++)
    result[i] = lhs[i] / rhs[i];
  return result;
}

SmallVector<int64_t> tensorrt::shapeDivide(ArrayRef<int64_t> lhs, int64_t rhs) {
  assert(!lhs.empty());
  SmallVector<int64_t> result(lhs.size());
  for (auto [idx, dimSize] : llvm::enumerate(llvm::reverse(lhs))) {
    result[lhs.size() - 1 - idx] = std::min(dimSize, rhs);
    rhs = std::max<int64_t>(rhs / dimSize, 1);
  }
  return result;
}

SmallVector<int64_t> tensorrt::shapeMax(ArrayRef<int64_t> lhs, int64_t rhs) {
  SmallVector<int64_t> result;
  result.reserve(lhs.size());
  for (auto val : lhs)
    result.push_back(std::max(val, rhs));
  return result;
}

SmallVector<int64_t> tensorrt::shapeMultiply(ArrayRef<int64_t> lhs,
                                             ArrayRef<int64_t> rhs) {
  assert(lhs.size() == rhs.size() && "expected equal ranks");
  SmallVector<int64_t> result(lhs.size());
  for (unsigned i = 0; i < lhs.size(); i++)
    result[i] = lhs[i] * rhs[i];
  return result;
}

int64_t tensorrt::shapeVolume(ArrayRef<int64_t> input) {
  if (llvm::find(input, ShapedType::kDynamic) != input.end())
    return ShapedType::kDynamic;
  return std::accumulate(input.begin(), input.end(), 1, std::multiplies<>());
}

SmallVector<AffineExpr> tensorrt::computeElementwiseAdd(ArrayRef<AffineExpr> v1,
                                                        ArrayRef<int64_t> v2) {
  // Early exit if both are empty, let zip_equal fail if only 1 is empty.
  if (v1.empty() && v2.empty())
    return {};
  SmallVector<AffineExpr> result;
  for (auto [lhs, rhs] : llvm::zip_equal(v1, v2))
    result.push_back(lhs + rhs);
  return result;
}

SmallVector<AffineExpr>
tensorrt::computeElementwiseAdd(ArrayRef<AffineExpr> v1,
                                ArrayRef<AffineExpr> v2) {
  // Early exit if both are empty, let zip_equal fail if only 1 is empty.
  if (v1.empty() && v2.empty())
    return {};
  SmallVector<AffineExpr> result;
  for (auto [lhs, rhs] : llvm::zip_equal(v1, v2))
    result.push_back(lhs + rhs);
  return result;
}

SmallVector<AffineExpr> tensorrt::computeElementwiseMul(ArrayRef<AffineExpr> v1,
                                                        ArrayRef<int64_t> v2) {
  // Early exit if both are empty, let zip_equal fail if only 1 is empty.
  if (v1.empty() && v2.empty())
    return {};
  SmallVector<AffineExpr> result;
  for (auto [lhs, rhs] : llvm::zip_equal(v1, v2))
    result.push_back(lhs * rhs);
  return result;
}

//===----------------------------------------------------------------------===//
// Tiling utilities
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<int64_t>>
mlir::computeTileShape(ArrayRef<int64_t> outerShape, int64_t tileVolume,
                       std::optional<ArrayRef<int64_t>> tileOrder) {
  assert((!tileOrder || tileOrder->size() == outerShape.size()) &&
         "expected outerShape and tileOrder to be equal rank");

  SmallVector<int64_t> tileShape(outerShape.size(), 0);
  for (unsigned i = 0, e = outerShape.size(); i < e; i++) {
    unsigned idx = tileOrder ? (*tileOrder)[i] : (e - 1 - i);
    int64_t dimSize = outerShape[idx];
    if (tileVolume >= dimSize) {
      if (tileVolume % dimSize != 0)
        return failure();
      tileShape[idx] = dimSize;
      tileVolume /= dimSize;
      continue;
    }

    // If remaining tile volume is smaller,
    // then we divide dimSize by the remaining volume
    // it is divisible.
    if (dimSize % tileVolume != 0)
      return failure();
    tileShape[idx] = tileVolume;
    tileVolume = 1;
  }

  return tileShape;
}
