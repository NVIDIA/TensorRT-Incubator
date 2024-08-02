//===- ShapeUtils.h ---------------------------------------------*- c++ -*-===//
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
// Functions in this translation unit should only have dependencies on upstream
// MLIR.
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_UTILS_SHAPEUTILS
#define MLIR_TENSORRT_UTILS_SHAPEUTILS

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
class RewriterBase;
namespace tensorrt {

/// Tries to broadcast two shapes. Here "broadcastable" means that a) they have
/// equal rank and b) for each dimension i, either lhs_shape[i] == rhs_shape[i]
/// or one of the dimension size values is 1. In the case where both dimensions
/// are dynamic, then it is unknown whether the types are broadcastable and
/// this function will return success. The broadcasted shape is returned.
/// TODO: in the case of two unknown dimensions, we should have a constraint op
/// in place to verify the shapes are broadcastable. This is akin to what HLO
/// does with the shape dialect.
FailureOr<SmallVector<int64_t>> getBroadcastedShape(RankedTensorType lhs,
                                                    RankedTensorType rhs);
FailureOr<SmallVector<int64_t>> getBroadcastedShape(ArrayRef<int64_t> lhs,
                                                    ArrayRef<int64_t> rhs);

/// Shape as the above `getBroadcastedShape`, but calculates the broadcasted
/// shape of N different shapes.
FailureOr<SmallVector<int64_t>>
getBroadcastedShape(ArrayRef<ArrayRef<int64_t>> shapes);

/// A slightly simpler version of the above that returns failure if two shapes
/// are provably not broadcastable. Otherwise it returns success conservatively.
LogicalResult checkShapesBroadcastable(TensorType lhs, TensorType rhs);
LogicalResult checkShapesBroadcastable(ArrayRef<int64_t> lhs,
                                       ArrayRef<int64_t> rhs);

/// Similar to `checkShapesBroadcastable`, but only the `lhs` can have unit dims
/// that broadcast, not rhs.
LogicalResult checkLhsShapeBroadcastableToRhs(ArrayRef<int64_t> lhs,
                                              ArrayRef<int64_t> rhs);

/// Returns success if the reshape from `fromTensor` to `toTensor` is a rank
/// expanding reshape that only adds unit-dims.
LogicalResult isUnitDimRankExpanding(TensorType fromTensor,
                                     TensorType toTensor);

/// Returns success if the reshape from `fromTensor` to `toTensor` is a rank
/// reducing reshape that only removes unit-dims.
LogicalResult isUnitDimRankReducing(TensorType fromTensor, TensorType toTensor);

/// Returns true if the shape of `lhs` is equal to the shape of `rhs`, where
/// "equal" means that dimensions are either equal or cannot be shown to be not
/// equal (i.e. one or both are dynamic).
bool areShapesEquivalentUpToDynamicDims(ArrayRef<int64_t> lhs,
                                        ArrayRef<int64_t> rhs);
bool areShapesEquivalentUpToDynamicDims(ShapedType lhs, ShapedType rhs);

/// Return `true` if `target` is the same as source except that one or more
/// dynamic dimensions have been refined into static extents.
bool isTargetRefinementOfSource(ArrayRef<int64_t> source,
                                ArrayRef<int64_t> target);

/// Create an AffineMap from the given array and check if it is a permutation
bool isPermutationMap(ArrayRef<int64_t> permutation, MLIRContext *context);

/// Takes an array of integers representing a permutation and returns an
/// AfineMap representation.
AffineMap getAsPermutationMap(MLIRContext *ctx, ArrayRef<int64_t> perm);

/// Infer the result shape for a reshape operation with the specified
/// `inputType`. Emits diagnostics via `errorCallback`. Returns the inferred
/// result shape.
FailureOr<SmallVector<int64_t>> inferReshapeResultShape(
    RankedTensorType inputShape, ArrayRef<int64_t> reshapeSpec,
    bool zeroIsPlaceholder,
    std::function<LogicalResult(const std::string &msg)> errorCallback);

/// TensorRT padding modes
namespace shape_utils {
enum class TensorRTPaddingMode {
  EXPLICIT_ROUND_DOWN = 0,
  EXPLICIT_ROUND_UP = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
  CAFFE_ROUND_DOWN = 4,
  CAFFE_ROUND_UP = 5
};
}

/// Helper struct to organize input/output for Convolution, Deconvolution and
/// Pooling op
struct ConvDeconvPoolTensorShapeComponents {
  SmallVector<int64_t> spatialDims;
  SmallVector<int64_t> shape;
  int64_t batchSize;
  int64_t channels;
  bool isInput = false;

  /// 2D convolution/deconvolution/pooling input is assumed to be organized as
  /// [N, C_in, H_in, W_in]
  /// 3D convolution/deconvolution/pooling input is assumed to be organized as
  /// [N, C_in, D_in, H_in, W_in]
  /// where,
  /// N = batch size, C_in = input channels, H_in = input height,
  /// W_in = input width and D_in = input depth
  /// Dimensions after first two (N and C_in) are spatial dimensions.
  static FailureOr<ConvDeconvPoolTensorShapeComponents>
  createFromInputShape(ArrayRef<int64_t> inputShape);

  /// If `ConvDeconvPoolTensorShapeComponents` represents an input tensor to the
  /// operation, this function returns input tensor shape otherwise returns
  /// shape of the output tensor of the operation.
  ///
  /// 2D convolution/deconvolution/pooling output is organized as
  /// [N, C_out, H_out, W_out]
  /// 3D convolution/deconvolution/pooling output is organized as
  /// [N, C_out, D_out, H_out, W_out]
  /// where,
  /// N = batch size, C_out = output channels, H_out = output height,
  /// W_out = output width and D_out = output depth
  /// Dimensions after first two (N and C_out) are spatial dimensions.
  FailureOr<SmallVector<int64_t>> getShape();

  ConvDeconvPoolTensorShapeComponents &setShape(ArrayRef<int64_t> shape);
  ConvDeconvPoolTensorShapeComponents &
  setSpatialDims(ArrayRef<int64_t> spatialDims);
  ConvDeconvPoolTensorShapeComponents &setBatchSize(int64_t batchSize);
  ConvDeconvPoolTensorShapeComponents &setChannels(int64_t channels);
};

/// Helper struct to organize Convolution or Deconvolution kernel parameters.
struct ConvDeconvKernelShapeComponents {
  SmallVector<int64_t> spatialDims;
  int64_t inChannels;
  int64_t outChannels;
  bool isOpConv = true;

  /// 2D convolution kernel is assumed to be organized as [C_out,
  /// C_in/num_groups, K_0, K_1]
  /// 3D convolution kernel is assumed to be organized as [C_out,
  /// C_in/num_groups, K_0, K_1, K_2]
  /// 2D deconvolution kernel assumed to be is organized as [C_in,
  /// C_out/num_groups, K_0, K_1]
  /// 3D deconvolution kernel assumed to be is organized as [C_in,
  /// C_out/num_groups, K_0, K_1, K_2]
  /// Bias, if present, is assumed to be organized as [C_out]
  /// where, C_out = output channels, C_in = input channels K_{i} = ith
  /// filter spatial dimension.
  /// Dimensions after first two (input and output channels) are spatial
  /// dimensions.
  static FailureOr<ConvDeconvKernelShapeComponents>
  createFromKernelShape(ArrayRef<int64_t> kernelShape, bool isOpConv,
                        int32_t numGroups);

  ConvDeconvKernelShapeComponents &
  setSpatialDims(ArrayRef<int64_t> spatialDims);
  ConvDeconvKernelShapeComponents &setOutChannels(int64_t outChannels);
  ConvDeconvKernelShapeComponents &setInChannels(int64_t inChannels);
  ConvDeconvKernelShapeComponents &setOpType(bool isOpConv);
};

/// Helper struct to organize Convolution or Deconvolution layer attributes.
struct ConvDeconvPoolLayerComponents {
  SmallVector<int64_t> stride;
  SmallVector<int64_t> prePadding;
  SmallVector<int64_t> postPadding;
  SmallVector<int64_t> poolingWindow;           // NOT used in conv/deconv
  std::optional<SmallVector<int64_t>> dilation; // NOT used for pooling
  int32_t numGroups = 1;                        // NOT used for pooling
  shape_utils::TensorRTPaddingMode paddingMode =
      shape_utils::TensorRTPaddingMode::EXPLICIT_ROUND_DOWN;

  ConvDeconvPoolLayerComponents &setStride(ArrayRef<int64_t> stride);
  ConvDeconvPoolLayerComponents &setPrePadding(ArrayRef<int64_t> prePadding);
  ConvDeconvPoolLayerComponents &setPostPadding(ArrayRef<int64_t> postPadding);
  ConvDeconvPoolLayerComponents &
  setPoolingWindow(ArrayRef<int64_t> poolingWindow);
  ConvDeconvPoolLayerComponents &
  setDilation(std::optional<ArrayRef<int64_t>> dilation);
  ConvDeconvPoolLayerComponents &
  setPaddingMode(shape_utils::TensorRTPaddingMode paddingMode);
  ConvDeconvPoolLayerComponents &setNumGroups(int32_t numGroups);
};

/// Computes the output shape of convolution or deconvolution op based on
/// `ConvDeconvPoolTensorShapeComponents`, `ConvDeconvKernelShapeComponents`,
/// and
// `ConvDeconvPoolLayerComponents` and return as
// `ConvDeconvPoolTensorShapeComponents`.
/// Shorthands:
/// O_{j} = jth output dimension
/// I_{j} = jth input dimension
/// A_{i} = prepadding added to the ith spatial dimension. The default is 0.
/// B_{i} = postpadding added to the ith spatial dimension . The default is 0.
/// M_{i} = I_{j} + A_{i} + B_{i}
/// K_{i} = ith spatial dimension of kernel
/// D_{i} = dilation value for the ith spatial dimension. The default is 1.
/// DK_{i} = 1 + D_{i} * (K_{i} - 1)
/// S_{i} = stride value for the ith spatial dimension. The default is 1.
/// N = batch size
/// C_out = output channels
///
/// First two dimensions of output are [N, C_out]. Output dimension `O_{j}`, for
/// each remaining spatial dimension `i` in the range `[0, num_spatial_dims]`
/// with `j` = `i` + 2 is computed based on layer attributes as follows,
///
/// A. Padding mode == EXPLICIT_ROUND_DOWN
/// O_{j} = floor((M_{i} - DK_{i}) / S_{i}) + 1
/// B. Padding mode == EXPLICIT_ROUND_UP
/// O_{j} = ceil((M_{i} - DK_{i}) / S_{i}) + 1
///
/// TODO: Add support for more padding modes.
FailureOr<ConvDeconvPoolTensorShapeComponents>
getConvDeconvOpOutputShape(ConvDeconvPoolTensorShapeComponents &inpShapeComp,
                           ConvDeconvKernelShapeComponents &kernelShapeComp,
                           ConvDeconvPoolLayerComponents &layerComp,
                           bool isConv = true);

/// Computes the output shape of pooling op based on
/// `ConvDeconvPoolTensorShapeComponents`, and `ConvDeconvPoolLayerComponents`
/// and return as `ConvDeconvPoolTensorShapeComponents`.
/// Shorthands:
/// O_{j} = jth output dimension
/// I_{j} = jth input dimension
/// A_{i} = prepadding added to the ith spatial dimension. The default is 0.
/// B_{i} = postpadding added to the ith spatial dimension . The default is 0.
/// M_{i} = I_{j} + A_{i} + B_{i}
/// W_{i} = ith spatial dimension of pooling window
/// S_{i} = stride value for the ith spatial dimension. The default is 1.
/// N = batch size
/// C_in = input channels
///
/// First two dimensions of output are [N, C_in]
/// Output dimension `O_{j}`, for each remaining spatial dimension `i` in the
/// range `[0, num_spatial_dims]` with `j` = `i` + 2 is computed based on layer
/// attributes as follows,
///
/// A. Padding mode == EXPLICIT_ROUND_DOWN
/// O_{j} = floor((M_{i} - W_{i}) / S_{i}) + 1
/// B. Padding mode == EXPLICIT_ROUND_UP
/// O_{j} = ceil((M_{i} - W_{i}) / S_{i}) + 1
///
/// TODO: Add support for more padding modes.
FailureOr<ConvDeconvPoolTensorShapeComponents>
getPoolingOpOutputShape(ConvDeconvPoolTensorShapeComponents &inpShapeComp,
                        ConvDeconvPoolLayerComponents &layerComp);

//===----------------------------------------------------------------------===//
// Common Shape Arithmetic Operations
//===----------------------------------------------------------------------===//

/// Divide `lhs` by `rhs` in an element-wise manner.
SmallVector<int64_t> shapeDivide(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs);

/// Divide each element of `lhs` by the `rhs`.
SmallVector<int64_t> shapeDivide(ArrayRef<int64_t> lhs, int64_t rhs);

/// Compute the elementwise max of `lhs` and `rhs`.
SmallVector<int64_t> shapeMax(ArrayRef<int64_t> lhs, int64_t rhs);

/// Compute the elementwise product of `lhs` and `rhs`.
SmallVector<int64_t> shapeMultiply(ArrayRef<int64_t> lhs,
                                   ArrayRef<int64_t> rhs);

/// Compute the number of elements in `input` shape.
int64_t shapeVolume(ArrayRef<int64_t> input);

//===----------------------------------------------------------------------===//
// AffineExpr utils
//===----------------------------------------------------------------------===//

/// Return the AffineExprs representing the elementwise sum of `v1` and `v2`.
SmallVector<AffineExpr> computeElementwiseAdd(ArrayRef<AffineExpr> v1,
                                              ArrayRef<AffineExpr> v2);

/// Return the AffineExprs representing the elementwise sum of `v1` and `v2`.
SmallVector<AffineExpr> computeElementwiseAdd(ArrayRef<AffineExpr> v1,
                                              ArrayRef<int64_t> v2);

/// Return the AffineExprs representing the elementwise product of `v1` and
/// `v2`.
SmallVector<AffineExpr> computeElementwiseMul(ArrayRef<AffineExpr> v1,
                                              ArrayRef<int64_t> v2);

} // namespace tensorrt

//===----------------------------------------------------------------------===//
// Tiling utilities
//===----------------------------------------------------------------------===//

/// Compute tile shape when distributing tile volume (which could be the number
/// of processors) to `outerShape` in the order given by `tileOrder`.
/// Example: ((16, 1024), (0, 1), 256) -> (16, 16)
FailureOr<SmallVector<int64_t>>
computeTileShape(ArrayRef<int64_t> outerShape, int64_t tileVolume,
                 std::optional<ArrayRef<int64_t>> tileOrder = {});

} // namespace mlir

#endif // MLIR_TENSORRT_UTILS_SHAPEUTILS
