//===- Utils.h ---------------------------------------------------*- C++-*-===//
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
#ifndef MLIR_TENSORRT_DIALECT_TENSORRT_UTILS_UTILS
#define MLIR_TENSORRT_DIALECT_TENSORRT_UTILS_UTILS

#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"

namespace mlir {
class FunctionOpInterface;
namespace tensorrt {

/// Return true if the func argument at `argIndex` has a valid arg attribute of
/// type ShapeProfileAttr representing bounds on the tensor's shape.
bool hasArgumentShapeProfile(FunctionOpInterface op, unsigned argIndex);

/// Return true if the func argument at `argIndex` has a valid arg attribute
/// representing the bounds of a host tensor values.
bool hasHostTensorValueBounds(FunctionOpInterface, unsigned argIndex);

/// Retrieve the ShapeProfileAttr attribute `tensorrt.shape_profile` or return
/// failure.
FailureOr<ShapeProfileAttr> getArgumentShapeProfile(FunctionOpInterface op,
                                                    unsigned argIndex);

/// Retrieve the arg attribute under `tensorrt.value_bounds` or return
/// failure.
FailureOr<ShapeProfileAttr> getArgumentValueBounds(FunctionOpInterface op,
                                                   unsigned argIndex);

/// Given a function-like operation and an argument index, retrieve the arg
/// attribute (`tensorrt.shape_profile`) that describes the dynamic shape range
/// and return a ShapeProfileAttr object. If the argument type has a static
/// shape, then it returns a new ShapeProfileAttr describing the static shape.
/// If the argument is only unknown in the first dimension and `batchSizeRange`
/// is provided, then the profile will be generated assuming `batchSizeRange`
/// describes the first dimension bounds. This function does **not** update the
/// shape profile arg attribute of `op`.
FailureOr<ShapeProfileAttr>
inferArgShapeProfile(FunctionOpInterface op, unsigned argIndex,
                     std::optional<DynamicDimensionBounds> batchSizeRange);

/// Create a constant shape tensor (`tensor<rank x i32>` filled with given
/// values).
TypedValue<RankedTensorType>
createConstShapeTensor(RewriterBase &b, Location loc, ArrayRef<int32_t> values);

/// Create a shape tensor from a known constant 'baseShape' by inserting
/// `update` (of scalar i32 tensor type) into the `scatterDim` position. This is
/// accomplished using concatenation.
TypedValue<RankedTensorType>
scatterShapeTensor(RewriterBase &b, Location loc, ArrayRef<int64_t> baseShape,
                   int32_t scatterDim, TypedValue<RankedTensorType> update);

} // namespace tensorrt
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_TENSORRT_UTILS_UTILS
