//===- ShapeInfo.h ---------------------------------------------*- C++ -*-===//
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
/// Declarations for callback types are used to abstract away how to infer
/// shape knowledge from a pass or transformation. For example, a pass operating
/// on StableHlo IR may need to check whether the *values* of tensor A represent
/// the actual *shape* of tensor B, whose shape may not be known statically at
/// compile time.
///
/// The specific mechanism that one may use to determine the validity of a
/// specific proposition like the example above (which must be reported as
/// "unknown", "true", or "false") may depend on the context. In the case
/// of the StableHlo example above, we could try to naively pattern match
/// whether tensor A is the result of `stablehlo.concat` of appropriate
/// `stablehlo.get_dimensions_size %A, dim = ...` results. In other cases,
/// we may have access to an analysis that assists with more robustly
/// checking the proposition.
///
/// This file just contains callback types that a Pass or rewrite/transform can
/// accept as a parameter, allowing the creator or caller to hand in a
/// particular implementation.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_UTILS_SHAPEINFO
#define MLIR_TENSORRT_UTILS_SHAPEINFO

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace mlir {

/// TensorElementValue identifies a particular scalar element value of a
/// statically-shaped tensor.
struct TensorElementValue {
  TensorElementValue(Value value, ArrayRef<int64_t> coord);

  TypedValue<RankedTensorType> getTensor() const { return tensor; }
  int64_t getLinearIndex() const { return linearIndex; }

  /// A value of type (must be statically-shaped) RankedTensorType.
  TypedValue<RankedTensorType> tensor;

  /// The linear coordinate of the value.
  int64_t linearIndex;
};

/// TensorShapeDimExtent identifies a (potentially dynamically shaped) size
/// of a particular dimension of a tensor's shape.
struct TensorShapeDimExtent {
  TensorShapeDimExtent(Value value, int64_t dim);

  std::optional<int64_t> getConstantSize() const;

  /// A value of type  RankedTensorType.
  TypedValue<RankedTensorType> tensor;

  /// The dimension.
  int64_t dim;
};

struct ShapeInfoCallbacks {
  // Check whether 'tensorElementValue' is provably equivalent to
  // `tensorShapeDimExtent`. Returning 'nullopt' means "unknown", true means
  // "equal", false means "not equal".
  std::function<std::optional<bool>(TensorElementValue tensorElementValue,
                                    TensorShapeDimExtent tensorShapeDimExtent)>
      isElementValueEqualToShapeDimExtent;

  // Check whether 'tensorElementValue' is provably equivalent to the given
  // static value. Returning 'nullopt' means "unknown", true means "equal",
  // false means "not equal".
  std::function<std::optional<bool>(TensorElementValue tensorElementValue,
                                    Attribute constantValue)>
      isElementValueEqualToConstant;
};

} // namespace mlir

#endif // MLIR_TENSORRT_UTILS_SHAPEINFO
