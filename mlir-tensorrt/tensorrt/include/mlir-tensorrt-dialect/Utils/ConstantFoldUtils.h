//===- ConstantFoldUtils.h --------------------------------------*- C++ -*-===//
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
/// Utilities to assist with common constant-folding operations. Each of these
/// utilities should gracefully handle the `dense_resource<__elided__>` case
/// to simulate folding if possible. These functions also don't do any size
/// checks to rule out cases that are too costly. That is the responsibility
/// of the caller.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_UTILS_CONSTANTFOLDUTILS_H
#define MLIR_TENSORRT_UTILS_CONSTANTFOLDUTILS_H

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {

/// Transpose elements `attr` from input type to `outputType` using the
/// specified permutation.
ElementsAttr constantFoldTranspose(ElementsAttr attr, AffineMap permutation);

/// Fold reshape.
ElementsAttr constantFoldReshape(ShapedType newType, ElementsAttr attr);

/// Fold element type conversions (e.g. elementwise
/// `arith.extf|truncf|sitofp|fptosi`).
ElementsAttr constantFoldConvert(Type newElementType, ElementsAttr attr);

/// Fold slice operations parameterized by constant offset, limit, and stride.
ElementsAttr constantFoldSliceOffsetLimitStride(ElementsAttr attr,
                                                RankedTensorType outputType,
                                                ArrayRef<int64_t> offsets,
                                                ArrayRef<int64_t> limits,
                                                ArrayRef<int64_t> strides);

} // namespace mlir

#endif // MLIR_TENSORRT_UTILS_CONSTANTFOLDUTILS_H
