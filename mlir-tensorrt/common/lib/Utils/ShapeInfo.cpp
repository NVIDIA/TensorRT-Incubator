//===- ShapeInfo.cpp -----------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-tensorrt-common/Utils/ShapeInfo.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

TensorElementValue::TensorElementValue(Value value, ArrayRef<int64_t> coord)
    : tensor(cast<TypedValue<RankedTensorType>>(value)),
      linearIndex(mlir::linearize(
          coord, mlir::computeSuffixProduct(tensor.getType().getShape()))) {}

TensorShapeDimExtent::TensorShapeDimExtent(Value value, int64_t dim)
    : tensor(cast<TypedValue<RankedTensorType>>(value)), dim(dim) {
  assert(dim > 0 && dim < tensor.getType().getRank() &&
         "dim must be > 0 and < tensor rank");
}

std::optional<int64_t> TensorShapeDimExtent::getConstantSize() const {
  if (tensor.getType().isDynamicDim(dim))
    return {};
  return tensor.getType().getDimSize(dim);
}
