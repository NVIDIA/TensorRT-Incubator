//===- Attributes.cpp -----------------------------------------------------===//
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
/// Definitions of helper functions, used to write python or other language
/// bindings for MLIR TensorRT attributes.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect-c/TensorRTAttributes.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include <optional>

#define DEFINE_ATTR_GETTER_FROM_STRING(attrName)                               \
  MlirAttribute tensorrt##attrName##AttrGet(MlirContext ctx,                   \
                                            MlirStringRef value) {             \
    std::optional<mlir::tensorrt::attrName> var##attrName =                    \
        mlir::tensorrt::symbolize##attrName(unwrap(value));                    \
    if (!var##attrName)                                                        \
      llvm_unreachable("Invalid string value for the attribute.");             \
    return wrap(mlir::tensorrt::attrName##Attr::get(unwrap(ctx),               \
                                                    var##attrName.value()));   \
  }

#define DEFINE_IS_ATTR(attrName)                                               \
  bool tensorrtIs##attrName##Attr(MlirAttribute attr) {                        \
    return llvm::isa<mlir::tensorrt::attrName##Attr>(unwrap(attr));            \
  }

#define DEFINE_STRING_GETTER_FROM_ATTR(attrName)                               \
  MlirStringRef tensorrt##attrName##AttrGetValue(MlirAttribute attr) {         \
    return wrap(mlir::tensorrt::stringify##attrName(                           \
        llvm::cast<mlir::tensorrt::attrName##Attr>(unwrap(attr)).getValue())); \
  }

DEFINE_ATTR_GETTER_FROM_STRING(ActivationType)
DEFINE_IS_ATTR(ActivationType)
DEFINE_STRING_GETTER_FROM_ATTR(ActivationType)

DEFINE_ATTR_GETTER_FROM_STRING(PaddingMode)
DEFINE_IS_ATTR(PaddingMode)
DEFINE_STRING_GETTER_FROM_ATTR(PaddingMode)

DEFINE_ATTR_GETTER_FROM_STRING(PoolingType)
DEFINE_IS_ATTR(PoolingType)
DEFINE_STRING_GETTER_FROM_ATTR(PoolingType)

DEFINE_ATTR_GETTER_FROM_STRING(ElementWiseOperation)
DEFINE_IS_ATTR(ElementWiseOperation)
DEFINE_STRING_GETTER_FROM_ATTR(ElementWiseOperation)

DEFINE_ATTR_GETTER_FROM_STRING(GatherMode)
DEFINE_IS_ATTR(GatherMode)
DEFINE_STRING_GETTER_FROM_ATTR(GatherMode)

DEFINE_ATTR_GETTER_FROM_STRING(UnaryOperation)
DEFINE_IS_ATTR(UnaryOperation)
DEFINE_STRING_GETTER_FROM_ATTR(UnaryOperation)

DEFINE_ATTR_GETTER_FROM_STRING(ReduceOperation)
DEFINE_IS_ATTR(ReduceOperation)
DEFINE_STRING_GETTER_FROM_ATTR(ReduceOperation)

DEFINE_ATTR_GETTER_FROM_STRING(SliceMode)
DEFINE_IS_ATTR(SliceMode)
DEFINE_STRING_GETTER_FROM_ATTR(SliceMode)

DEFINE_ATTR_GETTER_FROM_STRING(TopKOperation)
DEFINE_IS_ATTR(TopKOperation)
DEFINE_STRING_GETTER_FROM_ATTR(TopKOperation)

DEFINE_ATTR_GETTER_FROM_STRING(MatrixOperation)
DEFINE_IS_ATTR(MatrixOperation)
DEFINE_STRING_GETTER_FROM_ATTR(MatrixOperation)

DEFINE_ATTR_GETTER_FROM_STRING(ResizeMode)
DEFINE_IS_ATTR(ResizeMode)
DEFINE_STRING_GETTER_FROM_ATTR(ResizeMode)

DEFINE_ATTR_GETTER_FROM_STRING(ResizeCoordinateTransformation)
DEFINE_IS_ATTR(ResizeCoordinateTransformation)
DEFINE_STRING_GETTER_FROM_ATTR(ResizeCoordinateTransformation)

DEFINE_ATTR_GETTER_FROM_STRING(ResizeSelector)
DEFINE_IS_ATTR(ResizeSelector)
DEFINE_STRING_GETTER_FROM_ATTR(ResizeSelector)

DEFINE_ATTR_GETTER_FROM_STRING(ResizeRoundMode)
DEFINE_IS_ATTR(ResizeRoundMode)
DEFINE_STRING_GETTER_FROM_ATTR(ResizeRoundMode)

DEFINE_ATTR_GETTER_FROM_STRING(LoopOutput)
DEFINE_IS_ATTR(LoopOutput)
DEFINE_STRING_GETTER_FROM_ATTR(LoopOutput)

DEFINE_ATTR_GETTER_FROM_STRING(TripLimit)
DEFINE_IS_ATTR(TripLimit)
DEFINE_STRING_GETTER_FROM_ATTR(TripLimit)

DEFINE_ATTR_GETTER_FROM_STRING(FillOperation)
DEFINE_IS_ATTR(FillOperation)
DEFINE_STRING_GETTER_FROM_ATTR(FillOperation)

DEFINE_ATTR_GETTER_FROM_STRING(ScatterMode)
DEFINE_IS_ATTR(ScatterMode)
DEFINE_STRING_GETTER_FROM_ATTR(ScatterMode)
