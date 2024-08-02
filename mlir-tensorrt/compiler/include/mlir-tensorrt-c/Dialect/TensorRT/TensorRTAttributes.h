//===- Attributes.h -----------------------------------------------*- C -*-===//
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
/// Declaration of helper functions, used to write python bindings for MLIR
/// TensorRT attributes.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_C_DIALECT_TENSORRT_TENSORRTATTRIBUTES
#define MLIR_TENSORRT_C_DIALECT_TENSORRT_TENSORRTATTRIBUTES

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO: Add a generator for this in `MlirTensorRTTablegen.cpp` tool.
#define DECLARE_ATTR_GETTER_FROM_STRING(attrName)                              \
  MLIR_CAPI_EXPORTED MlirAttribute tensorrt##attrName##AttrGet(                \
      MlirContext ctx, MlirStringRef value);

#define DECLARE_IS_ATTR(attrName)                                              \
  MLIR_CAPI_EXPORTED bool tensorrtIs##attrName##Attr(MlirAttribute attr);

#define DECLARE_STRING_GETTER_FROM_ATTR(attrName)                              \
  MLIR_CAPI_EXPORTED MlirStringRef tensorrt##attrName##AttrGetValue(           \
      MlirAttribute attr);

//===----------------------------------------------------------------------===//
// ActivationType
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(ActivationType)
DECLARE_IS_ATTR(ActivationType)
DECLARE_STRING_GETTER_FROM_ATTR(ActivationType)

//===----------------------------------------------------------------------===//
// PaddingMode
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(PaddingMode)
DECLARE_IS_ATTR(PaddingMode)
DECLARE_STRING_GETTER_FROM_ATTR(PaddingMode)

//===----------------------------------------------------------------------===//
// PoolingType
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(PoolingType)
DECLARE_IS_ATTR(PoolingType)
DECLARE_STRING_GETTER_FROM_ATTR(PoolingType)

//===----------------------------------------------------------------------===//
// ElementWiseOperation
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(ElementWiseOperation)
DECLARE_IS_ATTR(ElementWiseOperation)
DECLARE_STRING_GETTER_FROM_ATTR(ElementWiseOperation)

//===----------------------------------------------------------------------===//
// GatherMode
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(GatherMode)
DECLARE_IS_ATTR(GatherMode)
DECLARE_STRING_GETTER_FROM_ATTR(GatherMode)

//===----------------------------------------------------------------------===//
// UnaryOperation
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(UnaryOperation)
DECLARE_IS_ATTR(UnaryOperation)
DECLARE_STRING_GETTER_FROM_ATTR(UnaryOperation)

//===----------------------------------------------------------------------===//
// ReduceOperation
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(ReduceOperation)
DECLARE_IS_ATTR(ReduceOperation)
DECLARE_STRING_GETTER_FROM_ATTR(ReduceOperation)

//===----------------------------------------------------------------------===//
// SliceMode
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(SliceMode)
DECLARE_IS_ATTR(SliceMode)
DECLARE_STRING_GETTER_FROM_ATTR(SliceMode)

//===----------------------------------------------------------------------===//
// TopKOperation
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(TopKOperation)
DECLARE_IS_ATTR(TopKOperation)
DECLARE_STRING_GETTER_FROM_ATTR(TopKOperation)

//===----------------------------------------------------------------------===//
// MatrixOperation
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(MatrixOperation)
DECLARE_IS_ATTR(MatrixOperation)
DECLARE_STRING_GETTER_FROM_ATTR(MatrixOperation)

//===----------------------------------------------------------------------===//
// ResizeMode
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(ResizeMode)
DECLARE_IS_ATTR(ResizeMode)
DECLARE_STRING_GETTER_FROM_ATTR(ResizeMode)

//===----------------------------------------------------------------------===//
// ResizeCoordinateTransformation
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(ResizeCoordinateTransformation)
DECLARE_IS_ATTR(ResizeCoordinateTransformation)
DECLARE_STRING_GETTER_FROM_ATTR(ResizeCoordinateTransformation)

//===----------------------------------------------------------------------===//
// ResizeSelector
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(ResizeSelector)
DECLARE_IS_ATTR(ResizeSelector)
DECLARE_STRING_GETTER_FROM_ATTR(ResizeSelector)

//===----------------------------------------------------------------------===//
// ResizeRoundMode
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(ResizeRoundMode)
DECLARE_IS_ATTR(ResizeRoundMode)
DECLARE_STRING_GETTER_FROM_ATTR(ResizeRoundMode)

//===----------------------------------------------------------------------===//
// LoopOutput
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(LoopOutput)
DECLARE_IS_ATTR(LoopOutput)
DECLARE_STRING_GETTER_FROM_ATTR(LoopOutput)

//===----------------------------------------------------------------------===//
// TripLimit
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(TripLimit)
DECLARE_IS_ATTR(TripLimit)
DECLARE_STRING_GETTER_FROM_ATTR(TripLimit)

//===----------------------------------------------------------------------===//
// FillOperation
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(FillOperation)
DECLARE_IS_ATTR(FillOperation)
DECLARE_STRING_GETTER_FROM_ATTR(FillOperation)

//===----------------------------------------------------------------------===//
// ScatterMode
//===----------------------------------------------------------------------===//

DECLARE_ATTR_GETTER_FROM_STRING(ScatterMode)
DECLARE_IS_ATTR(ScatterMode)
DECLARE_STRING_GETTER_FROM_ATTR(ScatterMode)

#ifdef __cplusplus
}
#endif
#endif // MLIR_TENSORRT_C_DIALECT_TENSORRT_TENSORRTATTRIBUTES
