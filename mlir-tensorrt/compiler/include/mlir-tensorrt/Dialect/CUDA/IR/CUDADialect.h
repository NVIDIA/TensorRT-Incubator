//===- CUDADialect.h --------------------------------------------*- C++ -*-===//
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
/// CUDA dialect declarations
///
//===----------------------------------------------------------------------===//

#ifndef MLIR_TENSORRT_DIALECT_CUDA_IR_CUDA_H
#define MLIR_TENSORRT_DIALECT_CUDA_IR_CUDA_H

#include "mlir/Bytecode/BytecodeOpInterface.h"           // IWYU pragma: keep
#include "mlir/IR/Dialect.h"                             // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"                        // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"                    // IWYU pragma: keep
#include "mlir/Interfaces/DestinationStyleOpInterface.h" // IWYU pragma: keep
#include "mlir/Interfaces/FunctionInterfaces.h"          // IWYU pragma: keep
#include "mlir/Interfaces/InferTypeOpInterface.h"        // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h"        // IWYU pragma: keep

//===----------------------------------------------------------------------===//
// CUDA Dialect
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// CUDA Dialect Enums
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAEnums.h.inc"

//===----------------------------------------------------------------------===//
// CUDA Dialect Attributes
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAAttributes.h.inc"
#undef GET_ATTRDEF_CLASSES

//===----------------------------------------------------------------------===//
// CUDA Dialect Types
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// CUDA Op Interfaces
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// CUDA Dialect Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAOps.h.inc"

#endif // MLIR_TENSORRT_DIALECT_CUDA_IR_CUDA_H
