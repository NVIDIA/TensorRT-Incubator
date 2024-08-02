//===- TensorRTRuntime.h ----------------------------------------*- C++ -*-===//
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
/// Declarations for the TensorRTRuntime dialect.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_TENSORRTRUNTIME_H
#define MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_TENSORRTRUNTIME_H

#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace func {
class FuncOp;
}

namespace trtrt {
template <typename ConcreteType>
class TensorRTRuntimeOpTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, TensorRTRuntimeOpTrait> {
};
} // namespace trtrt
} // namespace mlir

//===----------------------------------------------------------------------===//
// TensorRT Dialect Declaration
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeOpsDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeOpsTypes.h.inc"
#define GET_OP_CLASSES
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeOps.h.inc"

#endif // MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_TENSORRTRUNTIME_H
