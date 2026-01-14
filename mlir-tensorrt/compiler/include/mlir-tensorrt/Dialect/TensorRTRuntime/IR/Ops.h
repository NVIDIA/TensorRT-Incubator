//===- Ops.h -------------------------------------------------------------===//
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
///
/// Declarations for the TensorRTRuntime dialect operations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_OPS
#define MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_OPS

#include "mlir-tensorrt-common/Interfaces/StreamSchedulableOpInterface.h" // IWYU pragma: keep
#include "mlir-tensorrt-common/Interfaces/TensorKindOpInterface.h" // IWYU pragma: keep
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"   // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"                        // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"                    // IWYU pragma: keep
#include "mlir/IR/RegionKindInterface.h"                 // IWYU pragma: keep
#include "mlir/IR/SymbolTable.h"                         // IWYU pragma: keep
#include "mlir/Interfaces/DestinationStyleOpInterface.h" // IWYU pragma: keep
#include "mlir/Interfaces/InferTypeOpInterface.h"        // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h"        // IWYU pragma: keep

#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/Dialect.h" // IWYU pragma: keep
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/Types.h" // IWYU pragma: keep

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

#define GET_OP_CLASSES
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeOps.h.inc"

#endif // MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_OPS
