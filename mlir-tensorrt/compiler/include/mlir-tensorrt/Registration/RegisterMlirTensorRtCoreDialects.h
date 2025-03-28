//===- RegisterMlirTensorRtCoreDialects.h -----------------------*- C++ -*-===//
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
/// Registration methods for the core dialects defined by this project.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Target/TensorRTEncodingImpl.h"
#include "mlir-tensorrt/Backends/Host/HostBackend.h"
#include "mlir-tensorrt/Backends/TensorRT/TensorRTBackend.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#ifdef MLIR_TRT_ENABLE_HLO
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#endif // MLIR_TRT_ENABLE_HLO
#ifdef MLIR_TRT_ENABLE_EXECUTOR
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/CUDA/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#endif // MLIR_TRT_ENABLE_EXECUTOR

namespace mlir {
class DialectRegistry;

/// Register core MLIR-TensorRT project dialects (dialects defined by this
/// project and any of their immediate dependencies.
inline void registerCoreMlirTensorRtDialects(DialectRegistry &registry) {
  registry.insert<tensorrt::TensorRTDialect, func::FuncDialect>();
  tensorrt::registerTensorRTEncodingOpInterfaceExternalModels(registry);
  tensorrt::registerTensorKindOpInterfaceExternalModels(registry);
  func::registerInlinerExtension(registry);

#ifdef MLIR_TRT_ENABLE_EXECUTOR
  registry.insert<executor::ExecutorDialect, cuda::CUDADialect,
                  trtrt::TensorRTRuntimeDialect, DLTIDialect>();
#endif // MLIR_TRT_ENABLE_EXECUTOR

#ifdef MLIR_TRT_ENABLE_HLO
  registry.insert<plan::PlanDialect>();
  mlir::plan::registerHostBackend(registry);
  mlir::plan::registerTensorRTBackend(registry);
#endif // MLIR_TRT_ENABLE_HLO
}

inline void
registerMlirTensorRtBufferizationInterfaces(DialectRegistry &registry) {
#ifdef MLIR_TRT_ENABLE_EXECUTOR
  trtrt::registerBufferizableOpInterfaceExternalModels(registry);
  cuda::registerBufferizableOpInterfaceExternalModels(registry);
#endif // MLIR_TRT_ENABLE_EXECUTOR
}

inline void registerMlirTensorRtTransformExtensions(DialectRegistry &registry) {
}

} // namespace mlir
