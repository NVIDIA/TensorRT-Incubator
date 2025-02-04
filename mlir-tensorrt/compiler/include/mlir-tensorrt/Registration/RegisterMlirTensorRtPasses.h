//===- RegisterMlirTensorRtPasses.h -----------------------------*- C++ -*-===//
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
// Registration for mlir-tensorrt passes
//===----------------------------------------------------------------------===//
#ifndef REGISTRATION_REGISTERMLIRTENSORRTPASSES_H
#define REGISTRATION_REGISTERMLIRTENSORRTPASSES_H

#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#ifdef MLIR_TRT_ENABLE_HLO
#include "mlir-tensorrt/Compiler/StablehloToExecutable/Passes.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/StablehloToExecutable.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Pipelines/StableHloInputPipelines.h"
#include "stablehlo/transforms/Passes.h"
#endif // MLIR_TRT_ENABLE_HLO

#ifdef MLIR_TRT_ENABLE_EXECUTOR
#include "mlir-executor/InitAllPasses.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#endif // MLIR_TRT_ENABLE_EXECUTOR

namespace mlir {
namespace tensorrt {

/// Register passes declared within this repo.
inline void registerAllMlirTensorRtPasses() {
  registerMLIRTensorRTConversionPasses();
  registerTensorRTPasses();
  registerMLIRTensorRTGenericTransformsPasses();
  mlir::registerTransformsPasses();
  mlir::registerConvertPDLToPDLInterp();
  mlir::emitc::registerEmitCPasses();

#ifdef MLIR_TRT_ENABLE_HLO
  mlirtrt::compiler::registerStablehloToExecutablePasses();
  mlirtrt::compiler::registerStablehloToExecutablePipelines();
  registerStableHloInputPipelines();
  stablehlo_ext::registerStableHloExtPasses();
  stablehlo::registerPasses();
  plan::registerPlanPasses();
  plan::registerPlanDialectPipelines();
#endif // MLIR_TRT_ENABLE_HLO

#ifdef MLIR_TRT_ENABLE_EXECUTOR
  registerConvertCUDAToExecutorPass();
  bufferization::registerBufferizationPasses();
  executor::registerAllPasses();
#endif // MLIR_TRT_ENABLE_EXECUTOR
}

} // namespace tensorrt
} // namespace mlir

#endif // REGISTRATION_REGISTERMLIRTENSORRTPASSES_H
