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

#include "mlir-executor/InitAllPasses.h"
#include "mlir-tensorrt-common/Conversion/Passes.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Features.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#ifdef MLIR_TRT_ENABLE_HLO
#include "mlir-tensorrt/Compiler/StablehloToExecutable/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#endif // MLIR_TRT_ENABLE_HLO

#ifdef MLIR_TRT_TARGET_TENSORRT
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/Passes.h"
#endif // MLIR_TRT_TARGET_TENSORRT

#ifdef MLIR_TRT_ENABLE_TORCH
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#endif // MLIR_TRT_ENABLE_TORCH

namespace mlirtrt::compiler {

/// Register passes declared within this repo.
inline void registerAllPasses() {
  mlir::emitc::registerEmitCPasses();
  mlir::plan::registerPlanDialectPipelines();
  mlir::plan::registerPlanPasses();
  mlir::registerLowerAffinePass();
  mlir::registerConvertPDLToPDLInterpPass();
  mlir::registerMLIRTensorRTConversionPasses();
  mlir::registerMLIRTensorRTGenericTransformsPasses();
  mlir::registerTransformsPasses();
  mlir::tensorrt::registerTensorRTPasses();
  mlir::registerConvertCUDAToExecutorPass();
  mlir::bufferization::registerBufferizationPasses();
  mlir::executor::registerAllPasses();
  mlir::registerMLIRTensorRTCommonConversionPasses();

  IF_MLIR_TRT_ENABLE_HLO({
    mlirtrt::compiler::registerStablehloToExecutablePasses();
    mlirtrt::compiler::registerStablehloToExecutablePipelines();
    mlirtrt::compiler::registerStableHloInputPipelines();
    mlir::stablehlo_ext::registerStableHloExtPasses();
    mlir::stablehlo::registerPasses();
    mlir::stablehlo::registerOptimizationPasses();
  });

  IF_MLIR_TRT_ENABLE_TORCH({
    mlir::torch::registerTorchPasses();
    mlir::torch::registerTorchConversionPasses();
    mlir::torch::registerConversionPasses();
    mlir::torch::TMTensor::registerPasses();
  });

  IF_MLIR_TRT_TARGET_TENSORRT(
      { mlirtrt::compiler::registerTensorRTToExecutablePipelines(); });
}

} // namespace mlirtrt::compiler

#endif // REGISTRATION_REGISTERMLIRTENSORRTPASSES_H
