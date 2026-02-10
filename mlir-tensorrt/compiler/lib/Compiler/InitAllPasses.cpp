//===- InitAllPasses.cpp --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Registration for MLIR-TensorRT passes.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/InitAllPasses.h"

#include "mlir-executor/InitAllPasses.h"
#include "mlir-kernel/Conversion/Passes.h"
#include "mlir-kernel/Kernel/Pipelines/Pipelines.h"
#include "mlir-kernel/Kernel/TransformSchedules/Passes.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir-tensorrt-common/Conversion/Passes.h"
#include "mlir-tensorrt-dialect/Target/Passes.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt/Backends/Host/Passes.h"
#include "mlir-tensorrt/Backends/Kernel/Passes.h"
#include "mlir-tensorrt/Compiler/Extensions/KernelGenExtension.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Features.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"

#ifdef MLIR_TRT_ENABLE_HLO
#include "mlir-tensorrt/Compiler/InputPipelines/StablehloInputPipeline.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#endif // MLIR_TRT_ENABLE_HLO

#ifdef MLIR_TRT_ENABLE_TORCH
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#endif // MLIR_TRT_ENABLE_TORCH

#ifdef MLIR_TRT_ENABLE_SHARDY
#include "shardy/dialect/sdy/transforms/passes.h"
#endif // MLIR_TRT_ENABLE_SHARDY

void mtrt::compiler::registerAllPasses() {
  mlir::arith::registerArithPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::emitc::registerEmitCPasses();
  mlir::executor::registerAllPasses();
  mlir::kernel::registerKernelPasses();
  mlir::kernel::registerKernelPipelines();
  mlir::kernel::registerKernelTransformSchedulesPasses();
  mlir::cuda::registerCUDAPasses();
  mlir::plan::registerPlanDialectPipelines();
  mlir::plan::registerPlanPasses();
  mlir::registerConvertCUDAToExecutorPass();
  mlir::registerConvertPDLToPDLInterpPass();
  mlir::registerKernelConversionPasses();
  mlir::registerLinalgElementwiseOpFusionPass();
  mlir::registerLinalgFoldUnitExtentDimsPass();
  mlir::registerLinalgGeneralizeNamedOpsPass();
  mlir::registerLinalgSpecializeGenericOpsPass();
  mlir::registerLowerAffinePass();
  mlir::registerMLIRTensorRTCommonConversionPasses();
  mlir::registerMLIRTensorRTConversionPasses();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerTransformsPasses();
  mlir::tensorrt::registerTensorRTPasses();
  mtrt::compiler::registerHostBackendPasses();
  mtrt::compiler::registerKernelBackendPasses();
  mtrt::registerMLIRTensorRTGenericTransformsPasses();

  IF_MLIR_TRT_ENABLE_HLO({
    mtrt::compiler::registerStableHloInputPipelines();
    mtrt::compiler::registerKernelGenExtensionPipelines();
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
      { mlir::tensorrt::registerTensorRTTranslationPasses(); });
  IF_MLIR_TRT_ENABLE_SHARDY({ mlir::sdy::registerAllSdyPassesAndPipelines(); });
}
