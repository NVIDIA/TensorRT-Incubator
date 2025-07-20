//===- TensorRTExtension.cpp ----------------------------------------------===//
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
/// TensorRT-specific compilation hooks.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/StablehloToExecutable/TensorRTExtension.h"
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlirtrt::compiler;
using namespace mlirtrt;
using namespace mlir;

//===----------------------------------------------------------------------===//
// TensorRTCompilerExtension
//===----------------------------------------------------------------------===//

void StablehloToExecutableTensorRTExtension::populatePasses(
    mlir::OpPassManager &pm, Phase phase,
    const StablehloToExecutableOptions &options) const {

  if (this->disabled)
    return;

  if (phase == Phase::PreClustering) {
    // We must materialize TRT plugion shape regions prior to clustering.
    pm.addNestedPass<func::FuncOp>(tensorrt::createInferPluginShapesPass());
    return;
  }

  if (phase == Phase::PostClustering) {
    ConvertStablehloToTensorRTPassOptions convertOpts{};
    convertOpts.preferEinsum = this->preferEinsum;
    pm.addNestedPass<tensorrt::TensorRTModuleOp>(
        mlir::createConvertStablehloToTensorRTPass(std::move(convertOpts)));
    return;
  }

  if (phase == Phase::PreBufferization) {

    tensorrt::TensorRTTranslationOptions translationOpts;
    if (useGlobalCLFlags) {
      translationOpts = tensorrt::TensorRTTranslationOptions::fromCLFlags();
    } else {
      translationOpts.saveTensorRTEnginesToDirectory =
          saveTensorRTEnginesToDirectory;
      translationOpts.saveTensorRTLayerInfoDirectory =
          saveTensorRTLayerInfoDirectory;
      translationOpts.tensorrtBuilderOptLevel = tensorrtBuilderOptLevel;
      translationOpts.enableStronglyTyped = enableStronglyTyped;
      translationOpts.timingCachePath = timingCachePath;
      translationOpts.workspaceMemoryPoolLimit = workspaceMemoryPoolLimit;
    }

    // Simplify and translate functions nested in `tensorrt.module` ops.
    auto &trtPM = pm.nest<tensorrt::TensorRTModuleOp>();

    tensorrt::ApplyWorkaroundsPassOptions trtWAROptions = {};
    trtWAROptions.tensorrtStronglyTyped = translationOpts.enableStronglyTyped;
    trtWAROptions.forceDefaultSliceInBounds = this->forceDefaultSliceInBounds;
    tensorrt::buildTensorRTModuleTransformationPipeline(trtPM, trtWAROptions);

    trtPM.addPass(
        tensorrt::createTranslateTensorRTPass(nullptr, translationOpts));

    pm.addPass(createConvertTensorRTToTensorRTRuntimePass());
    return;
  }

  if (phase == Phase::PostBufferization) {
    return;
  }

  if (phase == Phase::ExecutorLowering) {
    if (options.hostTarget == HostTarget::Executor) {
      ConvertTensorRTRuntimeToExecutorPassOptions toExecutorOpts;
      toExecutorOpts.indexBitwidth =
          options.get<ExecutorOptions>().indexBitwidth;
      toExecutorOpts.usePackedMemRefCConv =
          options.get<ExecutorOptions>().usePackedMemRefCConv;
      pm.addPass(createConvertTensorRTRuntimeToExecutorPass(
          std::move(toExecutorOpts)));
      return;
    }
    if (options.hostTarget == HostTarget::LLVM) {
      ConvertTensorRTRuntimeToLLVMPassOptions toLLVMOpts;
      toLLVMOpts.artifactsDirectory = options.artifactsDirectory;
      pm.addPass(createConvertTensorRTRuntimeToLLVMPass(std::move(toLLVMOpts)));
      return;
    }
  }
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(
    mlirtrt::compiler::StablehloToExecutableTensorRTExtension)
