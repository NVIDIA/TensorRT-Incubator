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
#include "mlir-tensorrt/Compiler/TensorRTExtension/TensorRTExtension.h"
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlirtrt::compiler;
using namespace mlirtrt;
using namespace mlir;

#define DEBUG_TYPE "tensorrt-extension"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

//===----------------------------------------------------------------------===//
// TensorRTCompilerExtension
//===----------------------------------------------------------------------===//
StableHLOToExecutableTensorRTExtension::
    StableHLOToExecutableTensorRTExtension() {}

void StableHLOToExecutableTensorRTExtension::populatePasses(
    mlir::OpPassManager &pm, Phase phase,
    const StableHLOToExecutableOptions &options) const {
  if (this->disabled)
    return;

  if (phase == Phase::PreClustering) {
    // We must materialize TRT plugin shape regions prior to clustering.
    pm.addNestedPass<func::FuncOp>(tensorrt::createInferPluginShapesPass());
    return;
  }

  if (phase == Phase::PostClustering) {
    pm.addNestedPass<tensorrt::TensorRTModuleOp>(
        mlir::createConvertStablehloToTensorRTPass());
    pm.addPass(createConvertTensorRTToTensorRTRuntimePass());
    return;
  }

  if (phase == Phase::PreBufferization) {
    // Simplify and translate functions nested in `tensorrt.module` ops.
    auto &trtPM = pm.nest<tensorrt::TensorRTModuleOp>();
    tensorrt::buildTensorRTModuleTransformationPipeline(
        trtPM, translationOptions.enableStronglyTyped);
    trtPM.addPass(tensorrt::createTranslateTensorRTPass(
        nullptr, options.layerMetadataCallback, translationOptions));
    return;
  }

  if (phase == Phase::PostBufferization) {
    return;
  }

  if (phase == Phase::ExecutorLowering) {
    ConvertTensorRTRuntimeToExecutorPassOptions toExecutorOpts;
    toExecutorOpts.indexBitwidth = options.get<ExecutorOptions>().indexBitwidth;
    toExecutorOpts.usePackedMemRefCConv =
        options.get<ExecutorOptions>().usePackedMemRefCConv;
    pm.addPass(createConvertTensorRTRuntimeToExecutorPass(toExecutorOpts));
    return;
  }
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(
    mlirtrt::compiler::StableHLOToExecutableTensorRTExtension)
