//===- TensorRTExtension.cpp ----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-tensorrt/Compiler/Extensions/TensorRTExtension.h"
#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt/Compiler/Pipeline.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "tensorrt-extension"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE << "] "

using namespace mtrt::compiler;
using namespace mtrt;
using namespace mlir;

//===----------------------------------------------------------------------===//
// TensorRTCompilerExtension
//===----------------------------------------------------------------------===//

void TensorRTExtension::populatePasses(mlir::OpPassManager &pm,
                                       Phase phase) const {
  const MainOptions &options = this->getOptions();

  // Currently we only use the Linalg input kind for testing kernel generation
  // and other features. We implicitly disable TensorRT-related functionality in
  // order to avoid the overhead of creating a TRT builder instance in test
  // pipelines. In the future, we can remove this if we want to mix Linalg IR
  // and TensorRT IR at the input level.
  if (options.inputKind == plan::InputKind::Linalg ||
      options.disableTensorRTExtension || options.disableAllExtensions)
    return;

  // Clustering phases are only applicable to Stablehlo input.
  if (options.inputKind == plan::InputKind::Stablehlo) {
    if (phase == Phase::PreClustering) {
      // We must materialize TRT plugin shape regions prior to clustering.
      pm.addNestedPass<func::FuncOp>(tensorrt::createInferPluginShapesPass());
      return;
    }

    if (phase == Phase::PostClustering) {
      ConvertStablehloToTensorRTPassOptions convertOpts{};
      convertOpts.preferEinsum =
          options.get<TensorRTOptions>().tensorrtPreferEinsum;
      pm.addNestedPass<tensorrt::TensorRTModuleOp>(
          mlir::createConvertStablehloToTensorRTPass(std::move(convertOpts)));
      return;
    }
  }

  if (phase == Phase::PreBufferization) {

    // Simplify and translate functions nested in `tensorrt.module` ops.
    auto &trtPM = pm.nest<tensorrt::TensorRTModuleOp>();

    tensorrt::ApplyWorkaroundsPassOptions trtWAROptions = {};
    trtWAROptions.tensorrtStronglyTyped =
        options.get<TensorRTOptions>().getTranslationOptions().stronglyTyped;
    trtWAROptions.forceDefaultSliceInBounds =
        options.get<TensorRTOptions>().forceDefaultSliceInBounds;
    tensorrt::buildTensorRTModuleTransformationPipeline(trtPM, trtWAROptions);

    trtPM.addPass(tensorrt::createTranslateTensorRTPass(
        nullptr, options.get<TensorRTOptions>().getTranslationOptions()));

    pm.addPass(createConvertTensorRTToTensorRTRuntimePass());
    return;
  }

  if (phase == Phase::PostBufferization)
    return;

  if (phase == Phase::ExecutorLowering) {
    switch (options.hostTarget) {
    case HostTarget::Executor: {
      ConvertTensorRTRuntimeToExecutorPassOptions toExecutorOpts;
      toExecutorOpts.indexBitwidth =
          options.get<ExecutorOptions>().indexBitwidth;
      pm.addPass(createConvertTensorRTRuntimeToExecutorPass(
          std::move(toExecutorOpts)));
      return;
    }
    case HostTarget::LLVM: {
      pm.addPass(createConvertTensorRTRuntimeToLLVMPass());
      return;
    }
    case HostTarget::EmitC: {
      // No actions for EmitC since 'host-to-emitc' handles TensorRTRuntime
      // dialect ops directly.
      return;
    }
    }
  }
}

//===----------------------------------------------------------------------===//
// Extension Registration
//===----------------------------------------------------------------------===//

void mtrt::compiler::registerTensorRTExtension() {
  mtrt::compiler::registerExtension<TensorRTExtension>();
}
