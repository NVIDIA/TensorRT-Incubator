//===- TensorRTToExecutable.cpp ---------------------------------*- C++ -*-===//
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
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/TensorRTToExecutable.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/Passes.h"
#include "mlir-tensorrt/Conversion/CUDAToExecutor/CUDAToExecutor.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mtrt::compiler;

static plan::PlanBufferizationOptions
convertBufferizationOptions(const TensorRTToExecutableOptions &pipelineOpts) {
  const BufferizationOptions &opts = pipelineOpts.get<BufferizationOptions>();
  plan::PlanBufferizationOptions bufferizationOpts{};
  bufferizationOpts.forceEntrypointsReturnAllocs =
      opts.forceEntrypointsReturnAllocs;
  bufferizationOpts.deallocationPrivateFuncDynamicOwnership =
      opts.deallocationPrivateFuncDynamicOwnership;
  bufferizationOpts.enableBufferLoopHoisting = opts.enableBufferLoopHoisting;
  bufferizationOpts.enableBufferHoisting = opts.enableBufferHoisting;
  bufferizationOpts.enablePinnedMemoryPromotion =
      opts.enablePinnedMemoryPromotion;
  return bufferizationOpts;
}

//===----------------------------------------------------------------------===//
// TensorRTToExecutableTask
//===----------------------------------------------------------------------===//

TensorRTToExecutableTask::TensorRTToExecutableTask(
    MLIRContext *ctx, std::unique_ptr<TensorRTToExecutableOptions> options)
    : Pipeline(ctx, std::move(options)) {}

void TensorRTToExecutableTask::populatePassManager() {
  PassManager &pm = *this;
  const TensorRTToExecutableOptions &options = getOptions();

  pm.addPass(plan::createVerifyInputAndAssignSlotsPass());
  pm.addPass(createOutlineTensorRTOpPass());

  // Simplify and translate functions nested in `tensorrt.module` ops.
  {
    auto &trtPM = pm.nest<tensorrt::TensorRTModuleOp>();

    tensorrt::ApplyWorkaroundsPassOptions bugWAROptions = {};
    bugWAROptions.tensorrtStronglyTyped =
        options.get<TensorRTOptions>().enableStronglyTyped;
    bugWAROptions.forceDefaultSliceInBounds =
        options.get<TensorRTOptions>().forceDefaultSliceInBounds;

    tensorrt::buildTensorRTModuleTransformationPipeline(trtPM, bugWAROptions);
    trtPM.addPass(tensorrt::createTranslateTensorRTPass(
        nullptr, options.get<TensorRTOptions>().options()));
  }

  pm.addPass(createConvertTensorRTToTensorRTRuntimePass());

  pm.addNestedPass<func::FuncOp>(plan::createPostClusteringValidationPass());

  pm.addPass(createCanonicalizerPass());

  pm.addPass(createInlinerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // We then perform some final simplification on the top-level func.func ops
  // (e.g. public entrypoint functions).
  pm.addNestedPass<func::FuncOp>(mtrt::createSCFDetensorizePass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Pre-bufferization
  plan::buildPlanBufferizationPipeline(pm,
                                       convertBufferizationOptions(options));

  // Post-bufferization
  pm.addPass(createConvertMemRefToCUDAPass());
  pm.addPass(createConvertPlanToExecutorPass());
  pm.addPass(executor::createExecutorAllocsToGlobalsPass());
  pm.addNestedPass<func::FuncOp>(
      executor::createExecutorPopulateFunctionMetadataPass());

  // Executor lowering
  ConvertTensorRTRuntimeToExecutorPassOptions toExecutorOpts;
  toExecutorOpts.indexBitwidth = options.get<ExecutorOptions>().indexBitwidth;
  pm.addPass(createConvertTensorRTRuntimeToExecutorPass(toExecutorOpts));

  ConvertCUDAToExecutorPassOptions cudaToExecutorOpts;
  cudaToExecutorOpts.indexBitwidth =
      options.get<ExecutorOptions>().indexBitwidth;
  pm.addPass(createConvertCUDAToExecutorPass(cudaToExecutorOpts));

  pm.addPass(createDropNestedModulesPass());

  mlir::executor::ConvertStdToExecutorPassOptions stdToExecOpts;
  stdToExecOpts.indexBitwidth = options.get<ExecutorOptions>().indexBitwidth;
  mlir::executor::buildExecutorLoweringPipeline(
      pm, stdToExecOpts, [](mlir::TypeConverter &typeConverter) {
        mlir::populateCUDAToExecutorTypeConversions(typeConverter);
      });
}

void mtrt::compiler::registerTensorRTToExecutableTask() {
  registerPipelineWithNoExtensions<TensorRTToExecutableTask,
                                   TensorRTToExecutableOptions>(
      "tensorrt-to-executable");
}
