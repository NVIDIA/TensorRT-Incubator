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
#ifdef MLIR_TRT_TARGET_TENSORRT

#include "mlir-tensorrt/Compiler/TensorRTToExecutable/TensorRTToExecutable.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "mlir-tensorrt/Compiler/OptionsRegistry.h"
#include "mlir-tensorrt/Compiler/TensorRTToExecutable/Passes.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlirtrt::compiler;

//===----------------------------------------------------------------------===//
// TensorRTToExecutableOptions
//===----------------------------------------------------------------------===//

TensorRTToExecutableOptions::TensorRTToExecutableOptions(
    TaskExtensionRegistry extensions) {
  // TODO (pranavm): We don't need extensions - remove from constructor and add
  // `setExtensions` to base class.
  assert(extensions.extensions.size() == 0);
}

//===----------------------------------------------------------------------===//
// TensorRTToExecutableTask
//===----------------------------------------------------------------------===//

TensorRTToExecutableTask::TensorRTToExecutableTask(
    MLIRContext *ctx, const TensorRTToExecutableOptions &options)
    : CompilationTask(ctx, options) {
  options.get<DebugOptions>().applyToPassManager(*this);
}

void TensorRTToExecutableTask::buildTensorRTClusteringPipeline(
    OpPassManager &pm, const TensorRTToExecutableOptions &opts) {
  pm.addPass(createOutlineTensorRTOpPass());
}

void TensorRTToExecutableTask::buildPostClusteringPipeline(
    OpPassManager &pm, const TensorRTToExecutableOptions &options) {
  // Simplify and translate functions nested in `tensorrt.module` ops.
  auto &trtPM = pm.nest<tensorrt::TensorRTModuleOp>();
  tensorrt::buildTensorRTModuleTransformationPipeline(
      trtPM, options.get<TensorRTOptions>().options.enableStronglyTyped);
  trtPM.addPass(tensorrt::createTranslateTensorRTPass(
      nullptr, options.get<TensorRTOptions>().options));

  pm.addPass(createConvertTensorRTToTensorRTRuntimePass());

  pm.addNestedPass<func::FuncOp>(plan::createPostClusteringValidationPass());

  pm.addPass(createCanonicalizerPass());

  pm.addPass(createInlinerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // We then perform some final simplification on the top-level func.func ops
  // (e.g. public entrypoint functions).
  pm.addNestedPass<func::FuncOp>(createSCFDetensorizeLoopsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Pre-bufferization
  pm.addPass(createMemRefCastEliminationPass());
  plan::PlanAllocTensorsPassOptions allocTensorOpts{};
  allocTensorOpts.forceEntrypointsReturnAllocs =
      options.get<PlanAllocOptions>().forceEntrypointsReturnAllocs;
  pm.addPass(plan::createPlanAllocTensorsPass(allocTensorOpts));
  pm.addPass(plan::createPlanBufferizePass());
  pm.addPass(createMemRefCastEliminationPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  plan::buildPlanBufferOptimizationPipeline(pm);
  plan::buildPlanBufferDeallocationPipeline(
      pm, bufferization::DeallocationOptions{
              /*privateFuncDynamicOwnership=*/false});

  // Post-bufferization
  pm.addPass(createConvertMemRefToCUDAPass());
  pm.addPass(createConvertPlanToExecutorPass());
  pm.addPass(executor::createExecutorAllocsToGlobalsPass());
  pm.addNestedPass<func::FuncOp>(
      executor::createExecutorPopulateFunctionMetadataPass());

  // Executor lowering
  ConvertTensorRTRuntimeToExecutorPassOptions toExecutorOpts;
  toExecutorOpts.indexBitwidth = options.get<ExecutorOptions>().indexBitwidth;
  toExecutorOpts.usePackedMemRefCConv =
      options.get<ExecutorOptions>().usePackedMemRefCConv;
  pm.addPass(createConvertTensorRTRuntimeToExecutorPass(toExecutorOpts));

  ConvertCUDAToExecutorPassOptions cudaToExecutorOpts;
  cudaToExecutorOpts.indexBitwidth =
      options.get<ExecutorOptions>().indexBitwidth;
  cudaToExecutorOpts.usePackedMemRefCConv =
      options.get<ExecutorOptions>().usePackedMemRefCConv;
  pm.addPass(createConvertCUDAToExecutorPass(cudaToExecutorOpts));

  pm.addPass(createDropNestedModulesPass());
}

void TensorRTToExecutableTask::populatePassManager(
    mlir::PassManager &pm, const TensorRTToExecutableOptions &options) {
  buildTensorRTClusteringPipeline(pm, options);

  buildPostClusteringPipeline(pm, options);

  mlir::executor::ConvertStdToExecutorPassOptions stdToExecOpts;
  stdToExecOpts.indexBitwidth = options.get<ExecutorOptions>().indexBitwidth;
  stdToExecOpts.usePackedMemRefCConv = true;
  mlir::executor::buildExecutorLoweringPipeline(pm, stdToExecOpts);
}

void mlirtrt::compiler::registerTensorRTToExecutableTask() {
  registerCompilationTaskWithNoExtensions<TensorRTToExecutableTask,
                                          TensorRTToExecutableOptions>(
      "tensorrt-to-executable");
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlirtrt::compiler::TensorRTToExecutableTask)

#endif
