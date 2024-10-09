//===- Passes.cpp
//----------------------------------------------------------===//
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
/// Definitions of Plan dialect pipelines.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::plan;

void plan::buildPlanSegmentationPipeline(
    OpPassManager &pm, const plan::StablehloClusteringPassOptions &opts) {
  pm.addNestedPass<func::FuncOp>(
      plan::createMaterializeShapeCalculationsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(plan::createPlanRefineTypesPass());
  if (!opts.disableCreateShapeFuncPass)
    pm.addPass(createPlanCreateShapeFuncsPass());
  pm.addNestedPass<func::FuncOp>(
      plan::createPlanPopulateFunctionBoundsAttributesPass());
  pm.addPass(plan::createStablehloClusteringPass(opts));
  pm.addPass(plan::createCreateClosedRegionsPass());
  pm.addPass(plan::createOutlineClustersPass());
  pm.addPass(mlir::createFuncExtDuplicateFunctionEliminationPass());
  pm.addPass(plan::createEliminateShapeOpsPass());
}

void plan::buildPlanBufferizationPipeline(OpPassManager &pm) {
  pm.addPass(createInlinerPass());
  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  pm.addPass(plan::createPlanAllocTensorsPass());
  pm.addPass(plan::createPlanBufferizePass());
  pm.addPass(mlir::createMemRefCastEliminationPass());
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
}

void plan::buildPlanBufferOptimizationPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferLoopHoistingPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferHoistingPass());
  bufferization::BufferDeallocationPipelineOptions opts;
  opts.privateFunctionDynamicOwnership = false;
  bufferization::buildBufferDeallocationPipeline(pm, opts);
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

namespace {
struct ClusteringPipelineCliOpts
    : public PassPipelineOptions<ClusteringPipelineCliOpts> {
  Option<std::string> entrypoint{*this, "entrypoint", llvm::cl::init(""),
                                 llvm::cl::desc("name of entrypoint function")};
  Option<bool> disallowHostTensorsInTensorRTClusters{
      *this, "disallow-host-tensors-in-tensorrt-clusters",
      llvm::cl::desc("don't allow host tensor inputs to tensorrt clusters"),
      llvm::cl::init(false)};
  Option<int64_t> trtMajorVersion{
      *this, "trt-major-version",
      llvm::cl::desc("target TensorRT version for segmentation pipeline"),
      llvm::cl::init(10)};
};
} // namespace

// Register pipelines.

void plan::registerPlanDialectPipelines() {
  PassPipelineRegistration<> executorBufferizationPipeline(
      "plan-bufferize-pipeline",
      "perform bufferization and standard pre/post processing passes",
      [](OpPassManager &pm) {
        buildPlanBufferizationPipeline(pm);
        buildPlanBufferOptimizationPipeline(pm);
      });

  PassPipelineRegistration<ClusteringPipelineCliOpts> segPipelineRegistration(
      "plan-segmentation-pipeline",
      "apply the Plan Dialect segmentation pipeline",
      [](OpPassManager &pm, const ClusteringPipelineCliOpts &opts) {
        StablehloClusteringPassOptions clusterOpts{};
        clusterOpts.trtMajorVersion = opts.trtMajorVersion;
        clusterOpts.disallowHostTensorsInTensorRTClusters =
            opts.disallowHostTensorsInTensorRTClusters;
        clusterOpts.entrypoint = opts.entrypoint;
        buildPlanSegmentationPipeline(pm, clusterOpts);
      });
}
