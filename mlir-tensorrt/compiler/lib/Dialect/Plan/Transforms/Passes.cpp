//===- Passes.cpp --------------------------------------------------------===//
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
/// Definitions of Plan dialect pipelines.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::plan;

void plan::buildPlanSegmentationPipeline(
    OpPassManager &pm, const plan::ClusteringPassOptions &opts) {
  pm.addNestedPass<func::FuncOp>(
      plan::createMaterializeShapeCalculationsPass({opts.inputKind}));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(plan::createPlanRefineTypesPass({opts.inputKind}));
  pm.addPass(createCanonicalizerPass());
  if (!opts.disableCreateShapeFuncPass)
    pm.addPass(createPlanCreateShapeFuncsPass());
  pm.addNestedPass<func::FuncOp>(
      plan::createPlanPopulateFunctionBoundsAttributesPass());
  pm.addPass(plan::createClusteringPass(opts));
  plan::CreateClosedRegionsPassOptions closedRegionOptions{};
  closedRegionOptions.forceEntrypointsReturnAllocs =
      opts.forceEntrypointsReturnAllocs;
  closedRegionOptions.inputKind = opts.inputKind;
  pm.addPass(plan::createCreateClosedRegionsPass(closedRegionOptions));
  pm.addPass(plan::createOutlineClustersPass());
  pm.addPass(mlir::createFuncExtDuplicateFunctionEliminationPass());
  pm.addPass(plan::createEliminateShapeOpsPass());
}

static void buildPlanOneShotBufferizePipelinePipeline(
    OpPassManager &pm, const plan::PlanAllocTensorsPassOptions &opts) {
  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  pm.addPass(plan::createPlanAssignMemorySpacesPass());
  pm.addNestedPass<func::FuncOp>(plan::createPlanOptimizeMemorySpacesPass());
  pm.addNestedPass<func::FuncOp>(
      plan::createPlanPromoteHostTensorsToHostPinnedPass());
  pm.addNestedPass<func::FuncOp>(
      plan::createPlanMaterializeExplicitTransfersPass());
  pm.addPass(plan::createPlanAllocTensorsPass(opts));

  // Perform inlining after memory space selection and optimizations since we
  // rely on function-scoped attributes to determine the default bufferization
  // memory space.
  pm.addPass(createInlinerPass());

  pm.addPass(plan::createPlanModuleBufferizePass());
  pm.addPass(mlir::createMemRefCastEliminationPass());

  // We must canonicalize prior to `buffer-results-to-out-params` in order to
  // eliminate loop-carried arguments that bufferize in-place. Otherwise, we may
  // append too many output argumetns to functions that return loop results.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // If inlining does not occur (e.g. because a function is marked no_inline),
  // then we need to transform to DPS style and move allocation to the caller.
  // Otherwise the deallocation pipeline will produce very inefficient code; its
  // alias analysis is very conservative  and it can't see across calls
  // boundaries.
  pm.addPass(plan::createPlanBufferResultsToOutParamsPass(
      plan::PlanBufferResultsToOutParamsPassOptions{
          /*ignorePublicFunctions=*/opts.forceEntrypointsReturnAllocs}));

  /// TODO: Currently we must canonicalize prior to dropping buffer results
  /// since it helps to identify return values that are actually block
  /// arguments. Loop memref results that feed into returns, for example, are
  /// one such case where canonicalization will drop the loop memref results if
  /// it is loop invariant. This can result in establishing that the result
  /// value is actually a block argument. Ideally this sort of
  /// simplification/phase ordering should be eliminated.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(plan::createPlanRemoveEquivalentBufferResultsPass());
}

static void buildPlanBufferOptimizationPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferLoopHoistingPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferHoistingPass());
}

static void buildPlanBufferDeallocationPipeline(
    OpPassManager &pm, const bufferization::DeallocationOptions &options) {
  pm.addPass(memref::createExpandReallocPass(/*emitDeallocs=*/false));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(plan::createPlanOwnershipBasedBufferDeallocationPass(
      plan::PlanOwnershipBasedBufferDeallocationPassOptions{
          options.privateFuncDynamicOwnership}));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(bufferization::createLowerDeallocationsPass());
  pm.addNestedPass<func::FuncOp>(
      mlir::createConvertBufferizationToMemRefPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

namespace {
struct ClusteringPipelineCliOpts
    : public PassPipelineOptions<ClusteringPipelineCliOpts> {
  Option<std::string> entrypoint{*this, "entrypoint", llvm::cl::init(""),
                                 llvm::cl::desc("name of entrypoint function")};
  Option<bool> forceEntrypointsReturnAllocs{
      *this, "force-entrypoints-return-allocs", llvm::cl::init(false),
      llvm::cl::desc(
          "Require entrypoint functions to return allocations corresponding to"
          " the original tensor results, otherwise they are transformed"
          " into destination arguments whenever possible.")};
};

struct PlanBufferizationPipelineCliOpts
    : public PassPipelineOptions<PlanBufferizationPipelineCliOpts> {
  Option<bool> forceEntrypointsReturnAllocs{
      *this, "force-entrypoints-return-allocs", llvm::cl::init(false),
      llvm::cl::desc(
          "Require entrypoint functions to return allocations corresponding to"
          " the original tensor results, otherwise they are transformed"
          " into destination arguments whenever possible.")};
};

} // namespace

void plan::buildPlanBufferizationPipeline(
    OpPassManager &pm, const plan::PlanAllocTensorsPassOptions &options,
    const bufferization::DeallocationOptions &deallocationOptions) {
  buildPlanOneShotBufferizePipelinePipeline(pm, options);
  buildPlanBufferOptimizationPipeline(pm);
  buildPlanBufferDeallocationPipeline(pm, deallocationOptions);
}

// Register pipelines.
void plan::registerPlanDialectPipelines() {
  PassPipelineRegistration<PlanBufferizationPipelineCliOpts>
      executorBufferizationPipeline(
          "plan-bufferize-pipeline",
          "perform one-shot-bufferization, optimizations, and dallocation",
          [](OpPassManager &pm, const PlanBufferizationPipelineCliOpts &opts) {
            PlanAllocTensorsPassOptions allocTensorOpts{};
            allocTensorOpts.forceEntrypointsReturnAllocs =
                opts.forceEntrypointsReturnAllocs;
            buildPlanBufferizationPipeline(
                pm, allocTensorOpts, bufferization::DeallocationOptions{false});
          });

  PassPipelineRegistration<ClusteringPipelineCliOpts> segPipelineRegistration(
      "plan-segmentation-pipeline",
      "apply the Plan Dialect segmentation pipeline",
      [](OpPassManager &pm, const ClusteringPipelineCliOpts &opts) {
        ClusteringPassOptions clusterOpts{};
        clusterOpts.entrypoint = opts.entrypoint;
        clusterOpts.disableCreateShapeFuncPass =
            opts.forceEntrypointsReturnAllocs;
        buildPlanSegmentationPipeline(pm, clusterOpts);
      });
}
