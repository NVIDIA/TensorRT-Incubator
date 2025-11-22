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
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-tensorrt/Conversion/Passes.h"
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
    OpPassManager &pm, const plan::ClusteringPassOptions &opts,
    int abiVersion) {
  pm.addNestedPass<func::FuncOp>(
      plan::createMaterializeShapeCalculationsPass({opts.inputKind}));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(plan::createPlanRefineTypesPass({opts.inputKind}));
  pm.addPass(createCanonicalizerPass());
  if (!opts.disableCreateShapeFuncPass)
    pm.addPass(
        createPlanCreateShapeFuncsPass(plan::PlanCreateShapeFuncsPassOptions{
            /*forceUndefOutputArgs=*/opts.forceEntrypointsReturnAllocs,
            /*abiVersion=*/abiVersion,
        }));
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

static void
buildPlanOneShotBufferizePipeline(OpPassManager &pm,
                                  const plan::PlanBufferizationOptions &opts) {
  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  pm.addPass(plan::createPlanAssignMemorySpacesPass());
  pm.addNestedPass<func::FuncOp>(plan::createPlanOptimizeMemorySpacesPass());
  pm.addNestedPass<func::FuncOp>(
      plan::createPlanPromoteHostTensorsToHostPinnedPass());
  pm.addNestedPass<func::FuncOp>(
      plan::createPlanMaterializeExplicitTransfersPass());

  // Perform inlining after memory space selection and optimizations since we
  // rely on function-scoped attributes to determine the default bufferization
  // memory space.
  pm.addPass(createInlinerPass());

  plan::PlanAllocTensorsPassOptions allocOpts{};
  allocOpts.forceEntrypointsReturnAllocs = opts.forceEntrypointsReturnAllocs;
  pm.addPass(plan::createPlanAllocTensorsPass(allocOpts));

  pm.addPass(plan::createPlanModuleBufferizePass());
  pm.addNestedPass<func::FuncOp>(plan::createPlanConfirmArgumentDonationPass(
      plan::PlanConfirmArgumentDonationPassOptions{
          opts.failOnDonationArgumentRejection}));
  pm.addPass(mlir::createMemRefCastEliminationPass());

  // We must canonicalize prior to `plan-remove-equivalent-buffer-results`. This
  // helps to eliminate operations such as casts which would otherwise prevent
  // us from identifying that a returned value is exactly equivalent to an
  // argument.
  pm.addPass(createCanonicalizerPass());

  // 'plan-remove-equivalent-buffer-results': Drop function memref results that
  // are known to be exactly equivalent to function block arguments. This runs
  // on all functions, even in nested modules.
  pm.addPass(plan::createPlanRemoveEquivalentBufferResultsPass());

  // If functions are returning memref results that are not equivalent to
  // function arguments, then we rewrite the functions to use 'out' parameters
  // where possible. If we can prove that a function result is a new allocation,
  // then we also try to hoist that allocation to the caller (this depends on
  // also hoisting the size calculation, if present). If hoisting an allocation
  // is not possible, then we leave the returned value as-is. The buffer
  // deallocation pipeline will enforce conditions that the returned value is
  // a new allocation (e.g. by cloning the value if necessary) which does
  // not alias any other memref returned from the call or live-in to the
  // function. The caller can then assume ownership of any returned value.
  pm.addPass(plan::createPlanBufferResultsToOutParamsPass(
      plan::PlanBufferResultsToOutParamsPassOptions{
          /*ignorePublicFunctions=*/opts.forceEntrypointsReturnAllocs}));

  pm.addNestedPass<func::FuncOp>(createLowerLinalgCopiesPass());
}

static void buildPlanBufferOptimizationPipeline(
    OpPassManager &pm, const plan::PlanBufferizationOptions &options) {
  if (options.enableBufferLoopHoisting)
    pm.addNestedPass<func::FuncOp>(
        bufferization::createBufferLoopHoistingPass());
  if (options.enableBufferHoisting)
    pm.addNestedPass<func::FuncOp>(bufferization::createBufferHoistingPass());
}

static void buildPlanBufferDeallocationPipeline(
    OpPassManager &pm, const plan::PlanBufferizationOptions &options) {
  pm.addPass(memref::createExpandReallocPass(/*emitDeallocs=*/false));
  pm.addPass(createCanonicalizerPass());

  pm.addPass(plan::createPlanOwnershipBasedBufferDeallocationPass(
      plan::PlanOwnershipBasedBufferDeallocationPassOptions{
          options.deallocationPrivateFuncDynamicOwnership}));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(bufferization::createLowerDeallocationsPass());
  pm.addPass(executor::createExecutorLowerABIOpsPass());
  pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      mlir::createConvertBufferizationToMemRefPass());
  pm.addNestedPass<func::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
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
  Option<bool> failOnDonationArgumentRejection{
      *this, "fail-on-donation-argument-rejection", llvm::cl::init(false),
      llvm::cl::desc(
          "Function argument attribute `plan.aliasing_output`, if present, "
          "hints argument is donated. However, it is not guaranteed that "
          "donated argument is bufferized to the same buffer as that of result "
          "it is donated for, after `-plan-module-bufferize` pass. If this "
          "flag is set to true, pass `-plan-confirm-argument-donation` fails "
          "if true donation doesn't happen. By default, this flag is set to "
          "false.")};
};

} // namespace

void plan::buildPlanBufferizationPipeline(
    OpPassManager &pm, const plan::PlanBufferizationOptions &options) {
  buildPlanOneShotBufferizePipeline(pm, options);
  buildPlanBufferOptimizationPipeline(pm, options);
  buildPlanBufferDeallocationPipeline(pm, options);
}

// Register pipelines.
void plan::registerPlanDialectPipelines() {
  PassPipelineRegistration<PlanBufferizationPipelineCliOpts>
      executorBufferizationPipeline(
          "plan-bufferize-pipeline",
          "perform one-shot-bufferization, optimizations, and dallocation",
          [](OpPassManager &pm, const PlanBufferizationPipelineCliOpts &opts) {
            plan::PlanBufferizationOptions bufferizationOpts{};
            bufferizationOpts.forceEntrypointsReturnAllocs =
                opts.forceEntrypointsReturnAllocs;
            bufferizationOpts.failOnDonationArgumentRejection =
                opts.failOnDonationArgumentRejection;
            buildPlanBufferizationPipeline(pm, bufferizationOpts);
          });

  PassPipelineRegistration<ClusteringPipelineCliOpts> segPipelineRegistration(
      "plan-segmentation-pipeline",
      "apply the Plan Dialect segmentation pipeline",
      [](OpPassManager &pm, const ClusteringPipelineCliOpts &opts) {
        ClusteringPassOptions clusterOpts{};
        clusterOpts.entrypoint = opts.entrypoint;
        clusterOpts.disableCreateShapeFuncPass =
            opts.forceEntrypointsReturnAllocs;
        buildPlanSegmentationPipeline(pm, clusterOpts, 1);
      });
}
