//===- StableHloToExecutable.cpp ------------------------------------------===//
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
/// MLIR-TensorRT CompilerClient API definitions.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/StablehloToExecutable/StablehloToExecutable.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt/Compiler/Client.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/OptionsProviders.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/Passes.h"
#include "mlir-tensorrt/Compiler/StablehloToExecutable/TensorRTExtension.h"
#include "mlir-tensorrt/Conversion/CUDAToExecutor/CUDAToExecutor.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <memory>

using namespace mlirtrt;
using namespace mlirtrt::compiler;
using namespace mlir;

#ifdef MLIR_TRT_ENABLE_HLO

static plan::PlanBufferizationOptions
convertBufferizationOptions(const StablehloToExecutableOptions &pipelineOpts) {
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
// StableHloToExecutableTask
//===----------------------------------------------------------------------===//

StablehloToExecutableTask::StablehloToExecutableTask(
    MLIRContext *ctx, std::unique_ptr<StablehloToExecutableOptions> options)
    : CompilationTask(ctx, std::move(options)) {}

static void populateExtensionPasses(mlir::OpPassManager &pm,
                                    const StablehloToExecutableOptions &options,
                                    Phase phase, ExtensionList &extensions) {
  for (auto &[key, ext] : extensions) {
    ext->populatePasses(pm, phase);
  }
}

static void addCleanupPasses(OpPassManager &pm) {
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void StablehloToExecutableTask::populatePassManager() {
  const StablehloToExecutableOptions &options = getOptions();
  mlir::PassManager &pm = *this;

  assert(pm.getPasses().empty() && "expected empty pass manager");

  pm.addPass(createPopulateDefaultBackendMetadataPass(
      PopulateDefaultBackendMetadataPassOptions{
          /*defaultBackends=*/SmallVector<std::string>(
              options.defaultBackends.begin(),
              options.defaultBackends.end())}));

  // StableHLO Preprocessing
  mlirtrt::compiler::StableHloInputOptions opts{};
  opts.legalizeControlFlowToSCF = true;
  opts.preserveChloErf = true;
  opts.preserveChloTopK = true;
  opts.unrollThreshold = options.unrollThreshold;
  {
    FailureOr<stablehlo_ext::TargetSpecificCanonicalizationOptions> parsed =
        stablehlo_ext::TargetSpecificCanonicalizationOptions::parse(
            options.stablehloDisableOptimizationPatternSet);
    if (failed(parsed)) {
      llvm::report_fatal_error("Invalid target-specific Stablehlo optimization "
                               "pattern set names.");
    }
    opts.targetSpecificOptions = std::move(*parsed);
    opts.constantFoldSizeLimit =
        options.stablehloInputRewriteConstantFoldVolumeLimit;
  }

  mlirtrt::compiler::buildStablehloPreProcessingPipeline(
      pm, opts,
      [&](mlir::OpPassManager &pm, const StableHloInputOptions &opts) {
        pm.addNestedPass<func::FuncOp>(stablehlo_ext::createConstantFoldingPass(
            stablehlo_ext::ConstantFoldingPassOptions{
                opts.constantFoldSizeLimit}));
        populateExtensionPasses(pm, options, Phase::ConstantFolding,
                                extensions);
      });

  // Add pre-clustering extension passes
  populateExtensionPasses(pm, options, Phase::PreClustering, extensions);

  plan::ClusteringPassOptions clusteringOpts{};
  clusteringOpts.entrypoint = options.entrypoint;
  clusteringOpts.inputKind = plan::InputKind::Stablehlo;
  plan::buildPlanSegmentationPipeline(pm, clusteringOpts);

  // Compile outlined scalarizable host clusters.
  pm.addNestedPass<func::FuncOp>(createProcessStablehloHostClustersPass());
  pm.addNestedPass<func::FuncOp>(createConvertStablehloConstantsToArithPass());

  populateExtensionPasses(pm, options, Phase::PostClustering, extensions);

  pm.addNestedPass<func::FuncOp>(plan::createPostClusteringValidationPass());

  // We then perform some final simplification on the top-level func.func ops
  // (e.g. public entrypoint functions).
  pm.addNestedPass<func::FuncOp>(createSCFDetensorizeLoopsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  populateExtensionPasses(pm, options, Phase::PreBufferization, extensions);

  // Perform bufferization.
  plan::buildPlanBufferizationPipeline(pm,
                                       convertBufferizationOptions(options));

  populateExtensionPasses(pm, options, Phase::PostBufferization, extensions);
  const HostTarget hostTarget = options.hostTarget;

  if (hostTarget == HostTarget::Executor && options.hoistAllocsToGlobals) {
    pm.addPass(executor::createExecutorAllocsToGlobalsPass());
  }

  pm.addPass(createConvertMemRefToCUDAPass());

  if (hostTarget == HostTarget::Executor) {
    pm.addPass(createConvertPlanToExecutorPass());
    pm.addNestedPass<func::FuncOp>(
        executor::createExecutorPopulateFunctionMetadataPass());
  }

  populateExtensionPasses(pm, options, Phase::ExecutorLowering, extensions);

  if (hostTarget == HostTarget::Executor) {
    ConvertCUDAToExecutorPassOptions cudaToExecutorOpts;
    cudaToExecutorOpts.indexBitwidth =
        options.get<ExecutorOptions>().indexBitwidth;
    pm.addPass(createConvertCUDAToExecutorPass(cudaToExecutorOpts));
  }

  pm.addPass(createDropNestedModulesPass());

  if (hostTarget == HostTarget::Executor) {
    mlir::executor::ConvertStdToExecutorPassOptions stdToExecOpts;
    stdToExecOpts.indexBitwidth = options.get<ExecutorOptions>().indexBitwidth;
    mlir::executor::buildExecutorLoweringPipeline(
        pm, stdToExecOpts, [](mlir::TypeConverter &typeConverter) {
          mlir::populateCUDAToExecutorTypeConversions(typeConverter);
        });
    return;
  }

  // LLVM and EmitC targets will execute a common set of passes except for the
  // SCF-to-CF pass.
  if (hostTarget == HostTarget::LLVM || hostTarget == HostTarget::EmitC) {
    pm.addPass(createConvertComplexToStandardPass());

    // For EmitC lowering, we rely on preserving control flow. Otherwise the C
    // code could be very unreadable.
    if (hostTarget != HostTarget::EmitC)
      pm.addPass(mlir::createSCFToControlFlowPass());

    pm.addPass(memref::createFoldMemRefAliasOpsPass());
    pm.addPass(memref::createExpandOpsPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    addCleanupPasses(pm);
    pm.addPass(affine::createAffineExpandIndexOpsPass());
    pm.addPass(mlir::createLowerAffinePass());
    addCleanupPasses(pm);

    if (hostTarget == HostTarget::LLVM) {
      pm.addPass(LLVM::createRequestCWrappersPass());
      ConvertCUDAToLLVMPassOptions cudaToLLVMOpts;
      cudaToLLVMOpts.artifactsDirectory = options.artifactsDirectory;
      pm.addPass(createConvertCUDAToLLVMPass(std::move(cudaToLLVMOpts)));
      pm.addPass(createConvertHostToLLVMPass());

      // Generally there is a lot of cleanup that can happen here after creation
      // of LLVM IR (e.g. insert|extract forwarding).
      addCleanupPasses(pm);
    }

    // For EmitC, just run Host-to-EmitC followed
    // by cleanup and expression forming.
    if (hostTarget == HostTarget::EmitC) {
      pm.addPass(createConvertHostToEmitCPass({options.artifactsDirectory}));
      addCleanupPasses(pm);
      // The EmitC "form-expressions" pass combines operations into
      // "expression regions" where possible, which allows the C++ translation
      // to be more concise.
      pm.addPass(emitc::createFormExpressionsPass());
    }
    return;
  }
}

void mlirtrt::compiler::registerStableHloToExecutableTask() {
  registerCompilationTaskWithNoExtensions<StablehloToExecutableTask,
                                          StablehloToExecutableOptions>(
      mlirtrt::compiler::StablehloToExecutableTask::getName());
}

#endif // MLIR_TRT_ENABLE_HLO
