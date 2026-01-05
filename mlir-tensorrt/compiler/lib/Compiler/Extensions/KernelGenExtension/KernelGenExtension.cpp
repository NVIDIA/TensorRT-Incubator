//===- KernelGenExtension.cpp ---------------------------------------------===//
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
/// Defines the Stablehlo-to-Executable codegen extension.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/Extensions/KernelGenExtension.h"
#include "mlir-kernel/Conversion/Passes.h"
#include "mlir-kernel/Kernel/Pipelines/Pipelines.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir-tensorrt/Backends/Kernel/KernelBackend.h"
#include "mlir-tensorrt/Backends/Kernel/Passes.h"
#include "mlir-tensorrt/Compiler/InputPipelines/StablehloInputPipeline.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir-tensorrt/Compiler/Pipeline.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/Utils.h"
#include "mlir-tensorrt/Features.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "codegen-extension"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "]"

using namespace mtrt::compiler;
using namespace mtrt;
using namespace mlir;

//===----------------------------------------------------------------------===//
// KernelGenCompilerExtension
//===----------------------------------------------------------------------===//

static void addCleanupPasses(OpPassManager &pm) {
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

static int64_t getScalarBitWidth(Type type) {
  assert(type.isIntOrIndexOrFloat() && "expected scalar type");
  // Index type doesn't have a bitwidth, so we treat as 64 bit. This will allow
  // folding of conversions that are index casts to 32bit or 64bit.
  return isa<IndexType>(type) ? 64 : type.getIntOrFloatBitWidth();
}

static int64_t getElementTypeBitWidth(Type type) {
  if (auto complexType = dyn_cast<ComplexType>(type))
    return 2 * getScalarBitWidth(complexType.getElementType());
  if (auto vectorType = dyn_cast<VectorType>(type))
    return getScalarBitWidth(vectorType.getElementType()) *
           vectorType.getNumElements();
  return getScalarBitWidth(type);
}

static void addInternalStablehloConstantFoldingPasses(mlir::OpPassManager &pm) {
  pm.addPass(
      plan::createOutlineConstantFoldableSubgraphsPass([](Operation *op) {
        // Skip `stablehlo.broadcastInDim`.
        if (isa<stablehlo::BroadcastInDimOp>(op))
          return true;

        if (auto convertOp = dyn_cast<stablehlo::ConvertOp>(op)) {
          Type destType = convertOp.getResult().getType().getElementType();
          Type srcType = convertOp.getOperand().getType().getElementType();

          // Allow folding cast to or from index type.
          if (destType.isIndex() || srcType.isIndex())
            return false;

          // As of now, we skip all convert ops casting from low to high
          // bitwidths.
          if (getElementTypeBitWidth(destType) >
              getElementTypeBitWidth(srcType))
            return true;
        }
        // `stablehlo.scatter` op is supported via `kernel.scatter`.
        if (isa<stablehlo::ScatterOp>(op) || stablehlo::canConvertToLinalg(op))
          return false;
        return true;
      }));
  pm.addPass(mtrt::createFuncExtDuplicateFunctionEliminationPass());
  pm.addPass(stablehlo::createStablehloConvertToSignlessPass());
  pm.addPass(mtrt::createPlanExecuteConstantFoldableSubgraphsPass());
  pm.addPass(createInlinerPass());
}

void KernelGenExtension::populatePasses(mlir::OpPassManager &pm,
                                        ExtensionPoint point) const {
  const MainOptions &options = this->getOptions();

  if (options.disableKernelGenExtension || options.disableAllExtensions)
    return;

  if (point == ExtensionPoint::ConstantFolding) {
    // Currently V2 constant folding is only supported for Stablehlo input.
    if (options.enableV2constantFolding &&
        options.inputKind == plan::InputKind::Stablehlo)
      addInternalStablehloConstantFoldingPasses(pm);
    return;
  }

  if (point == ExtensionPoint::PreClustering) {
    return;
  }

  if (point == ExtensionPoint::PostClustering) {
    // TODO: Verify whether this is still needed or whether it can be moved to
    // preprocessing stage.
    if (options.inputKind == plan::InputKind::Stablehlo)
      pm.addPass(stablehlo::createStablehloConvertToSignlessPass());

    // Case (2). Groups of StableHLO ops that are targeted to a single kernel
    // are outlined into functions tagged with 'cluster.codegen'. Convert
    // StableHlo ops in these functions to linalg using a dynamic pass
    // pipeline, then generate kernels using the transform IR generation
    // pipeline.
    mtrt::compiler::buildKernelGenReclusteringPipeline(pm, options.inputKind);

    const auto &deviceOpts = options.get<DeviceOptions>();
    const auto &kernelGenOpts = options.get<KernelGenOptions>();

    // Perform Transform IR transforms on codegen clusters.
    kernel::buildTransformIRPipeline(
        pm, mtrt::compiler::getKernelGenClusterAttrName(),
        deviceOpts.computeCapability, deviceOpts.maxSharedMemoryPerBlockKb,
        deviceOpts.maxRegistersPerBlock, kernelGenOpts.generatorBenefit);

    // Lower kernel.sort operations to generated merge sort kernels
    pm.addPass(kernel::createLowerKernelSortPass());

    kernel::SetGPUTargetPassOptions setTargetOptions{};
    setTargetOptions.inferTargetFromHost = deviceOpts.shouldInferFromHost;
    setTargetOptions.chip =
        "sm_" + std::to_string(deviceOpts.computeCapability);
    setTargetOptions.maxRegistersPerBlock = deviceOpts.maxRegistersPerBlock;
    setTargetOptions.maxSharedMemoryPerBlockKb =
        deviceOpts.maxSharedMemoryPerBlockKb;
    pm.addPass(kernel::createSetGPUTargetPass(setTargetOptions));

    // Annotate kernel entrypoints which weren't already annotated.
    pm.addPass(kernel::createAnnotateKernelEntrypointsPass());

    // Run pre-bufferization transformations on all GPU modules. This pass
    // dispatches dynamic pipelines as directed by the GPU module's
    // attributes.
    pm.addNestedPass<gpu::GPUModuleOp>(
        kernel::createDispatchGPUModuleCompilationPass(
            kernel::DispatchGPUModuleCompilationPassOptions{
                kernel::GPUModuleLoweringPhase::PreBufferization}));

    return;
  }

  if (point == ExtensionPoint::PostBufferization) {
    const auto &kernelGenOpts = options.get<KernelGenOptions>();

    // Apply complex type conversion to the entire module BEFORE GPU module
    // processing to ensure both host code (kernel.call operations) and GPU
    // kernels are converted together
    pm.addPass(mlir::createConvertComplexToStandardPass());

    pm.addPass(mlir::createConvertComplexToStandardExt(
        mlir::ConvertComplexToStandardExtOptions{
            /*convertOpGenerically=*/llvm::SmallVector<std::string>{
                "kernel.call", "memref.cast"}}));

    // Run post-bufferization transformations on all GPU modules. This pass
    // dispatches dynamic pipelines as directed by the GPU module's attributes.
    pm.addNestedPass<gpu::GPUModuleOp>(
        kernel::createDispatchGPUModuleCompilationPass(
            kernel::DispatchGPUModuleCompilationPassOptions{
                kernel::GPUModuleLoweringPhase::PostBufferization}));

    // Refine argument types based on uses.
    pm.addPass(kernel::createKernelRefineArgumentLayoutsPass());
    pm.addNestedPass<gpu::GPUModuleOp>(
        kernel::createKernelExpandMemRefArgsPass());

    // Lower code in `gpu.module` to NVVM.
    pm.addNestedPass<gpu::GPUModuleOp>(
        kernel::createDispatchGPUModuleCompilationPass(
            kernel::DispatchGPUModuleCompilationPassOptions{
                kernel::GPUModuleLoweringPhase::LowerToNVVM,
                /*debugDumpDir=*/kernelGenOpts.dumpPtxDir}));

    // Lower `kernel.call` to appropriate CUDA launch method.
    pm.addPass(createConvertKernelToCUDAPass());
    return;
  }

  if (point == ExtensionPoint::ExecutorLowering) {
    return;
  }
}

//===----------------------------------------------------------------------===//
// Extension Registration
//===----------------------------------------------------------------------===//

void mtrt::compiler::registerKernelGenExtension() {
  mtrt::compiler::registerExtension<KernelGenExtension>();
}

//===----------------------------------------------------------------------===//
// Extended Pipeline Registrations
//===----------------------------------------------------------------------===//

static void linalgOptimizationPasses(OpPassManager &pm) {
  addCleanupPasses(pm);
  pm.addPass(mtrt::createLinalgElementwiseFusionPass());
  pm.addPass(mtrt::createLinalgSimplifyExtractSlicePass());
  pm.addPass(mlir::createLinalgFoldUnitExtentDimsPass());
}

/// Builds a pipeline for lowering StableHLO to linalg-on-tensors.
static void buildStableHloToLinalgPipeline(
    OpPassManager &pm, const mtrt::compiler::StablehloInputOptions &opts) {
  // Inline functions.
  pm.addPass(createInlinerPass());

  // Do StableHlo-> Std conversion
  buildStablehloInputPipeline(
      pm, opts, [](mlir::OpPassManager &pm, const StablehloInputOptions &opts) {
        pm.addNestedPass<func::FuncOp>(stablehlo_ext::createConstantFoldingPass(
            stablehlo_ext::ConstantFoldingPassOptions{
                opts.constantFoldSizeLimit}));
        addInternalStablehloConstantFoldingPasses(pm);
      });
  auto &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(mlir::createStablehloToLinalgPass());
  addCleanupPasses(funcPM);

  pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());

  // Do elementwise fusion and simplification in a loop
  constexpr unsigned kNumOptimizationIterations = 2;
  for (unsigned i = 0; i < kNumOptimizationIterations; i++)
    linalgOptimizationPasses(funcPM);
  addCleanupPasses(funcPM);
}

void mtrt::compiler::registerStablehloToLinalgPipeline() {
  PassPipelineRegistration<>(
      "stablehlo-to-linalg-pipeline", "apply StableHLO to Linalg passes",
      [](OpPassManager &pm) {
        // This pipeline is used for testing purposes and test results can be
        // sensitive to changes to defaults values of the options. Fix the
        // relevant options here.
        LocalScopedOptionsGroup<mtrt::compiler::StablehloInputOptions> opts;
        opts.get().legalizeControlFlowToSCF.setValue(true);
        opts.get().preserveChloErf.setValue(false);
        opts.get().preserveChloTopK.setValue(true);
        opts.get().disableInliner.setValue(false);
        opts.get().constantFoldSizeLimit.setValue(128);
        buildStableHloToLinalgPipeline(pm, opts);
      });
}

void mtrt::compiler::buildKernelGenReclusteringPipeline(
    OpPassManager &pm, plan::InputKind inputKind) {

  OpPassManager &funcPM = pm.nest<func::FuncOp>();
  if (inputKind == plan::InputKind::Stablehlo) {
    funcPM.addPass(mlir::createStablehloToKernelPass());
    funcPM.addPass(mlir::createStablehloToLinalgPass());
  }
  funcPM.addPass(mlir::createLinalgElementwiseOpFusionPass());
  funcPM.addPass(mtrt::createLinalgSimplifyExtractSlicePass());

  pm.addPass(mlir::createConvertTensorToLinalgPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mtrt::compiler::createKernelSegmentationPass());
  pm.addPass(createFuncExtDuplicateFunctionEliminationPass());
}

void mtrt::compiler::registerKernelGenExtensionPipelines() {
  // The 'plan-ext-stablehlo-to-linalg' should be run after
  // stablehlo-clustering. It lowers stablehlo to linalg and then re-clusters
  // linalg operations into discrete kernels.
  PassPipelineRegistration<>(
      "plan-ext-stablehlo-to-linalg-kernels", "", [](OpPassManager &pm) {
        buildKernelGenReclusteringPipeline(pm, plan::InputKind::Stablehlo);
      });
}
