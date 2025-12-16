//===- BufferizationTestPass.cpp ------------------------------------------===//
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
#include "mlir-kernel/Kernel/Pipelines/Pipelines.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir-kernel/Kernel/Transforms/Transforms.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::kernel;

namespace mlir::kernel {
void registerKernelBufferizationTestPass();
}

namespace {
/// Performs bufferization on a `gpu.module` operation.
struct KernelBufferizationTestPass
    : PassWrapper<KernelBufferizationTestPass,
                  OperationPass<mlir::gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KernelBufferizationTestPass)

  KernelBufferizationTestPass(const KernelBufferizationTestPass &other)
      : KernelBufferizationTestPass(
            other.functionBoundaryTypeConversion.getValue()) {}

  KernelBufferizationTestPass(
      const bufferization::LayoutMapOption &functionBoundaryTypeConversion) {
    this->functionBoundaryTypeConversion = functionBoundaryTypeConversion;
  }

  KernelBufferizationTestPass() = default;

  StringRef getArgument() const final {
    return "test-kernel-one-shot-bufferize";
  }
  StringRef getDescription() const final {
    return "Runs one-shot-bufferization on a gpu.module using Kernel-dialect "
           "specific options";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    gpu::GPUModuleOp op = getOperation();
    if (failed(kernel::bufferizeKernelModule(op,
                                             functionBoundaryTypeConversion))) {
      emitError(op->getLoc()) << "kernel one-shot-module-bufferize failed";
      return signalPassFailure();
    }
  }

  Pass::Option<bufferization::LayoutMapOption> functionBoundaryTypeConversion{
      *this, "function-boundary-type-conversion",
      llvm::cl::desc(
          "Controls layout maps when bufferizing function signatures."),
      llvm::cl::init(
          ::mlir::bufferization::LayoutMapOption::FullyDynamicLayoutMap),
      ::mlir::kernel::detail::createBufferizationLayoutMapClOptions()};
};
} // namespace

/// Constructs a pipeline for bufferization of `gpu.module` and optimizations on
/// memrefs.
static void buildKernelBufferizationAndOptimizationPipeline(
    OpPassManager &modulePM,
    bufferization::LayoutMapOption functionBoundaryTypeConversion) {
  // Apply one-shot bufferization on the module level
  auto &kernelModulePM = modulePM.nest<gpu::GPUModuleOp>();
  kernelModulePM.addPass(createKernelExpandOpsPass());
  kernelModulePM.addPass(std::make_unique<KernelBufferizationTestPass>(
      functionBoundaryTypeConversion));
  buildKernelMemRefOptimizationPipeline(kernelModulePM);
}

namespace {
/// Options for Kernel test pipelines.
struct KernelCodegenPipelineOptions
    : public PassPipelineOptions<KernelCodegenPipelineOptions> {
  Option<std::string> funcFilter{
      *this, "func-filter",
      llvm::cl::desc(
          "In the pass to initial transform schedule, this option specifies "
          "on funcs with which attr to generate the transform schedules."),
      llvm::cl::init("")};

  ListOption<std::string> generatorBenefit{
      *this, "generator-benefit",
      llvm::cl::desc("A list of 'name:benefit' pairs to adjust generator "
                     "benefits for kernel generation.")};

  Option<int64_t> deviceComputeCapability{
      *this, "device-compute-capability",
      llvm::cl::desc("target device compute capability (SM version)"),
      llvm::cl::init(60)};

  Option<int64_t> deviceMaxSharedMemoryPerBlockKb{
      *this, "device-max-smem-per-block",
      llvm::cl::desc("max shared memory per block (in kilobytes)"),
      llvm::cl::init(50)};

  Option<int64_t> deviceMaxRegistersPerBlock{
      *this, "device-max-registers-per-block",
      llvm::cl::desc("max (4 byte) registers per block"),
      llvm::cl::init(65536)};

  Option<std::string> dumpPtxPath{
      *this, "dump-ptx",
      llvm::cl::desc(
          "dump generated PTX as files in this directory; the directory "
          "will be created if it does not exist"),
      llvm::cl::init("")};

  Option<bufferization::LayoutMapOption> functionBoundaryTypeConversion{
      *this, "function-boundary-type-conversion",
      llvm::cl::desc(
          "Controls layout maps when bufferizing function signatures."),
      llvm::cl::init(
          ::mlir::bufferization::LayoutMapOption::FullyDynamicLayoutMap),
      ::mlir::kernel::detail::createBufferizationLayoutMapClOptions()};
};

} // namespace

void kernel::registerKernelBufferizationTestPass() {
  PassRegistration<KernelBufferizationTestPass>();

  PassPipelineRegistration<KernelCodegenPipelineOptions>(
      "kernel-linalg-codegen-pipeline",
      "performs  Transform IR generation/application and lowering of "
      "gpu.module",
      [](OpPassManager &pm, const KernelCodegenPipelineOptions &opts) {
        kernel::buildTransformIRPipeline(
            pm, opts.funcFilter, opts.deviceComputeCapability,
            opts.deviceMaxSharedMemoryPerBlockKb,
            opts.deviceMaxRegistersPerBlock, opts.generatorBenefit);

        kernel::SetGPUTargetPassOptions setTargetOpts{};
        setTargetOpts.chip =
            "sm_" + std::to_string(opts.deviceComputeCapability);
        setTargetOpts.maxRegistersPerBlock = opts.deviceMaxRegistersPerBlock;
        setTargetOpts.maxSharedMemoryPerBlockKb =
            opts.deviceMaxSharedMemoryPerBlockKb;
        pm.addPass(kernel::createSetGPUTargetPass(setTargetOpts));

        buildKernelBufferizationAndOptimizationPipeline(
            pm, opts.functionBoundaryTypeConversion);
        auto &kernelModulePM = pm.nest<gpu::GPUModuleOp>();
        buildKernelLowerToPTXPipeline(kernelModulePM, opts.dumpPtxPath);
      });

  // Register test pipeline for gpu.module bufferization + optimizations.
  PassPipelineRegistration<>(
      "kernel-module-bufferization-pipeline",
      "performs bufferization and optimizations on memrefs",
      [](OpPassManager &modulePM) {
        buildKernelBufferizationAndOptimizationPipeline(
            modulePM, bufferization::LayoutMapOption::FullyDynamicLayoutMap);
      });

  // Register test pipeline for kernel transform IR generation.
  PassPipelineRegistration<KernelCodegenPipelineOptions>(
      "kernel-transform-ir-pipeline",
      "performs initial Transform IR generation/application",
      [](OpPassManager &pm, const KernelCodegenPipelineOptions &opts) {
        buildTransformIRPipeline(
            pm, opts.funcFilter, opts.deviceComputeCapability,
            opts.deviceMaxSharedMemoryPerBlockKb,
            opts.deviceMaxRegistersPerBlock, opts.generatorBenefit);
      });
}
