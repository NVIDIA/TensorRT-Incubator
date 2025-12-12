//===- DispatchGPUModuleCompilation.cpp -----------------------------------===//
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
///
/// Dispatch the GPU module compilation to the appropriate strategy.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"

namespace mlir::kernel {
#define GEN_PASS_DEF_DISPATCHGPUMODULECOMPILATIONPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace mlir::kernel
using namespace mlir;
using namespace mlir::kernel;

using PipelineCache = llvm::DenseMap<Attribute, OpPassManager>;

static std::optional<OpPassManager *>
getCachedPipeline(Attribute gpuModuleKind, PipelineCache &pipelineCache) {
  auto it = pipelineCache.find(gpuModuleKind);
  if (it != pipelineCache.end())
    return const_cast<OpPassManager *>(&it->second);
  return std::nullopt;
}

namespace {
struct DispatchGPUModuleCompilationPass
    : public kernel::impl::DispatchGPUModuleCompilationPassBase<
          DispatchGPUModuleCompilationPass> {
  using Base::Base;

  /// A cache of pipelines for each builtin GPU module kind and phase.
  /// Note: This cache is copied when the pass is cloned. We can't share
  /// a single cache among all instances of the pass because OpPassManager
  /// instances are stateful.
  PipelineCache pipelineCache;

  LogicalResult initialize(MLIRContext *context) override {

    /// For now, we populate the pass manager cache for "builtin" GPU module
    /// kinds native to the Kernel dialect. Otherwise, we would require another
    /// registry.
    ///
    /// Potentially we can modify the cache in `runOnOperation`. Otherwise,
    /// external users with custom GPULoweringAttrInterface implementations can
    /// setup their own pass for better caching.
    for (Attribute kind : SmallVector<Attribute>{
             kernel::DefaultGPUModuleKindAttr::get(context)}) {
      OpPassManager pm(gpu::GPUModuleOp::getOperationName());
      std::optional<LogicalResult> result =
          cast<kernel::GPUModuleLoweringAttrInterface>(kind).addPhasePasses(
              nullptr, pm, this->phase, this->debugDumpDir);
      if (!result)
        continue;
      if (failed(*result))
        return failure();
      pipelineCache[kind] = std::move(pm);
    }

    return success();
  }

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();

    auto gpuModuleKind =
        module->getAttrOfType<kernel::GPUModuleLoweringAttrInterface>(
            KernelDialect::getGpuModuleKindAttrName());
    if (!gpuModuleKind) {
      emitError(module.getLoc())
          << "gpu.module " << module.getSymNameAttr()
          << " does not have a valid gpu module kind attribute";
      return signalPassFailure();
    }

    if (std::optional<OpPassManager *> pm =
            getCachedPipeline(gpuModuleKind, pipelineCache)) {
      if (failed(runPipeline(**pm, module)))
        return signalPassFailure();
      return;
    }

    // Fallback to constructing a new temporary pass manager.
    OpPassManager pm(gpu::GPUModuleOp::getOperationName());
    auto result = gpuModuleKind.addPhasePasses(module, pm, this->phase,
                                               this->debugDumpDir);
    if (!result)
      return;
    if (failed(*result)) {
      emitError(module.getLoc())
          << "failed to populate compilation pipeline for GPU module \""
          << module.getName() << "\"";
      return signalPassFailure();
    }

    if (failed(runPipeline(pm, module))) {
      emitError(module.getLoc())
          << "failed to run compilation pipeline for GPU module \""
          << module.getName() << "\"";
      return signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    // Initialize LLVM NVPTX backend.
    if (this->phase == GPUModuleLoweringPhase::LowerToNVVM) {
      LLVMInitializeNVPTXTarget();
      LLVMInitializeNVPTXTargetInfo();
      LLVMInitializeNVPTXTargetMC();
      LLVMInitializeNVPTXAsmPrinter();
      registerLLVMDialectTranslation(registry);
      registerNVVMDialectTranslation(registry);
      registerGPUDialectTranslation(registry);
    }
  }
};
} // namespace
