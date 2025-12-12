//===- GPUModuleLoweringImpl.cpp ------------------------------------------===//
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
/// This file contains the implementation of the GPUModuleLoweringAttrInterface
/// for the Kernel dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Attributes.h"
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir-kernel/Kernel/Pipelines/Pipelines.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::kernel;

namespace {
struct DefaultGPUModuleLowering
    : GPUModuleLoweringAttrInterface::FallbackModel<DefaultGPUModuleLowering> {
  std::optional<LogicalResult> addPhasePasses(Attribute attr, gpu::GPUModuleOp,
                                              OpPassManager &pm,
                                              GPUModuleLoweringPhase phase,
                                              StringRef debugDumpDir) const {
    switch (phase) {
    case GPUModuleLoweringPhase::PreBufferization:
      pm.addPass(kernel::createKernelExpandOpsPass());
      return success();
    case GPUModuleLoweringPhase::PostBufferization:
      kernel::buildKernelMemRefOptimizationPipeline(pm);
      return success();
    case GPUModuleLoweringPhase::LowerToNVVM:
      kernel::buildKernelLowerToPTXPipeline(pm, debugDumpDir);
      return success();
    }
    return std::nullopt;
  }

  KernelArgLayoutMapOption
  getMemRefArgumentDefaultLayoutMap(Attribute attr,
                                    gpu::GPUModuleOp gpuModule) const {
    return KernelArgLayoutMapOption::FullyDynamic;
  }

  bool
  shouldParticipateInModuleBufferization(Attribute attr,
                                         gpu::GPUModuleOp gpuModule) const {
    return true;
  }
};

} // namespace

void mlir::kernel::registerGPUModuleLoweringAttrExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, KernelDialect *) {
    kernel::DefaultGPUModuleKindAttr::attachInterface<DefaultGPUModuleLowering>(
        *ctx);
  });
}
