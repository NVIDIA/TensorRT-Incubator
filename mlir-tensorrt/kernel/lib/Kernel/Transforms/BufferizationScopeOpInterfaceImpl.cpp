//===- BufferizationScopeOpInterfaceImpl.cpp ------------------------------===//
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
///===- BufferizationScopeOpInterfaceImpl.cpp -----------------------------===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// BufferizationScopeOpInterface implementation for 'gpu.module'
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Interfaces.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/Transforms.h"
#include "mlir-tensorrt-common/Interfaces/BufferizationScopeInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;

namespace {
class KernelModuleBufferizationScopeOpIfaceImpl
    : public BufferizationScopeOpInterface::ExternalModel<
          KernelModuleBufferizationScopeOpIfaceImpl, gpu::GPUModuleOp> {
public:
  std::optional<bufferization::OneShotBufferizationOptions>
  getBufferizationOptions(Operation *op_) const {
    auto op = cast<gpu::GPUModuleOp>(op_);
    auto kind = op->getAttrOfType<kernel::GPUModuleLoweringAttrInterface>(
        mlir::kernel::KernelDialect::getGpuModuleKindAttrName());
    assert(kind && "expected GPU module kind attribute");
    auto options = kernel::getKernelModuleBufferizationOptions();
    if (!kind.shouldParticipateInModuleBufferization(op))
      // Don't bufferize any operations.
      options.opFilter.denyOperation([](Operation *op) { return true; });
    return options;
  }

  LogicalResult performPostBufferizationActions(Operation *op_,
                                                IRRewriter &rewriter) const {
    auto op = cast<gpu::GPUModuleOp>(op_);
    auto kind = op->getAttrOfType<kernel::GPUModuleLoweringAttrInterface>(
        mlir::kernel::KernelDialect::getGpuModuleKindAttrName());
    if (!kind)
      return op->emitError("unknown GPU module kind");

    // Skip post-bufferization actions for modules that don't participate in
    // bufferization.
    if (!kind.shouldParticipateInModuleBufferization(op))
      return success();

    for (auto funcOp : op->getRegion(0).getOps<mlir::FunctionOpInterface>()) {
      if (funcOp.isDeclaration() || funcOp.isExternal())
        continue;
      SymbolTableCollection symbolTables;
      SymbolUserMap userMap(symbolTables, op);
      return kernel::runKernelModulePostBufferizationActions(
          cast<gpu::GPUModuleOp>(op), userMap);
    }

    return success();
  }
};
} // namespace

void mlir::kernel::registerKernelBufferizationScopeOpInterfaceImpls(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, gpu::GPUDialect *dialect) {
    gpu::GPUModuleOp::attachInterface<
        KernelModuleBufferizationScopeOpIfaceImpl>(*ctx);
  });
}
