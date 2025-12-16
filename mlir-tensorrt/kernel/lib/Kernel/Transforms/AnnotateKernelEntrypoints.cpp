//===- AnnotateKernelEntrypoints.cpp --------------------------------------===//
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
/// Implementation of `kernel-annotate-entrypoints` pass that annotates
/// kernel entrypoints with the 'gpu.kernel' attribute by checking which
/// `gpu.module` functions are called from the host.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"

namespace mlir {
namespace kernel {
#define GEN_PASS_DEF_ANNOTATEKERNELENTRYPOINTSPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace kernel
} // namespace mlir

using namespace mlir;
using namespace mlir::kernel;

namespace {
class AnnotateKernelEntrypointsPass
    : public kernel::impl::AnnotateKernelEntrypointsPassBase<
          AnnotateKernelEntrypointsPass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    SymbolTableCollection symbolTable;
    SymbolUserMap userMap(symbolTable, module);

    // Iterate over all gpu.module operations
    for (auto gpuModule : module.getOps<gpu::GPUModuleOp>()) {
      // Iterate over all functions in the gpu.module
      for (auto func : gpuModule.getOps<func::FuncOp>()) {
        // Check if this function is called from host code and/or device code
        bool isCalledFromHost = false;
        bool isCalledFromDevice = false;
        for (Operation *user : userMap.getUsers(func)) {
          // Check if the user is a CallOpInterface operation
          if (auto callOp = dyn_cast<CallOpInterface>(user)) {
            // Check if the call is NOT nested in a gpu.module (i.e., it's in
            // host code)
            if (callOp->getParentOfType<gpu::GPUModuleOp>() == nullptr) {
              isCalledFromHost = true;
            } else {
              // Call is nested in a gpu.module, so it's a device call
              isCalledFromDevice = true;
            }
          }
        }

        // Error if function is called from both host and device
        if (isCalledFromHost && isCalledFromDevice) {
          func->emitError("kernel function '")
              << func.getName()
              << "' cannot be called from both host and device code";
          return signalPassFailure();
        }

        // If the function is called from host code, annotate it with gpu.kernel
        if (isCalledFromHost) {
          func->setAttr("gpu.kernel", UnitAttr::get(&getContext()));
        }
      }
    }
  }
};
} // namespace
