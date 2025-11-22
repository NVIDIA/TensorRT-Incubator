//===- Bufferize.cpp ------------------------------------------------------===//
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
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace plan {
#define GEN_PASS_DEF_PLANOWNERSHIPBASEDBUFFERDEALLOCATIONPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace plan
} // namespace mlir

using namespace mlir;
using namespace mlir::plan;

//===----------------------------------------------------------------------===//
// PlanOwnershipBasedBufferDeallocationPass
//===----------------------------------------------------------------------===//

namespace {

/// The actual buffer deallocation pass that inserts and moves dealloc nodes
/// into the right positions. Furthermore, it inserts additional clones if
/// necessary. It uses the algorithm described at the top of the file.
struct PlanOwnershipBasedBufferDeallocationPass
    : public plan::impl::PlanOwnershipBasedBufferDeallocationPassBase<
          PlanOwnershipBasedBufferDeallocationPass> {
  using Base::Base;

  void runOnOperation() override {
    bufferization::DeallocationOptions options;
    options.privateFuncDynamicOwnership = privateFuncDynamicOwnership;
    SmallVector<FunctionOpInterface> hostFuncs =
        llvm::to_vector(getOperation().getOps<FunctionOpInterface>());

    for (auto func : hostFuncs) {
      if (func.isExternal())
        continue;
      if (failed(bufferization::deallocateBuffersOwnershipBased(func, options)))
        return signalPassFailure();
    }
  }
};
} // namespace
