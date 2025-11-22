//===- ModuleBufferizationUtils.cpp ---------------------------------------===//
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
/// This file contains definitions for some simple utilities needed
/// for bufferization. We use the CallGraph to find functions which
/// cannot call each other circularly and enumerate them in the order
/// of 'fewer incoming call edges' to 'more incoming call edges'.
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-executor/Executor/Transforms/BufferizationOpInterfaceImpls.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/ModuleBufferization/ModuleBufferization.h"
#include "mlir-tensorrt/Utils/ModuleUtils.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using OneShotBufferizationOptions = bufferization::OneShotBufferizationOptions;

LogicalResult plan::fixupHostModule(
    ModuleLikeOp module,
    const bufferization::OneShotBufferizationOptions &options) {
  IRRewriter rewriter(module);

  auto walkResult = module->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (ModuleLikeOp(op))
      return op == module ? WalkResult::advance() : WalkResult::skip();

    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (executor::abi::isABIWrapperFunction(funcOp)) {
        if (failed(executor::bufferizeABIWrapperFunctionType(funcOp, options)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      MemRefType fromType = loadOp.getMemRefType();
      auto space = dyn_cast<plan::MemorySpaceAttr>(fromType.getMemorySpace());
      if (!space || space.isHostVisible())
        return WalkResult::skip();
      emitError(loadOp.getLoc()) << "load on device memory is not supported";
      return WalkResult::interrupt();
    }

    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      MemRefType fromType = storeOp.getMemRefType();
      auto space = dyn_cast<plan::MemorySpaceAttr>(fromType.getMemorySpace());
      if (!space || space.isHostVisible())
        return WalkResult::skip();
      emitError(storeOp.getLoc()) << "store on device memory is not supported";
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return success(!walkResult.wasInterrupted());
}
