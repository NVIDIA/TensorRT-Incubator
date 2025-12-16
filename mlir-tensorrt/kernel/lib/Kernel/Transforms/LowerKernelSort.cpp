//===- LowerKernelSort.cpp ------------------------------------------------===//
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
/// This file implements the lowering pass for kernel.sort operations.
/// It generates GPU merge sort kernels and replaces kernel.sort operations
/// with calls to the generated dispatch function.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/GenerateSort.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::kernel {
#define GEN_PASS_DEF_LOWERKERNELSORTPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace mlir::kernel

using namespace mlir;
using namespace mlir::kernel;

/// Get or create a gpu.module in the target module with the given name
static gpu::GPUModuleOp getOrCreateGpuModule(OpBuilder &builder,
                                             ModuleOp module, StringRef name) {

  // Create new gpu.module
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto gpuModule = builder.create<gpu::GPUModuleOp>(module.getLoc(), name);

  // Add the required gpu_module_kind attribute
  auto moduleKindAttr =
      kernel::DefaultGPUModuleKindAttr::get(builder.getContext());
  gpuModule->setAttr(kernel::KernelDialect::getGpuModuleKindAttrName(),
                     moduleKindAttr);

  return gpuModule;
}

static TypedValue<RankedTensorType>
castToDynamic(RewriterBase &rewriter, Location loc,
              TypedValue<RankedTensorType> input) {
  RankedTensorType type = input.getType();
  if (static_cast<int64_t>(type.getNumDynamicDims()) == type.getRank())
    return input;
  SmallVector<int64_t> shape(type.getRank(), ShapedType::kDynamic);
  auto dynamicType = RankedTensorType::get(shape, type.getElementType());
  return cast<TypedValue<RankedTensorType>>(
      rewriter.create<tensor::CastOp>(loc, dynamicType, input).getResult());
}

/// Lower kernel.sort to func.call by generating and inserting
/// the merge sort kernels.
static LogicalResult lowerSortOp(RewriterBase &rewriter, kernel::SortOp sortOp,
                                 ModuleOp module, gpu::GPUModuleOp gpuModule,
                                 SymbolTableCollection &symbolTables) {

  Location loc = sortOp->getLoc();

  // Extract configuration from the sort op
  MergeSortConfig config;
  config.blockThreads = sortOp.getBlockThreads();
  config.itemsPerThread = sortOp.getItemsPerThread();

  // Get key and value types
  auto inputs = sortOp.getInputs();
  if (inputs.empty())
    return success();

  auto keyTensor = cast<RankedTensorType>(inputs[0].getType());
  Type keyType = keyTensor.getElementType();

  Type valueType;
  config.keysOnly = inputs.size() == 1;
  if (!config.keysOnly)
    valueType = cast<RankedTensorType>(inputs[1].getType()).getElementType();

  // Currently we only support 1D tensors of FloatType or IntegerType.
  if (llvm::any_of(inputs.getTypes(), [](Type type) {
        auto tensorType = cast<RankedTensorType>(type);
        return tensorType.getRank() != 1 ||
               !isa<FloatType, IntegerType>(tensorType.getElementType());
      }))
    return sortOp.emitOpError(
        "only 1D tensors of FloatType or IntegerType are supported");

  // Apply type-based scaling to items per thread
  // This optimizes register usage across different data types
  int64_t actualItemsPerThread = config.getActualItemsPerThread(keyType);
  MergeSortConfig scaledConfig = config;
  scaledConfig.itemsPerThread = actualItemsPerThread;

  // Generate the merge sort kernels with scaled configuration
  FailureOr<MergeSortKernelResult> kernelResult =
      MergeSortKernelGenerator::createMergeSortKernels(
          rewriter, loc, keyType, valueType, module, gpuModule, symbolTables,
          scaledConfig);

  if (failed(kernelResult))
    return sortOp.emitOpError(
        "failed to generate sort kernels and dispatch function");

  // Clone the dispatch function into the target module
  func::FuncOp dispatchFunc = kernelResult->dispatchFunc;

  // Set insertion point to the sort op location
  rewriter.setInsertionPoint(sortOp);

  // Get the count (array size)
  auto keysTensor = cast<TypedValue<RankedTensorType>>(inputs[0]);
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value count = rewriter.create<tensor::DimOp>(loc, keysTensor, c0);
  Value countI32 =
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), count);

  // Build arguments for the dispatch function call
  SmallVector<Value> callArgs;
  callArgs.push_back(castToDynamic(rewriter, loc, keysTensor)); // keys
  callArgs.push_back(countI32);                                 // count
  if (!config.keysOnly)
    callArgs.push_back(
        castToDynamic(rewriter, loc,
                      cast<TypedValue<RankedTensorType>>(inputs[1]))); // values

  // Create func.call to the dispatch function
  auto callOp = rewriter.create<func::CallOp>(loc, dispatchFunc, callArgs);

  SmallVector<Value> replacements;
  for (auto [callResult, sortResult] :
       llvm::zip_equal(callOp.getResults(), sortOp->getResults())) {
    if (callResult.getType() == sortResult.getType()) {
      replacements.push_back(callResult);
      continue;
    }
    replacements.push_back(
        rewriter.create<tensor::CastOp>(loc, sortResult.getType(), callResult));
  }
  rewriter.replaceOp(sortOp, replacements);

  return success();
}

namespace {

struct LowerKernelSortPass
    : public kernel::impl::LowerKernelSortPassBase<LowerKernelSortPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    IRRewriter rewriter(ctx);

    ModuleOp module = getOperation();
    SymbolTableCollection symbolTables;
    SymbolTable hostSymTable(module);

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      auto walkResult = func.walk([&](kernel::SortOp sortOp) {
        gpu::GPUModuleOp gpuModule =
            getOrCreateGpuModule(rewriter, module, "merge_sort");
        hostSymTable.insert(gpuModule);
        if (failed(lowerSortOp(rewriter, sortOp, module, gpuModule,
                               symbolTables))) {
          sortOp.emitOpError() << "failed to lower sort op";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted())
        return signalPassFailure();
    }
  }
};

} // namespace
