//===- KernelToCUDA.cpp  ----------------------------------------------===//
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
///  Implementation of `convert-kernel-to-cuda` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTKERNELTOCUDAPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::cuda;

static constexpr StringRef kPTXDataAttrName = "kernel.ptx_data";

/// Return a symbol reference to a external function declared at top of module,
/// creating a new declaration if necessary.
static std::optional<ElementsAttr> getPtxData(gpu::GPUModuleOp module) {
  std::string name = (module.getName() + "_ptx_data").str();
  Attribute ptxData = module->getAttr(kPTXDataAttrName);
  if (!ptxData)
    return std::nullopt;
  return cast<ElementsAttr>(ptxData);
}

/// Retrieve the `cuda.compiled_module` with the specified name if it already
/// exists, otherwise create it.
static cuda::CompiledModuleOp createCompiledModuleOp(RewriterBase &rewriter,
                                                     ElementsAttr data,
                                                     gpu::GPUModuleOp module,
                                                     SymbolTable &symbolTable) {
  std::string name = (module.getName() + "_cuModule").str();
  unsigned counter = 0;
  SmallString<32> uniqueName = symbolTable.generateSymbolName<32>(
      name,
      [&](StringRef candidate) { return symbolTable.lookup(name) != nullptr; },
      counter);

  auto globalOp = rewriter.create<cuda::CompiledModuleOp>(module.getLoc(),
                                                          uniqueName, data);
  symbolTable.insert(globalOp);
  return globalOp;
}

/// Verify that all `gpu.module` operations have the serialized PTX data
/// required for this pass to run.
static LogicalResult verifyKernelModulesHavePtxResource(Operation *op) {
  auto walkResult = op->walk([](gpu::GPUModuleOp kernelModule) {
    std::optional<ElementsAttr> ptxData = getPtxData(kernelModule);
    if (!ptxData) {
      emitError(kernelModule.getLoc())
          << gpu::GPUModuleOp::getOperationName() << " \""
          << kernelModule.getSymName() << "\" is missing serialized PTX IR";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
}

static LogicalResult decomposeMemRef(OpBuilder &b, Location loc,
                                     TypedValue<MemRefType> value,
                                     SmallVectorImpl<Value> &dynamicValues) {
  auto extractOp = b.create<memref::ExtractStridedMetadataOp>(loc, value);
  dynamicValues.push_back(extractOp.getBaseBuffer());
  if (isa<Value>(extractOp.getConstifiedMixedOffset()))
    dynamicValues.push_back(extractOp.getOffset());
  for (OpFoldResult size : extractOp.getConstifiedMixedSizes())
    if (isa<Value>(size))
      dynamicValues.push_back(cast<Value>(size));
  for (OpFoldResult stride : extractOp.getConstifiedMixedStrides())
    if (isa<Value>(stride))
      dynamicValues.push_back(cast<Value>(stride));
  return success();
}

namespace {

template <typename OpType>
struct KernelCallLikeOpConversionPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  KernelCallLikeOpConversionPattern(
      llvm::SmallDenseMap<gpu::GPUModuleOp, cuda::CompiledModuleOp> &map,
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<OpType>(typeConverter, ctx),
        kernelModuleToCudaBinary(map) {}

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumResults() > 0)
      return rewriter.notifyMatchFailure(op,
                                         "only allowed after bufferization");

    Location loc = op.getLoc();
    SymbolTableCollection collection;
    FunctionOpInterface funcOp = op.getKernelCallee(collection);
    auto kernelModule = funcOp->getParentOfType<gpu::GPUModuleOp>();
    assert(kernelModule &&
           "could not find KernelModule associated with kernel.call callee");

    cuda::CompiledModuleOp moduleGlobal =
        kernelModuleToCudaBinary.lookup(kernelModule);
    Value cudaFunc = rewriter.create<cuda::GetFunctionOp>(
        loc, FlatSymbolRefAttr::get(moduleGlobal),
        op.getKernelSym().getLeafReference());

    Type cudaIndexType = rewriter.getI32Type();
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(cudaIndexType, 1));
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(cudaIndexType, 0));

    // Utility to pad the grid/block shape out to exactly three values. The
    // verifier of `kernel.call` ensures the number of vals is in [1, 3].
    // Depending on index type conversion, these may be i64 integers, but we
    // require i32 for the executor op. Truncate if required.
    auto padToThreeValues = [&](ValueRange vals) {
      SmallVector<Value> padded =
          llvm::map_to_vector(vals, [&](Value v) -> Value {
            return rewriter.create<arith::IndexCastOp>(
                loc, rewriter.getI32Type(), v);
          });
      if (padded.size() == 3)
        return padded;
      padded.append(3 - padded.size(), one);
      return padded;
    };

    // Our ABI currently is that memrefs are passed in unpacked form, and only
    // dynamic values are passed.
    SmallVector<Value> args;
    for (Value arg : op.getArgOperands()) {
      auto memrefType = dyn_cast<MemRefType>(arg.getType());
      if (!memrefType) {
        args.push_back(arg);
        continue;
      }
      if (failed(decomposeMemRef(rewriter, loc,
                                 cast<TypedValue<MemRefType>>(arg), args)))
        return failure();
    }

    SmallVector<Value> gridShape = padToThreeValues(adaptor.getGridSize());
    SmallVector<Value> blockShape = padToThreeValues(adaptor.getBlockSize());

    // Enqueue the kernel on on the default stream.
    Value device = rewriter.create<cuda::GetActiveDeviceOp>(loc);
    Value stream = rewriter.create<cuda::GetGlobalStreamOp>(loc, device, 0);
    rewriter.replaceOpWithNewOp<cuda::LaunchOp>(
        op, cudaFunc, gridShape[0], gridShape[1], gridShape[2], blockShape[0],
        blockShape[1], blockShape[2],
        /*dynamic_shared_mem=*/zero, stream, args);
    return success();
  }

private:
  llvm::SmallDenseMap<gpu::GPUModuleOp, cuda::CompiledModuleOp>
      &kernelModuleToCudaBinary;
};

class ConvertKernelToCUDAPass
    : public mlir::impl::ConvertKernelToCUDAPassBase<ConvertKernelToCUDAPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp op = getOperation();
    MLIRContext *ctx = &getContext();

    if (failed(verifyKernelModulesHavePtxResource(op)))
      return signalPassFailure();

    TypeConverter typeConverter;
    typeConverter.addConversion([&](Type t) { return t; });
    ConversionTarget target(*ctx);
    // Kernel modules and anything nested under them are legal. All other kernel
    // ops are illegal.
    target.addDynamicallyLegalDialect<gpu::GPUDialect>(
        [](Operation *op) { return isa<gpu::GPUModuleOp>(op); });
    target.addLegalDialect<cuda::CUDADialect, arith::ArithDialect,
                           memref::MemRefDialect>();

    // Tracks mappings `gpu.module` to `cuda.compiled_module` ops.
    llvm::SmallDenseMap<gpu::GPUModuleOp, cuda::CompiledModuleOp>
        kernelModuleToCudaBinary;

    SymbolTable symbolTable(op);
    IRRewriter rewriter(ctx);
    for (auto kernelModule : op.getOps<gpu::GPUModuleOp>()) {
      rewriter.setInsertionPointToStart(op.getBody());
      std::optional<ElementsAttr> ptxData = getPtxData(kernelModule);
      if (!ptxData) {
        emitError(kernelModule.getLoc())
            << kernelModule.getSymName() << " is missing serialized PTX IR";
        return signalPassFailure();
      }
      cuda::CompiledModuleOp compiledModuleOp =
          createCompiledModuleOp(rewriter, *ptxData, kernelModule, symbolTable);
      kernelModuleToCudaBinary[kernelModule] = compiledModuleOp;
    }

    RewritePatternSet patterns(ctx);
    patterns.insert<KernelCallLikeOpConversionPattern<kernel::CallOp>,
                    KernelCallLikeOpConversionPattern<kernel::ExtCallOp>>(
        kernelModuleToCudaBinary, typeConverter, ctx);
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      emitError(op->getLoc())
          << "failed to run " << getArgument() << " conversion patterns";
      return signalPassFailure();
    }
  }
};
} // namespace
