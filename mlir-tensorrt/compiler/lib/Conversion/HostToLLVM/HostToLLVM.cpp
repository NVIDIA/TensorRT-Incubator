//===- HostToLLVM.cpp -----------------------------------------------------===//
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
/// Implementation of the `convert-host-to-llvm` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-tensorrt/Conversion/CUDAToLLVM/CUDAToLLVM.h"
#include "mlir-tensorrt/Conversion/LLVMCommon/LLVMCommon.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTHOSTTOLLVMPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct LowerPrintOp : public ConvertOpToLLVMPattern<executor::PrintOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(executor::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getFormat())
      return failure();
    MLIRContext *ctx = op->getContext();
    FailureOr<LLVM::LLVMFuncOp> printFunc = LLVM::lookupOrCreateFn(
        rewriter, op->getParentOfType<ModuleOp>(), "printf",
        {LLVM::LLVMPointerType::get(ctx)}, rewriter.getI32Type(), true);
    if (failed(printFunc))
      return failure();
    Value str = insertLLVMStringLiteral(
        rewriter, op.getLoc(), (*op.getFormat() + "\n").str(), "literal");
    SmallVector<Value> args = {str};
    for (Value v : adaptor.getArguments()) {
      // Any float value should be extended to f64 (I guess this is inserted
      // automatically by any modern C/C++ compiler).
      if (isa<FloatType>(v.getType()) &&
          v.getType().getIntOrFloatBitWidth() < 64)
        v = rewriter.create<LLVM::FPExtOp>(v.getLoc(), rewriter.getF64Type(),
                                           v);
      args.push_back(v);
    }
    rewriter.create<LLVM::CallOp>(op.getLoc(), *printFunc, args);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

/// Cleanup attributes that are not used by the LLVM dialect.
static void cleanupPlanDialectModuleAttributes(ModuleOp module) {
  SmallVector<StringRef> attributesToRemove = {
      mlir::plan::PlanDialect::kBackendsAttrName,
      mlir::plan::PlanDialect::kMemorySpaceConstraintAttrName,
  };

  for (auto attr : module->getDiscardableAttrs()) {
    if (llvm::is_contained(attributesToRemove, attr.getName())) {
      module->removeAttr(attr.getName());
    }
  }
}

namespace {
class HostToLLVMPass
    : public mlir::impl::ConvertHostToLLVMPassBase<HostToLLVMPass> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
    registerConvertToLLVMDependentDialectLoading(registry);
  }

  // Create the rewrite pattern set using all loaded dialects.
  LogicalResult initialize(MLIRContext *context) final {
    RewritePatternSet tempPatterns(context);
    auto target = std::make_shared<ConversionTarget>(*context);
    target->addLegalDialect<LLVM::LLVMDialect>();
    target->addIllegalDialect<cuda::CUDADialect>();
    target->addIllegalDialect<trtrt::TensorRTRuntimeDialect>();
    auto typeConverter = std::make_shared<LLVMTypeConverter>(context);

    // Normal mode: Populate all patterns from all dialects that implement the
    // interface.
    for (Dialect *dialect : context->getLoadedDialects()) {
      // First time we encounter this dialect: if it implements the interface,
      // let's populate patterns !
      auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
      if (!iface)
        continue;
      iface->populateConvertToLLVMConversionPatterns(*target, *typeConverter,
                                                     tempPatterns);
    }

    // Add a lowering pattern for `executor.print` since most of our tests rely
    // on this.
    // TODO: fix this so that this pass doesn't depend on anything in
    // "executor".
    tempPatterns.add<LowerPrintOp>(*typeConverter);
    target->addIllegalOp<executor::PrintOp>();

    this->patterns =
        std::make_unique<FrozenRewritePatternSet>(std::move(tempPatterns));
    this->target = target;
    this->typeConverter = typeConverter;
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(IRRewriter::atBlockEnd(module.getBody()));
    SymbolTableCollection symbolTables;
    if (failed(lowerCUDAGlobalsToLLVM(rewriter, module, symbolTables)))
      return signalPassFailure();

    if (failed(applyPartialConversion(getOperation(), *target, *patterns))) {
      emitError(getOperation()->getLoc())
          << "failed to apply conversion in " << getArgument();
      return signalPassFailure();
    }

    cleanupPlanDialectModuleAttributes(module);
  }

  std::shared_ptr<const FrozenRewritePatternSet> patterns;
  std::shared_ptr<const ConversionTarget> target;
  std::shared_ptr<const LLVMTypeConverter> typeConverter;
};
} // namespace
