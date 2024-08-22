//===- LowerToRuntimeBuiltins.cpp -----------------------------------------===//
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
#include "mlir-executor/Conversion/ConvertToExecutorCommon.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::executor {
#define GEN_PASS_DEF_EXECUTORLOWERTORUNTIMEBUILTINSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using namespace mlir::executor;

/// Replace any op with `LowerToFuncCallTrait` with a `func.call` operation.
struct LowerToFuncCallTraitPattern
    : OpTraitConversionPattern<LowerToFuncCallTrait> {
  using OpTraitConversionPattern::OpTraitConversionPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();
    // For now just do simple substitution to get the name.
    std::string funcName =
        llvm::join(llvm::split(op->getName().getStringRef(), "."), "_");

    // Insert the declaration.
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes)))
      return failure();

    mlir::SymbolRefAttr ref = [&] {
      auto *context = moduleOp.getContext();
      if (moduleOp.lookupSymbol<executor::FuncOp>(funcName))
        return SymbolRefAttr::get(context, funcName);

      // Insert the private function declaration into the body of the parent
      // module.
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto funcOp = rewriter.create<FuncOp>(
          op->getLoc(), funcName,
          ExecutorFunctionType::get(context,
                                    llvm::to_vector(TypeRange(adaptor)),
                                    resultTypes, UnitAttr{}));
      funcOp.setSymVisibility("private");
      return SymbolRefAttr::get(context, funcName);
    }();
    // Replace with call.
    rewriter.replaceOpWithNewOp<executor::CallOp>(
        op, resultTypes, ref.getLeafReference(), adaptor);
    return success();
  }
};

/// Convert operations that are 'lowerable to runtime builtin' to an
/// `executor.call` operation while also creating the `executor.func`
/// declaration if it does not already exist. This function assumes that the
/// desired function name is `executor_[op mnemonic]_[type1]_..._[typeN]` where
/// all types are simple scalar types.
struct LowerOpToBuiltin
    : public OpInterfaceConversionPattern<RuntimeBuiltinInterface> {

  LowerOpToBuiltin(ExecutorTypeConverter &typeConverter, MLIRContext *ctx,
                   PatternBenefit benefit = 1)
      : OpInterfaceConversionPattern(typeConverter, ctx, benefit),
        executorTypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(RuntimeBuiltinInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    FailureOr<CallOpInterface> callOp =
        op.lowerToCall(operands, rewriter, moduleOp, *typeConverter,
                       executorTypeConverter.getDataLayout());
    if (failed(callOp))
      return failure();
    rewriter.replaceOp(op, *callOp);
    return success();
  }

  const ExecutorTypeConverter &executorTypeConverter;
};

namespace {
class LowerToRuntimeBuiltinsPass
    : public executor::impl::ExecutorLowerToRuntimeBuiltinsPassBase<
          LowerToRuntimeBuiltinsPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalDialect<executor::ExecutorDialect>(
        [&](Operation *op) {
          return !op->hasTrait<LowerToFuncCallTrait>() &&
                 !isa<RuntimeBuiltinInterface>(op);
        });
    LowerToExecutorOptions opts;
    opts.indexType = IntegerType::get(ctx, indexBitwidth);
    opts.memrefArgPassingConvention =
        usePackedMemRefCConv ? executor::MemRefArgPassingConvention::Packed
                             : executor::MemRefArgPassingConvention::Unpacked;
    Operation *op = getOperation();
    FailureOr<DataLayout> dataLayout =
        executor::setDataLayoutSpec(op, indexBitwidth, 64);
    if (failed(dataLayout)) {
      emitError(op->getLoc())
          << "failed to set DataLayout; op has DLTI spec that is "
             "inconsistent with provided options";
      return signalPassFailure();
    }
    ExecutorTypeConverter typeConverter(ctx, opts, std::move(*dataLayout));
    RewritePatternSet patterns(ctx);
    patterns.add<LowerOpToBuiltin, LowerToFuncCallTraitPattern>(typeConverter,
                                                                ctx);
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      emitError(op->getLoc())
          << "failed to apply conversion in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
