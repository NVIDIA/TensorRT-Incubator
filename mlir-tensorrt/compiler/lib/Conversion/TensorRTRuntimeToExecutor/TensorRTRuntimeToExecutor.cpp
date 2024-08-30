//===- TensorRTRuntimeToExecutor.cpp --------------------------------------===//
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
#include "mlir-executor/Executor/Utils/Utils.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORRTRUNTIMETOEXECUTORPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::executor;
using namespace mlir::cuda;

static ExecutorOpaqueType getTrtOutputsOpaqueType(MLIRContext *ctx) {
  return ExecutorOpaqueType::get(ctx, "trtrt_outputs");
}
static ExecutorOpaqueType getTrtRuntimeOpaqueType(MLIRContext *ctx) {
  return ExecutorOpaqueType::get(ctx, "trtrt_runtime");
}
static ExecutorOpaqueType getTrtContextOpaqueType(MLIRContext *ctx) {
  return ExecutorOpaqueType::get(ctx, "trtrt_context");
}
static ExecutorOpaqueType getTrtEngineOpaqueType(MLIRContext *ctx) {
  return ExecutorOpaqueType::get(ctx, "trtrt_engine");
}
static PointerType getCudaStreamOpaqueType(MLIRContext *ctx) {
  return PointerType::get(ctx, MemoryType::host);
}

/// Return a symbol reference to a external function declared at top of module,
/// creating a new declaration if necessary.
static ConstantResourceOp
getOrCreateSerializedTrtEngineDeclaration(RewriterBase &rewriter,
                                          func::FuncOp trtFunc) {
  std::string name = (trtFunc.getName() + "_engine_data").str();
  auto parentModule = trtFunc->getParentOfType<ModuleOp>();
  auto trtModule = trtFunc->getParentOfType<tensorrt::TensorRTModuleOp>();
  assert(trtModule && "expected valid tensorrt module");
  auto engineData = trtFunc->getAttrOfType<ElementsAttr>("tensorrt.engine");
  assert(engineData && "expected valid serialized data");
  ConstantResourceOp resourceOp = getOrCreateConstantResourceDeclaration(
      rewriter, trtFunc.getLoc(), parentModule, name, engineData);
  return resourceOp;
}

static GlobalOp getOrCreateRuntimeGlobalOp(RewriterBase &rewriter,
                                           ModuleOp op) {
  std::string name = "tensorrt_runtime";
  Type opaqueType = getTrtRuntimeOpaqueType(rewriter.getContext());
  return getOrCreateGlobalOp(
      rewriter, op.getLoc(), op, name, opaqueType, false,
      [&](OpBuilder &nested, Location loc) {
        ImplicitLocOpBuilder ib(loc, nested);
        Value runtime = ib.create<trtrt::CreateRuntimeOp>();
        auto runtimeCasted =
            ib.create<UnrealizedConversionCastOp>(opaqueType, runtime);
        ib.create<ReturnOp>(runtimeCasted.getResult(0));
      });
}

static ExecutorCallBuilder getLoadEngineBuilder(MLIRContext *ctx) {
  return ExecutorCallBuilder{
      ctx,
      "_trtrt_load",
      executor::ExecutorOpaqueType::get(ctx, "trtrt_engine"),
      {executor::ExecutorOpaqueType::get(ctx, "trtrt_runtime"),
       executor::PointerType::get(ctx, MemoryType::host)}};
}

static ExecutorCallBuilder getCreateExecutionContextBuilder(MLIRContext *ctx) {
  return ExecutorCallBuilder{
      ctx, "_trtrt_create_context",
      executor::ExecutorOpaqueType::get(ctx, "trtrt_context"),
      executor::ExecutorOpaqueType::get(ctx, "trtrt_engine")};
}

/// Create a `executor.global` to load the TensorRT engine/execution context.
static GlobalOp getOrCreateExecutionContextGlobal(RewriterBase &rewriter,
                                                  func::FuncOp trtFunc,
                                                  ConstantResourceOp resourceOp,
                                                  GlobalOp runtimeGlobal) {
  std::string name = (trtFunc.getName() + "_exec_ctx").str();
  auto parentModule = trtFunc->getParentOfType<ModuleOp>();
  SymbolTable symbolTable(parentModule);
  ExecutorOpaqueType execOpaqueType =
      getTrtContextOpaqueType(rewriter.getContext());
  return getOrCreateGlobalOp(
      rewriter, trtFunc.getLoc(), parentModule, name, execOpaqueType, true,
      [&](OpBuilder &nested, Location loc) {
        ImplicitLocOpBuilder ib(loc, nested);
        Value data = ib.create<ConstantResourceLoadOp>(
            FlatSymbolRefAttr::get(resourceOp));
        Value runtime =
            ib.create<GetGlobalOp>(getTrtRuntimeOpaqueType(ib.getContext()),
                                   FlatSymbolRefAttr::get(runtimeGlobal));

        ExecutorCallBuilder callBuilder =
            getLoadEngineBuilder(rewriter.getContext());
        ExecutorCallBuilder createContextBuilder =
            getCreateExecutionContextBuilder(rewriter.getContext());

        Value engine =
            callBuilder
                .create(ib, trtFunc.getLoc(), parentModule, {runtime, data})
                .getResult(0);
        Value context =
            createContextBuilder
                .create(ib, trtFunc.getLoc(), parentModule, {engine})
                .getResult(0);
        ib.create<ReturnOp>(context);
      });
}

namespace {

/// Convert `tensorrt.compile` to a set of globals declarations/global
/// retrievals representing the deserialization of the TensorRT engine and the
/// retrieval of the execution context. The primary prequisite for this pattern
/// is that the serialized engine data must be available on the reference
/// TensorRT engine.
struct ConvertCompile : public ConvertOpToExecutorPattern<trtrt::CompileOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(trtrt::CompileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    func::FuncOp trtFunc =
        dyn_cast_or_null<func::FuncOp>(module.lookupSymbol(op.getTrtFunc()));

    ConstantResourceOp resourceGlobal =
        getOrCreateSerializedTrtEngineDeclaration(rewriter, trtFunc);
    GlobalOp runtimeGlobal = getOrCreateRuntimeGlobalOp(rewriter, module);
    GlobalOp executionContextGlobal = getOrCreateExecutionContextGlobal(
        rewriter, trtFunc, resourceGlobal, runtimeGlobal);
    resourceGlobal->moveAfter(runtimeGlobal);
    executionContextGlobal->moveAfter(resourceGlobal);

    rewriter.setInsertionPoint(op);
    Value context = rewriter.create<executor::GetGlobalOp>(
        op.getLoc(), getTrtContextOpaqueType(getContext()),
        FlatSymbolRefAttr::get(executionContextGlobal));
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, trtrt::ExecutionContextType::get(getContext()), context);
    return success();
  }
};

/// Convert `tensorrt.enqueue` to `executor.call`.
struct ConvertEnqueueToCall
    : public ConvertOpToExecutorPattern<trtrt::EnqueueOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(trtrt::EnqueueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // For now just do simple substitution to get the name.
    std::string funcName;
    funcName =
        "_" + llvm::join(llvm::split(op->getName().getStringRef(), "."), "_");

    SmallVector<Value> newOperands = {adaptor.getExecutionContext(),
                                      adaptor.getStream()};
    SmallVector<Value> argTablePack;

    auto createMemRefAndExractPtr = [&](Value oldVal, Value newVal) {
      auto memrefType = cast<MemRefType>(oldVal.getType());
      if (!memrefType)
        return failure();
      assert(isa<TableType>(newVal.getType()));
      executor::MemRefDescriptor memref(newVal, memrefType);
      Value offset =
          convertOffsetInElementsToBytes(b, memref.offset(b), memrefType);

      // Append the aligned pointer and offset.
      argTablePack.append({memref.alignedPtr(b), offset});

      // Append the rank followed by the shape integers.
      argTablePack.push_back(
          this->createIndexConstant(b, memrefType.getRank()));
      for (int64_t i = 0; i < memrefType.getRank(); i++)
        argTablePack.push_back(memref.size(b, i));

      return success();
    };

    for (auto [oldVal, newVal] :
         llvm::zip(op.getInputs(), adaptor.getInputs())) {
      if (failed(createMemRefAndExractPtr(oldVal, newVal)))
        return failure();
    }

    // Create the table containing the pointer/offset args and append it to the
    // arguments for the call op.
    Value args = b.create<executor::CreateTableOp>(
        executor::TableType::get(rewriter.getContext(),
                                 llvm::to_vector(TypeRange(argTablePack))),
        argTablePack);
    newOperands.push_back(args);

    SmallVector<Type, 4> resultTypes(op->getResultTypes().begin(), op->getResultTypes().end());

    auto parentModule = op->getParentOfType<ModuleOp>();
    auto enqueueFunc = getOrInsertFuncDeclaration(
        rewriter, op.getLoc(), parentModule, funcName,
        ExecutorFunctionType::get(rewriter.getContext(),
                                  {adaptor.getExecutionContext().getType(),
                                   adaptor.getStream().getType()},
                                  resultTypes, rewriter.getUnitAttr()));

    rewriter.replaceOpWithNewOp<CallOp>(
        op, op->getResultTypes(), enqueueFunc.getLeafReference(), newOperands);

    return success();
  }
};

class RemoveBufferizationClonePattern : public OpRewritePattern<bufferization::CloneOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::CloneOp op,
                                PatternRewriter &rewriter) const override {
    // Replace all uses of the clone op with its input
    rewriter.replaceAllUsesWith(op.getResult(), op.getInput());
    
    // Erase the clone op
    rewriter.eraseOp(op);
    
    return success();
  }
};

struct ConvertTrtrtOpToCall : public ConvertToExecutorPattern {
  ConvertTrtrtOpToCall(ExecutorTypeConverter &typeConverter,
                       MLIRContext *context, PatternBenefit benefit = 1)
      : ConvertToExecutorPattern(typeConverter, MatchAnyOpTypeTag(), benefit,
                                 context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<trtrt::CreateRuntimeOp>(op))
      return failure();

    SmallVector<Type> newResultTypes =
        llvm::to_vector(llvm::map_range(op->getResultTypes(), [&](Type t) {
          auto result = getTypeConverter()->convertType(t);
          assert(result && "expected converted type");
          return result;
        }));
    SmallVector<Type> newArgTypes = llvm::to_vector(TypeRange(operands));

    auto funcType = ExecutorFunctionType::get(getContext(), newArgTypes,
                                              newResultTypes, UnitAttr());

    std::string funcName;
    funcName =
        "_" + llvm::join(llvm::split(op->getName().getStringRef(), "."), "_");
    auto module = op->getParentOfType<ModuleOp>();
    SymbolRefAttr callee = executor::getOrInsertFuncDeclaration(
        rewriter, op->getLoc(), module, funcName, funcType);

    rewriter.replaceOpWithNewOp<executor::CallOp>(
        op, newResultTypes, callee.getLeafReference(), operands);

    return success();
  }
};

void populateRemoveBufferizationClonePatterns(RewritePatternSet &patterns) {
  patterns.add<RemoveBufferizationClonePattern>(patterns.getContext());
}


} // namespace

namespace {
class TensorRTRuntimeToExecutorPass
    : public mlir::impl::ConvertTensorRTRuntimeToExecutorPassBase<
          TensorRTRuntimeToExecutorPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LowerToExecutorOptions opts;
    opts.indexType = IntegerType::get(ctx, indexBitwidth);
    opts.memrefArgPassingConvention =
        usePackedMemRefCConv ? MemRefArgPassingConvention::Packed
                             : MemRefArgPassingConvention::Unpacked;
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

    typeConverter.addConversion([](trtrt::ExecutionContextType t) {
      return getTrtContextOpaqueType(t.getContext());
    });
    typeConverter.addConversion([](trtrt::RuntimeType t) {
      return getTrtRuntimeOpaqueType(t.getContext());
    });
    typeConverter.addConversion([](trtrt::EngineType t) {
      return getTrtEngineOpaqueType(t.getContext());
    });
    typeConverter.addConversion([](cuda::StreamType t) {
      return getCudaStreamOpaqueType(t.getContext());
    });

    // Convert `trtrt.compile` to globals that create execution context from
    // serialized TensorRT engine data.
    {
      ConversionTarget target(*ctx);
      target.addIllegalOp<trtrt::CompileOp>();
      target.addLegalDialect<ExecutorDialect, trtrt::TensorRTRuntimeDialect>();
      target.addLegalOp<UnrealizedConversionCastOp>();

      RewritePatternSet patterns(&getContext());
      patterns.add<ConvertCompile>(typeConverter, ctx);
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        return signalPassFailure();
    }

    {
      RewritePatternSet patterns(&getContext());
      populateRemoveBufferizationClonePatterns(patterns);

      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                        std::move(patterns))))
        return signalPassFailure();
    }

    // Convert `trtrt.enqueue|create_runtime|execution_context|load` to
    // `executor.call` and function declarations.
    {
      ConversionTarget target(*ctx);
      target.addIllegalOp<trtrt::EnqueueOp, trtrt::CompileOp,
                          trtrt::CreateRuntimeOp>();
      target.addLegalDialect<ExecutorDialect, CUDADialect>();

      RewritePatternSet patterns(&getContext());
      patterns.add<ConvertEnqueueToCall, ConvertTrtrtOpToCall>(typeConverter,
                                                               ctx);
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        return signalPassFailure();
    }
  }
};
} // namespace
