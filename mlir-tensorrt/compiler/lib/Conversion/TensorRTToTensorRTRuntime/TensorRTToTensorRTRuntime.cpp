//===- TensorRTToTensorRTRuntime.cpp ----------------------------*- C++ -*-===//
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
/// Implementation of a pass that converts TensorRT operations in the "host IR
/// module" to TensorRTRuntime and Executor dialect operations.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt-dialect/Interface/TensorKindOpInterface.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORRTTOTENSORRTRUNTIMEPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::cuda;

namespace {
/// Rewrite `tensorrt.constant` to `arith.constant`.
struct RewriteConstants : public OpRewritePattern<tensorrt::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    // Only convert ops not nested in a TensorRT module.
    if (op->getParentOfType<tensorrt::TensorRTModuleOp>())
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getWeights());
    return success();
  }
};

struct TensorRTCallAndEngineConverter {
  TensorRTCallAndEngineConverter(ModuleOp module);
  ModuleOp module;
  MLIRContext *ctx{module->getContext()};
  SymbolTableCollection symbolTables;
  SymbolTable &symbolTable{symbolTables.getSymbolTable(module)};
  SymbolUserMap userMap{symbolTables, module};
  IRRewriter rewriter{ctx};

  LogicalResult convert();
  LogicalResult convert(tensorrt::TensorRTModuleOp op);
  LogicalResult convert(func::FuncOp func, ArrayRef<int64_t> hostTensorArgs);
  LogicalResult convert(tensorrt::CallOp callOp, trtrt::CompiledFuncOp globalOp,
                        ArrayRef<int64_t> hostTensorArgs);
  LogicalResult convert(tensorrt::CallAllocOp callOp,
                        trtrt::CompiledFuncOp globalOp,
                        ArrayRef<int64_t> hostTensorArgs);
};
} // namespace

/// Return a symbol reference to a external function declared at top of module,
/// creating a new declaration if necessary.
static FailureOr<trtrt::CompiledFuncOp>
createSerializedEngineGlobal(RewriterBase &rewriter, ModuleOp module,
                             SymbolTable &symbolTable, func::FuncOp trtFunc) {
  OpBuilder::InsertionGuard g(rewriter);
  std::string name = (trtFunc.getName() + "_engine_data").str();
  assert(trtFunc->getParentOfType<tensorrt::TensorRTModuleOp>() &&
         "expected valid tensorrt module");
  auto engineData = trtFunc->getAttrOfType<ElementsAttr>("tensorrt.engine");
  if (!engineData)
    return trtFunc->emitError("TensorRT function has not been translated");
  rewriter.setInsertionPointToStart(module.getBody());
  auto result = rewriter.create<trtrt::CompiledFuncOp>(
      trtFunc.getLoc(), (trtFunc.getName() + "_engine_data").str(), engineData);
  symbolTable.insert(result);
  return result;
}

TensorRTCallAndEngineConverter::TensorRTCallAndEngineConverter(ModuleOp module)
    : module(module) {}

LogicalResult
TensorRTCallAndEngineConverter::convert(tensorrt::CallOp op,
                                        trtrt::CompiledFuncOp globalOp,
                                        ArrayRef<int64_t> hostTensorArgs) {
  Location loc = op.getLoc();
  Value executionContext =
      rewriter.create<trtrt::GetFunctionOp>(loc, globalOp.getSymName());
  Value stream = rewriter.create<cuda::GetGlobalStreamOp>(loc, 0);
  ValueRange inputs = op.getInputs();
  ValueRange outputs = op.getOutputs();

  auto enqueueOp = rewriter.create<trtrt::EnqueueOp>(
      loc, executionContext, stream, inputs, outputs,
      /*host_tensors_args=*/hostTensorArgs.empty()
          ? DenseI64ArrayAttr{}
          : DenseI64ArrayAttr::get(ctx, hostTensorArgs));
  rewriter.replaceOp(op, enqueueOp->getResults());
  return success();
}

LogicalResult
TensorRTCallAndEngineConverter::convert(tensorrt::CallAllocOp op,
                                        trtrt::CompiledFuncOp globalOp,
                                        ArrayRef<int64_t> hostTensorArgs) {
  Location loc = op.getLoc();
  Value executionContext =
      rewriter.create<trtrt::GetFunctionOp>(loc, globalOp.getSymName());
  Value stream = rewriter.create<cuda::GetGlobalStreamOp>(loc, 0);
  ValueRange inputs = op.getInputs();
  auto enqueueOp = rewriter.create<trtrt::EnqueueAllocOp>(
      loc, op.getResultTypes(), executionContext, stream, inputs,
      /*host_tensors_args=*/hostTensorArgs.empty()
          ? DenseI64ArrayAttr{}
          : DenseI64ArrayAttr::get(ctx, hostTensorArgs));
  rewriter.replaceOp(op, enqueueOp->getResults());
  return success();
}

LogicalResult
TensorRTCallAndEngineConverter::convert(func::FuncOp func,
                                        ArrayRef<int64_t> hostTensorArgs) {
  FailureOr<trtrt::CompiledFuncOp> globalOp =
      createSerializedEngineGlobal(rewriter, module, symbolTable, func);
  if (failed(globalOp))
    return failure();

  for (Operation *user : userMap.getUsers(func)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(user);
    if (auto callOp = dyn_cast<tensorrt::CallOp>(user)) {
      if (failed(convert(callOp, *globalOp, hostTensorArgs)))
        return failure();
      continue;
    }
    if (auto callOp = dyn_cast<tensorrt::CallAllocOp>(user)) {
      if (failed(convert(callOp, *globalOp, hostTensorArgs)))
        return failure();
      continue;
    }
    return user->emitError(
        "unexpected dangling reference to TensorRT function");
  }

  // There are no longer references to the original func, we can remove it.
  rewriter.eraseOp(func);
  return success();
}

LogicalResult
TensorRTCallAndEngineConverter::convert(tensorrt::TensorRTModuleOp op) {
  /// TODO: there is an upstream bug where the dataflow is not properly
  /// initialized if the root op is a "function", in only works when the root op
  /// is a "module". If that is fixed, then we could move this into the func
  /// converter for simplicity.
  DataFlowConfig config;
  config.setInterprocedural(false);
  DataFlowSolver solver(config);
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  SymbolTableCollection collection;
  solver.load<TensorKindAnalysis>(collection);
  if (failed(solver.initializeAndRun(op)))
    return op.emitError() << "failed to run TensorKindAnalysis";

  for (func::FuncOp func :
       llvm::make_early_inc_range(op.getOps<func::FuncOp>())) {
    SmallVector<int64_t> hostTensorArgs;
    for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
      const TensorKindLattice *kind =
          solver.lookupState<TensorKindLattice>(arg);
      RankedTensorType rtt = cast<RankedTensorType>(arg.getType());
      // To be conservative, we only do this if type is i32 and num elements
      // <= 8.
      if (kind && !kind->getValue().isUninitialized() &&
          kind->getValue().isHostVisible() &&
          rtt.getElementType().isInteger(32) && rtt.getNumElements() <= 8)
        hostTensorArgs.push_back(idx);
    }
    if (failed(convert(func, hostTensorArgs)))
      return failure();
  }

  // All references to the module op should be gone. We can't use the original
  // symbol user map because it is now invalidated.
  std::optional<SymbolTable::UseRange> remainingUses =
      SymbolTable::getSymbolUses(op, module);
  if (!remainingUses || remainingUses->empty())
    rewriter.eraseOp(op);

  return success();
}

LogicalResult TensorRTCallAndEngineConverter::convert() {
  for (tensorrt::TensorRTModuleOp trtModule : llvm::make_early_inc_range(
           module.getOps<tensorrt::TensorRTModuleOp>())) {
    if (failed(convert(trtModule)))
      return emitError(trtModule.getLoc())
             << "failed to convert 'tensorrt.module' operation";
  }
  return success();
}

namespace {
class ConvertTensorRTToRuntimePass
    : public mlir::impl::ConvertTensorRTToTensorRTRuntimePassBase<
          ConvertTensorRTToRuntimePass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // Rewrite `tensorrt.constant` to `arith.constant`.
    RewritePatternSet patterns(ctx);
    patterns.add<RewriteConstants>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }

    TensorRTCallAndEngineConverter converter(module);
    if (failed(converter.convert()))
      return signalPassFailure();
  }
};
} // namespace
