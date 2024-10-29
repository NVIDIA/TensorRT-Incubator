//===- TensorRTToTensorRTRuntime.cpp ----------------------------*- C++ -*-===//
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
    return failure();
  }
};

class ConvertTensorRTToRuntimePass
    : public mlir::impl::ConvertTensorRTToTensorRTRuntimePassBase<
          ConvertTensorRTToRuntimePass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // Rewrite `tensorrt.constant` to `arith.constant`.
    RewritePatternSet patterns(ctx);
    patterns.add<RewriteConstants>(ctx);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }

    IRRewriter rewriter(ctx);

    SmallVector<tensorrt::TensorRTModuleOp> trtModules =
        llvm::to_vector(module.getOps<tensorrt::TensorRTModuleOp>());
    if (trtModules.empty())
      return;
    if (trtModules.size() != 1) {
      emitError(module.getLoc())
          << "at most one embedded tensorrt.module is supported";
      return signalPassFailure();
    }

    SmallVector<tensorrt::CallOp> callOps;
    module.walk(
        [&](tensorrt::CallOp compileOp) { callOps.push_back(compileOp); });

    for (auto callOp : llvm::make_early_inc_range(callOps)) {
      Location loc = callOp.getLoc();
      func::FuncOp trtFunc = dyn_cast_or_null<func::FuncOp>(
          module.lookupSymbol(callOp.getCallee()));

      SymbolTableCollection symbolTable;
      DataFlowSolver solver;
      solver.load<dataflow::DeadCodeAnalysis>();
      solver.load<dataflow::SparseConstantPropagation>();
      solver.load<TensorKindAnalysis>(symbolTable);
      if (failed(solver.initializeAndRun(trtFunc))) {
        trtFunc.emitError() << "failed to run TensorKindAnalysis";
        return signalPassFailure();
      }

      // Check which tensors should be host tensors.
      SmallVector<int64_t> hostTensorArgs;
      for (auto [idx, arg] : llvm::enumerate(trtFunc.getArguments())) {
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

      rewriter.setInsertionPoint(callOp);
      Value executionContext = rewriter.create<trtrt::CompileOp>(
          loc, SymbolRefAttr::get(
                   rewriter.getStringAttr(trtModules.front().getSymName()),
                   {FlatSymbolRefAttr::get(trtFunc)}));
      Value stream = rewriter.create<cuda::GetGlobalStreamOp>(loc, 0);
      auto enqueueOp = rewriter.create<trtrt::EnqueueOp>(
          loc, executionContext, stream, callOp.getInputs(),
          callOp.getOutputs(),
          /*host_tensors_args=*/hostTensorArgs.empty()
              ? DenseI64ArrayAttr{}
              : DenseI64ArrayAttr::get(ctx, hostTensorArgs));
      rewriter.setInsertionPointAfter(enqueueOp);
      rewriter.replaceOp(callOp, enqueueOp->getResults());
    }
  }
};
} // namespace
