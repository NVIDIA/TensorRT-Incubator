//===- RemoveEquivalentBufferResults.cpp ---------------------------------===//
//
// Modified from upstream 'DropEquivalentBufferResults.cpp', part of the LLVM
// Project, under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Utils/ModuleUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::plan {
#define GEN_PASS_DEF_PLANREMOVEEQUIVALENTBUFFERRESULTSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

/// Drop function buffer results that are equivalent to block arguments.
/// TODO: this function logic is borrowed from upstream because the upstream
/// does not support invoking `mlir::dropEquivalentBufferResults` on ops other
/// than `builtin.module`.
static LogicalResult
dropEquivalentFuncBufferResults(RewriterBase &rewriter, func::FuncOp funcOp,
                                const SymbolUserMap &symbolUseMap) {
  if (funcOp.isDeclaration() || !funcOp.getBody().hasOneBlock())
    return success();

  func::ReturnOp returnOp =
      cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());

  // Compute erased results.
  SmallVector<Value> newReturnValues;
  BitVector erasedResultIndices(funcOp.getFunctionType().getNumResults());
  DenseMap<int64_t, int64_t> resultToArgs;
  for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
    bool erased = false;
    for (BlockArgument bbArg : funcOp.getArguments()) {
      Value val = it.value();
      while (auto castOp = val.getDefiningOp<memref::CastOp>())
        val = castOp.getSource();

      if (val == bbArg) {
        resultToArgs[it.index()] = bbArg.getArgNumber();
        erased = true;
        break;
      }
    }

    if (erased) {
      erasedResultIndices.set(it.index());
    } else {
      newReturnValues.push_back(it.value());
    }
  }

  // Update function.
  funcOp.eraseResults(erasedResultIndices);
  returnOp.getOperandsMutable().assign(newReturnValues);

  // Update function calls.
  for (Operation *symbolUser : symbolUseMap.getUsers(funcOp)) {
    func::CallOp callOp = dyn_cast<func::CallOp>(symbolUser);
    if (!callOp)
      continue;

    rewriter.setInsertionPoint(callOp);
    auto newCallOp = rewriter.create<func::CallOp>(callOp.getLoc(), funcOp,
                                                   callOp.getOperands());
    SmallVector<Value> newResults;
    int64_t nextResult = 0;
    for (int64_t i = 0; i < callOp.getNumResults(); ++i) {
      if (!resultToArgs.count(i)) {
        // This result was not erased.
        newResults.push_back(newCallOp.getResult(nextResult++));
        continue;
      }

      // This result was erased.
      Value replacement = callOp.getOperand(resultToArgs[i]);
      Type expectedType = callOp.getResult(i).getType();
      if (replacement.getType() != expectedType) {
        // A cast must be inserted at the call site.
        replacement = rewriter.create<memref::CastOp>(
            callOp.getLoc(), expectedType, replacement);
      }
      newResults.push_back(replacement);
    }
    rewriter.replaceOp(callOp, newResults);
  }

  return success();
}

/// Drop function buffer results that are equivalent to block arguments.
static LogicalResult
dropEquivalentFuncBufferResults(RewriterBase &rewriter, ModuleLikeOp module,
                                const SymbolUserMap &symbolUseMap) {
  for (auto funcOp : module.getOps<func::FuncOp>())
    if (failed(dropEquivalentFuncBufferResults(rewriter, funcOp, symbolUseMap)))
      return failure();

  return success();
}

namespace {
struct PlanRemoveEquivalentBufferResultsPass
    : public plan::impl::PlanRemoveEquivalentBufferResultsPassBase<
          PlanRemoveEquivalentBufferResultsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    SymbolTableCollection symbolTables;
    SymbolUserMap userMap(symbolTables, op);
    IRRewriter rewriter(op->getContext());

    auto walkResult = op->walk([&](Operation *nested) {
      if (ModuleLikeOp(nested)) {
        if (failed(dropEquivalentFuncBufferResults(
                rewriter, ModuleLikeOp(nested), userMap)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace
