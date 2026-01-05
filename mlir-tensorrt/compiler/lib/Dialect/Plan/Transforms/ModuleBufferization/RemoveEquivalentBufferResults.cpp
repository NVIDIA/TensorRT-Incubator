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
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-tensorrt-common/Utils/ModuleUtils.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::plan {
#define GEN_PASS_DEF_PLANREMOVEEQUIVALENTBUFFERRESULTSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

/// Given a memref value, look back through all view-like operations. This may
/// return an `!executor.ptr` value or another memref value.
static Value lookBackThroughView(Value val) {
  while (true) {
    if (auto viewOp = val.getDefiningOp<ViewLikeOpInterface>()) {
      val = viewOp.getViewSource();
      continue;
    }
    if (auto sendOp = val.getDefiningOp<executor::ABISendOp>()) {
      val = sendOp.getValue();
      continue;
    }
    if (auto recvOp = val.getDefiningOp<executor::ABIRecvOp>()) {
      val = recvOp.getPtr();
      continue;
    }
    break;
  }
  return val;
}

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
  for (auto [idx, returnedValue] : llvm::enumerate(returnOp.getOperands())) {

    bool erased = false;
    if (auto sendOp = returnedValue.getDefiningOp<executor::ABISendOp>();
        sendOp && isa<MemRefType>(sendOp.getValue().getType())) {
      Value val = lookBackThroughView(sendOp.getValue());
      if (val == sendOp.getPtr()) {
        resultToArgs[idx] = cast<BlockArgument>(sendOp.getPtr()).getArgNumber();
        erasedResultIndices.set(idx);
        continue;
      }
      newReturnValues.push_back(sendOp.getResult());
      continue;
    }

    for (BlockArgument bbArg : funcOp.getArguments()) {
      Value val = lookBackThroughView(returnedValue);

      if (val == bbArg) {
        if ((isa<MemRefType>(val.getType()) &&
             isa<MemRefType>(bbArg.getType()) &&
             memref::CastOp::areCastCompatible(returnedValue.getType(),
                                               bbArg.getType()))) {
          resultToArgs[idx] = bbArg.getArgNumber();
          erased = true;
          break;
        }
      }
    }

    if (erased) {
      erasedResultIndices.set(idx);
    } else {
      newReturnValues.push_back(returnedValue);
    }
  }

  // Update function.
  if (failed(funcOp.eraseResults(erasedResultIndices)))
    return failure();
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
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    if (failed(dropEquivalentFuncBufferResults(rewriter, funcOp, symbolUseMap)))
      return failure();
  }

  return success();
}

namespace {
struct PlanRemoveEquivalentBufferResultsPass
    : public plan::impl::PlanRemoveEquivalentBufferResultsPassBase<
          PlanRemoveEquivalentBufferResultsPass> {
  using Base::Base;

  bool canScheduleOn(RegisteredOperationName opName) const override {
    return opName.hasTrait<OpTrait::SymbolTable>();
  }

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
