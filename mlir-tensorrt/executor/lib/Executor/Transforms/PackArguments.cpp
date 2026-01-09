//===- PackArguments.cpp --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of the `executor-pack-arguments` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "executor-pack-arguments"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

namespace mlir::executor {
#define GEN_PASS_DEF_EXECUTORPACKARGUMENTSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using namespace mlir::executor;

/// A mapping of FuncOps to their callers.
using FuncCallerMap = DenseMap<func::FuncOp, DenseSet<Operation *>>;

/// Return the func::FuncOp called by `callOp`.
static func::FuncOp getCalledFunction(func::CallOp callOp) {
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// Modified from
/// `third_party/llvm-project/mlir/lib/Dialect/Bufferization/Transforms/OneShotModuleBufferize.cpp`:
/// Store all functions of the `moduleOp` in `orderedFuncOps`, sorted by
/// callee-caller order (i.e. callees without callers first).
/// Store the map of FuncOp to all its callers in `callerMap`.
/// Return `failure()` if a cycle of calls is detected or if we are unable to
/// retrieve the called FuncOp from any func::CallOp.
static LogicalResult
getFuncOpsOrderedByCalls(Operation *moduleOp,
                         SmallVectorImpl<func::FuncOp> &orderedFuncOps,
                         FuncCallerMap &callerMap) {
  // For each FuncOp, the set of functions called by it (i.e. the union of
  // symbols of all nested func::CallOp).
  DenseMap<func::FuncOp, DenseSet<func::FuncOp>> calledBy;
  // For each FuncOp, the number of func::CallOp it contains.
  DenseMap<func::FuncOp, unsigned> numberCallOpsContainedInFuncOp;
  WalkResult res = moduleOp->walk([&](func::FuncOp funcOp) -> WalkResult {
    // Collect function calls and populate the caller map.
    numberCallOpsContainedInFuncOp[funcOp] = 0;
    return funcOp.walk([&](func::CallOp callOp) -> WalkResult {
      func::FuncOp calledFunction = getCalledFunction(callOp);
      assert(calledFunction && "could not retrieved called func::FuncOp");
      callerMap[calledFunction].insert(callOp);
      if (calledBy[calledFunction].insert(funcOp).second)
        numberCallOpsContainedInFuncOp[funcOp]++;
      return WalkResult::advance();
    });
  });
  if (res.wasInterrupted())
    return failure();
  // Iteratively remove function operations that do not call any of the
  // functions remaining in the callCounter map and add them to the worklist.
  while (!numberCallOpsContainedInFuncOp.empty()) {
    auto it = llvm::find_if(numberCallOpsContainedInFuncOp,
                            [](auto entry) { return entry.getSecond() == 0; });
    if (it == numberCallOpsContainedInFuncOp.end())
      return moduleOp->emitOpError(
          "expected callgraph to be free of circular dependencies.");
    orderedFuncOps.push_back(it->getFirst());
    for (auto callee : calledBy[it->getFirst()])
      numberCallOpsContainedInFuncOp[callee]--;
    numberCallOpsContainedInFuncOp.erase(it);
  }
  return success();
}

namespace {
class PackArgumentsPass
    : public executor::impl::ExecutorPackArgumentsPassBase<PackArgumentsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    if (failed(checkIsModuleLike(op)))
      return signalPassFailure();

    // find functions that need to be repacked and an adaptor generated.
    SmallVector<func::FuncOp> funcs;
    FuncCallerMap callerMap;
    if (failed(getFuncOpsOrderedByCalls(op, funcs, callerMap)))
      return signalPassFailure();

    IRRewriter rewriter(op);
    MLIRContext *ctx = op->getContext();
    for (func::FuncOp func : funcs) {
      if (func.getNumArguments() <= maxArguments) {
        LLVM_DEBUG(DBGS() << func.getName()
                          << " num arguments = " << func.getNumArguments()
                          << " <= " << maxArguments << "\n");
        continue;
      }
      if (callerMap[func].size() > 0) {
        LLVM_DEBUG(DBGS() << func.getName() << " num callers = "
                          << callerMap[func].size() << "\n");
        continue;
      }

      FunctionMetadataAttr metadata = func->getAttrOfType<FunctionMetadataAttr>(
          ExecutorDialect::kFunctionMetadataAttrName);
      if (!metadata || metadata.getCconv() != CallingConvention::unpacked) {
        LLVM_DEBUG(
            DBGS() << "no metadata or metadata convention is not unpacked");
        continue;
      }

      auto tableType = TableType::get(ctx, func.getArgumentTypes());
      FunctionType newFuncType =
          FunctionType::get(ctx, tableType, func.getResultTypes());
      rewriter.setInsertionPoint(func);
      auto newFunc = rewriter.create<func::FuncOp>(func.getLoc(),
                                                   func.getName(), newFuncType);
      rewriter.inlineRegionBefore(func.getBody(), newFunc.getBody(),
                                  newFunc.getBody().end());
      rewriter.setInsertionPointToStart(&newFunc.getBody().front());
      newFunc.getArgument(0).setType(tableType);
      for (BlockArgument &arg : newFunc.getBody().getArguments()) {
        if (arg.use_empty())
          continue;
        auto extract = rewriter.create<executor::ExtractTableValueOp>(
            arg.getLoc(), newFunc.getArgument(0),
            rewriter.getI64IntegerAttr(arg.getArgNumber()));
        rewriter.replaceAllUsesExcept(arg, extract.getResult(), extract);
      }
      unsigned originNumArgs = newFunc.getBody().front().getNumArguments();
      newFunc.getBody().front().eraseArguments(1, originNumArgs - 1);

      for (NamedAttribute attr : func->getAttrDictionary()) {
        if (attr.getName() == func.getFunctionTypeAttrName() ||
            attr.getName() == func.getArgAttrsAttrName() ||
            attr.getName() == func.getResAttrsAttrName() ||
            attr.getName() == ExecutorDialect::kFunctionMetadataAttrName)
          continue;
        newFunc->setAttr(attr.getName(), attr.getValue());
      }

      // Overwrite the metadata attr
      auto attr = FunctionMetadataAttr::get(
          ctx, metadata.getArgs(), metadata.getResults(),
          metadata.getNumOutputArgs(), metadata.getArgBounds(),
          metadata.getResultBounds(), metadata.getShapeFunc(),
          CallingConvention::packed);
      newFunc->setAttr(ExecutorDialect::kFunctionMetadataAttrName, attr);

      rewriter.eraseOp(func);
    }
  }
};
} // namespace
