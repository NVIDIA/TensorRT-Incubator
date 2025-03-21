//===- LowerGlobals.cpp ---------------------------------------------------===//
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
/// Definition of the ` executor-lower-globals` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace executor {
#define GEN_PASS_DEF_EXECUTORLOWERGLOBALSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace executor
} // namespace mlir

using namespace mlir;
using namespace mlir::executor;

/// For each `GlobalOp` in the module, if the global op has an initialization
/// region, append that code into a function called `executor_globals_init` and
/// set the gloal with a `executor.set_global` operation.
static LogicalResult rewriteGlobalInitializers(RewriterBase &rewriter,
                                               Operation *op) {
  auto globals = llvm::to_vector(op->getRegion(0).getOps<GlobalOp>());
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&op->getRegion(0).front());
  MLIRContext *ctx = op->getContext();
  auto initFunc = func::FuncOp::create(op->getLoc(), "executor_init_globals",
                                       FunctionType::get(ctx, {}, {}));
  initFunc.setPrivate();
  SymbolTable moduleSymbolTable(op);
  moduleSymbolTable.insert(initFunc);

  // We may have name collision. In either case, we should always need to tell
  // the module what functino is the global init function.
  op->setAttr(getExecutorGlobalInitializerFuncNameAttr(),
              FlatSymbolRefAttr::get(initFunc));

  rewriter.setInsertionPointToStart(initFunc.addEntryBlock());
  auto termOp = rewriter.create<func::ReturnOp>(op->getLoc());
  for (GlobalOp globalOp : globals) {
    // For data with initial_value attributes, lower into DataSegmentOp and
    // a load.
    auto initialValueAttr = globalOp.getInitialValueAttr();
    if (initialValueAttr && isa<ElementsAttr>(initialValueAttr)) {
      rewriter.setInsertionPoint(globalOp);
      auto dataSegmentOp = DataSegmentOp::create(
          globalOp.getLoc(), (globalOp.getSymName() + "_initializer").str(),
          cast<ElementsAttr>(initialValueAttr),
          /*constant=*/true,
          /*uninitialized=*/false, IntegerAttr{});
      moduleSymbolTable.insert(dataSegmentOp);

      rewriter.setInsertionPoint(termOp);
      // Create the load and set in the initializer function.
      Value resourceValue = rewriter.create<ConstantResourceLoadOp>(
          globalOp.getLoc(), FlatSymbolRefAttr::get(dataSegmentOp));
      rewriter.create<SetGlobalOp>(op->getLoc(), resourceValue,
                                   globalOp.getSymName());

      // Globals should now return ptrs.
      rewriter.setInsertionPoint(globalOp);
      globalOp->removeAttr(globalOp.getInitialValueAttrName());
      continue;
    }

    if (initialValueAttr)
      return globalOp.emitOpError(
          "has initial value that is not an ElementsAttr; this is currently "
          "unsupported");

    if (!globalOp.hasInitRegion())
      continue;

    Block *globalInitBlock = globalOp.getInitBody();
    // Change the terminator to a store op.
    Operation *initTerm = globalInitBlock->getTerminator();
    rewriter.setInsertionPoint(initTerm);
    rewriter.create<SetGlobalOp>(op->getLoc(), initTerm->getOperand(0),
                                 globalOp.getSymName());
    rewriter.eraseOp(initTerm);
    rewriter.setInsertionPoint(termOp);
    // Inline the initializer region into the end of the global init function.
    rewriter.inlineBlockBefore(globalInitBlock, termOp);

    // Drop the region.
    rewriter.setInsertionPoint(globalOp);
    rewriter.cloneWithoutRegions(globalOp);
    rewriter.eraseOp(globalOp);
  }
  return success();
}

namespace {
class LowerGlobalsPass
    : public executor::impl::ExecutorLowerGlobalsPassBase<LowerGlobalsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();
    if (failed(checkIsModuleLike(op)))
      return signalPassFailure();

    IRRewriter rewriter(ctx);
    if (failed(rewriteGlobalInitializers(rewriter, op))) {
      op->emitError() << "failed to create globals initializer function";
      return signalPassFailure();
    }
  }
};
} // namespace
