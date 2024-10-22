//===- EliminateShapeOps.cpp ----------------------------------------------===//
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
/// Implementation of the `plan-eliminate-shape-ops` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir::plan {
#define GEN_PASS_DEF_ELIMINATESHAPEOPSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

namespace {
/// Replace `plan.with_shape` operations with their operand.
struct RemoveWithShapeRewriter : public OpRewritePattern<plan::WithShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(plan::WithShapeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

/// Replace `plan.with_values` operations with their operand.
struct RemoveWithValuesRewriter : public OpRewritePattern<plan::WithValuesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(plan::WithValuesOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};
} // namespace

/// Get a map from `tensorrt.func` functions to associated `tensorrt.call`
/// and `tensorrt.call_alloc` operations.
static llvm::DenseMap<func::FuncOp, SmallVector<Operation *>>
getTensorRTFunctionCallMap(ModuleOp op, SymbolTableCollection &collection) {
  llvm::DenseMap<func::FuncOp, SmallVector<Operation *>> map;
  op->walk([&](Operation *callOp) {
    if (!isa<tensorrt::CallOp, tensorrt::CallAllocOp>(callOp))
      return;

    func::FuncOp func;
    if (auto call = dyn_cast<tensorrt::CallOp>(callOp))
      func = call.getFuncCallee(collection);
    else if (auto callAlloc = dyn_cast<tensorrt::CallAllocOp>(callOp))
      func = callAlloc.getFuncCallee(collection);
    else
      return;

    if (map.count(func))
      map[func].push_back(callOp);
    else
      map.insert({func, SmallVector<Operation *>{callOp}});
  });
  return map;
}

/// Remove unused arguments in a TensorRT function and adjust all the associated
/// `tensorrt.call` operations.
static LogicalResult removeUnusedArgs(SymbolTableCollection &collection,
                                      ModuleOp op, func::FuncOp funcOp,
                                      ArrayRef<Operation *> callOps) {
  llvm::SmallBitVector unusedArgs(funcOp.getNumArguments(), 0);
  for (BlockArgument arg : funcOp.getArguments()) {
    if (arg.use_empty())
      unusedArgs.set(arg.getArgNumber());
  }

  if (!unusedArgs.any())
    return success();

  for (int64_t i = unusedArgs.find_last(); i != -1;
       i = unusedArgs.find_prev(i)) {
    funcOp.eraseArgument(i);

    // Update the call ops.
    for (Operation *callOp : callOps) {
      if (auto call = dyn_cast<tensorrt::CallOp>(callOp)) {
        call.getInputsMutable().erase(i);
      } else if (auto callAlloc = dyn_cast<tensorrt::CallAllocOp>(callOp)) {
        callAlloc.getInputsMutable().erase(i);
      } else {
        llvm::errs() << "Unexpected operation type in callOps\n";
        callOp->dump();
      }
    }
  }

  return success();
}

namespace {
class EliminateShapeOpsPass
    : public plan::impl::EliminateShapeOpsPassBase<EliminateShapeOpsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module->getContext();

    // Eliminate `plan.with_shape` operations.
    RewritePatternSet patterns(ctx);
    patterns.add<RemoveWithShapeRewriter, RemoveWithValuesRewriter>(ctx);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      emitError(module->getLoc())
          << "failed to run plan.with_shape elimination patterns in "
          << getArgument();
      return signalPassFailure();
    }

    // The above transform will leave many unused index arguments in outlined
    // functions. Clean those up as well.
    SymbolTableCollection symbolTableCollection;
    auto callMapping =
        getTensorRTFunctionCallMap(module, symbolTableCollection);
    for (const auto &[funcOp, callOps] : callMapping) {
      if (failed(removeUnusedArgs(symbolTableCollection, module, funcOp,
                                  callOps))) {
        emitError(funcOp->getLoc())
            << "failed to drop unused arguments in " << getArgument();
        return signalPassFailure();
      }
    }
  }
};
} // namespace
