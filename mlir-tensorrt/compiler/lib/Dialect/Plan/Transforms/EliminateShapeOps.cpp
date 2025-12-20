//===- EliminateShapeOps.cpp ----------------------------------------------===//
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
/// Implementation of the `plan-eliminate-shape-ops` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

/// Remove unused arguments in a TensorRT function and adjust all the associated
/// `tensorrt.call` operations.
static LogicalResult removeUnusedArgs(IRRewriter &rewriter,
                                      SymbolTableCollection &collection,
                                      ModuleOp op, func::FuncOp funcOp,
                                      const SymbolUserMap &userMap) {
  SmallVector<int64_t> usedArgs;
  SmallVector<int64_t> unusedArgs;
  for (BlockArgument arg : funcOp.getArguments()) {
    if (!arg.use_empty())
      usedArgs.push_back(arg.getArgNumber());
    else
      unusedArgs.push_back(arg.getArgNumber());
  }
  if (usedArgs.size() == funcOp.getNumArguments())
    return success();

  for (int64_t idx : llvm::reverse(unusedArgs))
    if (failed(funcOp.eraseArgument(idx)))
      return failure();

  // Update the call ops.
  for (Operation *callOp : userMap.getUsers(funcOp)) {
    if (!isa<CallOpInterface>(callOp))
      continue;
    rewriter.setInsertionPoint(callOp);
    if (auto trtCall = dyn_cast<tensorrt::CallOp>(callOp)) {
      SmallVector<Value> newOperands;
      for (int64_t idx : usedArgs)
        newOperands.push_back(trtCall.getInputs()[idx]);
      rewriter.replaceOpWithNewOp<tensorrt::CallOp>(
          trtCall, trtCall->getResultTypes(), newOperands, trtCall.getOutputs(),
          trtCall.getCalleeAttr());
      continue;
    }
    if (auto allocCall = dyn_cast<tensorrt::CallAllocOp>(callOp)) {
      SmallVector<Value> newOperands;
      for (int64_t idx : usedArgs)
        newOperands.push_back(allocCall.getInputs()[idx]);
      rewriter.replaceOpWithNewOp<tensorrt::CallAllocOp>(
          allocCall, allocCall.getResultTypes(), newOperands,
          allocCall.getCalleeAttr());
      continue;
    }
    if (auto funcCall = dyn_cast<func::CallOp>(callOp)) {
      SmallVector<Value> newOperands;
      for (int64_t idx : usedArgs)
        newOperands.push_back(funcCall.getOperands()[idx]);
      rewriter.replaceOpWithNewOp<func::CallOp>(funcCall, funcOp, newOperands);
      continue;
    }
    llvm_unreachable("unexpected call operation type");
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
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      emitError(module->getLoc())
          << "failed to run plan.with_shape elimination patterns in "
          << getArgument();
      return signalPassFailure();
    }

    // The above transform will leave many unused index arguments in outlined
    // functions. Clean those up as well.
    SymbolTableCollection symbolTableCollection;
    IRRewriter rewriter(ctx);
    SymbolUserMap symbolUserMap(symbolTableCollection, module);

    llvm::SmallVector<func::FuncOp> candidates;
    module->walk([&](func::FuncOp func) {
      // Only consider functions that are not exported from the top module.
      // That means only functions in nested modules or private functions.
      if (func->getParentWithTrait<OpTrait::SymbolTable>() == module &&
          !func.isPrivate())
        return;

      unsigned numCallers = 0;
      for (Operation *user : symbolUserMap.getUsers(func)) {
        // Currently we have 'users' of functions that aren't call operations,
        // e.g. referring to a function symbol name in the 'plan.shape_function'
        // attribute. So we only reject an op if it has CallOpInterface users
        // that we don't recognize. Other users should not need to be updated.
        if (isa<CallOpInterface>(user)) {
          if (!isa<func::CallOp, tensorrt::CallOp, tensorrt::CallAllocOp>(user))
            return;
          numCallers++;
        }
      }
      // Don't both modifying functions that don't have callers. Those will get
      // removed during DCE.
      if (numCallers == 0)
        return;

      candidates.push_back(func);
    });
    for (func::FuncOp funcOp : candidates) {
      if (failed(removeUnusedArgs(rewriter, symbolTableCollection, module,
                                  funcOp, symbolUserMap))) {
        emitError(funcOp->getLoc())
            << "failed to drop unused arguments in " << getArgument();
        return signalPassFailure();
      }
    }
  }
};
} // namespace
