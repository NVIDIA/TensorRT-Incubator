//===- SCFDetensorizeLoops.cpp --------------------------------------------===//
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
/// Implementation of `scf-detensorize-loops` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SCFDETENSORIZELOOPSPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

static bool isHostTensor(Value value, const DataFlowSolver &solver) {
  const TensorKindLattice *lattice =
      solver.lookupState<TensorKindLattice>(value);
  assert(lattice && "expected valid lattice point");
  if (lattice->getValue().isUninitialized())
    return false;
  return lattice->getValue().isHostOnly();
}

/// Returns true if it is OK to scalarize the given loop-carried variable.
static bool isValidToScalarize(BlockArgument arg,
                               const DataFlowSolver &solver) {
  auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
  if (!tensorType || tensorType.getNumElements() != 1)
    return false;

  // Check that all uses are by a `tensor.extract` operation or the terminator.
  return isHostTensor(arg, solver);
}

namespace {
/// Attempts to rewrite `scf.while` operations to scalarize the loop-carried
/// variables if possible.
struct DetensorizeWhilePattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  DetensorizeWhilePattern(MLIRContext *ctx, const DataFlowSolver &solver)
      : OpRewritePattern(ctx), solver(solver) {}

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> iterArgsToUpdate;
    Region &after = op.getAfter();
    Region &before = op.getBefore();
    if (after.getArgumentTypes() != before.getArgumentTypes())
      return rewriter.notifyMatchFailure(
          op, "only scf.while with same before/after region argument types are "
              "supported");

    for (BlockArgument arg : after.getArguments()) {
      if (!isValidToScalarize(arg, solver) ||
          !isValidToScalarize(before.getArgument(arg.getArgNumber()), solver))
        continue;
      iterArgsToUpdate.push_back(arg.getArgNumber());
    }

    if (iterArgsToUpdate.empty())
      return failure();

    // Create the `tensor.extract` operations before the loop op.
    Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    SmallVector<Type> newTypes(op->getResultTypes());
    SmallVector<Value> newOperands(op.getOperands());
    for (int64_t idx : iterArgsToUpdate) {
      auto tensorType = cast<RankedTensorType>(newTypes[idx]);
      newTypes[idx] = tensorType.getElementType();
      newOperands[idx] = rewriter.create<tensor::ExtractOp>(
          op.getLoc(), newOperands[idx],
          SmallVector<Value>(tensorType.getRank(), zero));
    }

    // Update the `while` op by moving the regions to a new while op.
    auto whileOp = rewriter.create<WhileOp>(op.getLoc(), newTypes, newOperands);
    rewriter.inlineRegionBefore(before, whileOp.getBefore(),
                                whileOp.getBefore().end());
    rewriter.inlineRegionBefore(after, whileOp.getAfter(),
                                whileOp.getAfter().end());
    auto yield = cast<YieldOp>(whileOp.getAfterBody()->getTerminator());
    auto cond = cast<ConditionOp>(whileOp.getBeforeBody()->getTerminator());

    SmallVector<Value> newConditionArgs(cond.getArgs());
    SmallVector<Value> newYieldArgs(yield.getOperands());

    for (int64_t idx : iterArgsToUpdate) {
      // For each loop-carried arg being transformed, update the block argument
      // types.
      auto oldType = cast<RankedTensorType>(
          whileOp.getBeforeBody()->getArgument(idx).getType());
      whileOp.getBeforeBody()->getArgument(idx).setType(newTypes[idx]);
      whileOp.getAfterBody()->getArgument(idx).setType(newTypes[idx]);

      // Update uses. By design of preconditions.
      for (BlockArgument arg : {whileOp.getBeforeBody()->getArgument(idx),
                                whileOp.getAfterBody()->getArgument(idx)}) {
        rewriter.setInsertionPointToStart(arg.getOwner());

        tensor::FromElementsOp replacement =
            rewriter.create<tensor::FromElementsOp>(arg.getLoc(), oldType, arg);
        rewriter.replaceAllUsesExcept(arg, replacement, replacement);
      }

      // Update the terminator to be a `tensor.extract` if the yielded value is
      // not exactly the block argument.
      auto getCoord = [&](Location loc) {
        return oldType.getRank() == 0
                   ? Value{}
                   : rewriter.create<arith::ConstantIndexOp>(loc, 0)
                         .getResult();
      };
      if (isa<RankedTensorType>(cond.getArgs()[idx].getType())) {
        rewriter.setInsertionPoint(cond);
        Location loc = cond.getArgs()[idx].getLoc();
        Value coord = getCoord(loc);
        auto extractOp = rewriter.create<tensor::ExtractOp>(
            loc, cond.getArgs()[idx], coord ? ValueRange{coord} : ValueRange{});
        newConditionArgs[idx] = extractOp;
      }
      if (isa<RankedTensorType>(yield.getOperands()[idx].getType())) {
        rewriter.setInsertionPoint(yield);
        Location loc = yield.getOperand(idx).getLoc();
        Value coord = getCoord(loc);
        auto extractOp = rewriter.create<tensor::ExtractOp>(
            loc, yield.getOperand(idx),
            coord ? ValueRange{coord} : ValueRange{});
        newYieldArgs[idx] = extractOp;
      }
    }

    rewriter.modifyOpInPlace(
        yield, [&]() { yield.getResultsMutable().assign(newYieldArgs); });
    rewriter.modifyOpInPlace(
        cond, [&]() { cond.getArgsMutable().assign(newConditionArgs); });

    // Replace the loop with new values. Create the scalars as necessary.
    rewriter.setInsertionPointAfter(whileOp);
    SmallVector<Value> replacements(whileOp.getResults());
    for (int64_t idx : iterArgsToUpdate)
      replacements[idx] = rewriter.create<tensor::FromElementsOp>(
          op.getLoc(), op->getResult(idx).getType(), replacements[idx]);

    rewriter.replaceOp(op, replacements);
    return success();
  }

  const DataFlowSolver &solver;
};

class SCFDetensorizeLoopsPass
    : public impl::SCFDetensorizeLoopsPassBase<SCFDetensorizeLoopsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    Operation *op = getOperation();

    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TensorKindAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(op))) {
      emitError(op->getLoc()) << "failed to run TensorKindAnalysis";
      return signalPassFailure();
    }

    patterns.add<DetensorizeWhilePattern>(ctx, solver);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
