//===- DecomposeAggregateLoadsAndStores.cpp -------------------------------===//
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
/// Decompose loads and stores of aggregates into more primitive ops.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::executor {
#define GEN_PASS_DEF_EXECUTORDECOMPOSEAGGREGATELOADSANDSTORESPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using namespace mlir::executor;

/// Loads of tables aren't converted to builtin calls. We must lower to a series
/// of loads and table creation.
struct LoadTableToTableCreate : public OpRewritePattern<executor::LoadOp> {
  LoadTableToTableCreate(const DataLayout &dataLayout, MLIRContext *ctx,
                         PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), dataLayout(dataLayout) {}

  LogicalResult matchAndRewrite(executor::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<TableType>(op.getType());
    if (!type)
      return failure();

    SmallVector<Value> elements;
    elements.reserve(type.getBody().size());
    Type indexType = op.getOffset().getType();
    Location loc = op.getLoc();

    for (auto [idx, t] : llvm::enumerate(type.getBody())) {
      Value offset = op.getOffset();
      Value withinTableOffset = rewriter.create<executor::GetOffsetOp>(
          loc, indexType, op.getType(),
          ArrayRef<OpFoldResult>{rewriter.getI64IntegerAttr(0),
                                 rewriter.getI64IntegerAttr(idx)});
      offset =
          rewriter.create<executor::AddIOp>(loc, offset, withinTableOffset);
      Value integer = rewriter.create<executor::LoadOp>(op.getLoc(), t,
                                                        op.getPtr(), offset);
      elements.push_back(integer);
    }
    rewriter.replaceOpWithNewOp<CreateTableOp>(op, op.getType(), elements);
    return success();
  }

  const DataLayout &dataLayout;
};

/// Store of tables can be decomposed into smaller stores.
struct StoreTableDecomposition : public OpRewritePattern<executor::StoreOp> {

  StoreTableDecomposition(const DataLayout &dataLayout, MLIRContext *ctx,
                          PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), dataLayout(dataLayout) {}

  LogicalResult matchAndRewrite(executor::StoreOp op,
                                PatternRewriter &rewriter) const override {
    auto aggregateType = dyn_cast<TableType>(op.getValue().getType());
    if (!aggregateType)
      return failure();

    SmallVector<Value> elements;
    elements.reserve(aggregateType.getBody().size());
    Type indexType = op.getOffset().getType();
    Location loc = op.getLoc();
    for (auto [idx, t] : llvm::enumerate(aggregateType.getBody())) {
      Value val = rewriter.create<executor::ExtractTableValueOp>(
          loc, op.getValue(), idx);
      Value offset = op.getOffset();
      Value withinTableOffset = rewriter.create<executor::GetOffsetOp>(
          loc, indexType, op.getValue().getType(),
          ArrayRef<OpFoldResult>{rewriter.getI64IntegerAttr(0),
                                 rewriter.getI64IntegerAttr(idx)});
      offset =
          rewriter.create<executor::AddIOp>(loc, offset, withinTableOffset);
      rewriter.create<executor::StoreOp>(loc, op.getPtr(), offset, val);
    }
    rewriter.eraseOp(op);
    return success();
  }

  const DataLayout &dataLayout;
};

namespace {
class ExecutorDecomposeAggregateLoadsAndStoresPass
    : public executor::impl::ExecutorDecomposeAggregateLoadsAndStoresPassBase<
          ExecutorDecomposeAggregateLoadsAndStoresPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    DataLayoutAnalysis &analysis = getAnalysis<DataLayoutAnalysis>();
    const DataLayout &dataLayout = analysis.getAtOrAbove(getOperation());
    RewritePatternSet patterns(ctx);
    patterns.add<StoreTableDecomposition, LoadTableToTableCreate>(dataLayout,
                                                                  ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
