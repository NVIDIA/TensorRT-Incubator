//===- SCFForCyclicIterArgs.cpp -------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of `mtrt-scf-for-cyclic-iter-args` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Transforms/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mtrt {
#define GEN_PASS_DEF_SCFFORCYCLICITERARGSPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mtrt

using namespace mlir;

namespace {

/// Returns the iteration index for `iv` as `(iv - lb) / step`.
static Value buildIterationIndex(OpBuilder &builder, Location loc, Value iv,
                                 Value lb, Value step) {
  Value diff = builder.create<arith::SubIOp>(loc, iv, lb);
  return builder.create<arith::DivUIOp>(loc, diff, step);
}

static Value buildConstant(OpBuilder &builder, Location loc, Type type,
                           int64_t value) {
  assert(type.isIntOrIndex() && "expected index or integer type");
  IntegerAttr attr = type.isIndex() ? builder.getIndexAttr(value)
                                    : builder.getIntegerAttr(type, value);
  return builder.create<arith::ConstantOp>(loc, type, attr);
}

/// Returns the trip count of a scf.for as `select(ub > lb, ceildiv(ub - lb,
/// step), 0)`.
static Value buildTripCount(OpBuilder &builder, Location loc, Value lb,
                            Value ub, Value step) {
  Value diff = builder.create<arith::SubIOp>(loc, ub, lb);
  Value ceil = builder.create<arith::CeilDivUIOp>(loc, diff, step);
  Value zero = buildConstant(builder, loc, lb.getType(), 0);
  Value hasIters =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, ub, lb);
  return builder.create<arith::SelectOp>(loc, hasIters, ceil, zero);
}

/// Build a value that selects the rotation of `initValues` based on `offset`.
static Value buildRotatedSelect(OpBuilder &builder, Location loc, Value offset,
                                ArrayRef<Value> initValues, unsigned pos) {
  Value result = initValues[pos];
  for (unsigned r = 1; r < initValues.size(); ++r) {
    Value cmp = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, offset,
        buildConstant(builder, loc, offset.getType(), r));
    Value candidate = initValues[(pos + r) % initValues.size()];
    result = builder.create<arith::SelectOp>(loc, cmp, candidate, result);
  }
  return result;
}

/// Identify cycles in the mapping from iter args to yield operands.
static SmallVector<SmallVector<unsigned>>
findCyclicIterArgGroups(scf::ForOp forOp) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  unsigned numIterArgs = forOp.getInitArgs().size();

  SmallVector<int64_t> mapping(numIterArgs, -1);
  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value yielded = yieldOp.getOperand(i);
    auto arg = dyn_cast<BlockArgument>(yielded);
    if (!arg || arg.getOwner() != forOp.getBody())
      continue;
    if (arg.getArgNumber() == 0)
      continue;
    mapping[i] = static_cast<int64_t>(arg.getArgNumber() - 1);
  }

  SmallVector<bool> visited(numIterArgs, false);
  SmallVector<SmallVector<unsigned>> cycles;
  for (unsigned i = 0; i < numIterArgs; ++i) {
    if (visited[i] || mapping[i] < 0)
      continue;
    SmallVector<int64_t> path;
    llvm::DenseMap<int64_t, unsigned> indexInPath;
    int64_t cur = static_cast<int64_t>(i);
    while (cur >= 0 && !visited[cur] && !indexInPath.count(cur)) {
      indexInPath[cur] = path.size();
      path.push_back(cur);
      cur = mapping[cur];
    }
    if (cur >= 0 && indexInPath.count(cur)) {
      unsigned start = indexInPath[cur];
      SmallVector<unsigned> cycle;
      cycle.reserve(path.size() - start);
      for (unsigned idx = start; idx < path.size(); ++idx)
        cycle.push_back(static_cast<unsigned>(path[idx]));
      if (cycle.size() > 1)
        cycles.push_back(std::move(cycle));
    }
    for (int64_t node : path)
      visited[node] = true;
  }
  return cycles;
}

static LogicalResult rewriteForOp(scf::ForOp forOp) {
  SmallVector<SmallVector<unsigned>> cycles = findCyclicIterArgGroups(forOp);
  if (cycles.empty())
    return success();

  const unsigned numIterArgs = forOp.getInitArgs().size();
  SmallVector<bool> removeIterArg(numIterArgs, false);
  for (const auto &cycle : cycles)
    for (unsigned idx : cycle)
      removeIterArg[idx] = true;

  SmallVector<Value> newIterOperands;
  newIterOperands.reserve(numIterArgs);
  SmallVector<int64_t> oldToNewIndex(numIterArgs, -1);
  for (unsigned i = 0; i < numIterArgs; ++i) {
    if (removeIterArg[i])
      continue;
    oldToNewIndex[i] = static_cast<int64_t>(newIterOperands.size());
    newIterOperands.push_back(forOp.getInitArgs()[i]);
  }

  IRRewriter rewriter(forOp.getContext());
  rewriter.setInsertionPoint(forOp);
  scf::ForOp newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newIterOperands);

  Block *oldBody = forOp.getBody();
  Block *newBody = newForOp.getBody();
  if (!newBody->empty())
    rewriter.eraseOp(newBody->getTerminator());

  // Build replacement values for removed iter args inside the new loop body.
  DenseMap<unsigned, Value> replacementForArg;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(newBody);
  Value iterIndex =
      buildIterationIndex(rewriter, forOp.getLoc(), newBody->getArgument(0),
                          newForOp.getLowerBound(), newForOp.getStep());

  for (const auto &cycle : cycles) {
    unsigned cycleSize = cycle.size();
    Value cycleMod =
        buildConstant(rewriter, forOp.getLoc(), iterIndex.getType(), cycleSize);
    Value offset =
        rewriter.create<arith::RemUIOp>(forOp.getLoc(), iterIndex, cycleMod);
    SmallVector<Value> initValues;
    initValues.reserve(cycleSize);
    for (unsigned idx : cycle)
      initValues.push_back(forOp.getInitArgs()[idx]);
    for (unsigned pos = 0; pos < cycleSize; ++pos) {
      unsigned iterArgIndex = cycle[pos];
      replacementForArg[iterArgIndex] =
          buildRotatedSelect(rewriter, forOp.getLoc(), offset, initValues, pos);
    }
  }

  SmallVector<Value> argReplacements;
  argReplacements.reserve(oldBody->getNumArguments());
  argReplacements.push_back(newBody->getArgument(0));
  for (unsigned i = 0; i < numIterArgs; ++i) {
    if (removeIterArg[i]) {
      argReplacements.push_back(replacementForArg.lookup(i));
      continue;
    }
    argReplacements.push_back(
        newBody->getArgument(static_cast<unsigned>(oldToNewIndex[i]) + 1));
  }
  rewriter.mergeBlocks(oldBody, newBody, argReplacements);

  auto newYield = cast<scf::YieldOp>(newBody->getTerminator());
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(newIterOperands.size());
  for (unsigned i = 0; i < numIterArgs; ++i) {
    if (!removeIterArg[i])
      newYieldOperands.push_back(newYield.getOperand(i));
  }
  rewriter.setInsertionPoint(newYield);
  rewriter.replaceOpWithNewOp<scf::YieldOp>(newYield, newYieldOperands);

  // Replace loop results for removed iter args with computed values.
  rewriter.setInsertionPointAfter(newForOp);
  Value tripCount =
      buildTripCount(rewriter, forOp.getLoc(), newForOp.getLowerBound(),
                     newForOp.getUpperBound(), newForOp.getStep());
  DenseMap<unsigned, Value> finalValueForArg;
  for (const auto &cycle : cycles) {
    unsigned cycleSize = cycle.size();
    Value cycleMod =
        buildConstant(rewriter, forOp.getLoc(), tripCount.getType(), cycleSize);
    Value offset =
        rewriter.create<arith::RemUIOp>(forOp.getLoc(), tripCount, cycleMod);
    SmallVector<Value> initValues;
    initValues.reserve(cycleSize);
    for (unsigned idx : cycle)
      initValues.push_back(forOp.getInitArgs()[idx]);
    for (unsigned pos = 0; pos < cycleSize; ++pos) {
      unsigned iterArgIndex = cycle[pos];
      finalValueForArg[iterArgIndex] =
          buildRotatedSelect(rewriter, forOp.getLoc(), offset, initValues, pos);
    }
  }

  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value oldResult = forOp.getResult(i);
    if (removeIterArg[i]) {
      oldResult.replaceAllUsesWith(finalValueForArg.lookup(i));
      continue;
    }
    oldResult.replaceAllUsesWith(
        newForOp.getResult(static_cast<unsigned>(oldToNewIndex[i])));
  }

  rewriter.eraseOp(forOp);
  return success();
}

class SCFForCyclicIterArgsPass
    : public mtrt::impl::SCFForCyclicIterArgsPassBase<
          SCFForCyclicIterArgsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    SmallVector<scf::ForOp> forOps;
    getOperation()->walk<WalkOrder::PostOrder>(
        [&](scf::ForOp forOp) { forOps.push_back(forOp); });

    for (scf::ForOp forOp : forOps) {
      if (failed(rewriteForOp(forOp)))
        return signalPassFailure();
    }
  }
};

} // namespace
