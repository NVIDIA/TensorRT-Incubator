//===- MemRefCastElimination.cpp ------------------------------------------===//
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
/// Tries to eliminate memref cast operations where they can be statically
/// determined to be not needed.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_MEMREFCASTELIMINATIONPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// Given an `scf.if` operation, return the result indices such that both the
/// "then" and "else" terminator operands are produced by "compatible" cast
/// operations that can be moved to act on the result of the if. The types of
/// the new `scf.if` operation are returned as well.
static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<Type>>>
isCastEliminationCandidate(scf::IfOp op) {
  SmallVector<int64_t> resultIndices;
  SmallVector<Type> newResultTypes;
  if (op->getNumResults() == 0)
    return failure();

  auto thenYield = cast<scf::YieldOp>(op.thenBlock()->getTerminator());
  auto elseYield = cast<scf::YieldOp>(op.elseBlock()->getTerminator());

  for (auto [idx, v] : llvm::enumerate(op.getResults())) {
    MemRefType memrefType = dyn_cast<MemRefType>(v.getType());
    if (!memrefType) {
      newResultTypes.push_back(v.getType());
      continue;
    }
    auto thenCast =
        thenYield.getOperands()[idx].getDefiningOp<memref::CastOp>();
    auto elseCast =
        elseYield.getOperands()[idx].getDefiningOp<memref::CastOp>();

    if (!thenCast || !elseCast ||
        thenCast.getOperand().getType() != elseCast.getOperand().getType()) {
      newResultTypes.push_back(v.getType());
      continue;
    }

    newResultTypes.push_back(thenCast.getOperand().getType());
    resultIndices.push_back(idx);
  }
  if (resultIndices.empty())
    return failure();
  return std::make_pair(resultIndices, newResultTypes);
}

namespace {
/// Move compatible `memref.cast` on operands of `scf.yield` in an `scf.if` body
/// blocks to act on the result instead.
struct SimplifyScfIf : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<std::pair<SmallVector<int64_t>, SmallVector<Type>>> params =
        isCastEliminationCandidate(op);
    if (failed(params))
      return failure();
    auto [resultIndices, newTypes] = *params;
    scf::IfOp newOp = rewriter.create<scf::IfOp>(op.getLoc(), newTypes,
                                                 op.getCondition(), true);
    newOp->setAttrs(op->getAttrs());
    rewriter.eraseBlock(newOp.elseBlock());
    rewriter.eraseBlock(newOp.thenBlock());

    auto thenYield = cast<scf::YieldOp>(op.thenBlock()->getTerminator());
    auto elseYield = cast<scf::YieldOp>(op.elseBlock()->getTerminator());
    for (auto idx : resultIndices) {
      auto thenCast =
          thenYield.getOperands()[idx].getDefiningOp<memref::CastOp>();
      auto elseCast =
          elseYield.getOperands()[idx].getDefiningOp<memref::CastOp>();
      thenYield.setOperand(idx, thenCast.getOperand());
      elseYield.setOperand(idx, elseCast.getOperand());
    }
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    SmallVector<Value> replacements(newOp->getResults());
    for (auto idx : resultIndices) {
      replacements[idx] = rewriter.create<memref::CastOp>(
          op.getLoc(), op->getResultTypes()[idx], newOp->getResult(idx));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};
} // namespace

namespace {
class MemRefCastEliminationPass
    : public mlir::impl::MemRefCastEliminationPassBase<
          MemRefCastEliminationPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    memref::CastOp::getCanonicalizationPatterns(patterns, ctx);
    patterns.insert<SimplifyScfIf>(ctx);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc())
          << "Failed to run memref cast elimination patterns";
      return signalPassFailure();
    }
  }
};
} // namespace
