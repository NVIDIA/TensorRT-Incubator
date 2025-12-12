//===- AffineBoundsOptimization.cpp ---------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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

//===- AffineBoundsOptimization.cpp ---------------------------------------===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of the affine bounds optimization patterns.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::kernel;

/// Returns true if the expression is monotonic, meaning that it is
/// non-increasing or non-decreasing. The check is conservative and may give
/// false negatives. We use this to check whether we can find the extrema of the
/// function over an interval of the domain by checking the value of the
/// function at the domain endpoints. This is true for what we normally think of
/// as affine functions (allow *,+,-) operators. Note that we could also allow
/// ceilDiv and floorDiv, but it would become much more complicated  to check.
static bool isMonotonic(AffineExpr expr) {
  struct IsMonotonic : public AffineExprVisitor<IsMonotonic, WalkResult> {
    using AffineExprVisitor::visit;
    WalkResult visitModExpr(AffineBinaryOpExpr expr) {
      return WalkResult::interrupt();
    }
    WalkResult visitCeilDivExpr(AffineBinaryOpExpr expr) {
      return WalkResult::interrupt();
    }
    WalkResult visitFloorDivExpr(AffineBinaryOpExpr expr) {
      return WalkResult::interrupt();
    }
  };
  return !IsMonotonic().walkPostOrder(expr).wasInterrupted();
}

/// Find the bound on the given `values` using the `type` of bound.
static FailureOr<SmallVector<int64_t>>
populateBounds(ValueRange values, presburger::BoundType type,
               bool closedUB = true) {
  SmallVector<int64_t> bounds;
  for (Value v : values) {
    FailureOr<int64_t> bound = ValueBoundsConstraintSet::computeConstantBound(
        type, v, nullptr, closedUB);
    if (failed(bound))
      return failure();
    bounds.push_back(*bound);
  }
  return bounds;
}

/// Attempt to constant fold the map using the given `values` to substitute for
/// the dimension and symbols of the map.
static FailureOr<SmallVector<int64_t>>
constantFoldMap(RewriterBase &rewriter, AffineMap map,
                ArrayRef<int64_t> values) {
  SmallVector<Attribute> operands = llvm::map_to_vector(
      values, [&](int64_t v) -> Attribute { return rewriter.getIndexAttr(v); });
  SmallVector<int64_t> res;
  bool hasPoison = false;
  map.partialConstantFold(operands, &res, &hasPoison);
  if (hasPoison || res.size() != map.getNumResults())
    return failure();
  return res;
}

/// Return a lower and upper bound for each result expression of `map` over the
/// domain defined by the `operands`. We do this by finding lower/upper bounds
/// of the operands and then folding the map over these bounds. This is only
/// true when each map expression is monotonic. If the map is not provably
/// monotonic or if we cannot find bounds on the operands, then failure is
/// returned.
static LogicalResult
getExpressionBounds(RewriterBase &rewriter, AffineMap map, ValueRange operands,
                    SmallVectorImpl<int64_t> &lowerBounds,
                    SmallVectorImpl<int64_t> &upperBounds) {
  if (!llvm::all_of(map.getResults(), isMonotonic))
    return failure();

  FailureOr<SmallVector<int64_t>> ubs =
      populateBounds(operands, presburger::BoundType::UB);
  if (failed(ubs))
    return failure();
  FailureOr<SmallVector<int64_t>> lbs =
      populateBounds(operands, presburger::BoundType::LB);
  if (failed(lbs))
    return failure();

  FailureOr<SmallVector<int64_t>> valuesFromUb =
      constantFoldMap(rewriter, map, *ubs);
  if (failed(valuesFromUb))
    return failure();
  FailureOr<SmallVector<int64_t>> valuesFromLb =
      constantFoldMap(rewriter, map, *lbs);
  if (failed(valuesFromLb))
    return failure();

  for (auto [valFromLb, valFromUb] :
       llvm::zip_equal(*valuesFromLb, *valuesFromUb)) {
    lowerBounds.push_back(std::min(valFromLb, valFromUb));
    upperBounds.push_back(std::max(valFromLb, valFromUb));
  }

  return success();
}

namespace {
/// Simplify `affine.min(exprs...)` if one `expr` has a range which is < the
/// lower bound of all the other expressions.
struct AffineMinSimplificationPattern
    : public OpRewritePattern<affine::AffineMinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineMinOp minOp,
                                PatternRewriter &rewriter) const override {
    // Align affine map results with dims/symbols in the constraint set.
    AffineMap map = minOp.getAffineMap();

    SmallVector<int64_t> minValues;
    SmallVector<int64_t> maxValues;
    if (failed(getExpressionBounds(rewriter, map, minOp.getOperands(),
                                   minValues, maxValues)))
      return rewriter.notifyMatchFailure(
          minOp, "failed to get expression values extremes");

    std::optional<unsigned> minBranchIdx = [&]() -> std::optional<unsigned> {
      for (auto [i, maxVal] : llvm::enumerate(maxValues)) {
        unsigned j = 0;
        for (unsigned e = map.getNumResults(); j < e; ++j) {
          if (i == j)
            continue;
          if (maxVal >= minValues[j])
            break;
        }
        // Return the first expression which is provably < all the other
        // expressions.
        if (j == map.getNumResults())
          return i;
      }
      return {};
    }();
    if (!minBranchIdx)
      return rewriter.notifyMatchFailure(minOp, "no dominator found");

    AffineMap newMap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                       map.getResults()[*minBranchIdx], rewriter.getContext());
    rewriter.replaceOpWithNewOp<affine::AffineApplyOp>(minOp, newMap,
                                                       minOp.getOperands());
    return success();
  }
};

/// Simplify `cmp<eq>(affine.min(exprs...), constVal)` to `false` if `constVal`
/// is provably not in the range of any of the `exprs`.
struct CmpIConstantSimplificationPattern
    : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = cmpOp.getLhs();
    Value rhs = cmpOp.getRhs();
    if (cmpOp.getPredicate() != arith::CmpIPredicate::eq)
      return failure();
    if (!lhs.getType().isIndex())
      return failure();
    if (ValueBoundsConstraintSet::compare(
            lhs, ValueBoundsConstraintSet::ComparisonOperator::LT, rhs) ||
        ValueBoundsConstraintSet::compare(
            lhs, ValueBoundsConstraintSet::ComparisonOperator::GT, rhs)) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          cmpOp, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      return success();
    }

    if (!matchPattern(rhs, m_Zero()))
      return failure();

    auto minOp = cmpOp.getLhs().getDefiningOp<affine::AffineMinOp>();
    if (!minOp)
      return failure();

    auto map = minOp.getAffineMap();
    SmallVector<int64_t> minValues;
    SmallVector<int64_t> maxValues;
    if (failed(getExpressionBounds(rewriter, map, minOp.getOperands(),
                                   minValues, maxValues)))
      return rewriter.notifyMatchFailure(
          minOp, "failed to get expression values extremes");

    // If all expressions can only be positive, then the comparison can never be
    // zero.
    for (int64_t val : minValues) {
      if (val <= 0)
        return rewriter.notifyMatchFailure(cmpOp, "comparison could be true");
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        cmpOp, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
    return success();
  }
};

struct GPUDimOpSimplificationPattern : public OpRewritePattern<gpu::BlockIdOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(gpu::BlockIdOp blockIdOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<int64_t> bound = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB, blockIdOp.getResult(), nullptr, true);
    if (failed(bound))
      return failure();
    if (*bound == 0) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          blockIdOp, rewriter.getZeroAttr(blockIdOp.getResult().getType()));
      return success();
    }
    return failure();
  }
};

} // namespace

void kernel::populateAffineBoundsOptimizationPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<AffineMinSimplificationPattern, CmpIConstantSimplificationPattern,
           GPUDimOpSimplificationPattern>(patterns.getContext());
}
