//===- ReshapeElimination.cpp ---------------------------------------------===//
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
/// Definition of reshape elimination pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace tensorrt {
#define GEN_PASS_DEF_RESHAPEELIMINATIONPASS
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h.inc"
} // namespace tensorrt
} // namespace mlir

#define DEBUG_TYPE "tensorrt-reshape-elimination"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;
using namespace mlir::tensorrt;

// Checks if reduction of an input tensor by a reshape `op` is valid,
// considering output of the reshape `op` is fed to a matmul as the LHS input.
// Reduction is valid if K is not reduced and M is reduced. LHS input to a
// matmul has shape [batch ...]xMxK.
static bool isLhsReductionValid(ReshapeOp op) {
  RankedTensorType originalType = op.getInput().getType();
  RankedTensorType collapsedType = op.getResult().getType();
  return (originalType.getShape()[originalType.getRank() - 1] ==
          collapsedType.getShape()[collapsedType.getRank() - 1]) &&
         (originalType.getShape()[originalType.getRank() - 2] !=
          collapsedType.getShape()[collapsedType.getRank() - 2]);
}

// Checks if reduction of an input tensor by a reshape `op` is valid,
// considering output of the reshape `op` is fed to a matmul as the RHS input.
// Reduction is valid if K is not reduced. RHS input to a matmul has shape
// [batch ...]xKxN.
static bool isRhsReductionValid(ReshapeOp op) {
  RankedTensorType originalType = op.getInput().getType();
  RankedTensorType collapsedType = op.getResult().getType();
  return originalType.getShape()[originalType.getRank() - 2] ==
         collapsedType.getShape()[collapsedType.getRank() - 2];
}

namespace {
/// Tries to eliminate reshape ops before and after matmul in the following
/// case.
/// %1 = reshape(%0)    // collapse batch dims and M
/// %2 = matmul(%1, %k)
/// %3 = reshape(%2)    // expand to previously collapsed batch dims and M
/// to
/// %1 = expand(%k)     // expand by adding 1's
/// %2 = matmul(%0, %1)
///
/// In short, we try to keep original shape of %1 (i.e. %0) by expanding rank of
/// %k. LHS operand to the matmul has shape [batch ...]xMxK. Matmul reduction
/// dimension i.e. K should not be collapsed. Matrix operation for both operands
/// is set to `MatrixOperation::kNONE`. This pattern does not apply when input
/// to the parent reshape op of matmul is dynamic.
struct EliminateReshapeBeforeAndAfterMatmulLHS : OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Check if expansion happens with this reshape
    if (op.getInput().getType().getRank() >= op.getResult().getType().getRank())
      return failure();

    auto matmul = op.getInput().getDefiningOp<MatrixMultiplyOp>();
    if (!matmul || matmul.getOp0() != MatrixOperation::kNONE ||
        matmul.getOp1() != MatrixOperation::kNONE)
      return failure();

    // Get expanded shape (by inserting unit dimensions in batch dimensions) of
    // matmul input lhs/rhs tensor to match the rank of the other side.
    auto getExpandedShapeOfRhs = [](RankedTensorType tensorToExpandType,
                                    int64_t otherSideRank) {
      assert(otherSideRank > tensorToExpandType.getRank() &&
             "operand to expand should have a smaller rank");
      // -2 for contraction dims
      int64_t numOnesToAdd = otherSideRank - 2;
      SmallVector<int64_t> expandedShape;
      if (tensorToExpandType.getRank() > 2) {
        ArrayRef<int64_t> tensorToExpandBatchDims =
            tensorToExpandType.getShape().drop_back(2);
        llvm::append_range(expandedShape, tensorToExpandBatchDims);
        numOnesToAdd -= tensorToExpandBatchDims.size();
      }
      expandedShape.insert(expandedShape.end(), numOnesToAdd, 1);
      // Finally add contraction dims
      llvm::append_range(expandedShape,
                         tensorToExpandType.getShape().take_back(2));
      return RankedTensorType::get(expandedShape,
                                   tensorToExpandType.getElementType());
    };

    // If lhsParentReshape exists, we need to make sure same dimensions as that
    // of collapsed by `lhsParentReshape` are expanded by `op` and input to the
    // `lhsParentReshape` is not dynamic.
    auto lhsParentReshape = matmul.getInput0().getDefiningOp<ReshapeOp>();
    if (!lhsParentReshape ||
        !lhsParentReshape.getInput().getType().getShape().drop_back(1).equals(
            op.getResult().getType().getShape().drop_back(1)) ||
        !isLhsReductionValid(lhsParentReshape) ||
        !lhsParentReshape.getInput().getType().hasStaticShape())
      return failure();

    auto rhsExpandedType =
        getExpandedShapeOfRhs(matmul.getInput1().getType(),
                              lhsParentReshape.getInput().getType().getRank());
    auto rhsExpanded = rewriter
                           .create<ExpandRankOp>(op->getLoc(),
                                                 /*result=*/rhsExpandedType,
                                                 /*input=*/matmul.getInput1())
                           .getResult();

    rewriter.replaceOpWithNewOp<MatrixMultiplyOp>(
        op,
        /*input0=*/lhsParentReshape.getInput(),
        /*input1=*/rhsExpanded,
        /*op0=*/matmul.getOp0(),
        /*op1=*/matmul.getOp1());
    return success();
  }
};

/// Simplify reshape ops by eliminating one reshape in the following case.
/// %1 = reshape(%0)    // collapse batch dims but not K
/// %2 = matmul(%k, %1)
/// %3 = reshape(%2)    // expand to previously collapsed batch dims
/// to
/// %1 = reshape(%k)
/// %2 = matmul(%1, %0)
///
/// LHS operand to the matmul has shape [batch ...]xMxK and RHS operand has
/// shape [batch ...]xKxN. Unlike `EliminateReshapeBeforeAndAfterMatmulLHS`
/// case, we can't expand `k` and keep original shape of %1 input (i.e. %0).
/// This is because for RHS input, reshape can only strictly change batch
/// dimension and not any of contracting dims (e.g. for LHS, M could be
/// collapsed). It means few or all only batch dims are collapsed for RHS. LHS
/// %k needs to be reshaped to have same batch dims as %0 (collapsed batch dims
/// of %0 are same as %k thus reshape works). This rewriter works only when
/// matrix operation for both operands is set to `MatrixOperation::kNONE`. This
/// pattern does not apply when input to the parent reshape op of matmul is
/// dynamic.
struct SimplifyReshapeBeforeAndAfterMatmulRHS : OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Check if expansion happens with this reshape
    if (op.getInput().getType().getRank() >= op.getResult().getType().getRank())
      return failure();

    auto matmul = op.getInput().getDefiningOp<MatrixMultiplyOp>();
    if (!matmul || matmul.getOp0() != MatrixOperation::kNONE ||
        matmul.getOp1() != MatrixOperation::kNONE)
      return failure();

    auto rhsParentReshape = matmul.getInput1().getDefiningOp<ReshapeOp>();
    // If rhsParentReshape exists, we need to make sure same dimensions as that
    // of collapsed by `rhsParentReshape` are expanded by `op` and input to the
    // `rhsParentShape` is not dynamic.
    if (!rhsParentReshape ||
        !rhsParentReshape.getInput().getType().getShape().drop_back(2).equals(
            op.getResult().getType().getShape().drop_back(2)) ||
        !isRhsReductionValid(rhsParentReshape) ||
        !rhsParentReshape.getInput().getType().hasStaticShape())
      return failure();

    SmallVector<int64_t> lhsReshapedShape(
        rhsParentReshape.getInput().getType().getShape().drop_back(2));
    SmallVector<int64_t> lhsReductionDims(
        matmul.getInput0().getType().getShape().drop_front(
            matmul.getInput0().getType().getRank() - 2));
    llvm::append_range(lhsReshapedShape, lhsReductionDims);
    auto lhsReshapedType = RankedTensorType::get(
        lhsReshapedShape, matmul.getInput0().getType().getElementType());
    auto lhsReshaped = rewriter
                           .create<ReshapeOp>(op->getLoc(),
                                              /*result=*/lhsReshapedType,
                                              /*input=*/matmul.getInput0())
                           .getResult();
    rewriter.replaceOpWithNewOp<MatrixMultiplyOp>(
        op,
        /*input0=*/lhsReshaped,
        /*input1=*/rhsParentReshape.getInput(),
        /*op0=*/matmul.getOp0(),
        /*op1=*/matmul.getOp1());
    return success();
  }
};
} // namespace

namespace {
class ReshapeEliminationPass
    : public tensorrt::impl::ReshapeEliminationPassBase<
          ReshapeEliminationPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();

    RewritePatternSet patterns(ctx);
    patterns.insert<EliminateReshapeBeforeAndAfterMatmulLHS,
                    SimplifyReshapeBeforeAndAfterMatmulRHS>(ctx);
    ReshapeOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
