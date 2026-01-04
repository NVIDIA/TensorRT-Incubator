//===- CanonicalizeDotGeneral.cpp -------------------------------*- C++ -*-===//
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
/// Implementation of `stablehlo-canonicalize-dot-general`.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Patterns.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_CANONICALIZEDOTGENERALPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

using namespace mlir;
using namespace mlir::stablehlo;

namespace {

/// Since StableHLO operations may have a more general result type than what
/// could be statically determined, all patterns in this file must use this base
/// class in order to verify the static shape precondition. Otherwise, some of
/// the logic may result in incorrect IR. It is up to other passes to
/// canonicalize/refine operations whose shapes can be refined.
struct DotGeneralCanonicalizerBase : public RewritePattern {
  DotGeneralCanonicalizerBase(MLIRContext *context, PatternBenefit benefit = 1,
                              ArrayRef<StringRef> generatedNames = {})
      : RewritePattern(DotGeneralOp::getOperationName(), benefit, context,
                       generatedNames) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    auto dotGeneralOp = cast<DotGeneralOp>(op);
    if (!dotGeneralOp.getType().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "op does not have static shape");
    return matchAndRewrite(dotGeneralOp, rewriter);
  }

  virtual LogicalResult matchAndRewrite(DotGeneralOp op,
                                        PatternRewriter &rewriter) const = 0;
};

/// Rewrite `stablehlo.dot_general` so that batching dimensions are contiguous
/// and are at the front.
struct RewriteDotGeneral : public DotGeneralCanonicalizerBase {
  using DotGeneralCanonicalizerBase::DotGeneralCanonicalizerBase;

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    stablehlo::DotDimensionNumbersAttr dimNums = op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchDims = dimNums.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchDims = dimNums.getRhsBatchingDimensions();

    auto getPerm = [](ArrayRef<int64_t> bds,
                      int64_t rank) -> std::optional<SmallVector<int64_t>> {
      llvm::SmallSetVector<int64_t, 4> result(bds.begin(), bds.end());
      for (int64_t dim = 0; dim < rank; dim++) {
        if (!result.contains(dim))
          result.insert(dim);
      }
      if (llvm::equal(result, llvm::seq<int64_t>(0, rank)))
        return std::nullopt;
      return llvm::to_vector(
          llvm::iterator_range(result.begin(), result.end()));
    };

    auto lhsPermutation =
        getPerm(lhsBatchDims, op.getLhs().getType().getRank());
    auto rhsPermutation =
        getPerm(rhsBatchDims, op.getRhs().getType().getRank());
    if (!lhsPermutation && !rhsPermutation)
      return failure();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    if (lhsPermutation)
      lhs = rewriter.create<stablehlo::TransposeOp>(op.getLoc(), lhs,
                                                    *lhsPermutation);
    if (rhsPermutation)
      rhs = rewriter.create<stablehlo::TransposeOp>(op.getLoc(), rhs,
                                                    *rhsPermutation);

    auto getContractDims = [](ArrayRef<int64_t> contractDims,
                              ArrayRef<int64_t> perm) {
      SmallVector<int64_t> result;
      result.reserve(contractDims.size());
      for (int64_t x : contractDims)
        result.push_back(std::distance(perm.begin(), llvm::find(perm, x)));
      return result;
    };

    auto newDotOp = rewriter.replaceOpWithNewOp<stablehlo::DotGeneralOp>(
        op, op.getType(), SmallVector<Value>{lhs, rhs}, op->getAttrs());
    SmallVector<int64_t> newBatchDims =
        llvm::to_vector(llvm::seq<int64_t>(0, lhsBatchDims.size()));
    newDotOp.setDotDimensionNumbersAttr(stablehlo::DotDimensionNumbersAttr::get(
        getContext(), newBatchDims, newBatchDims,
        lhsPermutation ? getContractDims(dimNums.getLhsContractingDimensions(),
                                         *lhsPermutation)
                       : llvm::to_vector(dimNums.getLhsContractingDimensions()),
        rhsPermutation
            ? getContractDims(dimNums.getRhsContractingDimensions(),
                              *rhsPermutation)
            : llvm::to_vector(dimNums.getRhsContractingDimensions())));
    return success();
  }
};
} // namespace

/// Return true if the `seq` is a stride-1 sequence `[start, stop)`.
static bool isSequence(ArrayRef<int64_t> seq, int64_t start, int64_t stop) {
  return llvm::equal(seq, llvm::seq<int64_t>(start, stop));
}

/// Collapses contracting and outer dimensions of the given `dotOpOperand` using
/// a sequence of transpose and reshape. To keep the canonical form,
/// irrespective of whether `dotOpOperand` is LHS or RHS, `leftSideIndices` are
/// indices of outer (result) dimensions and `rightSideIndices` are indices of
/// the contraction dimensions. First transpose makes sure to have
/// `leftSideIndices` before `rightSideIndices`. Reshape after that collapses
/// both `leftSIdeIndices` and `rightSideIndices`. Almost every time, there is a
/// single non contracting outer dimension since we isolate batch dimensions.
static Value transposeAndReshapeOperand(PatternRewriter &rewriter, Location loc,
                                        ArrayRef<int64_t> batchIndices,
                                        ArrayRef<int64_t> leftSideIndices,
                                        ArrayRef<int64_t> rightSideIndices,
                                        Value dotOpOperand) {
  RankedTensorType dotOpOperandType =
      cast<RankedTensorType>(dotOpOperand.getType());
  int64_t collapsedLeftSideShape = 1;
  for (const int64_t &idx : leftSideIndices)
    collapsedLeftSideShape *= dotOpOperandType.getDimSize(idx);

  int64_t collapsedRightSideShape = 1;
  for (const int64_t &idx : rightSideIndices)
    collapsedRightSideShape *= dotOpOperandType.getDimSize(idx);

  SmallVector<int64_t> transposePermutation(batchIndices);
  transposePermutation.reserve(dotOpOperandType.getRank());
  llvm::append_range(transposePermutation, leftSideIndices);
  llvm::append_range(transposePermutation, rightSideIndices);

  Value transpose = rewriter
                        .create<stablehlo::TransposeOp>(loc, dotOpOperand,
                                                        transposePermutation)
                        .getResult();
  SmallVector<int64_t> reshapeShape(
      dotOpOperandType.getShape().take_front(batchIndices.size()));
  reshapeShape.push_back(collapsedLeftSideShape);
  reshapeShape.push_back(collapsedRightSideShape);

  RankedTensorType reshapeShapeType =
      RankedTensorType::get(reshapeShape, dotOpOperandType.getElementType());
  return rewriter.create<stablehlo::ReshapeOp>(loc, reshapeShapeType, transpose)
      .getResult();
}

/// Process a single operand of dot general by adding a transpose and a reshape.
static Value processOperand(PatternRewriter &rewriter, Location loc,
                            int64_t numBatchDims,
                            ArrayRef<int64_t> contractDims,
                            Value dotOpOperand) {
  RankedTensorType dotOpOperandType =
      cast<RankedTensorType>(dotOpOperand.getType());
  SmallVector<int64_t> operandIndices =
      llvm::to_vector(llvm::seq<int64_t>(0, dotOpOperandType.getRank()));
  ArrayRef<int64_t> operandIndicesRef = ArrayRef<int64_t>(operandIndices);
  ArrayRef<int64_t> batchIndices = operandIndicesRef.take_front(numBatchDims);
  ArrayRef<int64_t> nonBatchIndices =
      operandIndicesRef.drop_front(numBatchDims);

  // Find non batched outer(result) dimension indices since only those are
  // operated upon. Outer dimensions are non batching and non contracting
  // dimensions.
  SmallVector<bool> isDimensionOuter(dotOpOperandType.getRank(), true);
  for (const int64_t &dim : contractDims)
    isDimensionOuter[dim] = false;
  SmallVector<int64_t> outerDimensions;
  for (const int64_t &nonBatchIndex : nonBatchIndices) {
    if (isDimensionOuter[nonBatchIndex])
      outerDimensions.push_back(nonBatchIndex);
  }
  return transposeAndReshapeOperand(
      rewriter, loc, batchIndices, outerDimensions, contractDims, dotOpOperand);
}

namespace {
/// This pattern collapses contracting and outer (result) dimensions by applying
/// transpose and reshape to LHS and RHS operand. At the end of this
/// transform, dot general lhs operand will be in the form [b1, ..bn, m, k] and
/// rhs operand will be in the form [b1, ..bn, n, k], with lhs contracting
/// dimension of `shape(lhs)[-1]` and rhs contracting dimension of
/// `shape(rhs)[-1]`. To be in canonical form, contracting dimensions are always
/// at the end.
struct DotGeneralCollapsingRewrite : public DotGeneralCanonicalizerBase {
  using DotGeneralCanonicalizerBase::DotGeneralCanonicalizerBase;
  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    //  TODO: Dynamic shape can be handled for non-contracting dims without any
    //  work except more tests and removing some of these restrictions, and
    //  dynamic shapes could be handled for contracting dims with a little extra
    //  work.
    if (!op.getLhs().getType().hasStaticShape() ||
        !op.getRhs().getType().hasStaticShape() ||
        !op.getType().hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "either lhs, rhs or result has dynamic shape");

    stablehlo::DotDimensionNumbersAttr cfg = op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchDims = cfg.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchDims = cfg.getRhsBatchingDimensions();
    // This pattern expects canonical batch dims (all leading dims).
    // Another pattern above takes care of putting the batching dims into this
    // form.
    if (!isSequence(lhsBatchDims, 0, lhsBatchDims.size()) ||
        rhsBatchDims != lhsBatchDims)
      return rewriter.notifyMatchFailure(
          op, "this pattern expects canonical batch dims");

    // Empty contraction dims is mapped to `multiply` by
    // `DotGeneralToMulRewriter` pattern.
    ArrayRef<int64_t> rhsContractDims = cfg.getRhsContractingDimensions();
    ArrayRef<int64_t> lhsContractDims = cfg.getLhsContractingDimensions();
    if (lhsContractDims.empty())
      return rewriter.notifyMatchFailure(
          op, "This pattern does not apply if contraction dimension is empty");

    const int64_t numBatchDims = lhsBatchDims.size();
    ArrayRef<int64_t> mExtents =
        op.getLhs().getType().getShape().drop_front(numBatchDims).drop_back(1);
    ArrayRef<int64_t> nExtents =
        op.getRhs().getType().getShape().drop_front(numBatchDims).drop_back(1);
    if (mExtents.size() == 1 && nExtents.size() == 1)
      return rewriter.notifyMatchFailure(op, "nothing to simplify");

    Value newLhs = processOperand(rewriter, op->getLoc(), lhsBatchDims.size(),
                                  lhsContractDims, op.getLhs());
    RankedTensorType newLhsType = cast<RankedTensorType>(newLhs.getType());
    Value newRhs = processOperand(rewriter, op->getLoc(), lhsBatchDims.size(),
                                  rhsContractDims, op.getRhs());
    RankedTensorType newRhsType = cast<RankedTensorType>(newRhs.getType());

    SmallVector<int64_t> newOutputShape(
        op.getLhs().getType().getShape().take_front(lhsBatchDims.size()));
    newOutputShape.push_back(newLhsType.getShape().take_back(2)[0]);
    newOutputShape.push_back(newRhsType.getShape().take_back(2)[0]);
    auto newConfig = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), cfg.getLhsBatchingDimensions(),
        cfg.getRhsBatchingDimensions(),
        SmallVector<int64_t>{newLhsType.getRank() - 1},
        SmallVector<int64_t>{newRhsType.getRank() - 1});
    Value replacement = rewriter.create<stablehlo::DotGeneralOp>(
        op->getLoc(), op.getType().clone(newOutputShape), newLhs, newRhs,
        newConfig, op.getPrecisionConfigAttr(), op.getAlgorithmAttr());
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(),
                                                      replacement);
    return success();
  }
};

} // namespace

void stablehlo_ext::populateCanonicalizeStablehloDotGeneralPatterns(
    RewritePatternSet &patterns) {
  patterns.add<RewriteDotGeneral, DotGeneralCollapsingRewrite>(
      patterns.getContext());
  stablehlo_ext::populateStablehloDotGeneralToMultiplyPatterns(patterns);
}

namespace {
class CanonicalizeDotGeneralPass
    : public stablehlo_ext::impl::CanonicalizeDotGeneralPassBase<
          CanonicalizeDotGeneralPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    stablehlo_ext::populateCanonicalizeStablehloDotGeneralPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc())
          << "failed to apply rewrite patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
