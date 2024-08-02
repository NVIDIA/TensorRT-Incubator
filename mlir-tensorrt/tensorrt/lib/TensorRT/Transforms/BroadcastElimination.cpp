//===- BroadcastElimination.cpp----------------------------------*- c++ -*-===//
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
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tensorrt {
#define GEN_PASS_DEF_BROADCASTELIMINATIONPASS
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h.inc"
} // namespace tensorrt
} // namespace mlir

using namespace mlir;
using namespace mlir::tensorrt;

/// Consider
/// ```
/// %0 = broadcast %arg0 : tensor<1x1x10xf32> to tensor<1x100x10xf32>
/// %1 = collapse_rank %0 : tensor<1x100x10xf32> to tensor<100x10xf32>
/// ```
/// In the above case, the first removed dimension is "0".
/// the `broadcast` op will tell us that [0, 1, 2] are broadcasted dims.
/// Since 0 is a member of the broadcast dimension, we know that the
/// corresponding dimension in %arg0 is a unit dim, so we can eliminate
/// it early.
/// ```
/// %0 = collapse_rank %arg0 : tensor<1x1x10xf32> to tensor<1x10xf32>
/// %1 = broadcast %0 : tensor<100x10xf32>
/// ```
/// This function returns the new broadcast operation (the replacement op).
static FailureOr<BroadcastOp> exchangeCollapseRankAndBroadcast(
    RewriterBase &rewriter, CollapseRankOp collapseOp, BroadcastOp bcastOp) {
  // Check which indices are removed.
  SmallVector<int64_t> removedDims =
      collapseOp.getInputShapeDimIndicesOfRemovedDims();
  if (removedDims.empty())
    return failure();

  // Let's just focus on the first removed dimension.
  int64_t focusDim = removedDims.front();
  SmallVector<int64_t> bcastInputShape(bcastOp.getInput().getType().getShape());
  SmallVector<int64_t> broadcastDims(bcastOp.getBroadcastDims());

  // If it is broadcasted, then we must remove it at the input. Drop this dim
  // from the list, and all indices higher than this must be decremented.
  int64_t *bcastDimIter = llvm::find(broadcastDims, focusDim);
  // TODO: can we handle this case?
  if (bcastDimIter == broadcastDims.end())
    return failure();

  // Find which input dimension this corresponds to.
  unsigned inputShapeDimIdx =
      std::distance(broadcastDims.begin(), bcastDimIter);
  assert(bcastInputShape[inputShapeDimIdx] == 1);

  // Erase this broadcast dimension.
  bcastInputShape.erase(bcastInputShape.begin() + inputShapeDimIdx);
  broadcastDims.erase(bcastDimIter);
  // Adjust all the other broadcast dimensions.
  for (auto &bcastDim : broadcastDims) {
    if (bcastDim > focusDim)
      bcastDim--;
  }

  Type newCollapseShapeType =
      RankedTensorType::Builder(
          bcastOp.getInput().getType().cast<RankedTensorType>())
          .setShape(bcastInputShape);

  Value newBcastInput = rewriter.create<CollapseRankOp>(
      bcastOp.getLoc(), newCollapseShapeType, bcastOp.getInput());
  auto newBroadcastOp = rewriter.create<BroadcastOp>(
      collapseOp.getLoc(), collapseOp.getType(), newBcastInput, broadcastDims);

  return newBroadcastOp;
}

namespace {
/// Given a `tensorrt.collapse_rank` operation, if the source operation is a
/// broadcast op, then commute the `collapse_rank` with the broadcast (i.e. push
/// down the broadcast so that it acts on the result of the collapse rank op).
struct PushDownBroadcastReduceRankOp : public OpRewritePattern<CollapseRankOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CollapseRankOp op,
                                PatternRewriter &rewriter) const override {
    // Check that the input of `collapse_rank` is produced by a broadcast
    // operation.
    auto broadcastOp = op.getInput().getDefiningOp<BroadcastOp>();
    if (!broadcastOp)
      return failure();

    FailureOr<BroadcastOp> newBroadcastOp =
        exchangeCollapseRankAndBroadcast(rewriter, op, broadcastOp);
    if (failed(newBroadcastOp))
      return failure();

    rewriter.replaceOp(op, newBroadcastOp->getResult());
    return success();
  }
};
} // namespace

namespace {
/// Create transpose + expand_rank on the input of a `tensorrt.broadcast` so
/// that the result has the same rank as the `tensorrt.broadcast` result and the
/// equivalent `broadcastDims` will preserve the ordering of the input dims.
struct SimplifyBroadcast : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    // Determine the ordering of the dimensions after the broadcast map.
    Location loc = op.getLoc();
    auto perm = op.getBroadcastDimsPermutation();
    TensorValue input = op.getInput();
    auto resultType = op.getResult().getType().cast<RankedTensorType>();

    if (perm.isIdentity() && resultType.getRank() == input.getType().getRank())
      return failure();

    // If input rank is > 0-D, then we need to transpose.
    if (input.getType().getRank() > 0) {
      auto transposeOp =
          rewriter.create<tensorrt::TransposeOp>(loc, input, perm);
      // This should just be sorted version of perm.
      SmallVector<int64_t> reorderedBroadcastDims =
          applyPermutationMap(perm, op.getBroadcastDims());

      // Calculate pre-expanded shape.
      SmallVector<int64_t> expandedShape(resultType.getRank());
      RankedTensorType transposeType = transposeOp.getResult().getType();
      unsigned inputIdx = 0;
      for (unsigned i = 0, e = expandedShape.size(); i < e; i++) {
        if (inputIdx < reorderedBroadcastDims.size() &&
            i == reorderedBroadcastDims[inputIdx]) {
          expandedShape[i] = transposeType.getDimSize(inputIdx++);
          continue;
        }
        expandedShape[i] = 1;
      }
      Value expanded = rewriter.create<ExpandRankOp>(
          loc, resultType.clone(expandedShape), transposeOp);
      rewriter.replaceOpWithNewOp<BroadcastOp>(
          op, op.getType(), expanded, op.getShape(),
          llvm::to_vector(llvm::seq<int64_t>(0, resultType.getRank())));
      return success();
    }
    // Otherwise, just reshape to all 1's directly.
    Type expandedRankType =
        RankedTensorType::Builder(resultType)
            .setShape(SmallVector<int64_t>(resultType.getRank(), 1));
    Value expanded =
        rewriter.create<ExpandRankOp>(loc, expandedRankType, input);
    rewriter.replaceOpWithNewOp<BroadcastOp>(
        op, op.getType(), expanded, op.getShape(),
        llvm::to_vector(llvm::seq<int64_t>(0, resultType.getRank())));
    return success();
  }
};
} // namespace

/// Returns true if the broadcast op does not do transposition and does not do
/// rank expansion.
static bool broadcastAbsorbPreconditions(BroadcastOp broadcastOp) {
  return broadcastOp.getInput().getType().getRank() ==
             broadcastOp.getType().getRank() &&
         broadcastOp.getBroadcastDimsPermutation().isIdentity();
}

namespace {
/// A `tensorrt.element_wise` operation can pull in `tensorrt.broadcast`
/// operations since the elementwise op has implicit broadcasting semantics.
struct ElementwiseAbsorbBroadcast : public OpRewritePattern<ElementWiseOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ElementWiseOp op,
                                PatternRewriter &rewriter) const override {
    TensorValue input1 = op.getInput1();
    TensorValue input2 = op.getInput2();

    if (auto broadcastOp = input1.getDefiningOp<BroadcastOp>()) {
      if (broadcastAbsorbPreconditions(broadcastOp))
        input1 = broadcastOp.getInput();
    }
    if (auto broadcastOp = input2.getDefiningOp<BroadcastOp>()) {
      if (broadcastAbsorbPreconditions(broadcastOp))
        input2 = broadcastOp.getInput();
    }

    // Return failure if nothing changed.
    if (input1 == op.getInput1() && op.getInput2() == input2)
      return failure();

    // You can't eliminate both broadcasts if the same unit-dim in both
    // operands is being broadcast to a larger value. We can do some further
    // simplification, but we leave that to other patterns.
    for (unsigned i = 0; i < input1.getType().getRank(); i++) {
      if (input1.getType().getDimSize(i) == 1 &&
          input2.getType().getDimSize(i) == 1 && op.getType().getDimSize(i) > 1)
        return failure();
    }
    rewriter.replaceOpWithNewOp<ElementWiseOp>(op, op.getType(), input1, input2,
                                               op.getElementwiseOperation());
    return success();
  }
};

/// A `tensorrt.select` operation can pull in `tensorrt.broadcast`
/// operations since the select op has implicit broadcasting semantics.
struct SelectAbsorbBroadcast : public OpRewritePattern<SelectOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    TensorValue cond = op.getCondition();
    TensorValue input1 = op.getThenInput();
    TensorValue input2 = op.getElseInput();
    if (auto broadcastOp = cond.getDefiningOp<BroadcastOp>()) {
      if (broadcastAbsorbPreconditions(broadcastOp))
        cond = broadcastOp.getInput();
    }
    if (auto broadcastOp = input1.getDefiningOp<BroadcastOp>()) {
      if (broadcastAbsorbPreconditions(broadcastOp))
        input1 = broadcastOp.getInput();
    }
    if (auto broadcastOp = input2.getDefiningOp<BroadcastOp>()) {
      if (broadcastAbsorbPreconditions(broadcastOp))
        input2 = broadcastOp.getInput();
    }

    // Early exit if nothing changed.
    if (input1 == op.getThenInput() && op.getElseInput() == input2 &&
        op.getCondition() == cond)
      return failure();

    // We can't eliminate broadcasts on all three operands if the same unit-dim
    // in all operands is being broadcast to a larger value. We can do some
    // further simplification, but we leave that to other patterns.
    for (unsigned i = 0, e = input1.getType().getRank(); i < e; i++) {
      if (input1.getType().getDimSize(i) == 1 &&
          input2.getType().getDimSize(i) == 1 &&
          cond.getType().getDimSize(i) == 1 && op.getType().getDimSize(i) > 1)
        return failure();
    }

    rewriter.replaceOpWithNewOp<SelectOp>(op, op.getType(), cond, input1,
                                          input2);

    return success();
  }
};
} // namespace

/// Assuming that the first 'numLeadingBatchDims' of the operand shape are
/// "batch dimensions", checks that the action of the broadcast is only to
/// broadcast the operand along those dimensions, the other dimensions are left
/// unchanged.
static bool isBroadcastAlongBatchDimsOnly(BroadcastOp op,
                                          unsigned numLeadingBatchDims) {
  // - ranks of operand and result must match
  // - Trailing non-batch dims of operand and result must match and be static
  //   (for dynamic dims of input, we do not know if it is 1 or > 1).
  // - Broadcast dimensions should be identity mapping
  if (op.getType().getRank() != op.getInput().getType().getRank())
    return false;
  for (auto [lhs, rhs] :
       llvm::zip_equal(op.getType().getShape().drop_front(numLeadingBatchDims),
                       op.getInput().getType().getShape().drop_front(
                           numLeadingBatchDims))) {
    if (ShapedType::isDynamic(lhs) || lhs != rhs)
      return false;
  }
  return llvm::equal(op.getBroadcastDims(),
                     llvm::seq<int64_t>(0, op.getType().getRank()));
}

namespace {
/// A `tensorrt.matrix_multiplication` operation can pull in
/// `tensorrt.broadcast` operations that are broadcasting operands along the
/// batch dim.
struct MatMulAbsorbBroadcast : public OpRewritePattern<MatrixMultiplyOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MatrixMultiplyOp op,
                                PatternRewriter &rewriter) const override {
    auto isVectorOp = [](MatrixOperation matOp) {
      return matOp == MatrixOperation::kVECTOR;
    };
    int64_t numBatchDims =
        op.getInput0().getType().getRank() - (isVectorOp(op.getOp0()) ? 1 : 2);
    Value newLhs = op.getInput0();
    Value newRhs = op.getInput1();

    if (auto lhs = op.getInput0().getDefiningOp<BroadcastOp>()) {
      if (isBroadcastAlongBatchDimsOnly(lhs, numBatchDims))
        newLhs = lhs.getInput();
    }
    if (auto rhs = op.getInput1().getDefiningOp<BroadcastOp>()) {
      if (isBroadcastAlongBatchDimsOnly(rhs, numBatchDims))
        newRhs = rhs.getInput();
    }

    if (newLhs == op.getInput0() && newRhs == op.getInput1())
      return failure();

    rewriter.modifyOpInPlace(op, [&]() {
      op.getInput0Mutable().assign(newLhs);
      op.getInput1Mutable().assign(newRhs);
    });
    return success();
  }
};

class BroadcastEliminationPass
    : public tensorrt::impl::BroadcastEliminationPassBase<
          BroadcastEliminationPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyBroadcast, ElementwiseAbsorbBroadcast,
                 PushDownBroadcastReduceRankOp, SelectAbsorbBroadcast,
                 MatMulAbsorbBroadcast>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to apply broadcast elimination patterns";
      return signalPassFailure();
    }
  }
};
} // namespace
