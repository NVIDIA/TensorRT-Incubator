//===- TransposeElimination.cpp -------------------------------------------===//
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
/// Definition of transpose elimination pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir-tensorrt-dialect/Utils/ConstantFoldUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <numeric>

namespace mlir {
namespace tensorrt {
#define GEN_PASS_DEF_TRANSPOSEELIMINATIONPASS
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h.inc"
} // namespace tensorrt
} // namespace mlir

#define DEBUG_TYPE "tensorrt-transpose-elimination"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;
using namespace mlir::tensorrt;

static int64_t memoryCost(TensorType type) {
  ArrayRef<int64_t> shape = type.getShape();
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

static TransposeOp getLowestTransposeCost(ElementWiseOp consumer,
                                          TransposeOp op1, TransposeOp op2) {
  // If there's only one transpose, then return it.
  if (!op1 || !op2)
    return op1 ? op1 : op2;
  // Otherwise, compute the lowest cost.
  int64_t cost1 = memoryCost(consumer.getType()) + memoryCost(op2.getType());
  int64_t cost2 = memoryCost(consumer.getType()) + memoryCost(op1.getType());
  LLVM_DEBUG(DBGS() << "cost1=" << cost1 << ", cost2=" << cost2 << "\n");
  return cost1 <= cost2 ? op1 : op2;
}

static std::pair<TransposeOp, TransposeOp>
getTransposeProducers(ElementWiseOp op) {
  auto producer1 = op.getInput1().getDefiningOp<TransposeOp>();
  auto producer2 = op.getInput2().getDefiningOp<TransposeOp>();
  return std::make_pair(producer1, producer2);
}

static TensorValue getOtherEwiseInput(ElementWiseOp op, Operation *producer) {
  assert(producer != nullptr && "expected valid producer");
  Operation *producer1 = op.getInput1().getDefiningOp();
  return producer1 == producer ? op.getInput2() : op.getInput1();
}

// If there is only one ewise branch with a transpose, the below pushdown
// pattern may not terminate by repeatedly ping-ponging. We avoid this by having
// a set of conditions on which to move transpose to the other branch. We do
// this if we know doing so will result in additional elimination patterns or a
// smaller transpose cost.
bool pushDownTransposePrecondition(ElementWiseOp op,
                                   TransposeOp transposeToPushdown) {
  TensorValue otherInput = getOtherEwiseInput(op, transposeToPushdown);
  Operation *otherProducer = otherInput.getDefiningOp();
  bool otherBranchHasSmallerCost =
      memoryCost(otherInput.getType()) <
      memoryCost(transposeToPushdown.getResult().getType());
  if (otherBranchHasSmallerCost)
    return true;
  // Even if the other branch has higher memory cost:
  // 1. If its a constant, then we can fold.
  // 2. If its a transpose, then we can combine transpose ops.
  return isa_and_nonnull<ConstantOp, TransposeOp>(otherProducer);
}

namespace {
/// This rewrite tries to eliminate transpose operations by rotating them
/// "forward" (e.g. pushing them past their user(s)) when they have a single
/// user and when that user is some sort of computation operation (e.g.
/// elementwise, unary, or convolution).
struct PushdownTransposeEwise : public OpRewritePattern<ElementWiseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ElementWiseOp op,
                                PatternRewriter &rewriter) const override {
    auto [transpose1, transpose2] = getTransposeProducers(op);
    // Exit if there are not transpose producers.
    if (!transpose1 && !transpose2)
      return failure();
    // Choose which transpose to push down. The one to push down should be the
    // one that results in the lowest memory cost. In all cases there will be
    // two transpose produced. Choose the one with the lowest cost. If they are
    // equal, choose either one.
    TransposeOp transposeToPushdown =
        getLowestTransposeCost(op, transpose1, transpose2);

    // If the other branch has smaller cost, we always move the transpose.
    if (!pushDownTransposePrecondition(op, transposeToPushdown))
      return rewriter.notifyMatchFailure(
          op, "does not meet transpose pushdown preconditions");

    LLVM_DEBUG(DBGS() << "pushing down transpose " << transposeToPushdown
                      << "\n");

    // Execute the transformation.
    AffineMap permutation = transposeToPushdown.getPermutation();
    AffineMap inversePerm = inversePermutation(permutation);
    Value otherInput = getOtherEwiseInput(op, transposeToPushdown);
    Value transposedOther =
        rewriter.create<TransposeOp>(op.getLoc(), otherInput, inversePerm);
    bool pushdownIsInput1 = op.getInput1() == transposeToPushdown.getResult();
    Value ewiseOutput = rewriter.create<ElementWiseOp>(
        op.getLoc(),
        pushdownIsInput1 ? transposeToPushdown.getInput() : transposedOther,
        !pushdownIsInput1 ? transposeToPushdown.getInput() : transposedOther,
        op.getElementwiseOperation());
    rewriter.replaceOpWithNewOp<TransposeOp>(op, ewiseOutput, permutation);
    return success();
  }
};

/// Rewrites `act(transpose(x))` to `transpose(act(x))`.
struct PushDownTransposeActivationRewriter
    : public OpRewritePattern<ActivationOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ActivationOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getInput().getDefiningOp<TransposeOp>();
    if (!producer)
      return failure();
    AffineMap permutation = producer.getPermutation();
    auto activationOp = rewriter.create<ActivationOp>(
        op.getLoc(), producer.getInput(), op.getActivationType(),
        op.getAlphaAttr(), op.getBetaAttr());
    rewriter.replaceOpWithNewOp<TransposeOp>(op, activationOp.getResult(),
                                             permutation);
    return success();
  }
};

/// Push transpose below `tensorrt.identity` if the identity consumer is an
/// elementwise op. After the "pushdown" phase, the "push up" phase will restore
/// the transpose above the identity if it could not be eliminated and the
/// source has a smaller memory cost.
struct PushdownTransposeIdentity : public OpRewritePattern<IdentityOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IdentityOp op,
                                PatternRewriter &rewriter) const override {
    TransposeOp transposeProducer = op.getInput().getDefiningOp<TransposeOp>();
    if (!transposeProducer)
      return failure();
    Type newIdentityType =
        RankedTensorType::Builder(
            transposeProducer.getInput().getType().cast<RankedTensorType>())
            .setElementType(op.getType().getElementType());
    Value newIdentityResult = rewriter.create<IdentityOp>(
        op.getLoc(), newIdentityType, transposeProducer.getInput());
    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, newIdentityResult, transposeProducer.getPermutation());
    return success();
  }
};

/// Rewrite transpose(unary(x)) to unary(transpose(x)).
template <typename OpType>
struct PushUpTransposeUnary : OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto unaryOp = op.getInput().getDefiningOp<OpType>();
    if (!unaryOp)
      return failure();
    Value newUnaryResult = rewriter.create<TransposeOp>(
        op.getLoc(), unaryOp.getInput(), op.getPermutation());
    rewriter.replaceOpWithNewOp<OpType>(op, op.getType(), newUnaryResult,
                                        unaryOp->getAttrs());
    return success();
  }
};

/// The cost of a `reshape` is always zero, because the implementation of a
/// reshape is always just rearrangement of metadata and it does not involve
/// launching a CUDA kernel. The cost of a `transpose` will be zero if its input
/// is a `ConstantOp`. However, such case doesn't occur since constant folding
///  eliminates transpose op even before this pass runs.
static int64_t
reshapeOrTransposeOrConstantCost(Operation *reshapeOrTransposeOrConstant) {
  if (isa<ReshapeOp>(reshapeOrTransposeOrConstant))
    return 0;
  if (isa<ConstantOp>(reshapeOrTransposeOrConstant))
    return 0;
  TransposeOp op = dyn_cast<TransposeOp>(reshapeOrTransposeOrConstant);
  assert(op);
  return memoryCost(op.getResult().getType());
}

/// Computes benefit of pushing transpose above elementwise and after
/// reshape/transpose producer.
/// Benefit = Original cast - New cost
/// Original cost = memoryCost(original reshape/transpose/constant) +
/// memoryCost(transpose after elementwise)
/// New cost = memoryCost(original reshape/transpose/constant) +
/// memoryCost(pushed up transpose) + memoryCost(newly added transpose on the
/// other side)
static int64_t pushUpBenefit(ElementWiseOp elementwise,
                             Operation *reshapeOrTransposeOrConstantProducer) {
  int64_t originalTransposeReshapeCost =
      reshapeOrTransposeOrConstantCost(reshapeOrTransposeOrConstantProducer);
  int64_t originalCost = originalTransposeReshapeCost +
                         memoryCost(elementwise.getResult().getType());
  // The cost of the pushed up transpose will be 0 in the following cases,
  // a. The parent of the elementwise input is a `reshape` op with a
  // `ConstantOp` input.
  // b. The parent of the elementwise input is a `ConstantOp`.
  // Such case won't happens for `transpose` op since its folded away even
  // before this pass.
  int64_t pushedUpTransposeCost =
      (matchPattern(reshapeOrTransposeOrConstantProducer,
                    m_Op<ReshapeOp>(m_Op<ConstantOp>())) ||
       isa<ConstantOp>(reshapeOrTransposeOrConstantProducer))
          ? 0
          : memoryCost(elementwise.getResult().getType());
  // If other side input to elementwise is coming from `ConstantOp`, cost of the
  // newly added transpose will be zero.
  TensorValue otherSideValue =
      getOtherEwiseInput(elementwise, reshapeOrTransposeOrConstantProducer);
  int64_t otherSideNewlyAddedTransposeCost =
      otherSideValue.getDefiningOp<ConstantOp>()
          ? 0
          : memoryCost(otherSideValue.getType());

  int64_t newCost = originalTransposeReshapeCost + pushedUpTransposeCost +
                    otherSideNewlyAddedTransposeCost;
  return originalCost - newCost;
}

/// Returns true if pushing transpose op above given `elementwise` op is
/// beneficial.
static bool shouldPushUpTransposeElementwise(
    ElementWiseOp elementwise,
    Operation *reshapeOrTransposeOrConstantProducer) {
  return (pushUpBenefit(elementwise, reshapeOrTransposeOrConstantProducer) > 0);
}

/// Push transpose above elementwise if both of the following conditions hold
/// true,
/// 1. Input to the transpose is the output of an elementwise op.
/// 2. One or both of the inputs to the elementwise op is an output of transpose
/// or reshape op.
/// Idea is, if transpose is pushed up in this case, transpose/transpose or
/// reshape/transpose pair at the input of elementwise op will result in a
/// single shuffle op (both reshape and transpose are canonicalized to the
/// shuffle op, later in the pipeline).
///
/// For example, subgraph in the form
///
/// %2 = [reshape, transpose](%0)
/// %3 = [reshape, transpose](%1)
/// %4 = elementwise(%2, %3)
/// %5 = transpose(%4)
///
/// where [reshape, transpose](%k) means if (%k can be passed without any
/// transformation as well) transformation is applied to %k, it will be either
/// reshape or transpose. However, if both %0 and %1 does not have any
/// transformation, this pattern doesn't apply. Both or any one of %0 and %1 can
/// be an output of constant op.
///
/// is rewritten to
///
/// %2 = [reshape, transpose](%0)
/// %4 = transpose1(%2)
/// %3 = [reshape, transpose](%1)
/// %5 = transpose2(%3)
/// %6 = elementwise(%4, %5)
///
/// In short, the transpose (called `pushed-up` transpose, hereafter) after
/// elementwise is pushed above elementwise, before lhs or rhs input, if input's
/// parent is reshape/transpose. A `new` transpose is added on the other side of
/// `pushed-up` transpose. Benefit of applying this pattern is computed based on
/// `memoryCost` difference in the original and modified subgraph.

struct PushUpTransposeElementwise : OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto elementwiseOp = op.getInput().getDefiningOp<ElementWiseOp>();
    if (!elementwiseOp)
      return failure();
    Operation *lhsParent = elementwiseOp.getInput1().getDefiningOp();
    Operation *rhsParent = elementwiseOp.getInput2().getDefiningOp();
    bool isLhsParentReshapeOrTransposeOrConstant =
        lhsParent && isa<ReshapeOp, TransposeOp, ConstantOp>(lhsParent);
    bool isRhsParentReshapeOrTransposeOrConstant =
        rhsParent && isa<ReshapeOp, TransposeOp, ConstantOp>(rhsParent);
    if (!isLhsParentReshapeOrTransposeOrConstant &&
        !isRhsParentReshapeOrTransposeOrConstant)
      return failure();

    auto addTransposeOp = [&](Value input, AffineMap perm) {
      return rewriter
          .create<TransposeOp>(op->getLoc(),
                               /*input=*/input,
                               /*permutation=*/perm)
          .getResult();
    };

    auto rewriteElementwiseOp = [&](Value lhs, Value rhs) {
      return rewriter
          .replaceOpWithNewOp<ElementWiseOp>(
              elementwiseOp,
              /*input1=*/lhs,
              /*input2=*/rhs,
              /*op=*/elementwiseOp.getElementwiseOperation())
          .getResult();
    };

    auto pushUpTransposeOnLhs = [&]() {
      // Add a transpose before RHS
      auto rhsTranspose =
          addTransposeOp(elementwiseOp.getInput2(), op.getPermutation());
      // Push up transpose from after elementwise to before elementwise on LHS
      auto pushedUpTransposeLhs =
          addTransposeOp(elementwiseOp.getInput1(), op.getPermutation());
      // Update elementwise op
      auto updatedElementwiseOp =
          rewriteElementwiseOp(pushedUpTransposeLhs, rhsTranspose);
      // Replace uses of old transpose
      rewriter.replaceAllUsesWith(op.getResult(), updatedElementwiseOp);
    };

    auto pushUpTransposeOnRhs = [&]() {
      // Add a transpose before LHS
      auto lhsTranspose =
          addTransposeOp(elementwiseOp.getInput1(), op.getPermutation());
      // Push up transpose from after elementwise to before elementwise on RHS
      auto pushedUpTransposeRhs =
          addTransposeOp(elementwiseOp.getInput2(), op.getPermutation());
      // Update elementwise op
      auto updatedElementwiseOp =
          rewriteElementwiseOp(lhsTranspose, pushedUpTransposeRhs);
      // Replace uses of old transpose
      rewriter.replaceAllUsesWith(op.getResult(), updatedElementwiseOp);
    };

    if (isLhsParentReshapeOrTransposeOrConstant &&
        shouldPushUpTransposeElementwise(elementwiseOp, lhsParent)) {
      pushUpTransposeOnLhs();
      return success();
    }

    if (isRhsParentReshapeOrTransposeOrConstant &&
        shouldPushUpTransposeElementwise(elementwiseOp, rhsParent)) {
      pushUpTransposeOnRhs();
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
/// Constant fold transpose
struct TransposeConstantFold : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    TensorType inputType = op.getInput().getType();

    // Fold  the input to a constant if possible, otherwise return.
    ElementsAttr inputConst;
    if (!matchPattern(op.getInput(), m_Constant(&inputConst)))
      return failure();
    assert(inputType.hasStaticShape() && "constants should have static shape");

    // Don't fold transpose if input has > 1 user and input is non-splat
    // constant.
    if (!inputConst.isSplat() && !op.getInput().hasOneUse())
      return failure();

    ElementsAttr result =
        constantFoldTranspose(inputConst, op.getPermutation());
    if (!result)
      return failure();
    rewriter.replaceOpWithNewOp<ConstantOp>(op, result);
    return success();
  }
};
} // namespace

namespace {
class TransposeEliminationPass
    : public tensorrt::impl::TransposeEliminationPassBase<
          TransposeEliminationPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();

    // First, we try to eliminate transpose operations by "pushing down" the
    // transpose operations. This involves performing rewrites of the form
    // "op(transpose(y))->transpose(op(y))". Often, this will eliminate most
    // transpose operations in CNN networks produced by frameworks that use NHWC
    // conventions (e.g. Tensorflow and often JAX/Flax models).
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<PushdownTransposeEwise, TransposeConstantFold,
                      PushdownTransposeIdentity,
                      PushDownTransposeActivationRewriter>(ctx);
      TransposeOp::getCanonicalizationPatterns(patterns, ctx);
      ExpandRankOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc())
            << "failed to apply pushdown patterns in " << getArgument();
        return signalPassFailure();
      }
    }

    // Second, we try to eliminate transpose operations by "pushing up" (commute
    // in the reverse direction). This can possible eliminate additional
    // transpose ops.
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<TransposeConstantFold, PushUpTransposeUnary<IdentityOp>,
                      PushUpTransposeUnary<UnaryOp>,
                      PushUpTransposeUnary<ActivationOp>,
                      PushUpTransposeElementwise>(ctx);
      TransposeOp::getCanonicalizationPatterns(patterns, ctx);
      ExpandRankOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc())
            << "failed to apply pushup patterns in " << getArgument();
        return signalPassFailure();
      }
    }
  }
};
} // namespace
