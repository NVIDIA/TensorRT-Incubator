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
#include <unordered_map>

namespace mlir {
namespace tensorrt {
#define GEN_PASS_DEF_TRANSPOSERESHAPEELIMINATIONPASS
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h.inc"
} // namespace tensorrt
} // namespace mlir

#define DEBUG_TYPE "tensorrt-transpose-reshape-elimination"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;
using namespace mlir::tensorrt;

// Set the max size of tensors which can be constant-folded to 131072 (0.5 MB
// for f32 constants).
constexpr int64_t kFoldOpEltLimit = 1 << 17;

static int64_t memoryCost(RankedTensorType type) {
  // If the type is dynamic, then return max.
  if (!type.hasStaticShape())
    return std::numeric_limits<int64_t>::max();
  return type.getNumElements();
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
  if (cost1 == 0 && cost2 == 0)
    return {};
  return cost1 <= cost2 ? op1 : op2;
}

static std::pair<TransposeOp, TransposeOp>
getTransposeProducers(ElementWiseOp op) {
  auto producer1 = op.getInput1().getDefiningOp<TransposeOp>();
  auto producer2 = op.getInput2().getDefiningOp<TransposeOp>();
  if (producer1 && producer1.getInput().getDefiningOp<ConstantOp>())
    producer1 = {};
  if (producer2 && producer2.getInput().getDefiningOp<ConstantOp>())
    producer2 = {};
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
static bool pushDownTransposePrecondition(ElementWiseOp op,
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
    auto newTranspose = rewriter.create<TransposeOp>(
        producer.getLoc(), activationOp.getResult(), permutation);
    rewriter.replaceOp(op, newTranspose.getResult());
    return success();
  }
};

// Rewrites unary(transpose(x)) to transpose(unary(x))
struct PushDownTransposeUnary : OpRewritePattern<UnaryOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getInput().getDefiningOp<TransposeOp>();
    if (!producer)
      return failure();
    AffineMap permutation = producer.getPermutation();
    auto unary = rewriter.create<UnaryOp>(op.getLoc(), producer.getInput(),
                                          op.getUnaryOperationAttr());
    auto newTranspose = rewriter.create<TransposeOp>(
        producer.getLoc(), unary.getResult(), permutation);
    rewriter.replaceOp(op, newTranspose.getResult());
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
            cast<RankedTensorType>(transposeProducer.getInput().getType()))
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

    Location loc = op->getLoc();
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
          .create<TransposeOp>(loc,
                               /*input=*/input,
                               /*permutation=*/perm)
          .getResult();
    };

    auto rewriteElementwiseOp = [&](Value lhs, Value rhs) {
      return rewriter
          .create<ElementWiseOp>(loc,
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
    if (!inputConst.isSplat() &&
        (!op.getInput().hasOneUse() ||
         inputConst.getNumElements() > kFoldOpEltLimit))
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
// Convert a matrix multiply to an einsum
// einsum allows is more flexible with the inputs, braodcasting dimensions and
// transposing. Hence, we can easily implement rewrites that merge transpose
// into einsum and push reshape through an einsum
class MatmulToEinsum : public OpRewritePattern<tensorrt::MatrixMultiplyOp> {
public:
  using OpRewritePattern<tensorrt::MatrixMultiplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::MatrixMultiplyOp op,
                                PatternRewriter &rewriter) const override {
    using tensorrt::MatrixOperation;

    int numBatchDims = op.getCollectionRank(0);
    if (numBatchDims != op.getCollectionRank(1))
      return failure(/* unknown number of batch dimensions */);

    std::string arg0Pattern = "", arg1Pattern = "", outPattern = "";
    char nextChar = 'a';
    for (int i = 0; i < numBatchDims; i++) {
      // einsum supports broadcasting, so we just add the batch dims to the
      // pattern
      arg0Pattern += nextChar;
      arg1Pattern += nextChar;
      outPattern += nextChar++;
    }

    char matrix0A, matrix0B, matrix1A, matrix1B, multiplyLetter;
    if (op.getOp0() == MatrixOperation::kVECTOR) {
      matrix0A = 0;
      multiplyLetter = matrix0B = nextChar++;
      arg0Pattern += matrix0B;
    } else if (op.getOp0() == MatrixOperation::kNONE) { // normal matrix
      matrix0A = nextChar++;
      multiplyLetter = matrix0B = nextChar++;
      arg0Pattern += matrix0A;
      arg0Pattern += matrix0B;
      outPattern += matrix0A;
    } else if (op.getOp0() == MatrixOperation::kTRANSPOSE) {
      multiplyLetter = matrix0A = nextChar++;
      matrix0B = nextChar++;
      arg0Pattern += matrix0A;
      arg0Pattern += matrix0B;
      outPattern += matrix0B;
    } else {
      return failure(/* unknown matrix operation */);
    }

    if (op.getOp1() == MatrixOperation::kVECTOR) {
      matrix1A = multiplyLetter;
      matrix1B = 0;
      arg1Pattern += matrix1A;
    } else if (op.getOp1() == MatrixOperation::kNONE) { // normal matrix
      matrix1A = multiplyLetter;
      matrix1B = nextChar++;
      arg1Pattern += matrix1A;
      arg1Pattern += matrix1B;
      outPattern += matrix1B;
    } else if (op.getOp1() == MatrixOperation::kTRANSPOSE) {
      matrix1A = nextChar++;
      matrix1B = multiplyLetter;
      arg1Pattern += matrix1A;
      arg1Pattern += matrix1B;
      outPattern += matrix1A;
    } else {
      return failure(/* unknown matrix operation */);
    }

    SmallVector<Value> args{op.getInput0(), op.getInput1()};
    std::string einsum = arg0Pattern + "," + arg1Pattern + "->" + outPattern;
    rewriter.replaceOpWithNewOp<tensorrt::EinsumOp>(op, op.getType(), args,
                                                    einsum);
    return success();
  }
};
} // namespace

namespace {
// convert tensorrt.shuffle to tensorrt.transpose and tensorrt.reshape
// tensorrt.shuffle is the "lower level" op that eventually gets converted to
// INetwork layers it is possible that tensorrt.shuffle already exist in the
// network, hence, convert it back to the "simpler" reshape and transpose ops
// shuffle -> transpose(reshape(transpose(x)))
class ShuffleToTransposeAndReshape
    : public OpRewritePattern<tensorrt::ShuffleOp> {
public:
  using OpRewritePattern<tensorrt::ShuffleOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();

    if (op.getZeroIsPlaceholder())
      return failure();

    input = rewriter.createOrFold<tensorrt::TransposeOp>(
        op.getLoc(), input,
        AffineMap::getPermutationMap(op.getFirstTranspose(), op.getContext()));
    if (op.getReshape()) {
      input = rewriter.createOrFold<tensorrt::ReshapeOp>(
          op.getLoc(),
          cast<RankedTensorType>(input.getType()).clone(*op.getReshape()),
          input);
    } else if (op.getDynamicReshape()) {
      SmallVector<int64_t> shape(ShapedType::kDynamic,
                                 op.getResult().getType().getRank());
      input = rewriter.createOrFold<tensorrt::ReshapeOp>(
          op.getLoc(), cast<RankedTensorType>(input.getType()).clone(shape),
          input, op.getDynamicReshape());
    }
    input = rewriter.createOrFold<tensorrt::TransposeOp>(
        op.getLoc(), input,
        AffineMap::getPermutationMap(op.getSecondTranspose(), op.getContext()));
    rewriter.replaceOp(op, input);
    return success();
  }
};
} // namespace

namespace {
template <typename OpType>
class RankChangeToReshape : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensorrt::ReshapeOp>(op, op.getType(),
                                                     op.getInput());
    return success();
  }
};
} // namespace

namespace {
struct EinsumEquation {
  std::string equation;
  SmallVector<std::string> lhsParts;
  std::string lhs;
  std::string rhs;

  LogicalResult parse(llvm::StringRef einsumEquation) {
    std::string e{einsumEquation};
    return parse(e);
  }

  LogicalResult parse(const std::string &einsumEquation) {
    size_t pos = einsumEquation.find("->");
    if (pos == std::string::npos)
      return failure();
    equation = einsumEquation;
    lhs = einsumEquation.substr(0, pos);
    rhs = einsumEquation.substr(pos + 2);
    std::istringstream lhsStream(lhs);
    std::string currentPart;
    while (std::getline(lhsStream, currentPart, ',')) {
      lhsParts.push_back(currentPart);
      for (char c : currentPart)
        if (!(c >= 'a' && c <= 'z'))
          return failure();
    }
    return success();
  }

  std::string generateEquation() const {
    std::string ret = lhsParts[0];
    for (size_t i = 1; i < lhsParts.size(); i++)
      ret += "," + lhsParts[i];
    ret += "->" + rhs;
    return ret;
  }
};
} // namespace

namespace {
// Control when fusing a transpose into another op.
// Currently always fuse
static bool shouldFuseTranspose(tensorrt::TransposeOp transposeOp,
                                mlir::Operation *targetFusion) {
  return true;
}
} // namespace

namespace {
// Push down transpose to into an einsum, rearranging the axes of the input
// tensors in the einsum as needed einsum(x1, transpose(x2), ...) -> einsum(x1,
// x2, ...)
class PushDownTransposeToEinsum : public OpRewritePattern<tensorrt::EinsumOp> {
public:
  using OpRewritePattern<tensorrt::EinsumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::EinsumOp op,
                                PatternRewriter &rewriter) const override {
    EinsumEquation einsumEquation;
    if (failed(einsumEquation.parse(op.getEquation())))
      return failure();

    bool hasTransposeInput = false;
    SmallVector<Value> newInputs;
    for (size_t i = 0; i < op.getInputs().size(); i++) {
      auto input = op.getInputs()[i];
      tensorrt::TransposeOp transpose =
          input.getDefiningOp<tensorrt::TransposeOp>();
      if (transpose && shouldFuseTranspose(transpose, op)) {
        AffineMap perm = transpose.getPermutation();
        if (!perm.isPermutation())
          return failure(/* Transpose is not a permutation */);
        SmallVector<int64_t> equation;
        for (char c : einsumEquation.lhsParts[i])
          equation.push_back(c);

        equation = inversePermutation(perm).compose(equation);
        einsumEquation.lhsParts[i] = "";
        for (size_t j = 0; j < equation.size(); j++)
          einsumEquation.lhsParts[i] += (char)equation[j];
        newInputs.push_back(transpose.getInput());
        hasTransposeInput = true;
      } else {
        newInputs.push_back(input);
      }
    }

    if (!hasTransposeInput)
      return failure();

    std::string newEinsumEquation = einsumEquation.generateEquation();
    assert(einsumEquation.rhs.size() == op.getType().getShape().size());
    rewriter.replaceOpWithNewOp<tensorrt::EinsumOp>(op, op.getType(), newInputs,
                                                    newEinsumEquation);
    return success();
  }
};
} // namespace

namespace {
// Push up transpose from an einsum, rearranging the axes of the output tensor
// in the einsum as needed transpose(einsum(x1, x2, ...)) -> einsum(x1, x2, ...)
class PushUpTransposeToEinsum : public OpRewritePattern<tensorrt::TransposeOp> {
public:
  using OpRewritePattern<tensorrt::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    AffineMap perm = op.getPermutation();
    if (!perm.isPermutation())
      return failure();

    auto einsum = op.getInput().getDefiningOp<tensorrt::EinsumOp>();
    if (!einsum)
      return failure();

    if (!einsum->hasOneUse())
      return failure();

    if (!shouldFuseTranspose(op, einsum))
      return failure();

    EinsumEquation einsumEquation;
    if (failed(einsumEquation.parse(einsum.getEquation())))
      return failure();

    SmallVector<int64_t> einsumRhs;
    for (char c : einsumEquation.rhs)
      einsumRhs.push_back(c);
    einsumRhs = perm.compose(einsumRhs);
    einsumEquation.rhs = "";
    for (size_t i = 0; i < einsumRhs.size(); i++)
      einsumEquation.rhs += (char)einsumRhs[i];

    std::string newEinsumEquation = einsumEquation.generateEquation();

    auto newEinsum = rewriter.create<tensorrt::EinsumOp>(
        op.getLoc(), op.getType(), einsum.getInputs(), newEinsumEquation);
    assert(einsumEquation.rhs.size() == newEinsum.getType().getShape().size());
    rewriter.replaceOp(op, newEinsum.getResult());
    return success();
  }
};
} // namespace

namespace {
// Create an new transpose op from an einsum.  Rearrange the output axes to
// match the ordering of the input axes This should enable converting the einsum
// back to a matmul einsum(x1, x2, ...) -> transpose(einsum(x1, x2, ...))
class EinsumPushDownTranspose : public OpRewritePattern<tensorrt::EinsumOp> {
public:
  using OpRewritePattern<tensorrt::EinsumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::EinsumOp op,
                                PatternRewriter &rewriter) const override {
    for (auto input : op.getInputs()) {
      if (input.getDefiningOp<tensorrt::TransposeOp>())
        return failure(); // Wait until the transpose is pushed into the einsum
                          // first
    }
    // determine the "best" order.
    // Ideally, we want the einsum to be reducable to a matmul.  So the batch
    // elements should appear first in the output

    EinsumEquation equation;
    if (failed(equation.parse(op.getEquation())))
      return failure();

    SmallVector<std::pair<char, int64_t>> outputAxes;
    for (size_t i = 0; i < equation.rhs.size(); i++)
      outputAxes.push_back(std::make_pair(equation.rhs[i], i));
    std::sort(outputAxes.begin(), outputAxes.end(),
              [&](const std::pair<char, int64_t> &a,
                  const std::pair<char, int64_t> &b) {
                for (std::string &eqLhs : equation.lhsParts) {
                  if (eqLhs.find(a.first) != std::string::npos) {
                    if (eqLhs.find(b.first) != std::string::npos) {
                      return eqLhs.find(a.first) < eqLhs.find(b.first);
                    } else {
                      return true;
                    }
                  } else if (eqLhs.find(b.first) != std::string::npos) {
                    return false;
                  }
                }
                return a.first < b.first;
              });

    LLVM_DEBUG({
      std::stringstream out;
      out << "outputAxes: [";
      for (auto x : outputAxes)
        out << x.first << "(" << x.second << ") ";
      out << "]\n";
      DBGS() << out.str();
    });

    SmallVector<int64_t> newEinsumShape;
    SmallVector<int64_t> forwardPerm;
    std::string newEinsumRhs = "";
    for (auto &[c, i] : outputAxes) {
      newEinsumRhs += c;
      newEinsumShape.push_back(op.getType().getDimSize(i));
      forwardPerm.push_back(i);
    }
    if (newEinsumRhs == equation.rhs)
      return failure(); // no change

    equation.rhs = newEinsumRhs;
    std::string newEinsumEquation = equation.generateEquation();

    auto newEinsum = rewriter.create<tensorrt::EinsumOp>(
        op.getLoc(), op.getType().clone(newEinsumShape), op.getInputs(),
        newEinsumEquation);
    assert(equation.rhs.size() == newEinsum.getType().getShape().size());

    auto forwardMap =
        AffineMap::getPermutationMap(forwardPerm, op.getLoc().getContext());

    auto newTranspose = rewriter.create<tensorrt::TransposeOp>(
        op.getLoc(), newEinsum.getResult(), inversePermutation(forwardMap));

    assert(op.getType() == newTranspose.getType());
    rewriter.replaceOp(op, newTranspose.getResult());

    return success();
  }
};
} // namespace

namespace {
// Create an new transpose op from an einsum.  Rearrange the input axes to match
// the ordering of the output axes. This should enable converting the einsum
// back to a matmul using the `EinsumToMatrixMultiply` pattern. einsum(x1, x2,
// ...) -> einsum(x1, transpose(x2), ...)
class EinsumPushUpTranspose : public OpRewritePattern<tensorrt::EinsumOp> {
public:
  using OpRewritePattern<tensorrt::EinsumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::EinsumOp op,
                                PatternRewriter &rewriter) const override {
    EinsumEquation equation;
    if (failed(equation.parse(op.getEquation())))
      return failure();

    llvm::SmallSetVector<char, 16> multipliedAxes;
    llvm::SmallSetVector<char, 16> uniqueAxes;
    for (size_t i = 0; i < equation.lhsParts.size(); i++) {
      for (size_t j = 0; j < equation.lhsParts[i].size(); j++) {
        if (equation.rhs.find(equation.lhsParts[i][j]) == std::string::npos) {
          multipliedAxes.insert(equation.lhsParts[i][j]);
        } else {
          // contained in rhs
          bool found = false;
          for (size_t k = 0; k < equation.lhsParts.size(); k++) {
            if (k != i) {
              if (equation.lhsParts[k].find(equation.lhsParts[i][j]) !=
                  std::string::npos) {
                found = true;
                break;
              }
            }
          }
          if (!found) {
            // contained in the i-th lhs, contained in the rhs and not contained
            // in any other lhs this is not a batch axis of the multiplication.
            // E.g in "abc,acd->abd" this would be "bd"
            uniqueAxes.insert(equation.lhsParts[i][j]);
          }
        }
      }
    }

    bool didChange = false;
    SmallVector<Value> newInputs;
    for (size_t i = 0; i < op.getInputs().size(); i++) {
      auto input = cast<TypedValue<RankedTensorType>>(op.getInputs()[i]);
      RankedTensorType inputType = input.getType();
      SmallVector<std::pair<char, int64_t>> inputAxes;
      for (int j = 0; j < inputType.getRank(); j++)
        inputAxes.push_back(std::make_pair(equation.lhsParts[i][j], j));
      std::sort(inputAxes.begin(), inputAxes.end(),
                [&](const std::pair<char, int64_t> &a,
                    const std::pair<char, int64_t> &b) {
                  size_t posA = equation.rhs.find(a.first);
                  size_t posB = equation.rhs.find(b.first);
                  if (posA != std::string::npos && posB != std::string::npos) {
                    // both letters are in the rhs, meaning that these are
                    // either batch or dims of the matrix try to match the order
                    // of the output so that these can become batch dims later
                    return posA < posB;
                  } else if (posA == std::string::npos &&
                             posB == std::string::npos) {
                    return a.second < b.second; // preserve the order if neither
                                                // is found in output
                  } else {
                    // one is in the output, and the other is not in the output
                    // if the character is one of the last two outputs, then we
                    // would rather preserve the ordering as the transpose
                    // property on the matrix multiply can be used to handle
                    if ((i == 0 && (posA == equation.rhs.size() - 2 ||
                                    posB == equation.rhs.size() - 2)) ||
                        (i == 1 && (posA == equation.rhs.size() - 1 ||
                                    posB == equation.rhs.size() - 1))) {
                      return a.second < b.second; // preserve ordering
                    }
                    // does not match expected pattern, put the ordering so that
                    // the one in the output is first
                    return posA != std::string::npos;
                  }
                });
      std::string newEquation = "";
      for (size_t j = 0; j < inputAxes.size(); j++)
        newEquation += inputAxes[j].first;
      if (newEquation != equation.lhsParts[i]) {
        equation.lhsParts[i] = newEquation;
        didChange = true;
        SmallVector<int64_t> perm;
        for (size_t j = 0; j < inputAxes.size(); j++)
          perm.push_back(inputAxes[j].second);
        auto newTranspose = rewriter.create<tensorrt::TransposeOp>(
            op.getLoc(), input,
            AffineMap::getPermutationMap(perm, op.getContext()));
        newInputs.push_back(newTranspose.getResult());
      } else {
        newInputs.push_back(input);
      }
    }

    if (!didChange)
      return failure();

    std::string newEquation = equation.generateEquation();
    assert(equation.rhs.size() == op.getType().getShape().size());
    rewriter.replaceOpWithNewOp<tensorrt::EinsumOp>(op, op.getType(), newInputs,
                                                    newEquation);
    return success();
  }
};
} // namespace

namespace {
// When one of the input axes are 1, then we can push that up as a reshape.
// For example,
//    %3 = tensorrt.einsum("abc,cd->abd", %1: tensor<1x2x3xf32>, %2:
//    tensor<3x4xf32>) -> tensor<1x2x4xf32>
// will become
//    %r1 = tensorrt.reshape %1 : tensor<1x2x3xf32> to tensor<2x3xf32> reshape
//    the input to remove the 1 dim %e1 = tensorrt.einsum("bc,cd->bd", %r1, %2)
//    -> tensor<2x4xf32> %3 = tensorrt.reshape %e1 : tensor<2x4xf32> to
//    tensor<1x2x4xf32>    reshape the output to add the 1 dim back
class EinsumEliminate1Axis : public OpRewritePattern<tensorrt::EinsumOp> {
public:
  using OpRewritePattern<tensorrt::EinsumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::EinsumOp op,
                                PatternRewriter &rewriter) const override {
    EinsumEquation einsumEquation;
    if (failed(einsumEquation.parse(op.getEquation())))
      return failure();

    bool madeChange = false;
    SmallVector<Value> newInputs;

    for (size_t i = 0; i < op.getInputs().size(); i++) {
      auto input = cast<TypedValue<RankedTensorType>>(op.getInputs()[i]);
      RankedTensorType inputType = input.getType();
      std::string equation = "";
      bool change = false;
      SmallVector<int64_t> newInputShape;
      for (int j = 0; j < inputType.getRank(); j++) {
        if (inputType.getDimSize(j) == 1) {
          // this axis is size 1, and not used in the multiplication, we can
          // remove it from the einsum
          madeChange = change = true;
        } else {
          equation += einsumEquation.lhsParts[i][j];
          newInputShape.push_back(inputType.getDimSize(j));
        }
      }
      if (change) {
        auto newInput =
            rewriter
                .create<tensorrt::ReshapeOp>(
                    op.getLoc(), inputType.clone(newInputShape), input)
                .getResult();
        newInputs.push_back(newInput);
        einsumEquation.lhsParts[i] = equation;
      } else {
        newInputs.push_back(input);
      }
    }

    if (!madeChange)
      return failure();

    RankedTensorType outputType = op.getType();
    EinsumEquation newEinsumEquation = einsumEquation;
    newEinsumEquation.rhs = "";
    SmallVector<int64_t> newOutputShape;
    bool changeOutput = false;
    for (int i = 0; i < outputType.getRank(); i++) {
      if (outputType.getDimSize(i) == 1) {
        // this axis is size 1, and not used in the multiplication, we can
        // remove it from the einsum
        changeOutput = true;
      } else {
        newEinsumEquation.rhs += einsumEquation.rhs[i];
        newOutputShape.push_back(outputType.getDimSize(i));
      }
    }
    std::string newEquation = newEinsumEquation.generateEquation();

    if (changeOutput) {
      auto newEinsum = rewriter.create<tensorrt::EinsumOp>(
          op.getLoc(), outputType.clone(newOutputShape), newInputs,
          newEquation);
      assert(newEinsumEquation.rhs.size() ==
             newEinsum.getType().getShape().size());
      auto outReshape =
          rewriter
              .create<tensorrt::ReshapeOp>(op.getLoc(), op.getType(),
                                           newEinsum.getResult())
              .getResult();
      assert(op.getType() == outReshape.getType());
      rewriter.replaceOp(op, outReshape);
      return success();
    } else {
      assert(newEinsumEquation.rhs.size() == op.getType().getShape().size());
      rewriter.replaceOpWithNewOp<tensorrt::EinsumOp>(op, op.getType(),
                                                      newInputs, newEquation);
      return success();
    }
  }
};
} // namespace

namespace {

// When one of the input axes has a 1 shaped input and there is a reshape on the
// input, then the reshape can be merged with the einsum. E.g.
//    %1 = tensorrt.reshape %0 : tensor<1x2x3xf32> to tensor<2x3xf32>
//    %2 = tensorrt.einsum("bc,cd->bd", %1, %2) -> tensor<2x4xf32>
// will become
//    %2 = tensorrt.einsum("abc,cd->bd", %0, %2) -> tensor<2x4xf32>
class EinsumMergeDown1Axis : public OpRewritePattern<tensorrt::EinsumOp> {
public:
  using OpRewritePattern<tensorrt::EinsumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::EinsumOp op,
                                PatternRewriter &rewriter) const override {
    EinsumEquation equation;
    if (failed(equation.parse(op.getEquation())))
      return failure();

    char nextChar = 'a';
    auto getNextChar = [&]() -> char {
      while (nextChar <= 'z') {
        char c = nextChar++;
        if (equation.equation.find(c) == std::string::npos)
          return c;
      }
      return 0;
    };

    SmallVector<Value> newInputs;
    bool madeChange = false;
    for (size_t i = 0; i < op.getInputs().size(); i++) {
      Value input = op.getInputs()[i];
      if (auto collapse = input.getDefiningOp<tensorrt::CollapseRankOp>()) {
        RankedTensorType inputType = collapse.getInput().getType();
        if (!inputType.hasStaticShape())
          return failure(/* collapse rank op with dynamic shape */);
        auto inputShape = inputType.getShape();
        std::string newEquation = "";
        size_t k = 0;
        for (size_t j = 0; j < inputShape.size(); j++) {
          if (inputShape[j] == 1) {
            char c = getNextChar();
            if (c == 0)
              return failure(/* no more einsum characters available */);
            newEquation += c;
          } else {
            newEquation += equation.lhsParts[i][k++];
          }
        }
        assert(k == equation.lhsParts[i].size());
        newInputs.push_back(collapse.getInput());
        equation.lhsParts[i] = newEquation;
        madeChange = true;
      } else {
        newInputs.push_back(input);
      }
    }

    if (!madeChange)
      return failure();

    std::string newEquation = equation.generateEquation();
    assert(equation.rhs.size() == op.getType().getShape().size());
    rewriter.replaceOpWithNewOp<tensorrt::EinsumOp>(op, op.getType(), newInputs,
                                                    newEquation);
    return success();
  }
};
} // namespace

namespace {

// In the case that the output of an einsum has a 1 shaped output, then the
// reshape can be merged with the einsum if there is an input that is also one
// shaped. E.g.
//    %1 = tensorrt.einsum("abc,cd->bd", %0 : tensor<1x2x3xf32>, %1) ->
//    tensor<2x4xf32> %2 = tensorrt.reshape %1 : tensor<2x4xf32> to
//    tensor<1x2x4xf32>
// will become
//    %2 = tensorrt.einsum("abc,cd->abd", %0, %1) -> tensor<1x2x4xf32>
class EinsumMergeUp1Axis : public OpRewritePattern<tensorrt::ExpandRankOp> {
public:
  using OpRewritePattern<tensorrt::ExpandRankOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::ExpandRankOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().hasStaticShape())
      return failure(/* only handle static expand rank */);
    auto einsum = op.getInput().getDefiningOp<tensorrt::EinsumOp>();
    if (!einsum)
      return failure();
    if (!einsum->hasOneUse())
      return failure(/* einsum used more than once, can't modify */);

    EinsumEquation equation;
    if (failed(equation.parse(einsum.getEquation())))
      return failure();

    llvm::SmallSetVector<char, 16> oneAxisChars;
    llvm::SmallSetVector<char, 16> nonOneAxisChars;
    for (size_t i = 0; i < einsum.getInputs().size(); i++) {
      auto inputShape =
          cast<RankedTensorType>(einsum.getInputs()[i].getType()).getShape();
      for (size_t j = 0; j < inputShape.size(); j++) {
        if (inputShape[j] == 1)
          oneAxisChars.insert(equation.lhsParts[i][j]);
        else
          nonOneAxisChars.insert(equation.lhsParts[i][j]);
      }
    }

    // an axis can hvae 1 and non-1 shapes associated in the case that the axis
    // is broadcast.  In which case, it is not a 1 shaped axis on the output.
    oneAxisChars.remove_if([&](char c) { return nonOneAxisChars.contains(c); });
    if (oneAxisChars.empty())
      return failure(/* no one axis inputs found */);

    auto einsumShape = op.getInput().getType().getShape();
    auto outputShape = op.getResult().getType().getShape();
    std::string newRhs = "";
    for (size_t i = 0, j = 0, k = 0; i < outputShape.size(); i++) {
      if (outputShape[i] == 1) {
        if (k >= oneAxisChars.size())
          return failure();
        newRhs += oneAxisChars[k++];
      } else {
        if (j >= equation.rhs.size())
          return failure();
        assert(einsumShape[j] == outputShape[i]);
        newRhs += equation.rhs[j++];
      }
    }

    std::string newEquation = equation.lhs + "->" + newRhs;
    assert(newRhs.size() == op.getType().getShape().size());
    rewriter.replaceOpWithNewOp<tensorrt::EinsumOp>(
        op, op.getType(), einsum.getInputs(), newEquation);
    return success();
  }
};
} // namespace

namespace {
// In the case of an einsum that is performing a broadcast, increase the rank of
// its inputs so that it can better match the matrix multiply pattern.
class EinsumPushUp1AxisReshape : public OpRewritePattern<tensorrt::EinsumOp> {
public:
  using OpRewritePattern<tensorrt::EinsumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::EinsumOp op,
                                PatternRewriter &rewriter) const override {
    EinsumEquation equation;
    if (failed(equation.parse(op.getEquation())))
      return failure();
    if (op->getNumOperands() != 2)
      return failure();

    assert(equation.rhs.size() == op.getType().getShape().size());

    char matrixAxes[2] = {0, 0};
    char multipliedAxis = 0;

    for (size_t i = 0; i < 2; i++) {
      for (int j = equation.lhsParts[i].size() - 1; j >= 0; j--) {
        char c = equation.lhsParts[i][j];
        if (multipliedAxis == 0 &&
            equation.lhsParts[1 - i].find(c) != std::string::npos &&
            equation.rhs.find(c) == std::string::npos)
          multipliedAxis = c;
        if (matrixAxes[i] == 0 &&
            equation.lhsParts[1 - i].find(c) == std::string::npos &&
            equation.rhs[equation.rhs.size() - 2 + i] == c)
          matrixAxes[i] = c;
      }
    }

    RankedTensorType inputType[2] = {
        cast<RankedTensorType>(op.getInputs()[0].getType()),
        cast<RankedTensorType>(op.getInputs()[1].getType())};
    if (!inputType[0].hasStaticShape() || !inputType[1].hasStaticShape())
      return failure();

    SmallVector<int64_t> newInputShapes[2] = {
        SmallVector<int64_t>{inputType[0].getShape()},
        SmallVector<int64_t>{inputType[1].getShape()}};
    EinsumEquation newEquation = equation;

    for (int i = 0; i < 2; i++) {
      for (char c : equation.lhsParts[i]) {
        if (c == multipliedAxis || c == matrixAxes[i] ||
            equation.rhs.find(c) == std::string::npos)
          continue;
        if (newEquation.lhsParts[1 - i].find(c) == std::string::npos) {
          // figure out the best place to insert "c"
          // Find the best index to insert 'c' into newEquation.lhsParts[1-i]
          // so that all letters to the left of 'c' in equation.lhsParts[i] are
          // to the left of 'c' and all letters to the right of 'c' in
          // equation.lhsParts[i] are to the right of 'c'
          size_t insertIdx = 0;
          // Find the leftmost position such that all letters in
          // equation.lhsParts[i] before 'c' are to the left of 'c' in
          // newEquation.lhsParts[1-i] and all letters after 'c' are to the
          // right We do this by finding the first position in
          // newEquation.lhsParts[1-i] where a letter that comes after 'c' in
          // equation.lhsParts[i] appears. If none, insert at the end.
          std::string &target = newEquation.lhsParts[1 - i];
          const std::string &src = newEquation.lhsParts[i];
          size_t cPos = src.find(c);
          for (insertIdx = 0; insertIdx <= target.size(); ++insertIdx) {
            bool valid = true;
            // Check all letters before c in src
            for (size_t l = 0; l < cPos; ++l) {
              char leftChar = src[l];
              size_t posInTarget = target.find(leftChar);
              if (posInTarget != std::string::npos &&
                  posInTarget >= insertIdx) {
                valid = false;
                break;
              }
            }
            if (!valid)
              continue;
            // Check all letters after c in src
            for (size_t r = cPos + 1; r < src.size(); ++r) {
              char rightChar = src[r];
              size_t posInTarget = target.find(rightChar);
              if (posInTarget != std::string::npos && posInTarget < insertIdx) {
                valid = false;
                break;
              }
            }
            if (valid)
              break;
          }
          target.insert(target.begin() + insertIdx, c);
          newInputShapes[1 - i].insert(
              newInputShapes[1 - i].begin() + insertIdx, 1);
          assert(target.size() == newInputShapes[1 - i].size());
        }
      }
    }

    RankedTensorType newInputTypes[2] = {inputType[0].clone(newInputShapes[0]),
                                         inputType[1].clone(newInputShapes[1])};

    if (newInputTypes[0] == inputType[0] && newInputTypes[1] == inputType[1])
      return failure(/* nothing changed */);

    assert(newInputShapes[0].size() == newEquation.lhsParts[0].size() &&
           newInputShapes[1].size() == newEquation.lhsParts[1].size());

    SmallVector<Value> reshapes{
        rewriter.createOrFold<ReshapeOp>(op.getLoc(), newInputTypes[0],
                                         op.getInputs()[0]),
        rewriter.createOrFold<ReshapeOp>(op.getLoc(), newInputTypes[1],
                                         op.getInputs()[1])};

    assert(newEquation.rhs.size() == op.getType().getShape().size());
    rewriter.replaceOpWithNewOp<EinsumOp>(op, op.getType(), reshapes,
                                          newEquation.generateEquation());

    return success();
  }
};
} // namespace

namespace {
// Push up a reshape through an einum to its inputs
// reshape(einsum(x1, x2, ..)) ->
//    einsum(reshape(transpose(x1)), reshape(transpose(x2)), ...)
class PushReshapeUpThroughEinsum
    : public OpRewritePattern<tensorrt::ReshapeOp> {
public:
  using OpRewritePattern<tensorrt::ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getResult().getType().hasStaticShape())
      return failure(/* only handle static reshapes */);

    auto einsum = op.getInput().getDefiningOp<tensorrt::EinsumOp>();
    if (!einsum)
      return failure();
    if (!einsum->hasOneUse())
      return failure();

    EinsumEquation equation;
    if (failed(equation.parse(einsum.getEquation())))
      return failure();

    char nextChar = 'a';
    auto getNextChar = [&]() -> char {
      while (nextChar <= 'z') {
        char c = nextChar++;
        if (equation.equation.find(c) == std::string::npos)
          return c;
      }
      return 0;
    };

    SmallVector<SmallVector<int64_t>> inputShapes;
    for (Value input : einsum.getInputs()) {
      SmallVector<int64_t> shape(
          cast<RankedTensorType>(input.getType()).getShape());
      inputShapes.push_back(shape);
    }

    auto reshapeInShape = op.getInput().getType().getShape();
    auto reshapeOutShape = op.getResult().getType().getShape();

    struct ReshapeInfo {
      std::string newAxes;
      SmallVector<int64_t> newShape;
      SmallVector<int64_t> oldShape;
    };

    bool hasNonTrivalReshape = false;
    std::unordered_map<std::string, ReshapeInfo> inputToReshapedMap;
    size_t inputNumElems = 1;
    size_t outputNumElems = 1;
    std::string inAxes = "";
    std::string outAxes = "";
    std::string prevInAxes = "";
    SmallVector<int64_t> outShape;
    SmallVector<int64_t> inShape;
    for (size_t i = 0, j = 0; i < reshapeOutShape.size(); i++) {
      if (reshapeOutShape[i] == 0)
        return failure(/* 0-shape not supported */);
      outputNumElems *= reshapeOutShape[i];
      outShape.push_back(reshapeOutShape[i]);
      char c = getNextChar();
      if (c == 0)
        return failure(/* no more einsum characters available */);
      outAxes += c;
      while (j < reshapeInShape.size() && inputNumElems < outputNumElems) {
        inputNumElems *= reshapeInShape[j];
        inShape.push_back(reshapeInShape[j]);
        inAxes += equation.rhs[j++];
      }
      if (inputNumElems == outputNumElems) {
        if (inAxes.empty()) {
          if (!prevInAxes.empty() && reshapeOutShape[i] == 1 &&
              outAxes.size() == 1) {
            auto &p = inputToReshapedMap[prevInAxes];
            p.newAxes.push_back(c);
            p.newShape.push_back(1);
            if (prevInAxes.size() != p.newAxes.size())
              hasNonTrivalReshape = true;
            outAxes = "";
            outShape.clear();
            inShape.clear();
          }
          continue;
        }
        if (inAxes.size() != outAxes.size())
          hasNonTrivalReshape = true;
        inputToReshapedMap[inAxes] = ReshapeInfo{
            .newAxes = outAxes, .newShape = outShape, .oldShape = inShape};
        outShape.clear();
        inShape.clear();
        prevInAxes = inAxes;
        inAxes = "";
        outAxes = "";
      }
    }
    if (inputNumElems != outputNumElems || !inAxes.empty() || !outAxes.empty())
      return failure(/* should not happen, unexpected reshape */);
    if (!hasNonTrivalReshape)
      return failure(/* reshape is only expanding rank */);

    llvm::SmallMapVector<char, std::string, 16> charToGroup;
    for (auto &[k, v] : inputToReshapedMap)
      for (auto c : k)
        charToGroup[c] = k;

    // check that all of the inputs are have the right groupping.  If this
    // doesn't happen then that means that the reshape can not get pushed
    // through
    for (std::string &eqLhs : equation.lhsParts) {
      for (char c : eqLhs) {
        auto it = charToGroup.find(c);
        if (it == charToGroup.end())
          continue;
        for (char c2 : it->second)
          if (eqLhs.find(c2) == std::string::npos)
            return failure(/* Not able to push reshape through einsum */);
      }
    }

    EinsumEquation newEquation = equation;
    newEquation.rhs = "";
    for (char c : equation.rhs) {
      assert(charToGroup.count(c));
      if (charToGroup[c][0] == c)
        newEquation.rhs += inputToReshapedMap[charToGroup[c]].newAxes;
    }

    // generate a `x` -> `reshape(transpose(x))` if necessary
    SmallVector<Value> newInputs;
    newEquation.lhsParts.clear();

    LLVM_DEBUG({
      std::stringstream out;
      out << "==== Einsum Reshape/Transpose Pushup Debug ====\n";
      for (const auto &entry : charToGroup) {
        out << "  charToGroup[" << entry.first << "] = " << entry.second
            << "\n";
      }
      for (const auto &entry : inputToReshapedMap) {
        out << "  inputToReshapedMap[" << entry.first
            << "]: axes = " << entry.second.newAxes << ", shape = [";
        for (size_t si = 0; si < entry.second.newShape.size(); ++si) {
          out << entry.second.newShape[si];
          if (si + 1 < entry.second.newShape.size())
            out << ", ";
        }
        out << "], old shape = [";
        for (size_t si = 0; si < entry.second.oldShape.size(); ++si) {
          out << entry.second.oldShape[si];
          if (si + 1 < entry.second.oldShape.size())
            out << ", ";
        }
        out << "]";
        out << "\n";
      }
      DBGS() << out.str();
    });

    // check that the input shape for all of the inputs match (that there are no
    // broadcasts happening on some inputs)
    for (auto &[inputAxes, reshapeInfo] : inputToReshapedMap) {
      // this is a single axis, so broadcasting is allowed in this case, hence
      // do not check
      if (inputAxes.size() == 1 && reshapeInfo.newAxes.size() == 1)
        continue;

      for (size_t i = 0; i < einsum.getInputs().size(); i++) {
        auto inputShape =
            cast<RankedTensorType>(einsum.getInputs()[i].getType()).getShape();
        for (size_t j = 0; j < inputAxes.size(); j++) {
          size_t pos = equation.lhsParts[i].find(inputAxes[j]);
          if (pos != std::string::npos &&
              inputShape[pos] != reshapeInfo.oldShape[j])
            return failure(/* input shape does not match output shape*/);
        }
      }
    }

    for (size_t i = 0; i < einsum.getInputs().size(); i++) {
      Value input = einsum.getInputs()[i];
      auto inputType = cast<RankedTensorType>(input.getType());
      std::string newInputEquation = "";
      SmallVector<int64_t> newInputShape;
      SmallVector<int64_t> newInputTranspose;
      for (int j = 0; j < inputType.getRank(); j++) {
        auto group = charToGroup.find(equation.lhsParts[i][j]);
        if (group == charToGroup.end()) {
          // this must be going into the multply, so it should just keep this
          // letter
          newInputEquation += equation.lhsParts[i][j];
          newInputTranspose.push_back(j);
          newInputShape.push_back(inputType.getDimSize(j));
        } else {
          // then there is some pattern that is getting consumed
          if (group->second[0] != equation.lhsParts[i][j])
            continue; // then this isn't the first character, so it should have
                      // already been processed
          // this is the first character in the group.  So process all of the
          // group
          for (char c : group->second)
            newInputTranspose.push_back(equation.lhsParts[i].find(c));
          newInputEquation += inputToReshapedMap[group->second].newAxes;
          for (int64_t v : inputToReshapedMap[group->second].newShape) {
            if (v != 1 && group->second.size() == 1 &&
                inputType.getDimSize(j) == 1) {
              // if the group is of size 1, then it can have different sizes for
              // each input due to broadcasting
              newInputShape.push_back(1);
            } else {
              newInputShape.push_back(v);
            }
          }
        }
      }

      // Debug print for this input's result
      LLVM_DEBUG({
        std::stringstream out;
        out << "Input #" << i << "  orig eq: " << equation.lhsParts[i]
            << "  new eq: " << newInputEquation << "\n";
        out << "  newInputTranspose: [";
        for (size_t ti = 0; ti < newInputTranspose.size(); ++ti) {
          out << newInputTranspose[ti];
          if (ti + 1 < newInputTranspose.size())
            out << ", ";
        }
        out << "]\n";
        out << "  newInputShape: [";
        for (size_t si = 0; si < newInputShape.size(); ++si) {
          out << newInputShape[si];
          if (si + 1 < newInputShape.size())
            out << ", ";
        }
        out << "]\n";
        out << "  oldShape: [";
        for (size_t si = 0; si < inputType.getShape().size(); ++si) {
          out << inputType.getShape()[si];
          if (si + 1 < inputType.getShape().size())
            out << ", ";
        }
        out << "]\n";
        DBGS() << out.str() << "\n";
      });

      auto newTranspose = rewriter.createOrFold<tensorrt::TransposeOp>(
          op.getLoc(), input,
          AffineMap::getPermutationMap(newInputTranspose,
                                       op.getLoc().getContext()));
      auto newReshape = rewriter.createOrFold<tensorrt::ReshapeOp>(
          op.getLoc(), inputType.clone(newInputShape), newTranspose);

      newInputs.push_back(newReshape);
      newEquation.lhsParts.push_back(newInputEquation);
    }
    std::string newEquationStr = newEquation.generateEquation();

    LLVM_DEBUG({
      DBGS() << newEquationStr << "\n"
             << "===============================================\n";
    });

    auto newEinsum = rewriter.create<tensorrt::EinsumOp>(
        einsum.getLoc(), op.getType(), newInputs, newEquationStr);
    assert(newEquation.rhs.size() == newEinsum.getType().getShape().size());
    assert(op.getType() == newEinsum.getType());
    rewriter.replaceOp(op, newEinsum.getResult());

    return success();
  }
};
} // namespace

namespace {

// if there are mutliple axes that are getting multiplied together in an einsum,
// push up a reshape so that there is only a single axis.  This will help with
// the conversion from einsum to matrix multiply. For example,
//    %0 = tensorrt.einsum {equation = "acd,bcd->ab"} ins(%arg0, %arg1)
// will become
//    %0 = tensorrt.reshape %arg0
//    %1 = tensorrt.reshape %arg1
//    %2 = tensorrt.einsum {equation = "ac,bc->ab"} ins(%0, %1)
class EinsumPushUpMultipleMulitipliedAxes
    : public OpRewritePattern<tensorrt::EinsumOp> {
public:
  using OpRewritePattern<tensorrt::EinsumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::EinsumOp op,
                                PatternRewriter &rewriter) const override {
    EinsumEquation equation;
    if (failed(equation.parse(op.getEquation())))
      return failure();

    std::string multipliedAxes = "";
    for (char c : equation.lhsParts[0])
      if (equation.rhs.find(c) == std::string::npos)
        multipliedAxes += c;
    if (multipliedAxes.size() <= 1)
      return failure(/* pattern does not match */);
    for (size_t i = 0; i < equation.lhsParts.size(); i++) {
      if (equation.lhsParts[i].find(multipliedAxes) == std::string::npos)
        return failure(/* pattern does not match */);
      if (!cast<RankedTensorType>(op.getInputs()[i].getType()).hasStaticShape())
        return failure();
    }
    char nextChar = 'a';
    while (nextChar <= 'z') {
      if (equation.equation.find(nextChar) == std::string::npos)
        break;
      nextChar++;
    }
    if (nextChar > 'z')
      return failure(/* No more characters available */);

    SmallVector<Value> newInputs;
    EinsumEquation newEquation = equation;
    for (size_t i = 0; i < equation.lhsParts.size(); i++) {
      auto inputType = cast<RankedTensorType>(op.getInputs()[i].getType());
      SmallVector<int64_t> newInputShape;
      std::string newInputEquation = "";
      size_t j = 0;
      while (j < equation.lhsParts[i].size() &&
             multipliedAxes.find(equation.lhsParts[i][j]) ==
                 std::string::npos) {
        newInputShape.push_back(inputType.getDimSize(j));
        newInputEquation += equation.lhsParts[i][j];
        j++;
      }

      int64_t combinedInputShape = 1;
      while (j < equation.lhsParts[i].size() &&
             multipliedAxes.find(equation.lhsParts[i][j]) != std::string::npos)
        combinedInputShape *= inputType.getDimSize(j++);
      newInputShape.push_back(combinedInputShape);
      newInputEquation += nextChar;
      while (j < equation.lhsParts[i].size()) {
        newInputShape.push_back(inputType.getDimSize(j));
        newInputEquation += equation.lhsParts[i][j];
        j++;
      }

      newEquation.lhsParts[i] = newInputEquation;
      auto reshape = rewriter.createOrFold<tensorrt::ReshapeOp>(
          op.getLoc(), inputType.clone(newInputShape), op.getInputs()[i]);
      newInputs.push_back(reshape);
    }

    assert(newEquation.rhs.size() == op.getType().getShape().size());
    rewriter.replaceOpWithNewOp<tensorrt::EinsumOp>(
        op, op.getType(), newInputs, newEquation.generateEquation());
    return success();
  }
};
} // namespace

namespace {
static uint64_t estimateShuffleCost(Value input) {
  // This is a heuristic.  One may wish to update this in the future depending
  // on their use case. This heuristic currently attempts to put shuffles
  // "together" which allows two shuffles to be merged, and put shuffles on
  // constant values which allows for them to be merged with the constant.

  Operation *op = input.getDefiningOp();
  bool foundShuffle = false;
  bool canMergeUp = true;
  for (int i = 0; op && i < 10; i++) {
    if (isa<tensorrt::ConstantOp>(op))
      return 0; // This has found a constant.  The constant can be
                // reshaped/rearranged as necessary to absorb the shuffle.
                // Hence, mark this as having 0 cost.
    if (canMergeUp &&
        isa<tensorrt::ShuffleOp, tensorrt::ReshapeOp, tensorrt::TransposeOp>(
            op))
      foundShuffle = true;
    if (!isa<tensorrt::UnaryOp, tensorrt::ActivationOp, tensorrt::IdentityOp,
             tensorrt::ReshapeOp, tensorrt::TransposeOp, tensorrt::ShuffleOp>(
            op))
      canMergeUp = false;
    if (op->getNumOperands() != 1)
      break;
    op = op->getOperand(0).getDefiningOp();
  }

  if (foundShuffle)
    return 100; // should be able to merge this op up into another shuffle.  So
                // it gets a lower cost

  // if there is no constant or no existing shuffle, then there is nothing to
  // merge with, so we are going to mark this as "high cost"
  return 1000;
}
} // namespace

namespace {
// Push a reshape down through an einsum
// einsum(reshape(x), y) -> transpose(reshape(einsum(x, reshape(transpose(y)),
// ...))
class PushReshapeDownThroughEinsum
    : public OpRewritePattern<tensorrt::EinsumOp> {
public:
  using OpRewritePattern<tensorrt::EinsumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::EinsumOp op,
                                PatternRewriter &rewriter) const override {
    // this needs to some "heuristic" to determine if a reshape should
    // get pushed down as reshapes might need to get added to other inputs to
    // make the shapes work
    bool hasReshapeInput = false;
    Location reshapeLoc = op.getLoc();
    for (auto input : op.getInputs()) {
      if (!cast<RankedTensorType>(input.getType()).hasStaticShape()) {
        return failure(/* dynamic input not supported */);
      }
      if (auto reshape = input.getDefiningOp<tensorrt::ReshapeOp>()) {
        if (!reshape.getInput().getType().hasStaticShape())
          return failure(/* dynamic reshape input not supported */);
        hasReshapeInput = true;
        reshapeLoc = reshape.getLoc();
      }
    }
    if (!hasReshapeInput)
      return failure();

    EinsumEquation equation;
    if (failed(equation.parse(op.getEquation())))
      return failure();

    char nextChar = 'a';
    auto getNextChar = [&]() -> char {
      while (nextChar <= 'z') {
        char c = nextChar++;
        if (equation.equation.find(c) == std::string::npos)
          return c;
      }
      return 0;
    };

    uint64_t currentEstimatedCost = 0;

    struct ReshapeInfo {
      SmallVector<int64_t> inputShape;
      SmallVector<int64_t> outputShape;
      std::string newEinsumStr;
    };
    std::unordered_map<std::string, ReshapeInfo> inputToReshapedMap;
    for (size_t i = 0; i < op.getInputs().size(); i++) {
      auto input = op.getInputs()[i];
      RankedTensorType einsumInputType =
          cast<RankedTensorType>(input.getType()); // reshape output type
      if (auto reshape = input.getDefiningOp<tensorrt::ReshapeOp>()) {
        currentEstimatedCost += estimateShuffleCost(input);
        size_t inputNumElems = 1;
        size_t outputNumElems = 1;
        SmallVector<int64_t> inputShape;
        SmallVector<int64_t> outputShape;
        std::string outputEinsumStr = "";
        RankedTensorType reshapeInputType = reshape.getInput().getType();
        for (int j = 0, k = 0; j < einsumInputType.getRank(); j++) {
          if (einsumInputType.getDimSize(j) <= 1) {
            // if 0-shape, then means the tensor is empty.  Annoying edge case
            // that not going to handle
            // TODO: if 1-shape, then need additional logic to handle this
            return failure(/* 0 or 1 dim not supported */);
          }
          outputNumElems *= einsumInputType.getDimSize(j);
          outputShape.push_back(einsumInputType.getDimSize(j));
          outputEinsumStr += equation.lhsParts[i][j];
          while (k < reshapeInputType.getRank() &&
                 inputNumElems < outputNumElems) {
            if (reshapeInputType.getDimSize(k) == 1)
              return failure(/* 1 dim not supported */);
            inputNumElems *= reshapeInputType.getDimSize(k);
            inputShape.push_back(reshapeInputType.getDimSize(k++));
          }
          if (inputNumElems == outputNumElems) {
            auto it = inputToReshapedMap.find(outputEinsumStr);
            if (it != inputToReshapedMap.end()) {
              if (it->second.inputShape != inputShape ||
                  it->second.outputShape != outputShape)
                return failure(
                    /* a single axis has multiple inconsistent reshapes */);
            } else {
              if (outputShape != inputShape) {
                std::string newEinsumStr = "";
                for (size_t l = 0; l < inputShape.size(); l++) {
                  char c = getNextChar();
                  if (c == 0)
                    return failure(/* no more characters available */);
                  newEinsumStr += c;
                }
                assert(outputEinsumStr.size() == outputShape.size());
                inputToReshapedMap[outputEinsumStr] =
                    ReshapeInfo{.inputShape = inputShape,
                                .outputShape = outputShape,
                                .newEinsumStr = newEinsumStr};
              } else {
                // do not register this as there is no change in the shape
                // so if something else requires a change, then it will get
                // registered for this symbol instead
                assert(outputShape.size() == 1);
              }
            }
            inputShape.clear();
            outputShape.clear();
            outputEinsumStr = "";
          }
        }
        assert(inputNumElems == outputNumElems);
      }
    }

    llvm::SmallMapVector<char, std::string, 16> charToGroup;
    for (auto &[k, v] : inputToReshapedMap) {
      for (char c : k) {
        auto it = charToGroup.find(c);
        if (it == charToGroup.end())
          charToGroup[c] = k;
        else
          return failure(
              /* a single axis has multiple inconsistent reshapes */);
      }
    }

    for (std::string &part : equation.lhsParts) {
      for (char c : part) {
        auto group = charToGroup.find(c);
        if (group == charToGroup.end())
          continue;
        for (char c2 : group->second) {
          if (part.find(c2) == std::string::npos)
            return failure(
                /* Missing dimensions that need to be reshaped together */);
        }
      }
    }

    for (char c : equation.rhs) {
      auto group = charToGroup.find(c);
      if (group == charToGroup.end())
        continue;
      for (char c2 : group->second) {
        if (equation.rhs.find(c2) == std::string::npos)
          return failure(
              /* Missing dimensions that need to be reshaped together */);
      }
    }

    size_t newEstimatedCost = 0;

    for (size_t i = 0; i < op.getInputs().size(); i++) {
      Value input = op.getInputs()[i];
      RankedTensorType inputType = cast<RankedTensorType>(input.getType());
      SmallVector<int64_t> newInputShape;
      for (int j = 0; j < inputType.getRank(); j++) {
        char c = equation.lhsParts[i][j];
        auto it = charToGroup.find(c);
        if (it == charToGroup.end()) {
          newInputShape.push_back(inputType.getDimSize(j));
        } else {
          if (it->second[0] != c)
            continue; // this will be processed on the first letter
          auto group = inputToReshapedMap.find(it->second);
          for (size_t k = 0; k < group->second.inputShape.size(); k++) {
            newInputShape.push_back(group->second.inputShape[k]);
          }
        }
      }
      Value reshapeIn = input;
      while (auto reshape = reshapeIn.getDefiningOp<tensorrt::ReshapeOp>()) {
        reshapeIn = reshape.getInput();
      }
      SmallVector<int64_t> reshapeInShape{
          cast<RankedTensorType>(reshapeIn.getType()).getShape()};
      if (reshapeInShape != newInputShape)
        newEstimatedCost += estimateShuffleCost(reshapeIn);
    }

    if (newEstimatedCost >= currentEstimatedCost)
      return failure(/* new cost is not better than current cost */);

    // done matching against the pattern.  Going to start modifying the MLIR at
    // this point

    SmallVector<Value> newInputs;
    EinsumEquation newEquation;
    for (size_t i = 0; i < op.getInputs().size(); i++) {
      Value input = op.getInputs()[i];
      RankedTensorType inputType = cast<RankedTensorType>(input.getType());
      SmallVector<int64_t> newInputShape;
      SmallVector<int64_t> newInputTranspose;
      std::string newEinsumStr = "";
      for (int j = 0; j < inputType.getRank(); j++) {
        char c = equation.lhsParts[i][j];
        auto it = charToGroup.find(c);
        if (it == charToGroup.end()) {
          newInputShape.push_back(inputType.getDimSize(j));
          newInputTranspose.push_back(j);
          newEinsumStr += c;
        } else {
          if (it->second[0] != c)
            continue; // this will be processed on the first letter
          auto group = inputToReshapedMap.find(it->second);
          newEinsumStr += group->second.newEinsumStr;
          for (size_t k = 0; k < group->second.inputShape.size(); k++) {
            newInputShape.push_back(group->second.inputShape[k]);
          }
          for (char c2 : group->first) {
            size_t pos = equation.lhsParts[i].find(c2);
            assert(pos != std::string::npos);
            newInputTranspose.push_back(pos);
          }
        }
      }

      Value reshapeIn = rewriter.createOrFold<tensorrt::TransposeOp>(
          op.getLoc(), input,
          AffineMap::getPermutationMap(newInputTranspose, op.getContext()));
      while (auto definingOp = reshapeIn.getDefiningOp<tensorrt::ReshapeOp>()) {
        // two sequential reshapes just results in the shape of the last
        // reshape.  There are canonicalization patterns that do this as well
        // but do it here so that the reshape op that was an input is no longer
        // used.
        reshapeIn = definingOp.getInput();
      }
      auto reshape = rewriter.createOrFold<tensorrt::ReshapeOp>(
          op.getLoc(), inputType.clone(newInputShape), reshapeIn);

      newInputs.push_back(reshape);
      newEquation.lhsParts.push_back(newEinsumStr);
    }

    RankedTensorType outputType = op.getType();
    SmallVector<int64_t> einsumOutputShape;
    SmallVector<int64_t> afterEinsumReshape;
    SmallVector<int64_t> afterReshapeTranspose;

    for (int j = 0; j < outputType.getRank(); j++) {
      char c = equation.rhs[j];
      auto it = charToGroup.find(c);
      if (it == charToGroup.end()) {
        einsumOutputShape.push_back(outputType.getDimSize(j));
        afterEinsumReshape.push_back(outputType.getDimSize(j));
        afterReshapeTranspose.push_back(j);
        newEquation.rhs += c;
      } else {
        if (it->second[0] != c)
          continue;
        auto group = inputToReshapedMap.find(it->second);
        newEquation.rhs += group->second.newEinsumStr;
        for (size_t k = 0; k < group->second.inputShape.size(); k++) {
          // the output shape of the einsum is the input shape of the reshape
          // now as the reshape will appear after the einsum
          einsumOutputShape.push_back(group->second.inputShape[k]);
        }
        for (size_t k = 0; k < group->second.outputShape.size(); k++)
          afterEinsumReshape.push_back(group->second.outputShape[k]);
        for (char c2 : it->second) {
          size_t pos = equation.rhs.find(c2);
          assert(pos != std::string::npos);
          afterReshapeTranspose.push_back(pos);
        }
      }
    }

    std::string newEinsumEquation = newEquation.generateEquation();

    auto newEinsum = rewriter.create<tensorrt::EinsumOp>(
        op.getLoc(), outputType.clone(einsumOutputShape), newInputs,
        newEinsumEquation);
    assert(newEquation.rhs.size() == newEinsum.getType().getShape().size());

    auto newReshape = rewriter.createOrFold<tensorrt::ReshapeOp>(
        reshapeLoc, outputType.clone(afterEinsumReshape),
        newEinsum.getResult());

    Value newOut = rewriter.createOrFold<tensorrt::TransposeOp>(
        reshapeLoc, newReshape,
        AffineMap::getPermutationMap(afterReshapeTranspose, op.getContext()));

    assert(op.getType() == newOut.getType());
    rewriter.replaceOp(op, newOut);

    return success();
  }
};
} // namespace

namespace {
// reshape(transpose(x)) -> transpose(reshape(x))
// NOTE: there are more cases that could be handled here
class MoveReshapeBeforeTranspose
    : public OpRewritePattern<tensorrt::ReshapeOp> {
public:
  using OpRewritePattern<tensorrt::ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto transpose = op.getInput().getDefiningOp<tensorrt::TransposeOp>();
    if (!transpose)
      return failure();

    RankedTensorType transposeInputType = transpose.getInput().getType();
    RankedTensorType reshapeInputType =
        op.getInput().getType(); // transpose output type
    RankedTensorType reshapeOutputType = op.getType();
    if (!reshapeInputType.hasStaticShape() ||
        !reshapeOutputType.hasStaticShape() ||
        !transposeInputType.hasStaticShape())
      return failure();

    SmallVector<int64_t> transposePerm;
    for (int i = 0; i < reshapeInputType.getRank(); i++)
      transposePerm.push_back(i);
    if (!transpose.getPermutation().isPermutation())
      return failure(/* Transpose is not a permutation */);

    transposePerm = transpose.getPermutation().compose(transposePerm);

    struct ReshapeGroup {
      SmallVector<int64_t> transposeInAxes;
      SmallVector<int64_t> transposeOutAxes;
      SmallVector<int64_t> reshapeOut;
      int64_t startOutputIdx;
    };
    SmallVector<ReshapeGroup> reshapeGroups;

    SmallVector<int64_t> transposeInAxes;
    SmallVector<int64_t> transposeOutAxes;
    SmallVector<int64_t> groupReshapeOut;
    size_t inputNumElems = 1;
    size_t outputNumElems = 1;
    int j = 0;
    for (int i = 0; i < reshapeInputType.getRank(); i++) {
      inputNumElems *= reshapeInputType.getDimSize(i);
      if (!transposeInAxes.empty() &&
          transposeInAxes.back() + 1 != transposePerm[i])
        return failure(/* the transpose and the reshape are not commutative */);
      transposeInAxes.push_back(transposePerm[i]);
      while (j < reshapeOutputType.getRank() &&
             inputNumElems > outputNumElems) {
        outputNumElems *= reshapeOutputType.getDimSize(j);
        groupReshapeOut.push_back(reshapeOutputType.getDimSize(j));
        transposeOutAxes.push_back(j++);
      }
      if (inputNumElems == outputNumElems) {
        reshapeGroups.push_back(ReshapeGroup{
            .transposeInAxes = transposeInAxes,
            .transposeOutAxes = transposeOutAxes,
            .reshapeOut = groupReshapeOut,
            .startOutputIdx = -1, // set later
        });
        transposeInAxes.clear();
        transposeOutAxes.clear();
        groupReshapeOut.clear();
      }
    }
    assert(inputNumElems == outputNumElems);
    while (j < reshapeOutputType.getRank()) {
      outputNumElems *= reshapeOutputType.getDimSize(j);
      groupReshapeOut.push_back(reshapeOutputType.getDimSize(j));
      transposeOutAxes.push_back(j++);
    }
    assert(inputNumElems == outputNumElems);
    assert(transposeInAxes.empty());
    if (!transposeOutAxes.empty() || !groupReshapeOut.empty())
      reshapeGroups.push_back(ReshapeGroup{
          .transposeInAxes = transposeInAxes,
          .transposeOutAxes = transposeOutAxes,
          .reshapeOut = groupReshapeOut,
          .startOutputIdx = -1, // set later
      });

    SmallVector<int64_t> newTranspose;
    SmallVector<int64_t> newReshape;

    std::sort(reshapeGroups.begin(), reshapeGroups.end(), [](auto &a, auto &b) {
      if (a.transposeInAxes.empty())
        return false;
      if (b.transposeInAxes.empty())
        return true;
      return a.transposeInAxes[0] < b.transposeInAxes[0];
    });

    for (auto &group : reshapeGroups) {
      group.startOutputIdx = newReshape.size();
      for (int64_t i : group.reshapeOut)
        newReshape.push_back(i);
    }

    std::sort(reshapeGroups.begin(), reshapeGroups.end(), [](auto &a, auto &b) {
      if (a.transposeOutAxes.empty())
        return false;
      if (b.transposeOutAxes.empty())
        return true;
      return a.transposeOutAxes[0] < b.transposeOutAxes[0];
    });

    LLVM_DEBUG({
      std::stringstream out;
      out << "Reshape Groups:\n";
      for (size_t idx = 0; idx < reshapeGroups.size(); ++idx) {
        const auto &group = reshapeGroups[idx];
        out << "  Group " << idx << ":\n";
        out << "    transposeInAxes: [";
        for (size_t i = 0; i < group.transposeInAxes.size(); ++i) {
          out << group.transposeInAxes[i];
          if (i + 1 < group.transposeInAxes.size())
            out << ", ";
        }
        out << "]\n";
        out << "    transposeOutAxes: [";
        for (size_t i = 0; i < group.transposeOutAxes.size(); ++i) {
          out << group.transposeOutAxes[i];
          if (i + 1 < group.transposeOutAxes.size())
            out << ", ";
        }
        out << "]\n";
        out << "    reshapeOut: [";
        for (size_t i = 0; i < group.reshapeOut.size(); ++i) {
          out << group.reshapeOut[i];
          if (i + 1 < group.reshapeOut.size())
            out << ", ";
        }
        out << "]\n";
        out << "    startOutputIdx: " << group.startOutputIdx << "\n";
      }
      DBGS() << out.str();
    });

    for (auto &group : reshapeGroups)
      for (size_t i = 0; i < group.reshapeOut.size(); i++)
        newTranspose.push_back(group.startOutputIdx + i);

    Value newReshapeOp = rewriter.createOrFold<tensorrt::ReshapeOp>(
        op.getLoc(), reshapeInputType.clone(newReshape), transpose.getInput());
    Value newTransposeOp = rewriter.createOrFold<tensorrt::TransposeOp>(
        transpose.getLoc(), newReshapeOp,
        AffineMap::getPermutationMap(newTranspose, op.getContext()));

    assert(op.getType() == newTransposeOp.getType());
    rewriter.replaceOp(op, newTransposeOp);

    return success();
  }
};
} // namespace

namespace {
// transpose(reshape(x)) -> reshape(transpose(x))
// NOTE: there are more cases that could be handled here
class MoveTransposeBeforeReshape
    : public OpRewritePattern<tensorrt::TransposeOp> {
public:
  using OpRewritePattern<tensorrt::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto reshape = op.getInput().getDefiningOp<tensorrt::ReshapeOp>();
    if (!reshape)
      return failure();

    RankedTensorType reshapeInputType = reshape.getInput().getType();
    RankedTensorType reshapeOutputType = reshape.getType();
    RankedTensorType transposeOutputType = op.getType();
    if (!reshapeInputType.hasStaticShape() ||
        !reshapeOutputType.hasStaticShape() ||
        !transposeOutputType.hasStaticShape())
      return failure();

    SmallVector<int64_t> transposePerm;
    for (int i = 0; i < reshapeOutputType.getRank(); i++) {
      transposePerm.push_back(i);
    }
    transposePerm =
        inversePermutation(op.getPermutation()).compose(transposePerm);

    struct ReshapeGroup {
      SmallVector<int64_t> inputAxes;
      SmallVector<int64_t> outputAxes;
      SmallVector<int64_t> reshapeOut;
    };
    SmallVector<ReshapeGroup> reshapeGroups;

    SmallVector<int64_t> inputAxes;
    SmallVector<int64_t> outputAxes;
    SmallVector<int64_t> groupReshapeOut;
    size_t inputNumElems = 1;
    size_t outputNumElems = 1;
    int j = 0;
    for (int i = 0; i < reshapeInputType.getRank(); i++) {
      inputNumElems *= reshapeInputType.getDimSize(i);
      inputAxes.push_back(i);
      while (j < reshapeOutputType.getRank() &&
             inputNumElems > outputNumElems) {
        outputNumElems *= reshapeOutputType.getDimSize(j);
        groupReshapeOut.push_back(reshapeOutputType.getDimSize(j));
        if (!outputAxes.empty() && outputAxes.back() + 1 != transposePerm[j])
          return failure(
              /* the transpose and the reshape are not commutative */);
        outputAxes.push_back(transposePerm[j++]);
      }
      if (inputNumElems == outputNumElems) {
        reshapeGroups.push_back(ReshapeGroup{
            .inputAxes = inputAxes,
            .outputAxes = outputAxes,
            .reshapeOut = groupReshapeOut,
        });
        inputAxes.clear();
        outputAxes.clear();
        groupReshapeOut.clear();
      }
    }
    assert(inputNumElems == outputNumElems);
    while (j < reshapeOutputType.getRank()) {
      outputNumElems *= reshapeOutputType.getDimSize(j);
      groupReshapeOut.push_back(reshapeOutputType.getDimSize(j));
      outputAxes.push_back(transposePerm[j++]);
    }

    assert(inputNumElems == outputNumElems);
    assert(inputAxes.empty());
    if (!outputAxes.empty() || !groupReshapeOut.empty())
      reshapeGroups.push_back(ReshapeGroup{
          .inputAxes = inputAxes,
          .outputAxes = outputAxes,
          .reshapeOut = groupReshapeOut,
      });

    SmallVector<int64_t> newTranspose;
    SmallVector<int64_t> newReshape;

    std::sort(reshapeGroups.begin(), reshapeGroups.end(), [](auto &a, auto &b) {
      if (a.outputAxes.empty())
        return false;
      if (b.outputAxes.empty())
        return true;
      return a.outputAxes[0] < b.outputAxes[0];
    });

    // Debug print of reshapeGroups
    LLVM_DEBUG({
      std::stringstream out;
      out << "reshapeGroups:\n";
      for (size_t idx = 0; idx < reshapeGroups.size(); ++idx) {
        const auto &group = reshapeGroups[idx];
        out << "  Group " << idx << ":\n";
        out << "    inputAxes: [";
        for (size_t i = 0; i < group.inputAxes.size(); ++i) {
          out << group.inputAxes[i];
          if (i + 1 < group.inputAxes.size())
            out << ", ";
        }
        out << "]\n";
        out << "    outputAxes: [";
        for (size_t i = 0; i < group.outputAxes.size(); ++i) {
          out << group.outputAxes[i];
          if (i + 1 < group.outputAxes.size())
            out << ", ";
        }
        out << "]\n";
        out << "    reshapeOut: [";
        for (size_t i = 0; i < group.reshapeOut.size(); ++i) {
          out << group.reshapeOut[i];
          if (i + 1 < group.reshapeOut.size())
            out << ", ";
        }
        out << "]\n";
      }
      DBGS() << out.str();
    });

    for (auto &group : reshapeGroups) {
      for (int64_t i : group.inputAxes)
        newTranspose.push_back(i);
      for (int64_t i : group.reshapeOut)
        newReshape.push_back(i);
    }

    Value newTransposeOp;
    if (newTranspose.empty()) {
      newTransposeOp = reshape.getInput(); // this can happen in the case of a
                                           // scalar tensor<f32> type
    } else {
      newTransposeOp = rewriter.createOrFold<tensorrt::TransposeOp>(
          op.getLoc(), reshape.getInput(),
          AffineMap::getPermutationMap(newTranspose, op.getContext()));
    }
    Value newReshapeOp = rewriter.createOrFold<tensorrt::ReshapeOp>(
        reshape.getLoc(), reshapeInputType.clone(newReshape), newTransposeOp);

    assert(op.getType() == newReshapeOp.getType());
    rewriter.replaceOp(op, newReshapeOp);
    return success();
  }
};
} // namespace

namespace {
// activation(reshape(x)) -> reshape(activation(x))
class PushDownReshapeActivationRewriter
    : public OpRewritePattern<tensorrt::ActivationOp> {
public:
  using OpRewritePattern<tensorrt::ActivationOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::ActivationOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getInput().getDefiningOp<tensorrt::ReshapeOp>();
    if (!producer)
      return failure();

    auto activationOp = rewriter.create<tensorrt::ActivationOp>(
        op.getLoc(), producer.getInput(), op.getActivationType(),
        op.getAlphaAttr(), op.getBetaAttr());
    auto reshapeOp = rewriter.createOrFold<tensorrt::ReshapeOp>(
        producer.getLoc(), op.getType(), activationOp.getResult(),
        producer.getShape());
    assert(op.getType() == reshapeOp.getType());
    rewriter.replaceOp(op, reshapeOp);
    return success();
  }
};
} // namespace

namespace {
// unary(reshape(x)) -> reshape(unary(x))
class PushDownReshapeUnaryRewriter
    : public OpRewritePattern<tensorrt::UnaryOp> {
public:
  using OpRewritePattern<tensorrt::UnaryOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::UnaryOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getInput().getDefiningOp<tensorrt::ReshapeOp>();
    if (!producer)
      return failure();

    auto unaryOp = rewriter.create<tensorrt::UnaryOp>(
        op.getLoc(), producer.getInput(), op.getUnaryOperationAttr());
    auto reshapeOp = rewriter.createOrFold<tensorrt::ReshapeOp>(
        producer.getLoc(), op.getType(), unaryOp.getResult(),
        producer.getShape());
    assert(op.getType() == reshapeOp.getType());
    rewriter.replaceOp(op, reshapeOp);
    return success();
  }
};
} // namespace

namespace {
// identity(reshape(x)) -> reshape(identity(x))
class PushDownReshapeIdentityRewriter
    : public OpRewritePattern<tensorrt::IdentityOp> {
public:
  using OpRewritePattern<tensorrt::IdentityOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::IdentityOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getInput().getDefiningOp<tensorrt::ReshapeOp>();
    if (!producer)
      return failure();

    RankedTensorType newIdentityType =
        producer.getInput().getType().clone(op.getType().getElementType());
    Value newIdentityResult = rewriter.create<IdentityOp>(
        op.getLoc(), newIdentityType, producer.getInput());
    auto reshapeOp = rewriter.createOrFold<tensorrt::ReshapeOp>(
        producer.getLoc(), op.getType(), newIdentityResult,
        producer.getShape());
    assert(op.getType() == reshapeOp.getType());
    rewriter.replaceOp(op, reshapeOp);
    return success();
  }
};
} // namespace

namespace {
// reshape(unary_OpType(x)) -> unary_OpType(reshape(x))
template <typename OpType>
class PushUpReshapeUnary : public OpRewritePattern<tensorrt::ReshapeOp> {
public:
  using OpRewritePattern<tensorrt::ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getInput().getDefiningOp<OpType>();
    if (!producer)
      return failure();

    Type reshapeType =
        op.getType().clone(producer.getInput().getType().getElementType());

    Value newReshapeResult = rewriter.create<tensorrt::ReshapeOp>(
        op.getLoc(), reshapeType, producer.getInput(), op.getShape());
    auto newOp =
        rewriter.createOrFold<OpType>(producer.getLoc(), op.getType(),
                                      newReshapeResult, producer->getAttrs());
    assert(op.getType() == newOp.getType());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};
} // namespace

namespace {
// op(dequantize(quantize(x))) -> dequantize(quantize(op(x)))
template <typename OpType>
class PushUpOpQuantizeDequantize : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto dequantizeOp =
        op.getInput().template getDefiningOp<tensorrt::DequantizeOp>();
    if (!dequantizeOp)
      return failure();
    Value scale = dequantizeOp.getScale();
    RankedTensorType scaleType = cast<RankedTensorType>(scale.getType());
    if (!(scaleType.getRank() == 0 ||
          (scaleType.getRank() == 1 && scaleType.getDimSize(0) == 1)) ||
        dequantizeOp.getAxis().has_value())
      return failure();
    auto quantizeOp =
        dequantizeOp.getInput().template getDefiningOp<tensorrt::QuantizeOp>();
    if (!quantizeOp)
      return failure();
    if (quantizeOp.getScale() != scale || quantizeOp.getAxis().has_value())
      return failure();

    auto input = quantizeOp.getInput();
    auto pushedOp = rewriter.create<OpType>(
        op.getLoc(), op.getResult().getType(), input, op->getAttrs());
    RankedTensorType newQuantizedType = pushedOp.getType().clone(
        quantizeOp.getResult().getType().getElementType());
    auto newQuantizeOp = rewriter.create<tensorrt::QuantizeOp>(
        quantizeOp.getLoc(), newQuantizedType, pushedOp, scale,
        quantizeOp.getAxisAttr());
    auto newDequantizeOp = rewriter.create<tensorrt::DequantizeOp>(
        dequantizeOp.getLoc(), op.getResult().getType(),
        newQuantizeOp.getResult(), scale, dequantizeOp.getAxisAttr());
    assert(op.getType() == newDequantizeOp.getType());
    rewriter.replaceOp(op, newDequantizeOp.getResult());
    return success();
  }
};
} // namespace

namespace {
// dequantize(quantize(op(x))) -> op(dequantize(quantize(x)))
template <typename OpType>
class PushDownOpQuantizeDequantize
    : public OpRewritePattern<tensorrt::DequantizeOp> {
public:
  using OpRewritePattern<tensorrt::DequantizeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::DequantizeOp dequantizeOp,
                                PatternRewriter &rewriter) const override {
    Value scale = dequantizeOp.getScale();
    auto scaleType = cast<RankedTensorType>(scale.getType());
    if (!(scaleType.getRank() == 0 ||
          (scaleType.getRank() == 1 && scaleType.getDimSize(0) == 1)) ||
        dequantizeOp.getAxis().has_value())
      return failure();
    auto quantizeOp =
        dequantizeOp.getInput().getDefiningOp<tensorrt::QuantizeOp>();
    if (!quantizeOp)
      return failure();
    if (quantizeOp.getScale() != scale || quantizeOp.getAxis().has_value())
      return failure();

    auto op = quantizeOp.getInput().getDefiningOp<OpType>();
    if (!op)
      return failure();

    auto input = op.getInput();
    RankedTensorType newQuantizedType = input.getType().clone(
        quantizeOp.getResult().getType().getElementType());
    auto newQuantizeOp = rewriter.create<tensorrt::QuantizeOp>(
        quantizeOp.getLoc(), newQuantizedType, input, scale,
        quantizeOp.getAxisAttr());
    RankedTensorType newDequantizedType = newQuantizedType.clone(
        dequantizeOp.getResult().getType().getElementType());
    auto newDequantizeOp = rewriter.create<tensorrt::DequantizeOp>(
        dequantizeOp.getLoc(), newDequantizedType, newQuantizeOp.getResult(),
        scale, dequantizeOp.getAxisAttr());
    auto newOp =
        rewriter.create<OpType>(op.getLoc(), dequantizeOp.getResult().getType(),
                                newDequantizeOp.getResult(), op->getAttrs());
    assert(dequantizeOp.getType() == newOp.getType());
    rewriter.replaceOp(dequantizeOp, newOp.getResult());
    return success();
  }
};
} // namespace

namespace {
// Convert einsum("bij,bjk->bik", %0, %1) -> matrix_multiply(%0, %1)
// by matching different einsum patterns that are supported by the
// matrix_multiply op
class EinsumToMatrixMultiply : public OpRewritePattern<tensorrt::EinsumOp> {
public:
  using OpRewritePattern<tensorrt::EinsumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::EinsumOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2)
      return failure();

    EinsumEquation equation;
    if (failed(equation.parse(op.getEquation())))
      return failure();

    char matrixAxis[2] = {0, 0};
    char multipliedAxis = 0;
    std::string batchAxes = "";

    Value inputs[2] = {op.getInputs()[0], op.getInputs()[1]};

    for (char c : equation.lhsParts[0]) {
      if (equation.lhsParts[1].find(c) == std::string::npos) {
        if (matrixAxis[0] != 0)
          return failure(/* einsum does not match matrix multiply format */);
        matrixAxis[0] = c;
      }
      if (equation.rhs.find(c) == std::string::npos) {
        if (multipliedAxis != 0)
          return failure(/* einsum does not match matrix multipliy format */);
        multipliedAxis = c;
      }
    }
    for (char c : equation.lhsParts[1]) {
      if (equation.lhsParts[0].find(c) == std::string::npos) {
        if (matrixAxis[1] != 0)
          return failure(/* einsum does not match matrix multiply format */);
        matrixAxis[1] = c;
      }
      if (equation.rhs.find(c) == std::string::npos) {
        if (multipliedAxis != 0 && multipliedAxis != c)
          return failure(/* einsum does not match matrix multiply format */);
        multipliedAxis = c;
      }
    }

    for (size_t i = 0; i < equation.rhs.size(); i++) {
      if (equation.lhsParts[0][i] == equation.rhs[i] &&
          equation.lhsParts[1][i] == equation.rhs[i])
        batchAxes += equation.rhs[i];
      else
        break;
    }

    if (multipliedAxis == 0)
      return failure();

    if (matrixAxis[0] != 0 && matrixAxis[1] != 0 &&
        equation.rhs.find(matrixAxis[0]) > equation.rhs.find(matrixAxis[1])) {
      // the order of the arguments need to get swapped as the order for a
      // matrix multiply requires the first matrix axis appears first
      std::swap(equation.lhsParts[0], equation.lhsParts[1]);
      std::swap(matrixAxis[0], matrixAxis[1]);
      std::swap(inputs[0], inputs[1]);
    }

    MatrixOperation opType[2];
    for (int i = 0; i < 2; i++) {
      if (matrixAxis[i] == 0) {
        if (equation.lhsParts[i] == batchAxes + multipliedAxis)
          opType[i] = MatrixOperation::kVECTOR;
        else
          return failure(/* einsum does not match matrix multiply format */);
      } else {
        if (equation.lhsParts[i] ==
            (batchAxes + matrixAxis[i]) + multipliedAxis)
          opType[i] = MatrixOperation::kNONE;
        else if (equation.lhsParts[i] ==
                 (batchAxes + multipliedAxis) + matrixAxis[i])
          opType[i] = MatrixOperation::kTRANSPOSE;
        else
          return failure(/* einsum does not match matrix multiply format */);
      }
    }

    switch (opType[1]) {
    case MatrixOperation::kTRANSPOSE:
      opType[1] = MatrixOperation::kNONE;
      break;
    case MatrixOperation::kNONE:
      opType[1] = MatrixOperation::kTRANSPOSE;
      break;
    default:;
    }

    rewriter.replaceOpWithNewOp<tensorrt::MatrixMultiplyOp>(
        op, op.getResult().getType(), inputs[0], inputs[1], opType[0],
        opType[1]);

    return success();
  }
};
} // namespace

namespace {

// Push down reshape through elementwise op
// elementwise(reshape(x), y) -> reshape(elementwise(x, reshape(y)))
class PushDownReshapeElementwise
    : public OpRewritePattern<tensorrt::ElementWiseOp> {
public:
  using OpRewritePattern<tensorrt::ElementWiseOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::ElementWiseOp op,
                                PatternRewriter &rewriter) const override {
    bool hasReshapeInput = false;
    uint64_t currentEstimatedCost = 0;
    Location reshapeLoc = op.getLoc();
    for (Value input : op.getOperands()) {
      if (!cast<RankedTensorType>(input.getType()).hasStaticShape()) {
        return failure();
      }
      if (auto reshape = input.getDefiningOp<tensorrt::ReshapeOp>()) {
        if (!reshape.getInput().getType().hasStaticShape())
          return failure();
        hasReshapeInput = true;
        reshapeLoc = reshape.getLoc();
        currentEstimatedCost += estimateShuffleCost(input);
      }
    }
    if (!hasReshapeInput)
      return failure();

    if (op.getInput1().getType().getShape() !=
        op.getInput2().getType().getShape())
      return failure();

    uint64_t newCost;
    auto reshape1 = op.getInput1().getDefiningOp<tensorrt::ReshapeOp>();
    auto reshape2 = op.getInput2().getDefiningOp<tensorrt::ReshapeOp>();
    bool useLhsShape = true;
    if (reshape1 && reshape2 &&
        reshape1.getInput().getType().getShape() ==
            reshape2.getInput().getType().getShape()) {
      newCost = 0; // should always do it
    } else {
      uint64_t cost1 = estimateShuffleCost(op.getInput1());
      uint64_t cost2 = estimateShuffleCost(op.getInput2());
      if (cost1 < cost2) {
        useLhsShape =
            false; // want to put the reshape on the rhs as its cost is lower
        newCost = cost1;
      } else {
        useLhsShape =
            true; // want to put the reshape on the lhs as its cost is lower
        newCost = cost2;
      }
    }

    if (newCost >= currentEstimatedCost) {
      return failure();
    }

    Value newLhs = op.getInput1();
    Value newRhs = op.getInput2();
    while (auto reshape = newLhs.getDefiningOp<tensorrt::ReshapeOp>())
      newLhs = reshape.getInput();
    while (auto reshape = newRhs.getDefiningOp<tensorrt::ReshapeOp>())
      newRhs = reshape.getInput();

    auto newShape = useLhsShape ? reshape1.getInput().getType().getShape()
                                : reshape2.getInput().getType().getShape();

    RankedTensorType newLhsType = op.getInput1().getType().clone(newShape);
    RankedTensorType newRhsType = op.getInput2().getType().clone(newShape);

    newLhs = rewriter.createOrFold<tensorrt::ReshapeOp>(op.getLoc(), newLhsType,
                                                        newLhs);
    newRhs = rewriter.createOrFold<tensorrt::ReshapeOp>(op.getLoc(), newRhsType,
                                                        newRhs);

    RankedTensorType elementwiseType = op.getResult().getType().clone(newShape);
    auto newElementwiseOp = rewriter.create<tensorrt::ElementWiseOp>(
        op.getLoc(), elementwiseType, newLhs, newRhs,
        op.getElementwiseOperation());
    auto newReshapeOp = rewriter.createOrFold<tensorrt::ReshapeOp>(
        reshapeLoc, op.getType(), newElementwiseOp.getResult());
    assert(op.getType() == newReshapeOp.getType());
    rewriter.replaceOp(op, newReshapeOp);
    return success();
  }
};
} // namespace

namespace {
// Push up reshape through elementwise op
// reshape(elementwise(x, y)) -> elementwise(reshape(x), reshape(y))
class PushUpReshapeElementwise : public OpRewritePattern<tensorrt::ReshapeOp> {
public:
  using OpRewritePattern<tensorrt::ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto elementwiseOp = op.getInput().getDefiningOp<tensorrt::ElementWiseOp>();
    if (!elementwiseOp)
      return failure();

    RankedTensorType type = op.getType();
    if (!type.hasStaticShape())
      return failure();

    if (elementwiseOp.getInput1().getType().getShape() !=
        elementwiseOp.getInput2().getType().getShape())
      return failure();

    // heuristic to check if should apply
    Operation *lhsParent = elementwiseOp.getInput1().getDefiningOp();
    Operation *rhsParent = elementwiseOp.getInput2().getDefiningOp();
    bool isLhsParentReshapeOrTransposeOrConstant =
        lhsParent && isa<ReshapeOp, TransposeOp, ConstantOp>(lhsParent);
    bool isRhsParentReshapeOrTransposeOrConstant =
        rhsParent && isa<ReshapeOp, TransposeOp, ConstantOp>(rhsParent);
    if (!isLhsParentReshapeOrTransposeOrConstant &&
        !isRhsParentReshapeOrTransposeOrConstant)
      return failure();

    RankedTensorType newLhsType =
        type.clone(elementwiseOp.getInput1().getType().getElementType());
    RankedTensorType newRhsType =
        type.clone(elementwiseOp.getInput2().getType().getElementType());
    auto newLhs = rewriter.createOrFold<tensorrt::ReshapeOp>(
        op.getLoc(), newLhsType, elementwiseOp.getInput1());
    auto newRhs = rewriter.createOrFold<tensorrt::ReshapeOp>(
        op.getLoc(), newRhsType, elementwiseOp.getInput2());

    auto newElementwiseOp = rewriter.create<tensorrt::ElementWiseOp>(
        elementwiseOp.getLoc(), op.getResult().getType(), newLhs, newRhs,
        elementwiseOp.getElementwiseOperation());
    assert(op.getType() == newElementwiseOp.getType());
    rewriter.replaceOp(op, newElementwiseOp.getResult());

    return success();
  }
};
} // namespace

namespace {
// This pattern matches matrix multiply operations whose arguments are produced
// by a transpose that swaps only the last two dimensions. In such cases, it
// absorbs the transpose into the matrix multiply operator by toggling the
// internal transpose flag on the corresponding input, eliminating unnecessary
// explicit transpose operations.
class MatrixMultiplyTransposedArguments
    : public OpRewritePattern<tensorrt::MatrixMultiplyOp> {
public:
  using OpRewritePattern<tensorrt::MatrixMultiplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::MatrixMultiplyOp op,
                                PatternRewriter &rewriter) const override {
    bool didChange = false;
    const auto replaceArg =
        [&](Value arg,
            MatrixOperation operation) -> std::tuple<Value, MatrixOperation> {
      if (operation == MatrixOperation::kVECTOR) {
        return std::make_tuple(arg, operation);
      }
      auto transpose = arg.getDefiningOp<tensorrt::TransposeOp>();
      if (transpose && shouldFuseTranspose(transpose, op)) {
        AffineMap perm = transpose.getPermutation();
        // Check if perm swaps its last two axes while keeping everything else
        // the same
        auto permVec = llvm::to_vector(perm.getResults());
        int64_t rank = permVec.size();
        if (rank < 2)
          return std::make_tuple(arg, operation);
        bool swapsLastTwo = true;
        for (int64_t i = 0; i < rank - 2; ++i) {
          auto expr = dyn_cast<AffineDimExpr>(permVec[i]);
          if (!expr || expr.getPosition() != i) {
            swapsLastTwo = false;
            break;
          }
        }
        if (swapsLastTwo) {
          auto expr1 = dyn_cast<AffineDimExpr>(permVec[rank - 2]);
          auto expr2 = dyn_cast<AffineDimExpr>(permVec[rank - 1]);
          if (!(expr1 && expr2 && expr1.getPosition() == rank - 1 &&
                expr2.getPosition() == rank - 2))
            swapsLastTwo = false;
        }
        if (swapsLastTwo) {
          didChange = true;
          return std::make_tuple(transpose.getInput(),
                                 operation == MatrixOperation::kTRANSPOSE
                                     ? MatrixOperation::kNONE
                                     : MatrixOperation::kTRANSPOSE);
        }
        return std::make_tuple(arg, operation);
      } else {
        return std::make_tuple(arg, operation);
      }
    };

    auto [newLhs, newLhsOp] = replaceArg(op.getInput0(), op.getOp0());
    auto [newRhs, newRhsOp] = replaceArg(op.getInput1(), op.getOp1());

    if (didChange) {
      rewriter.replaceOpWithNewOp<tensorrt::MatrixMultiplyOp>(
          op, op.getType(), newLhs, newRhs, newLhsOp, newRhsOp);
      return success();
    } else {
      return failure();
    }
  }
};
} // namespace

namespace {
// push up transpose through softmax
// softmax(transpose(x)) -> transpose(softmax(x))
class PushUpTransposeSoftmax : public OpRewritePattern<tensorrt::TransposeOp> {
public:
  using OpRewritePattern<tensorrt::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto softmax = op.getInput().getDefiningOp<tensorrt::SoftMaxOp>();
    if (!softmax)
      return failure();
    unsigned axis = softmax.getAxis();
    unsigned newAxis =
        inversePermutation(op.getPermutation()).getDimPosition(axis);
    auto newTranspose = rewriter.create<tensorrt::TransposeOp>(
        op.getLoc(), softmax.getInput(), op.getPermutation());
    auto newSoftmax = rewriter.create<tensorrt::SoftMaxOp>(
        softmax.getLoc(), newTranspose, newAxis);
    assert(op.getType() == newSoftmax.getType());
    rewriter.replaceOp(op, newSoftmax.getResult());
    return success();
  }
};
} // namespace

namespace {
// push down transpose through softmax
// transpose(softmax(x)) -> softmax(transpose(x))
class PushDownTransposeSoftmax : public OpRewritePattern<tensorrt::SoftMaxOp> {
public:
  using OpRewritePattern<tensorrt::SoftMaxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::SoftMaxOp op,
                                PatternRewriter &rewriter) const override {
    auto transpose = op.getInput().getDefiningOp<tensorrt::TransposeOp>();
    if (!transpose)
      return failure();
    unsigned axis = op.getAxis();
    unsigned newAxis = transpose.getPermutation().getDimPosition(axis);
    auto newSoftmax = rewriter.create<tensorrt::SoftMaxOp>(
        op.getLoc(), transpose.getInput(), newAxis);
    auto newTranspose = rewriter.create<tensorrt::TransposeOp>(
        transpose.getLoc(), newSoftmax, transpose.getPermutation());
    assert(op.getType() == newTranspose.getType());
    rewriter.replaceOp(op, newTranspose.getResult());
    return success();
  }
};
} // namespace

namespace {
// push up reshape through softmax
// softmax(reshape(x)) -> reshape(softmax(x))
class PushUpReshapeSoftmax : public OpRewritePattern<tensorrt::ReshapeOp> {
public:
  using OpRewritePattern<tensorrt::ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().hasStaticShape() ||
        !op.getInput().getType().hasStaticShape())
      return failure();
    auto softmax = op.getInput().getDefiningOp<tensorrt::SoftMaxOp>();
    if (!softmax)
      return failure();
    int axis = softmax.getAxis();
    int newAxis = -1;
    size_t numInputElements = 1;
    size_t numOutputElements = 1;
    auto inputType = op.getInput().getType();
    auto outputType = op.getType();
    for (int i = 0, j = 0; i < inputType.getRank(); i++) {
      numInputElements *= inputType.getDimSize(i);
      while (numOutputElements < numInputElements && j < outputType.getRank())
        numOutputElements *= outputType.getDimSize(j++);
      if (i == axis) {
        if (numInputElements != numOutputElements ||
            inputType.getDimSize(i) != outputType.getDimSize(j - 1)) {
          return failure(/* the reshape impacts the elements that are getting softmaxed */);
        } else {
          newAxis = j - 1;
          break;
        }
      }
    }
    if (newAxis == -1)
      return failure();
    auto newReshape = rewriter.create<tensorrt::ReshapeOp>(
        op.getLoc(), outputType, softmax.getInput());
    auto newSoftmax = rewriter.create<tensorrt::SoftMaxOp>(
        softmax.getLoc(), newReshape.getResult(), newAxis);
    assert(op.getType() == newSoftmax.getType());
    rewriter.replaceOp(op, newSoftmax.getResult());
    return success();
  }
};
} // namespace

namespace {
// push down reshape through softmax
// reshape(softmax(x)) -> softmax(reshape(x))
class PushDownReshapeSoftmax : public OpRewritePattern<tensorrt::SoftMaxOp> {
public:
  using OpRewritePattern<tensorrt::SoftMaxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::SoftMaxOp op,
                                PatternRewriter &rewriter) const override {
    auto reshapeOp = op.getInput().getDefiningOp<tensorrt::ReshapeOp>();
    if (!reshapeOp)
      return failure();
    int axis = op.getAxis();
    int newAxis = -1;
    size_t numInputElements = 1;
    size_t numOutputElements = 1;
    auto inputType = reshapeOp.getInput().getType();
    auto outputType = reshapeOp.getType();
    for (int i = 0, j = 0; i < outputType.getRank(); i++) {
      numOutputElements *= outputType.getDimSize(i);
      while (numInputElements < numOutputElements && j < inputType.getRank())
        numInputElements *= inputType.getDimSize(j++);
      if (i == axis) {
        if (numInputElements != numOutputElements ||
            inputType.getDimSize(j - 1) != outputType.getDimSize(i)) {
          return failure();
        } else {
          newAxis = j - 1;
          break;
        }
      }
    }
    if (newAxis == -1)
      return failure();
    auto newSoftmax = rewriter.create<tensorrt::SoftMaxOp>(
        op.getLoc(), reshapeOp.getInput(), newAxis);
    auto newReshape = rewriter.create<tensorrt::ReshapeOp>(
        reshapeOp.getLoc(), outputType, newSoftmax.getResult());
    assert(op.getType() == newReshape.getType());
    rewriter.replaceOp(op, newReshape.getResult());
    return success();
  }
};
} // namespace

namespace {

// If there is a transpose that is shuffling an axis that is a 1, then that
// transpose could instead ba a reshape A reshape is preferred over a transpose
// as it should not correspond with rearranging the tensor's memory
class SimpleTransposeToReshape
    : public OpRewritePattern<tensorrt::TransposeOp> {
public:
  using OpRewritePattern<tensorrt::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensorrt::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getInput().getType().hasStaticShape())
      return failure();
    auto transposeInputType = op.getInput().getType();
    SmallVector<int64_t> transposePerm;
    int nonOneCount = 0;
    for (int i = 0; i < transposeInputType.getRank(); i++) {
      transposePerm.push_back(nonOneCount);
      if (transposeInputType.getDimSize(i) != 1)
        nonOneCount++;
    }
    if (!op.getPermutation().isPermutation())
      return failure(/* Transpose is not a permutation */);

    transposePerm = op.getPermutation().compose(transposePerm);
    for (int i = 1; i < transposePerm.size(); i++)
      if (transposePerm[i - 1] > transposePerm[i])
        return failure(/* Pattern failed to match*/);

    rewriter.replaceOpWithNewOp<tensorrt::ReshapeOp>(op, op.getType(),
                                                     op.getInput());
    return success();
  }
};

} // namespace

namespace {
class TransposeReshapeEliminationPass
    : public tensorrt::impl::TransposeReshapeEliminationPassBase<
          TransposeReshapeEliminationPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();

    // 1), convert ops to a "simpler" form using einsums and reshapes and
    // transposes
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<MatmulToEinsum, ShuffleToTransposeAndReshape,
                      PushDownTransposeToEinsum, PushUpTransposeToEinsum,
                      RankChangeToReshape<tensorrt::ExpandRankOp>,
                      RankChangeToReshape<tensorrt::CollapseRankOp>>(ctx);
      ReshapeOp::getCanonicalizationPatternsSameOp(patterns, ctx);
      TransposeOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc())
            << "failed to apply simplification patterns in " << getArgument();
        return signalPassFailure();
      }
    }

    // 1.1), eliminate 1-axis einsums as these are reshapes that can be pushed
    // around further
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<EinsumEliminate1Axis>(ctx);
      if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc())
            << "failed to apply simplification patterns in " << getArgument();
        return signalPassFailure();
      }
    }

    // 2) we try to eliminate transpose operations by "pushing down" the
    // transpose operations. This involves performing rewrites of the form
    // "op(transpose(y))->transpose(op(y))". Often, this will eliminate most
    // transpose operations in CNN networks produced by frameworks that use NHWC
    // conventions (e.g. Tensorflow and often JAX/Flax models).
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<
          PushdownTransposeEwise, TransposeConstantFold,
          PushdownTransposeIdentity, PushDownTransposeActivationRewriter,
          PushDownTransposeUnary, PushDownTransposeToEinsum,
          MoveReshapeBeforeTranspose, PushDownReshapeActivationRewriter,
          PushDownReshapeUnaryRewriter, PushDownReshapeIdentityRewriter,
          PushDownOpQuantizeDequantize<tensorrt::TransposeOp>,
          PushDownOpQuantizeDequantize<tensorrt::ReshapeOp>,
          PushReshapeDownThroughEinsum, PushDownReshapeElementwise,
          PushDownTransposeSoftmax, PushDownReshapeSoftmax,
          SimpleTransposeToReshape>(ctx, PatternBenefit(1));
      patterns.insert<EinsumPushDownTranspose>(ctx, PatternBenefit(0));
      TransposeOp::getCanonicalizationPatterns(patterns, ctx);
      ExpandRankOp::getCanonicalizationPatterns(patterns, ctx);
      ReshapeOp::getCanonicalizationPatternsSameOp(patterns, ctx);
      if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc())
            << "failed to apply pushdown patterns in " << getArgument();
        return signalPassFailure();
      }
    }

    // 3) we try to eliminate transpose operations by "pushing up" (commute
    // in the reverse direction). This can possible eliminate additional
    // transpose ops.
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<
          TransposeConstantFold, PushUpTransposeUnary<IdentityOp>,
          PushUpTransposeUnary<UnaryOp>, PushUpTransposeUnary<ActivationOp>,
          PushUpTransposeElementwise, PushUpTransposeToEinsum,
          MoveTransposeBeforeReshape, PushUpReshapeUnary<IdentityOp>,
          PushUpReshapeUnary<UnaryOp>, PushUpReshapeUnary<ActivationOp>,
          PushUpOpQuantizeDequantize<tensorrt::TransposeOp>,
          PushUpOpQuantizeDequantize<tensorrt::ReshapeOp>,
          PushReshapeUpThroughEinsum, PushUpReshapeElementwise,
          PushUpTransposeSoftmax, PushUpReshapeSoftmax,
          SimpleTransposeToReshape>(ctx, PatternBenefit(2));
      patterns.insert<EinsumPushUpTranspose>(ctx, PatternBenefit(1));
      patterns.insert<EinsumPushUp1AxisReshape>(ctx, PatternBenefit(0));
      TransposeOp::getCanonicalizationPatterns(patterns, ctx);
      ExpandRankOp::getCanonicalizationPatterns(patterns, ctx);
      ReshapeOp::getCanonicalizationPatternsSameOp(patterns, ctx);
      if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc())
            << "failed to apply pushup patterns in " << getArgument();
        return signalPassFailure();
      }
    }

    // 4) convert einsums back to matrix multiplies
    // (Unsure if this is necessary as TensorRT seems to generate the same
    // matrix mulitiply kernels)
    {
      RewritePatternSet patterns(ctx);
      TransposeOp::getCanonicalizationPatterns(patterns, ctx);
      ExpandRankOp::getCanonicalizationPatterns(patterns, ctx);
      ReshapeOp::getCanonicalizationPatterns(
          patterns, ctx); // convert back to expand rank and collapse rank ops
      patterns.insert<EinsumToMatrixMultiply>(ctx, PatternBenefit(1));
      patterns.insert<
          MatrixMultiplyTransposedArguments, EinsumPushUp1AxisReshape,
          EinsumPushUpMultipleMulitipliedAxes, SimpleTransposeToReshape>(ctx);
      if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc())
            << "failed to apply convert back to matrix multiply pattern "
            << getArgument();
        return signalPassFailure();
      }
    }

    // 4.1) if there are any remaining einsums, merge the transposes back into
    // the einsum
    {
      RewritePatternSet patterns(ctx);
      TransposeOp::getCanonicalizationPatterns(patterns, ctx);
      ExpandRankOp::getCanonicalizationPatterns(patterns, ctx);
      ReshapeOp::getCanonicalizationPatterns(
          patterns, ctx); // convert back to expand rank and collapse rank ops
      patterns.insert<EinsumToMatrixMultiply>(ctx, PatternBenefit(1));
      patterns
          .insert<PushDownTransposeToEinsum, PushUpTransposeToEinsum,
                  EinsumMergeDown1Axis, EinsumMergeUp1Axis,
                  MatrixMultiplyTransposedArguments, SimpleTransposeToReshape>(
              ctx);
      if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
        emitError(op->getLoc())
            << "failed to apply merge stragler transposes to einsum "
            << getArgument();
        return signalPassFailure();
      }
    }
  }
};
} // namespace
