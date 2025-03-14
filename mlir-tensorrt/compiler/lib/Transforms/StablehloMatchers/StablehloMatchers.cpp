//===- StablehloMatchers.cpp ----------------------------------------------===//
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
/// Matchers for StableHLO.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Transforms/StablehloMatchers/StablehloMatchers.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Debug.h"

#define TEST_DEBUG_TYPE "test-stablehlo-matchers"
#define DEBUG_TYPE "stablehlo-matchers"

#define DBGS() (llvm::dbgs() << "\n[" DEBUG_TYPE "]: ")
#define TDBGS() (llvm::dbgs() << "\n[" TEST_DEBUG_TYPE "]: ")

namespace mlir {
#define GEN_PASS_DEF_TESTSTABLEHLOMATCHERSPASS
#include "mlir-tensorrt/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

// Return true if it is a sequence of 0,1, ..., rank-1
template <typename Range>
static bool isSeqOfRank(Range &&r, int64_t rank) {
  return llvm::equal(std::forward<Range>(r), llvm::seq<int64_t>(0, rank));
}

/// %1 = stabehlo.broadcast_in_dim %input dims = [0, 1, 2] : tensor<AxBxC> to
/// tensor<AxBxCx1>
template <typename BcastInDimOp>
static bool isRankExpansionByOne(BcastInDimOp op) {
  TensorType inputType = op.getOperand().getType();
  TensorType resultType = op.getType();
  auto dimensions = op.getBroadcastDimensions();
  if (inputType.getRank() != resultType.getRank() - 1 ||
      resultType.getShape().back() != 1)
    return false;
  return isSeqOfRank(dimensions, inputType.getRank());
}

/// %2 = stabehlo.broadcast_in_dim %1 dims = [0, 1, 2, 3] : tensor<AxBxCx1> to
/// tensor<AxBxCxD>
template <typename InputOp>
static bool isBroadcastInFinalDim(InputOp op) {
  TensorType inputType = op.getOperand().getType();
  TensorType resultType = op.getType();
  auto dimensions = op.getBroadcastDimensions();
  if (inputType.getRank() != resultType.getRank())
    return false;
  return isSeqOfRank(dimensions, inputType.getRank());
}

// clang-format off
/// Match:
/// %1 = stabehlo.broadcast_in_dim %input dims = [0, 1, 2] : tensor<AxBxC> to tensor<AxBxCx1>
/// %2 = stabehlo.broadcast_in_dim %1 dims = [0, 1, 2, 3] : tensor<AxBxCx1> to tensor<AxBxCxD>
// clang-format on
template <typename InputMatcher, typename BcastInDimOp>
bool matchers::detail::MatchExpandDimsAndBroadcastAlongFinalDim<
    InputMatcher, BcastInDimOp>::match(Operation *root) {
  // Match in reverse, starting from root (%2) and working up.
  if (!matchPattern(root, m_Op<BcastInDimOp>(m_Op<BcastInDimOp>(inputMatcher))))
    return false;

  // Check conditions on "last dim is expanded and broadcasted".
  auto bcastOp = cast<BcastInDimOp>(root);
  auto expandOp = cast<BcastInDimOp>(bcastOp.getOperand().getDefiningOp());
  // Reject "zero rank". You can't softmax on scalars.
  if (expandOp.getOperand().getType().getRank() == 0)
    return false;
  return isRankExpansionByOne(expandOp) && isBroadcastInFinalDim(bcastOp);
}

/// Retrieves the dimension over which reduction is performed.
template <typename ReduceOp>
static FailureOr<int64_t> getReductionDim(ReduceOp reduceOp) {
  Attribute dims = reduceOp.getDimensionsAttr();
  SmallVector<int64_t> dimensions;
  if (auto arrayAttr = dyn_cast<DenseI64ArrayAttr>(dims)) {
    dimensions = llvm::to_vector(arrayAttr.asArrayRef());
  } else if (auto arrayAttr = dyn_cast<DenseIntElementsAttr>(dims)) {
    dimensions = llvm::to_vector(arrayAttr.getValues<int64_t>());
  } else {
    llvm_unreachable("unsupported attribute type");
  }
  if (dimensions.size() != 1) {
    LLVM_DEBUG(DBGS() << "Has multiple reduction dimensions");
    // return defult incorrect value
    return failure();
  }
  return dimensions[0];
}

/// Checks if the final dimension is the reduced dimension
template <typename ReduceOp>
static bool isReductionDimFinalDim(ReduceOp reduceOp, int64_t reductionDim) {
  // check that the reduction dimension is the final dimension
  TensorType inputType =
      llvm::cast<TensorType>(reduceOp->getOperand(0).getType());
  if (reductionDim != inputType.getRank() - 1) {
    LLVM_DEBUG(DBGS() << "Reduction dim is not the final dimension");
    return false;
  }
  return true;
}

template <typename ReduceOp, typename BodyOp>
static bool isCorrectBodyOp(ReduceOp reduceOp) {
  // check that there are only two ops, max and return,
  // in the reduce op's body.
  Block *body = &reduceOp.getBody().front();
  unsigned numOpsInRedOp = body->getOperations().size() - 1;
  if (numOpsInRedOp != 1)
    return false;
  return isa<BodyOp>(body->getOperations().front());
}

/// Match ReduceOp with a max op in its body across its final dimension
/// The initial values are -inf
template <typename InputMatcher, typename ReduceOp, typename MaxOp>
bool matchers::detail::MatchReduceMaxOverLastDim<
    InputMatcher, ReduceOp, MaxOp>::match(Operation *root) {
  ReduceOp reduceOp = llvm::dyn_cast<ReduceOp>(root);
  // check that init matches -inf
  auto init = reduceOp->getOperand(1);
  if (!matchPattern(init, m_NegInfFloat())) {
    LLVM_DEBUG(DBGS() << "Initial value is not neg inf float");
    return false;
  }
  auto reductionDim = getReductionDim(reduceOp);
  if (!failed(reductionDim) && !isReductionDimFinalDim(reduceOp, *reductionDim))
    return false;

  if (!isCorrectBodyOp<ReduceOp, MaxOp>(reduceOp))
    return false;

  // If the function reaches here, the reduce op has a single op in its body
  // and has a single reduction dim ie the final dim and a single input.
  deducedSoftmaxInput = reduceOp->getOperand(0);
  redMaxDim = *reductionDim;
  return true;
}

/// Match ReduceOp with an Add op in its body across its final dimension
/// The initial values are zeros for Add.
template <typename InputMatcher, typename ReduceOp, typename AddOp>
bool matchers::detail::MatchReduceAddOverLastDim<
    InputMatcher, ReduceOp, AddOp>::match(Operation *root) {
  ReduceOp reduceOp = llvm::dyn_cast<ReduceOp>(root);

  // Since this is reduce-sum, the init constant should match constant zero
  // splat tensor. `m_AnyZeroFloat` should match `0.0` and `-0.0`.
  auto init = reduceOp->getOperand(1);
  if (!matchPattern(init, m_AnyZeroFloat())) {
    LLVM_DEBUG(DBGS() << "Initial value is not a zero float");
    return false;
  }

  auto reductionDim = getReductionDim(reduceOp);

  if (!failed(reductionDim) && !isReductionDimFinalDim(reduceOp, *reductionDim))
    return false;

  if (!isCorrectBodyOp<ReduceOp, AddOp>(reduceOp))
    return false;

  // If the function reaches here, the reduce op has a single op in its body
  // and has a single reduction dim ie the final dim and a single input.
  return true;
}

/// Match the following pattern of operations, rooted at DivOp
/// ReduceMax -> Broadcastx2 -> Subtract -> Exponential -> ReduceAdd
/// -> Broadcast x 2 -> Divide
template <typename BcastInDimOp, typename ReduceOp, typename SubOp,
          typename ExpnOp, typename DivideOp, typename MaxOp, typename AddOp>
bool matchers::detail::HLOToSoftmaxMatcher<BcastInDimOp, ReduceOp, SubOp,
                                           ExpnOp, DivideOp, MaxOp,
                                           AddOp>::match(Operation *op) {
  auto inputValue = m_Any();
  auto reduceMaxOp =
      m_matchReduceMaxOverFinalDim<decltype(inputValue), ReduceOp, MaxOp>(
          inputValue, softmaxInputOperand, softmaxAxis);
  auto broadCastedRedMaxOp =
      m_matchExpandDimsAndBroadcastAlongFinalDim<decltype(reduceMaxOp),
                                                 BcastInDimOp>(reduceMaxOp);
  auto subOp = m_Op<SubOp>(inputValue, broadCastedRedMaxOp);
  auto expOp = m_Op<ExpnOp>(subOp);
  auto redSumOp =
      m_matchReduceAddOverFinalDim<decltype(expOp), ReduceOp, AddOp>(expOp);
  auto broadcastedRedSumExp =
      m_matchExpandDimsAndBroadcastAlongFinalDim<decltype(redSumOp),
                                                 BcastInDimOp>(redSumOp);
  return matchPattern(op, m_Op<DivideOp>(expOp, broadcastedRedSumExp));
}
template bool matchers::detail::HLOToSoftmaxMatcher<
    stablehlo::BroadcastInDimOp, stablehlo::ReduceOp, stablehlo::SubtractOp,
    stablehlo::ExpOp, stablehlo::DivOp, stablehlo::MaxOp,
    stablehlo::AddOp>::match(Operation *);

namespace {
struct TestRaiseToSoftmax : public OpRewritePattern<stablehlo::DivOp> {
  using OpRewritePattern<stablehlo::DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DivOp op,
                                PatternRewriter &rewriter) const override {

    if (op->hasAttr("__matched__softmax__")) {
      LLVM_DEBUG(TDBGS() << "Softmax previously matched.");
      return failure();
    }
    Value deducedSoftmaxInp;
    int64_t softmaxAxis;
    if (matchPattern(op.getOperation(), matchers::m_StableHLOSoftmaxMatcher(
                                            deducedSoftmaxInp, softmaxAxis))) {
      LLVM_DEBUG(TDBGS() << "Softmax matched at stablehlo::DivOp at "
                         << op->getLoc());
      op->setAttr("__matched__softmax__", UnitAttr::get(op->getContext()));
      return success();
    }
    op->setAttr("__not__softmax__", UnitAttr::get(op->getContext()));
    return failure();
  }
};

/// This pass is used to raise the input IR to multiple recognized MHA patterns.
/// Eg: for stablehlo: stablehlo.dot_general -> tensorrt.softmax ->
/// stablehlo.dot_general
///     for tensorrt: tensorrt.einsum -> tensorrt.softmax -> tensorrt.einsum
/// In both of these cases, it is possible that tensorrt.softmax can be broken
/// down to more basic ops like subtract, exponential and divide.
class TestStablehloMatchersPass
    : public impl::TestStablehloMatchersPassBase<TestStablehloMatchersPass> {
  using Base::Base;

public:
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet mhaPatterns(ctx);
    mhaPatterns.add<TestRaiseToSoftmax>(mhaPatterns.getContext());
    if (failed(applyPatternsGreedily(op, std::move(mhaPatterns)))) {
      emitError(op->getLoc()) << "failed to convert patterns from "
                                 "stablehlo to tensorrt. ";
      return signalPassFailure();
    }
  }
};
} // namespace
