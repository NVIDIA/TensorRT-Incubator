//===- StablehloMatchers.cpp ---------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "mlir-tensorrt/Dialect/StablehloExt/Utils/StablehloMatchers.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Debug.h"

#define TEST_DEBUG_TYPE "test-stablehlo-matchers"
#define DEBUG_TYPE "stablehlo-matchers"

#define DBGS() (llvm::dbgs() << "\n[" DEBUG_TYPE "]: ")
#define TDBGS() (llvm::dbgs() << "\n[" TEST_DEBUG_TYPE "]: ")

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// Matchers
//===----------------------------------------------------------------------===//

// clang-format off
/// Example Match with stablehlo:
/// %1 = stabehlo.broadcast_in_dim %input dims = [0, 1, 2] : tensor<AxBxC> to tensor<AxBxCx1>
/// %2 = stabehlo.broadcast_in_dim %1 dims = [0, 1, 2, 3] : tensor<AxBxCx1> to tensor<AxBxCxD>
// clang-format on
template <typename InputMatcher, typename BcastInDimOp>
struct MatchExpandDimsAndBroadcastAlongFinalDim {
  InputMatcher inputMatcher;
  /// `finalShapeToMatch` is only used to match the type
  MatchExpandDimsAndBroadcastAlongFinalDim(InputMatcher inputValue)
      : inputMatcher(inputValue) {}
  bool match(Operation *root);
};

// clang-format off
/// Match:
/// %1 = stabehlo.broadcast_in_dim %input dims = [0, 1, 2] : tensor<AxBxC> to tensor<AxBxCx1>
/// %2 = stabehlo.broadcast_in_dim %1 dims = [0, 1, 2, 3] : tensor<AxBxCx1> to tensor<AxBxCxD>
// clang-format on
template <typename InputMatcher, typename BcastInDimOp>
bool MatchExpandDimsAndBroadcastAlongFinalDim<
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

/// Match a double broadcast pattern that first expands the dims
/// and then broadcasts the final dim.
template <typename InputMatcher, typename BcastInDimOp>
inline auto
m_matchExpandDimsAndBroadcastAlongFinalDim(InputMatcher inputValue) {
  return MatchExpandDimsAndBroadcastAlongFinalDim<InputMatcher, BcastInDimOp>(
      inputValue);
}

/// Match ReduceOp with a max op in its body and
/// reduces its final dimension
// clang-format off
/// Example match:
///  //Initialized to -inf
///  %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
///  %2 = stablehlo.reduce(%0 init: %1) across dimensions = [3] :
///            (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
///   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
///    %12 = stablehlo.maximum %arg2, %arg3 : tensor<f32> //Body Op is maximum
///    stablehlo.return %12 : tensor<f32>
///  }
// clang-format on
template <typename InputMatcher, typename ReduceOp, typename MaxOp>
struct MatchReduceMaxOverLastDim {
  InputMatcher inputMatcher;
  Value &deducedSoftmaxInput;
  int64_t &redMaxDim;
  MatchReduceMaxOverLastDim(InputMatcher inputValue, Value &deducedSoftmaxInput,
                            int64_t &deducedRedMaxDim)
      : inputMatcher(inputValue), deducedSoftmaxInput(deducedSoftmaxInput),
        redMaxDim(deducedRedMaxDim) {}
  bool match(Operation *root);
};

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
bool MatchReduceMaxOverLastDim<InputMatcher, ReduceOp, MaxOp>::match(
    Operation *root) {
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

/// Match ReduceOp with a Max op in its body and
/// reduces its final dimension
template <typename InputMatcher, typename ReduceOp, typename MaxOp>
inline auto m_matchReduceMaxOverFinalDim(InputMatcher inputValue,
                                         Value &deducedSoftmaxInput,
                                         int64_t &deducedRedMaxDim) {
  return MatchReduceMaxOverLastDim<InputMatcher, ReduceOp, MaxOp>(
      inputValue, deducedSoftmaxInput, deducedRedMaxDim);
}

/// Match ReduceOp with a Sum op in its body and
/// reduces its final dimension
// clang-format off
/// Example match:
///  // initialized to zeros
///  %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
///  %8 = stablehlo.reduce(%6 init: %7) across dimensions = [3] :
///          (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
///  reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
///   %12 = stablehlo.add %arg2, %arg3 : tensor<f32>
///   stablehlo.return %12 : tensor<f32>
/// }
// clang-format on
template <typename InputMatcher, typename ReduceOp, typename AddOp>
struct MatchReduceAddOverLastDim {
  InputMatcher inputMatcher;
  MatchReduceAddOverLastDim(InputMatcher inputValue)
      : inputMatcher(inputValue) {}
  bool match(Operation *root);
};

/// Match ReduceOp with an Add op in its body across its final dimension
/// The initial values are zeros for Add.
template <typename InputMatcher, typename ReduceOp, typename AddOp>
bool MatchReduceAddOverLastDim<InputMatcher, ReduceOp, AddOp>::match(
    Operation *root) {
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

/// Match ReduceOp with a Sum op in its body and
/// reduces its final dimension
template <typename InputMatcher, typename ReduceOp, typename AddOp>
inline auto m_matchReduceAddOverFinalDim(InputMatcher inputValue) {
  return MatchReduceAddOverLastDim<InputMatcher, ReduceOp, AddOp>(inputValue);
}

/// Match the following pattern of operations, rooted at DivOp
/// ReduceMax -> Broadcastx2 -> Subtract -> Exponential -> ReduceAdd
/// -> Broadcast x 2 -> Divide
template <typename BcastInDimOp, typename ReduceOp, typename SubOp,
          typename ExpnOp, typename DivideOp, typename MaxOp, typename AddOp>
bool stablehlo::detail::HLOToSoftmaxMatcher<BcastInDimOp, ReduceOp, SubOp,
                                            ExpnOp, DivideOp, MaxOp,
                                            AddOp>::match(Operation *op) {
  auto inputValue = mlir::matchers::m_Any();
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
template bool stablehlo::detail::HLOToSoftmaxMatcher<
    stablehlo::BroadcastInDimOp, stablehlo::ReduceOp, stablehlo::SubtractOp,
    stablehlo::ExpOp, stablehlo::DivOp, stablehlo::MaxOp,
    stablehlo::AddOp>::match(Operation *);
