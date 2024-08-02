//===- StablehloMatchers.h  -----------------------------------------------===//
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
/// This file defines different matchers for StableHLO.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_TRANSFORMS_STABLEHLOMATCHERS_STABLEHLOMATCHERS
#define MLIR_TENSORRT_TRANSFORMS_STABLEHLOMATCHERS_STABLEHLOMATCHERS

#include "mlir/IR/Matchers.h"
#include "stablehlo/dialect/StablehloOps.h"
namespace mlir {
namespace matchers {
namespace detail {

/// Matcher for stablehlo
/// reduce(max)->subtract->exponential->reduce(add)->divide The Softmax Matcher
/// is rooted at stablehlo::DivOp operation.
template <typename BcastInDimOp, typename ReduceOp, typename SubOp,
          typename ExpnOp, typename DivideOp, typename MaxOp, typename AddOp>
struct HLOToSoftmaxMatcher {
  HLOToSoftmaxMatcher(Value &softmaxInputDeduced, int64_t &reductionDim)
      : softmaxInputOperand(softmaxInputDeduced), softmaxAxis(reductionDim) {}
  Value &softmaxInputOperand;
  int64_t &softmaxAxis;
  bool match(Operation *op);
};

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

} // namespace detail

/// Match a pattern rooted at stablhehlo.divide to a softmax
/// operation.
inline auto m_StableHLOSoftmaxMatcher(Value &softmaxInputOperand,
                                      int64_t &softmax_axis) {
  return detail::HLOToSoftmaxMatcher<
      mlir::stablehlo::BroadcastInDimOp, mlir::stablehlo::ReduceOp,
      mlir::stablehlo::SubtractOp, mlir::stablehlo::ExpOp,
      mlir::stablehlo::DivOp, mlir::stablehlo::MaxOp, mlir::stablehlo::AddOp>(
      softmaxInputOperand, softmax_axis);
}

/// Match a double broadcast pattern that first expands the dims
/// and then broadcasts the final dim.
template <typename InputMatcher, typename BcastInDimOp>
inline auto
m_matchExpandDimsAndBroadcastAlongFinalDim(InputMatcher inputValue) {
  return detail::MatchExpandDimsAndBroadcastAlongFinalDim<InputMatcher,
                                                          BcastInDimOp>(
      inputValue);
}

/// Match ReduceOp with a Max op in its body and
/// reduces its final dimension
template <typename InputMatcher, typename ReduceOp, typename MaxOp>
inline auto m_matchReduceMaxOverFinalDim(InputMatcher inputValue,
                                         Value &deducedSoftmaxInput,
                                         int64_t &deducedRedMaxDim) {
  return detail::MatchReduceMaxOverLastDim<InputMatcher, ReduceOp, MaxOp>(
      inputValue, deducedSoftmaxInput, deducedRedMaxDim);
}

/// Match ReduceOp with a Sum op in its body and
/// reduces its final dimension
template <typename InputMatcher, typename ReduceOp, typename AddOp>
inline auto m_matchReduceAddOverFinalDim(InputMatcher inputValue) {
  return detail::MatchReduceAddOverLastDim<InputMatcher, ReduceOp, AddOp>(
      inputValue);
}

} // namespace matchers
} // namespace mlir

#endif // MLIR_TENSORRT_TRANSFORMS_STABLEHLOMATCHERS_STABLEHLOMATCHERS