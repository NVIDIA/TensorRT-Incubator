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
#ifndef MLIR_TENSORRT_DIALECT_STABLEHLOEXT_UTILS_STABLEHLOMATCHERS
#define MLIR_TENSORRT_DIALECT_STABLEHLOEXT_UTILS_STABLEHLOMATCHERS

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo {

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

} // namespace mlir::stablehlo

#endif // MLIR_TENSORRT_DIALECT_STABLEHLOEXT_UTILS_STABLEHLOMATCHERS
