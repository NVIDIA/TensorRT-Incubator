//===- Matchers.h -----------------------------------------------*- C++ -*-===//
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
// Implementation of `stablehlo` dialect IR pattern matching for stablehlo to
// TRT conversion.
//===----------------------------------------------------------------------===//
#ifndef CONVERSION_HLOTOTENSORRT_MATCHERS_H
#define CONVERSION_HLOTOTENSORRT_MATCHERS_H

#include "mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h"
#include "mlir/IR/Matchers.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace matchers {

// Methods in this namespace are extensions of the matchers provided by upstream
// `mlir/IR/Matchers.h`.
namespace detail {

/// Match a `stablehlo.compare` op with `direction` and other `matchers` on the
/// operands (optional).
template <typename... OperandMatchers>
struct StablehloComparisonOpMatcher {
  stablehlo::ComparisonDirection direction;
  std::tuple<OperandMatchers...> operandMatchers;

  StablehloComparisonOpMatcher(stablehlo::ComparisonDirection direction,
                               OperandMatchers... matchers)
      : direction(direction), operandMatchers(matchers...) {}

  bool match(Operation *op) {
    auto compareOp = dyn_cast<stablehlo::CompareOp>(op);
    if (!compareOp)
      return false;
    if (compareOp.getComparisonDirection() != direction)
      return false;
    bool res = true;
    enumerate(operandMatchers, [&](size_t index, auto &matcher) {
      res &= matchOperandOrValueAtIndex(op, index, matcher);
    });
    return res;
  }
};

template <stablehlo::ComparisonDirection direction, typename... Matchers>
inline auto m_stablehloComparison(Matchers... matchers) {
  return StablehloComparisonOpMatcher<Matchers...>(direction, matchers...);
}

/// A type of pattern representing ArgMinMax op as `stablehlo.reduce`
template <stablehlo::ComparisonDirection direction>
inline auto patternOneArgMinMax(Value lhsValue, Value lhsIndex, Value rhsValue,
                                Value rhsIndex) {
  static_assert(direction == stablehlo::ComparisonDirection::GE ||
                    direction == stablehlo::ComparisonDirection::LE,
                "expected GE or LE comparison direction");
  auto compareVal =
      m_stablehloComparison<direction>(m_Val(lhsValue), m_Val(rhsValue));

  // This is matching the following logic: If the values are equal, choose the
  // minimum index. Otherwise choose the value that corresponds to the
  // comparison above direction.
  auto eqCompare = m_stablehloComparison<stablehlo::ComparisonDirection::EQ>(
      m_Val(lhsValue), m_Val(rhsValue));
  auto eqBranch = m_Op<stablehlo::MinOp>(m_Val(lhsIndex), m_Val(rhsIndex));
  auto nonEqBranch =
      m_Op<stablehlo::SelectOp>(compareVal, m_Val(lhsIndex), m_Val(rhsIndex));
  auto idxPattern = m_Op<stablehlo::SelectOp>(eqCompare, eqBranch, nonEqBranch);

  if constexpr (direction == stablehlo::ComparisonDirection::GE)
    return m_Op<stablehlo::ReturnOp>(
        m_Op<stablehlo::MaxOp>(m_Val(lhsValue), m_Val(rhsValue)), idxPattern);
  else
    return m_Op<stablehlo::ReturnOp>(
        m_Op<stablehlo::MinOp>(m_Val(lhsValue), m_Val(rhsValue)), idxPattern);
}

/// Another type of pattern that implement ArgMinMax as `stablehlo.reduce`. This
/// is variation of pattern one in a way that ops are further broken down. This
/// pattern has little different reducer for float and int. Int reducer looks as
/// follows, reducer(%arg1: tensor<i32>, %arg3: tensor<i32>) (%arg2:
/// tensor<i32>, %arg4: tensor<i32>)  {
///    %2 = stablehlo.compare  LT, %arg1, %arg3,  SIGNED : (tensor<i32>,
///    tensor<i32>)
///    -> tensor<i1> %3 = stablehlo.compare  EQ, %arg1, %arg3,  SIGNED :
///    (tensor<i32>, tensor<i32>) -> tensor<i1> %4 = stablehlo.compare  LT,
///    %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> %5 =
///    stablehlo.and %3, %4 : tensor<i1> %6 = stablehlo.or %2, %5 : tensor<i1>
///    %7 = stablehlo.select %2, %arg1, %arg3 : tensor<i1>, tensor<i32> %8 =
///    stablehlo.select %6, %arg2, %arg4 : tensor<i1>, tensor<i32>
///    stablehlo.return %7, %8 : tensor<i32>, tensor<i32>
///}
template <stablehlo::ComparisonDirection direction>
auto patternTwoArgMinMaxInt(Value lhsValue, Value lhsIndex, Value rhsValue,
                            Value rhsIndex) {
  // Compare if lhsValue is greater or less than rhsValue
  auto compareVal =
      m_stablehloComparison<direction>(m_Val(lhsValue), m_Val(rhsValue));
  // Compare of lhsValue is equal to rhsValue
  auto equalityVal = m_stablehloComparison<stablehlo::ComparisonDirection::EQ>(
      m_Val(lhsValue), m_Val(rhsValue));
  // Compare of lhsIndex is smaller than rhsIndex
  // Applicable when lhsValue equals rhsValue
  auto smallerIdx = m_stablehloComparison<stablehlo::ComparisonDirection::LT>(
      m_Val(lhsIndex), m_Val(rhsIndex));
  // Check if lhsValue equal rhsValue and lhsIndex less than rhsIndex
  auto equalValAndSmallerIdx = m_Op<stablehlo::AndOp>(equalityVal, smallerIdx);
  auto compareValOrEqualValAndSmallerIdx =
      m_Op<stablehlo::OrOp>(compareVal, equalValAndSmallerIdx);
  auto idxMinMax = m_Op<stablehlo::SelectOp>(compareValOrEqualValAndSmallerIdx,
                                             m_Val(lhsIndex), m_Val(rhsIndex));
  static_assert(direction == stablehlo::ComparisonDirection::GT ||
                    direction == stablehlo::ComparisonDirection::LT,
                "expected GE or LE comparison direction");
  if constexpr (direction == stablehlo::ComparisonDirection::GT)
    return m_Op<stablehlo::ReturnOp>(
        m_Op<stablehlo::MaxOp>(m_Val(lhsValue), m_Val(rhsValue)), idxMinMax);
  else
    return m_Op<stablehlo::ReturnOp>(
        m_Op<stablehlo::MinOp>(m_Val(lhsValue), m_Val(rhsValue)), idxMinMax);
}

/// Float reducer for pattern two
/// reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4:
/// tensor<i32>)  {
///      %2 = stablehlo.compare  LT, %arg1, %arg3,  FLOAT : (tensor<f32>,
///      tensor<f32>) -> tensor<i1> %3 = stablehlo.compare  NE, %arg1, %arg1,
///      FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1> %4 = stablehlo.or %2,
///      %3 : tensor<i1> %5 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT :
///      (tensor<f32>, tensor<f32>) -> tensor<i1> %6 = stablehlo.compare  LT,
///      %arg2, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> %7 =
///      stablehlo.and %5, %6 : tensor<i1> %8 = stablehlo.or %4, %7 : tensor<i1>
///      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f32> %10 =
///      stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
///      stablehlo.return %9, %10 : tensor<f32>, tensor<i32>
/// }
/// NOTE* %3 = stablehlo.compare  NE, %arg1, %arg1 should return to false,
/// always.
/// TODO: (sagar) can this be optimized? Two almost similar patters are being
/// matched separately.
template <stablehlo::ComparisonDirection direction>
auto patternTwoArgMinMaxFloat(Value lhsValue, Value lhsIndex, Value rhsValue,
                              Value rhsIndex) {
  // Compare if lhsValue is greater or less than rhsValue
  auto compareVal =
      m_stablehloComparison<direction>(m_Val(lhsValue), m_Val(rhsValue));
  auto notEqualityLhs =
      m_stablehloComparison<stablehlo::ComparisonDirection::NE>(
          m_Val(lhsValue), m_Val(lhsValue));
  auto compareUpdatedVal = m_Op<stablehlo::OrOp>(compareVal, notEqualityLhs);
  // Compare of lhsValue is equal to rhsValue
  auto equalityVal = m_stablehloComparison<stablehlo::ComparisonDirection::EQ>(
      m_Val(lhsValue), m_Val(rhsValue));
  // Compare of lhsIndex is smaller than rhsIndex
  // Applicable when lhsValue equals rhsValue
  auto smallerIdx = m_stablehloComparison<stablehlo::ComparisonDirection::LT>(
      m_Val(lhsIndex), m_Val(rhsIndex));
  // Check if lhsValue equal rhsValue and lhsIndex less than rhsIndex
  auto equalValAndSmallerIdx = m_Op<stablehlo::AndOp>(equalityVal, smallerIdx);
  auto compareValOrEqualValAndSmallerIdx =
      m_Op<stablehlo::OrOp>(compareUpdatedVal, equalValAndSmallerIdx);
  auto valueMinMax = m_Op<stablehlo::SelectOp>(
      compareUpdatedVal, m_Val(lhsValue), m_Val(rhsValue));
  auto idxMinMax = m_Op<stablehlo::SelectOp>(compareValOrEqualValAndSmallerIdx,
                                             m_Val(lhsIndex), m_Val(rhsIndex));
  return m_Op<stablehlo::ReturnOp>(valueMinMax, idxMinMax);
}

/// Match `stablehlo.reduce(x, stablehlo.broadcast_in_dim(stablehlo.iota))` IR
/// that represents an argmin or argmax operation rooted at `stablehlo.reduce`.
template <stablehlo::ComparisonDirection direction>
struct StablehloArgMinMaxReduceMatcher {

  bool match(Operation *op) {
    auto reduceOp = dyn_cast<stablehlo::ReduceOp>(op);
    if (!reduceOp || reduceOp.getInputs().size() != 2)
      return false;
    if (reduceOp.getDimensions().size() != 1)
      return false;

    // We should have 4 arguments: 2 for the values and 2 for the indices.
    Block *reduceBody = &reduceOp.getBody().front();
    if (reduceBody->getNumArguments() != 4)
      return false;
    Value lhsValue = reduceBody->getArgument(0);
    Value lhsIndex = reduceBody->getArgument(1);
    Value rhsValue = reduceBody->getArgument(2);
    Value rhsIndex = reduceBody->getArgument(3);
    auto termOp = cast<stablehlo::ReturnOp>(reduceBody->getTerminator());

    // Pattern-match the IR using the `return` operands as the roots.
    if (matchPattern(termOp, patternOneArgMinMax<direction>(
                                 lhsValue, lhsIndex, rhsValue, rhsIndex)))
      return true;

    constexpr stablehlo::ComparisonDirection patternTwoDirection =
        (direction == stablehlo::ComparisonDirection::LE)
            ? stablehlo::ComparisonDirection::LT
            : stablehlo::ComparisonDirection::GT;
    if (isa<IntegerType>(
            cast<RankedTensorType>(lhsValue.getType()).getElementType()) &&
        matchPattern(termOp, patternTwoArgMinMaxInt<patternTwoDirection>(
                                 lhsValue, lhsIndex, rhsValue, rhsIndex)))
      return true;

    if (matchPattern(termOp, patternTwoArgMinMaxFloat<patternTwoDirection>(
                                 lhsValue, lhsIndex, rhsValue, rhsIndex)))
      return true;

    return false;
  }
};
} // namespace detail

/// Match a `stablehlo.compare` operation that specifies `direction` as the
/// comparison direction.
template <typename... Matchers>
auto m_StablehloComparisonWithDirection(
    stablehlo::ComparisonDirection direction, Matchers... matchers) {
  return detail::StablehloComparisonOpMatcher(direction, matchers...);
}

} // namespace matchers

/// Base pattern class for all Stablehlo to TensorRT rewrites. This class
/// will automatically skip target operations that are nested within stablehlo
/// op region bodies (e.g. for stablehlo.reduce-type ops). Conversion of
/// stablehlo ops with nested regions must be initiated at the level of the op
/// containing the region.
template <typename SourceOp>
class ConvertHloOpToTensorRTPattern
    : public ConvertOpToTensorRTPattern<SourceOp> {
public:
  using ConvertOpToTensorRTPattern<SourceOp>::ConvertOpToTensorRTPattern;

  using typename ConvertOpToTensorRTPattern<SourceOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Operation *parent = op->getParentOp();
    // Don't convert ops nested within stablehlo regions unless that region is a
    // if/while op.
    if (parent && isa<stablehlo::StablehloDialect>(parent->getDialect()) &&
        !isa<stablehlo::IfOp, stablehlo::WhileOp, stablehlo::CaseOp>(parent))
      return rewriter.notifyMatchFailure(op,
                                         "nested within HLO operation body");
    return matchAndRewrite(cast<SourceOp>(op),
                           OpAdaptor(operands, op->getAttrDictionary()),
                           rewriter);
  }
  virtual LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)op;
    (void)adaptor;
    (void)rewriter;
    llvm_unreachable("overwrite matchAndRewrite method");
    return success();
  }
};
} // namespace mlir

#endif // CONVERSION_HLOTOTENSORRT_MATCHERS_H
