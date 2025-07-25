//===- ConstantFolding.cpp  -----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
// Where noted below, some code is reproduced from upstream Stablehlo
// "aggressive folder" patterns. They can be removed once we fix upstream and
// can directly use those patterns. The copyright/license is reproduced below:
//
// Copyright 2024 The StableHLO Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of the `stablehlo-ext-constant-folding` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Utils/ConstantFoldUtils.h"
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir-tensorrt-dialect/Utils/StaticValueUtils.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Patterns.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/GatherScatterUtils.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/conversions/linalg/transforms/MapStablehloToScalarOp.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

namespace mlir {
namespace stablehlo_ext {
#define GEN_PASS_DEF_CONSTANTFOLDINGPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace stablehlo_ext
} // namespace mlir

using namespace mlir;
using namespace mlir::stablehlo;
using namespace mlir::stablehlo_ext;

/// Replace `originalOp` with `v` if the types match. Otherwise, insert a
/// `tensor.cast` of `v` if the types are "cast compatible", meaning that one
/// type is a generalization of the other. Otherwise, return failure.
static LogicalResult maybeCastAndReplace(RewriterBase &rewriter,
                                         Operation *originalOp,
                                         ValueRange replacements) {
  TypeRange originalTypes = originalOp->getResultTypes();
  TypeRange newTypes(replacements);
  if (originalTypes == newTypes) {
    rewriter.replaceOp(originalOp, replacements);
    return success();
  }

  // We should not be handling element type differences here. If this occurs,
  // something has gone wrong. Types should have matching ranks and one shape
  // should be a generalization of the other.
  if (originalTypes.size() != newTypes.size() ||
      llvm::any_of(llvm::zip(originalTypes, newTypes),
                   [](const std::tuple<Type, Type> &types) {
                     RankedTensorType originalType =
                         cast<RankedTensorType>(std::get<0>(types));
                     RankedTensorType newType =
                         cast<RankedTensorType>(std::get<1>(types));
                     return mlir::getElementTypeOrSelf(originalType) !=
                                mlir::getElementTypeOrSelf(newType) ||
                            (!tensorrt::isTargetRefinementOfSource(
                                 originalType.getShape(), newType.getShape()) &&
                             !tensorrt::isTargetRefinementOfSource(
                                 newType.getShape(), originalType.getShape()));
                   }))
    return failure();

  SmallVector<Value> finalReplacements;
  finalReplacements.reserve(originalTypes.size());
  for (auto [v, originalType, originalValue] :
       llvm::zip_equal(replacements, originalTypes, originalOp->getResults())) {
    if (v.getType() == originalType ||
        canUpdateTypeWithoutCast(originalValue)) {
      finalReplacements.push_back(v);
      continue;
    }
    finalReplacements.push_back(
        rewriter.create<tensor::CastOp>(originalOp->getLoc(), originalType, v));
  }
  rewriter.replaceOp(originalOp, finalReplacements);
  return success();
}

/// Like `rewriter.replaceOpWithNewOp`, except that casts are inserted if it
/// makes sense to match the original result types (see `maybeCastAndReplace`
/// above).
template <typename T, typename... Args>
static LogicalResult replaceOpWithNewOpAndMaybeCast(RewriterBase &rewriter,
                                                    Operation *originalOp,
                                                    Args &&...args) {
  auto newOp =
      rewriter.create<T>(originalOp->getLoc(), std::forward<Args>(args)...);
  return maybeCastAndReplace(rewriter, originalOp, newOp->getResults());
}

/// Converts an APFloat value to another APFloat value whose semantics match the
/// MLIR type returns nullopt if the conversion would lose information.
static APFloat roundToMatchingMlirType(FloatType mlirType,
                                       APFloat valueToConvert) {
  if (APFloat::SemanticsToEnum(mlirType.getFloatSemantics()) ==
      APFloat::SemanticsToEnum(valueToConvert.getSemantics()))
    return valueToConvert;
  // We ignore `losesInfo` and code on result of `convert` since this is a
  // lossy cast.
  bool losesInfo = false;
  valueToConvert.convert(mlirType.getFloatSemantics(),
                         APFloat::rmNearestTiesToEven, &losesInfo);
  return valueToConvert;
}

/// Fold elementwise `calculate(lhs, rhs)` and return the result as an
/// ElementsAttr.
/// NOTE: this is a slightly optimized version of `mlir::constantFoldBinaryOp`
/// which also allows the result element type to be different from the operand
/// element type.
template <typename ElementValueT, typename ResultValueT = ElementValueT>
static ElementsAttr constFoldBinaryOpImpl(
    ElementsAttr lhs, ElementsAttr rhs, RankedTensorType resultType,
    function_ref<std::optional<ResultValueT>(ElementValueT, ElementValueT)>
        calculate) {
  assert(lhs.getType() == rhs.getType() && "expected equal operand types");

  // Fast path if both operands are splat.
  if (lhs.isSplat() && rhs.isSplat()) {
    auto lhsValue = lhs.getSplatValue<ElementValueT>();
    auto rhsValue = rhs.getSplatValue<ElementValueT>();
    if (auto result = calculate(lhsValue, rhsValue))
      return DenseElementsAttr::get(resultType, *result);
    return {};
  }

  auto maybeLhsIt = lhs.try_value_begin<ElementValueT>();
  auto maybeRhsIt = rhs.try_value_begin<ElementValueT>();
  if (!maybeLhsIt || !maybeRhsIt)
    return {};
  auto lhsIt = *maybeLhsIt;
  auto rhsIt = *maybeRhsIt;
  SmallVector<ResultValueT, 4> elementResults;
  elementResults.reserve(lhs.getNumElements());
  for (size_t i = 0, e = lhs.getNumElements(); i < e; ++i, ++lhsIt, ++rhsIt) {
    auto elementResult = calculate(*lhsIt, *rhsIt);
    if (!elementResult)
      return {};
    elementResults.emplace_back(std::move(*elementResult));
  }
  return DenseElementsAttr::get(resultType, elementResults);
}

template <typename OpType>
struct StablehloExtFoldOpPattern : public OpRewritePattern<OpType> {
  StablehloExtFoldOpPattern(int64_t sizeLimit, MLIRContext *ctx,
                            PatternBenefit benefit = 1)
      : OpRewritePattern<OpType>(ctx, benefit), sizeLimit(sizeLimit) {}

  bool exceedsSizeLimit(ElementsAttr attr) const {
    return !attr.isSplat() && attr.getNumElements() > sizeLimit;
  }

  template <typename... Attrs>
  bool exceedsSizeLimit(ElementsAttr attr, Attrs... other) const {
    return this->exceedsSizeLimit(attr) || this->exceedsSizeLimit(other...);
  }

protected:
  int64_t sizeLimit;
};

namespace {

/// This is the base class for simple binary operations whose folding operation
/// can be described using a simple functor object like `std::multiplies<>`.
template <typename OpType, typename ArithOpFunctor>
struct EvalBinaryOpPattern : public StablehloExtFoldOpPattern<OpType> {
  using StablehloExtFoldOpPattern<OpType>::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    static_assert(OpType::template hasTrait<OpTrait::NOperands<2>::Impl>(),
                  "expected binary op");

    ElementsAttr lhs, rhs;
    if (!matchPattern(op->getOperand(0), m_Constant(&lhs)) ||
        !matchPattern(op->getOperand(1), m_Constant(&rhs)) ||
        this->exceedsSizeLimit(lhs, rhs))
      return failure();

    if (lhs.getElementType().isIntOrIndex() &&
        rhs.getElementType().isIntOrIndex()) {
      auto result = constFoldBinaryOpImpl<APInt, APInt>(
          lhs, rhs, op.getLhs().getType(),
          [](const APInt &a, const APInt &b) -> std::optional<APInt> {
            return ArithOpFunctor{}(std::move(a), b);
          });
      if (!result)
        return failure();
      return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(rewriter, op,
                                                                   result);
    }
    if (lhs.getElementType().isFloat() && rhs.getElementType().isFloat()) {
      auto result = constFoldBinaryOpImpl<APFloat, APFloat>(
          lhs, rhs, op.getLhs().getType(),
          [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
            return ArithOpFunctor{}(std::move(a), b);
          });
      if (!result)
        return failure();
      return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(rewriter, op,
                                                                   result);
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

/// Perform folding of `stablehlo.transpose(stablehlo.constant)`.
struct ConstFoldTranspose
    : public StablehloExtFoldOpPattern<stablehlo::TransposeOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputType = op.getOperand().getType();
    RankedTensorType resultType = op.getType();
    if (!inputType || !resultType)
      return failure();

    // Fold the input to a constant if possible, otherwise return.
    ElementsAttr inputConst;
    if (!matchPattern(op.getOperand(), m_Constant(&inputConst)) ||
        this->exceedsSizeLimit(inputConst))
      return failure();

    // Handle the zero-dim case.
    if (inputType.getRank() == 0) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, resultType,
                                                         inputConst);
      return success();
    }

    auto permRange = op.getPermutation();
    AffineMap perm = AffineMap::getPermutationMap(
        SmallVector<unsigned>(permRange.begin(), permRange.end()),
        rewriter.getContext());
    ElementsAttr result = mlir::constantFoldTranspose(inputConst, perm);
    if (!result)
      return failure();
    return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(rewriter, op,
                                                                 result);
  }
};

// Combine consecutive transpose.
struct CombineConsecutiveTranspose
    : StablehloExtFoldOpPattern<stablehlo::TransposeOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;
  LogicalResult matchAndRewrite(stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto parentTranspose = op.getOperand().getDefiningOp<TransposeOp>();
    if (!parentTranspose || !parentTranspose->hasOneUse())
      return failure();
    // Combine two permutations
    ArrayRef<int64_t> parentPermutation = parentTranspose.getPermutation();
    ArrayRef<int64_t> opPermutation = op.getPermutation();
    SmallVector<int64_t> newPermutation(parentPermutation.size());
    for (unsigned i = 0; i < newPermutation.size(); i++)
      newPermutation[i] = parentPermutation[opPermutation[i]];
    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
        op, op.getType(), parentTranspose.getOperand(), newPermutation);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MinOp and MaxOp
//===----------------------------------------------------------------------===//

/// Simplify `MinOp` if the operands are identical.
template <typename OpType>
struct SimplifyTrivialMinOrTrivalMax : StablehloExtFoldOpPattern<OpType> {
  using StablehloExtFoldOpPattern<OpType>::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    static_assert(std::is_same_v<OpType, MinOp> ||
                      std::is_same_v<OpType, MaxOp>,
                  "expected stablehlo::MinOp or stablehlo::MaxOp");
    if (op.getLhs() != op.getRhs())
      return failure();
    return maybeCastAndReplace(rewriter, op, op.getLhs());
  }
};

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

/// Fold `stablehlo.reshape` with the constant producer.
/// TODO: This pattern differs from the upstream in that it handles
/// DenseResourceElementsAttr. Move this upstream.
struct ConstFoldReshape
    : public StablehloExtFoldOpPattern<stablehlo::ReshapeOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;
  LogicalResult matchAndRewrite(stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    ElementsAttr elAttr{};
    if (!matchPattern(op.getOperand(), m_Constant(&elAttr)))
      return failure();
    ElementsAttr attr = mlir::constantFoldReshape(op.getType(), elAttr);
    if (!attr)
      return failure();
    return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(rewriter, op,
                                                                 attr);
  }
};

//===----------------------------------------------------------------------===//
// ReshapeOp -> BroadcastInDimOp -> ReshapeOp
//===----------------------------------------------------------------------===//

/// Replace the sequence of stablehlo.reshape, stablehlo.broadcast_in_dim, and
/// stablehlo.reshape with stablehlo.broadcast_in_dim.
struct SimplifyReshapeBroadcastInDimReshape
    : public StablehloExtFoldOpPattern<stablehlo::ReshapeOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;
  LogicalResult matchAndRewrite(stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto outputType = op.getType();
    if (!succeeded(tensorrt::isUnitDimRankReducing(op.getOperand().getType(),
                                                   outputType)))
      return failure();

    auto broadcastInDimOp =
        op.getOperand().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!broadcastInDimOp)
      return failure();

    auto reshapeOp =
        broadcastInDimOp.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshapeOp)
      return failure();
    auto inputType = reshapeOp.getOperand().getType();
    if (!succeeded(
            tensorrt::isUnitDimRankExpanding(inputType, reshapeOp.getType())))
      return failure();

    int64_t inputRank = inputType.getRank();
    int64_t outputRank = outputType.getRank();
    if (inputRank < outputRank)
      return failure();

    // Drop input operand's front to match for outputRank if and only if
    // dimSize=1
    int64_t gap = inputRank - outputRank;
    for (int64_t i = 0; i < gap; i++)
      if (inputType.getDimSize(i) != 1)
        return failure();

    auto getTargetReshapeOp = [&](int64_t gap) {
      if (gap == 0)
        return reshapeOp;

      // When gap > 0
      auto reshapeOp1 = rewriter.create<stablehlo::ReshapeOp>(
          reshapeOp.getLoc(),
          inputType.clone(inputType.getShape().drop_front(gap)),
          reshapeOp.getOperand());
      auto reshapeOp2 = rewriter.create<stablehlo::ReshapeOp>(
          reshapeOp.getLoc(), reshapeOp->getResultTypes(),
          reshapeOp1->getResults());
      rewriter.replaceOp(reshapeOp, reshapeOp2->getResults());
      return reshapeOp2;
    };

    auto isIncreasing = [](ArrayRef<int64_t> seq) {
      for (size_t i = 1; i < seq.size(); ++i)
        if (seq[i] <= seq[i - 1])
          return false;
      return true;
    };

    stablehlo::ReshapeOp targetReshapeOp = getTargetReshapeOp(gap);
    auto targetInputType = targetReshapeOp.getOperand().getType();

    if (!isIncreasing(broadcastInDimOp.getBroadcastDimensions()) ||
        !succeeded(tensorrt::checkLhsShapeBroadcastableToRhs(
            targetInputType.getShape(), outputType.getShape())))
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, outputType, targetReshapeOp.getOperand(),
        llvm::to_vector(llvm::seq<int64_t>(0, targetInputType.getRank())));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

/// Fold `stablehlo.convert` with the constant input.
struct ConstFoldConvert
    : public StablehloExtFoldOpPattern<stablehlo::ConvertOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;
  LogicalResult matchAndRewrite(stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    ElementsAttr operandValue{};
    if (!matchPattern(op.getOperand(), m_Constant(&operandValue)) ||
        this->exceedsSizeLimit(operandValue))
      return failure();

    auto attr =
        mlir::constantFoldConvert(op.getType().getElementType(), operandValue);
    if (!attr)
      return failure();
    return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(rewriter, op,
                                                                 attr);
  }
};

/// Rewrites two consecutive convert operations with a single, whenever
/// possible. For replacement to happen, first conversion should happen to
/// higher bit width data type.
struct EliminateCascadedConverts
    : public StablehloExtFoldOpPattern<stablehlo::ConvertOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;
  LogicalResult matchAndRewrite(stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto parentConvert = op.getOperand().getDefiningOp<ConvertOp>();
    if (!parentConvert)
      return failure();
    auto firstType = parentConvert.getOperand().getType().getElementType();
    auto secondType = op.getOperand().getType().getElementType();
    auto thirdType = op.getType().getElementType();
    // All types should be same and bit width should increase from the
    // `firstType` -> `secondType`.
    if (isa<FloatType>(firstType) && isa<FloatType>(secondType) &&
        isa<FloatType>(thirdType) &&
        secondType.getIntOrFloatBitWidth() >
            firstType.getIntOrFloatBitWidth()) {
      rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getType(),
                                             parentConvert.getOperand());
      return success();
    }
    if (isa<IntegerType>(firstType) && isa<IntegerType>(secondType) &&
        isa<IntegerType>(thirdType) &&
        secondType.getIntOrFloatBitWidth() >
            firstType.getIntOrFloatBitWidth()) {
      rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getType(),
                                             parentConvert.getOperand());
      return success();
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

/// Folds `stablehlo::SqrtOp` for float operands.
struct SqrtOpFolder : public StablehloExtFoldOpPattern<stablehlo::SqrtOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(stablehlo::SqrtOp op,
                                PatternRewriter &rewriter) const override {
    // This op can accept Float and Complex types. We only handle float here.
    FloatType elementType = dyn_cast<FloatType>(op.getType().getElementType());
    if (!elementType)
      return failure();

    // Check for constant operand.
    ElementsAttr inpAttr{};
    if (!matchPattern(op.getOperand(), m_Constant(&inpAttr)) ||
        this->exceedsSizeLimit(inpAttr))
      return failure();

    Attribute foldedResult =
        constFoldUnaryOpConditional<FloatAttr, FloatAttr::ValueType, void>(
            {inpAttr}, [&](const APFloat &v) -> std::optional<APFloat> {
              if (v.isNegative() || v.isNaN())
                return std::nullopt;
              if (elementType.isF64())
                return APFloat(sqrt(v.convertToDouble()));
              return roundToMatchingMlirType(
                  elementType, APFloat(sqrtf(v.convertToFloat())));
            });
    if (!foldedResult)
      return failure();
    return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(rewriter, op,
                                                                 foldedResult);
  }
};

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

/// Fold `stablehlo.rsqrt` with the constant producer.
struct RsqrtFolder : public StablehloExtFoldOpPattern<stablehlo::RsqrtOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;
  LogicalResult matchAndRewrite(stablehlo::RsqrtOp op,
                                PatternRewriter &rewriter) const override {
    // This op can accept Float and Complex types. We only handle float here.
    // Also, don't fold if we're over the size limit.
    FloatType elementType = dyn_cast<FloatType>(op.getType().getElementType());
    if (!elementType)
      return failure();

    // Check for constant operand.
    DenseElementsAttr attr{};
    if (!matchPattern(op.getOperand(), m_Constant(&attr)) ||
        this->exceedsSizeLimit(attr))
      return failure();

    Attribute foldedResult =
        constFoldUnaryOpConditional<FloatAttr, FloatAttr::ValueType, void>(
            attr, [&](const APFloat &v) -> std::optional<APFloat> {
              if (v.isNegative() || v.isNaN())
                return std::nullopt;
              if (elementType.isF64())
                return APFloat(1.0 / sqrt(v.convertToDouble()));
              return roundToMatchingMlirType(
                  elementType, APFloat(1.0f / sqrtf(v.convertToFloat())));
            });
    if (!foldedResult)
      return failure();
    return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(rewriter, op,
                                                                 foldedResult);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

namespace {

struct ConstFoldCompare
    : public StablehloExtFoldOpPattern<stablehlo::CompareOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(stablehlo::CompareOp op,
                                PatternRewriter &rewriter) const override {

    ElementsAttr lhsAttr{}, rhsAttr{};
    if (!matchPattern(op.getLhs(), m_Constant(&lhsAttr)) ||
        !matchPattern(op.getRhs(), m_Constant(&rhsAttr)))
      return rewriter.notifyMatchFailure(op->getLoc(), "operands not constant");

    if (exceedsSizeLimit(lhsAttr, rhsAttr))
      return failure();

    ComparisonDirection direction = op.getComparisonDirection();
    std::optional<ComparisonType> kind = op.getCompareType();

    if (lhsAttr.getElementType().isIntOrIndex() &&
        rhsAttr.getElementType().isIntOrIndex()) {
      // We the comparison kind is SIGNED unless:
      // - explictly set to UNSIGNED.
      // - the lhs operand type is unsigned integer and no kind
      //   has been explicitly set.
      bool isUnsigned = kind == ComparisonType::UNSIGNED;
      if (!kind && lhsAttr.getElementType().isUnsignedInteger())
        isUnsigned = true;
      std::optional<arith::CmpIPredicate> predicate =
          stablehlo::impl::getCmpPredicate<arith::CmpIPredicate>(direction,
                                                                 !isUnsigned);
      if (!predicate)
        return failure();
      Attribute result = constFoldBinaryOpImpl<APInt, APInt>(
          lhsAttr, rhsAttr, op.getLhs().getType().clone(rewriter.getI1Type()),
          [&](const APInt &lhs, const APInt &rhs) {
            return APInt(1,
                         mlir::arith::applyCmpPredicate(*predicate, lhs, rhs));
          });
      if (!result)
        return failure();
      return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(rewriter, op,
                                                                   result);
    }

    if (lhsAttr.getElementType().isFloat() &&
        rhsAttr.getElementType().isFloat()) {
      std::optional<arith::CmpFPredicate> predicate =
          stablehlo::impl::getCmpPredicate<arith::CmpFPredicate>(
              direction, /*isSigned=*/true);
      if (!predicate)
        return failure();

      auto maybeLhsIt = lhsAttr.try_value_begin<APFloat>();
      auto maybeRhsIt = rhsAttr.try_value_begin<APFloat>();
      if (!maybeLhsIt || !maybeRhsIt)
        return failure();
      auto lhsIt = *maybeLhsIt;
      auto rhsIt = *maybeRhsIt;
      SmallVector<APInt, 4> elementResults;
      elementResults.reserve(lhsAttr.getNumElements());
      for (size_t i = 0, e = lhsAttr.getNumElements(); i < e;
           ++i, ++lhsIt, ++rhsIt) {
        elementResults.push_back(APInt(
            1, mlir::arith::applyCmpPredicate(*predicate, *lhsIt, *rhsIt)));
      }
      auto result = DenseElementsAttr::get(
          op.getLhs().getType().clone(rewriter.getI1Type()), elementResults);
      return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(rewriter, op,
                                                                   result);
    }

    return failure();
  }
};

/// Perform constant folding for 'stablehlo.div'. Note that this only handles
/// floating-point element types since the upstream folder handles integer
/// element types.
struct ConstFoldDiv : public StablehloExtFoldOpPattern<stablehlo::DivOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(stablehlo::DivOp op,
                                PatternRewriter &rewriter) const override {
    DenseFPElementsAttr lhsAttr{}, rhsAttr{};
    if (!matchPattern(op.getLhs(), m_Constant(&lhsAttr)) ||
        !matchPattern(op.getRhs(), m_Constant(&rhsAttr)) ||
        this->exceedsSizeLimit(lhsAttr, rhsAttr))
      return failure();

    DenseFPElementsAttr result = llvm::dyn_cast_if_present<DenseFPElementsAttr>(
        constFoldBinaryOp<FloatAttr>(
            {lhsAttr, rhsAttr},
            [](const APFloat &a, const APFloat &b) { return a / b; }));
    if (!result)
      return failure();
    return replaceOpWithNewOpAndMaybeCast<ConstantOp>(rewriter, op, result);
  }
};

/// Perform constant folding for 'stablehlo.floor' (round toward negative
/// infinity).
struct ConstFoldFloor : public StablehloExtFoldOpPattern<stablehlo::FloorOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(stablehlo::FloorOp op,
                                PatternRewriter &rewriter) const override {
    DenseFPElementsAttr operandAttr{};
    if (!matchPattern(op.getOperand(), m_Constant(&operandAttr)) ||
        this->exceedsSizeLimit(operandAttr))
      return failure();

    DenseFPElementsAttr result = llvm::dyn_cast_if_present<DenseFPElementsAttr>(
        constFoldUnaryOp<FloatAttr>({operandAttr}, [](const APFloat &a) {
          APFloat result(a);
          result.roundToIntegral(llvm::RoundingMode::TowardNegative);
          return result;
        }));
    if (!result)
      return failure();

    return replaceOpWithNewOpAndMaybeCast<ConstantOp>(rewriter, op, result);
  }
};

/// Perform constant folding for 'stablehlo.sub'. Note that this only handles
/// floating-point element types since the upstream folder handles integer
/// element types.
struct ConstFoldSub : public StablehloExtFoldOpPattern<stablehlo::SubtractOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(stablehlo::SubtractOp op,
                                PatternRewriter &rewriter) const override {
    DenseFPElementsAttr lhsAttr{}, rhsAttr{};
    if (!matchPattern(op.getLhs(), m_Constant(&lhsAttr)) ||
        !matchPattern(op.getRhs(), m_Constant(&rhsAttr)) ||
        this->exceedsSizeLimit(lhsAttr, rhsAttr))
      return failure();

    DenseFPElementsAttr result = llvm::dyn_cast_if_present<DenseFPElementsAttr>(
        constFoldBinaryOp<FloatAttr>(
            {lhsAttr, rhsAttr},
            [](const APFloat &a, const APFloat &b) { return a - b; }));
    if (!result)
      return failure();
    return replaceOpWithNewOpAndMaybeCast<ConstantOp>(rewriter, op, result);
  }
};

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

struct FoldOrOp : public StablehloExtFoldOpPattern<stablehlo::OrOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(stablehlo::OrOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType resultType = op.getType();
    if (!resultType.hasStaticShape() ||
        resultType.getNumElements() > this->sizeLimit)
      return rewriter.notifyMatchFailure(
          op->getLoc(), "result type must be static and number of "
                        "elements less than `kFoldOpEltLimit`");
    // Fold op if both operands are constants.
    ElementsAttr lhsAttr{};
    matchPattern(op.getLhs(), m_Constant(&lhsAttr));
    ElementsAttr rhsAttr{};
    matchPattern(op.getRhs(), m_Constant(&rhsAttr));

    if (lhsAttr && lhsAttr.isSplat() &&
        lhsAttr.getSplatValue<IntegerAttr>().getValue().isAllOnes()) {
      return maybeCastAndReplace(rewriter, op, op.getLhs());
    }
    if (lhsAttr && lhsAttr.isSplat() &&
        lhsAttr.getSplatValue<IntegerAttr>().getValue().isZero()) {
      return maybeCastAndReplace(rewriter, op, op.getRhs());
    }
    if (rhsAttr && rhsAttr.isSplat() &&
        rhsAttr.getSplatValue<IntegerAttr>().getValue().isAllOnes()) {
      return maybeCastAndReplace(rewriter, op, op.getRhs());
    }
    if (rhsAttr && rhsAttr.isSplat() &&
        rhsAttr.getSplatValue<IntegerAttr>().getValue().isZero()) {
      return maybeCastAndReplace(rewriter, op, op.getLhs());
    }
    // At this point, both lhs and rhs attributes are needed for the bitwise
    // compute.
    if (!rhsAttr || !lhsAttr)
      return failure();

    // We always allow splat constant, otherwise don't fold if we're over the
    // limit.
    if (exceedsSizeLimit(rhsAttr, lhsAttr))
      return failure();

    // TODO: Handle DenseResourceElementsAttr
    if (dyn_cast<DenseResourceElementsAttr>(lhsAttr) ||
        dyn_cast<DenseResourceElementsAttr>(rhsAttr))
      return failure();
    Attribute foldedResult =
        constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(
            {lhsAttr, rhsAttr}, std::bit_or<>{});
    DenseElementsAttr denseFoldedResult =
        dyn_cast_or_null<DenseElementsAttr>(foldedResult);
    if (!denseFoldedResult)
      return failure();
    return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(
        rewriter, op, denseFoldedResult);
  }
};

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

struct FoldAndOp : public StablehloExtFoldOpPattern<stablehlo::AndOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(stablehlo::AndOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType resultType = op.getType();
    if (!resultType.hasStaticShape() ||
        resultType.getNumElements() > this->sizeLimit)
      return rewriter.notifyMatchFailure(
          op->getLoc(), "result type must be static and number of "
                        "elements less than `kFoldOpEltLimit`");
    // Fold op if both operands are constants.
    ElementsAttr lhsAttr{};
    matchPattern(op.getLhs(), m_Constant(&lhsAttr));
    ElementsAttr rhsAttr{};
    matchPattern(op.getRhs(), m_Constant(&rhsAttr));

    if (lhsAttr && lhsAttr.isSplat() &&
        lhsAttr.getSplatValue<IntegerAttr>().getValue().isAllOnes()) {
      return maybeCastAndReplace(rewriter, op, op.getRhs());
    }
    if (lhsAttr && lhsAttr.isSplat() &&
        lhsAttr.getSplatValue<IntegerAttr>().getValue().isZero()) {
      return maybeCastAndReplace(rewriter, op, op.getLhs());
    }
    if (rhsAttr && rhsAttr.isSplat() &&
        rhsAttr.getSplatValue<IntegerAttr>().getValue().isAllOnes()) {
      return maybeCastAndReplace(rewriter, op, op.getLhs());
    }
    if (rhsAttr && rhsAttr.isSplat() &&
        rhsAttr.getSplatValue<IntegerAttr>().getValue().isZero()) {
      return maybeCastAndReplace(rewriter, op, op.getRhs());
    }
    // At this point, both lhs and rhs attributes are needed for the bitwise
    // compute.
    if (!rhsAttr || !lhsAttr)
      return failure();

    // We always allow splat constant, otherwise don't fold if we're over the
    // limit.
    if (exceedsSizeLimit(lhsAttr, rhsAttr))
      return failure();

    // TODO: Handle DenseResourceElementsAttr
    if (dyn_cast<DenseResourceElementsAttr>(lhsAttr) ||
        dyn_cast<DenseResourceElementsAttr>(rhsAttr))
      return failure();
    Attribute foldedResult =
        constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(
            {lhsAttr, rhsAttr}, std::bit_and<>{});
    DenseElementsAttr denseFoldedResult =
        dyn_cast_or_null<DenseElementsAttr>(foldedResult);
    if (!denseFoldedResult)
      return failure();
    return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(
        rewriter, op, denseFoldedResult);
  }
};

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

/// Fold `stablehlo.slice(constant)`, inserting a cast if required.
struct ConstFoldStablehloSlice
    : public StablehloExtFoldOpPattern<stablehlo::SliceOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> start = op.getStartIndices();
    ArrayRef<int64_t> limit = op.getLimitIndices();
    ArrayRef<int64_t> stride = op.getStrides();

    DenseElementsAttr operandValue{};
    if (!matchPattern(op.getOperand(), m_Constant(&operandValue)))
      return failure();
    auto operandType = cast<RankedTensorType>(operandValue.getType());

    // Compute the expected result type using the shape inference methods.
    SmallVector<Type, 1> resultTypeVec;
    if (failed(hlo::inferSliceOp(std::nullopt, operandType, start, limit,
                                 stride, resultTypeVec)))
      return failure();
    assert(resultTypeVec.size() == 1 && "expected one result type");

    auto newResultType = cast<RankedTensorType>(resultTypeVec.front());
    if (!operandValue.isSplat() && !op.getOperand().hasOneUse() &&
        this->exceedsSizeLimit(operandValue))
      return rewriter.notifyMatchFailure(
          op, "result type num elements > fold limit");

    ElementsAttr resultValue = mlir::constantFoldSliceOffsetLimitStride(
        operandValue, newResultType, start, limit, stride);
    if (!resultValue)
      return rewriter.notifyMatchFailure(op, "could not compute slice");
    assert(resultValue.getShapedType().hasStaticShape());

    return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(rewriter, op,
                                                                 resultValue);
  }
};

/// Simplify trivial slices that represent an identity.
struct SimplifyTrivialSlice
    : public StablehloExtFoldOpPattern<stablehlo::SliceOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {

    ArrayRef<int64_t> start = op.getStartIndices();
    ArrayRef<int64_t> limit = op.getLimitIndices();
    ArrayRef<int64_t> stride = op.getStrides();
    RankedTensorType operandType = op.getOperand().getType();
    if (!operandType.hasStaticShape() || operandType.getNumElements() == 0)
      return rewriter.notifyMatchFailure(op,
                                         "operand has dynamic or empty shape");

    // Empty tensor is handled by a different pattern.
    if (op.getType().hasStaticShape() && op.getType().getNumElements() == 0)
      return rewriter.notifyMatchFailure(op, "result is empty tensor");

    for (auto [startIdx, stopIdx, strideIdx, dimSize] :
         llvm::zip_equal(start, limit, stride, operandType.getShape())) {
      // Empty tensors replacement is handled by a different pattern.
      if (startIdx == stopIdx)
        return failure();
      if (startIdx != 0 || strideIdx != 1 || stopIdx != dimSize)
        return rewriter.notifyMatchFailure(op, "not a trivial slice");
    }

    return maybeCastAndReplace(rewriter, op, op.getOperand());
  }
};

//===----------------------------------------------------------------------===//
// ConcatenateOp
// Note that the constant folding routine is upstream in StableHlo
// simplification patterns.
//===----------------------------------------------------------------------===//

/// If there is only one operand, just replace with itself.
/// TODO: This only differs from the equivalent upstream pattern in that it
/// inserts a cast if the types differ.
struct ConcatSingleSegment
    : public StablehloExtFoldOpPattern<stablehlo::ConcatenateOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1)
      return failure();
    return maybeCastAndReplace(rewriter, op, op.getInputs().front());
  }
};

//===----------------------------------------------------------------------===//
// ConstFoldGatherOnSplat
//===----------------------------------------------------------------------===//

/// Repalce `stablehlo.gather` with `stablehlo.constant` when the data operand
/// is a splat constant and the result type is statically shaped.
struct ConstFoldGatherOnSplat
    : public StablehloExtFoldOpPattern<stablehlo::GatherOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;
  LogicalResult matchAndRewrite(stablehlo::GatherOp op,
                                PatternRewriter &rewriter) const override {
    SplatElementsAttr splatConst{};
    RankedTensorType resultType = op.getType();
    if (!resultType.hasStaticShape() ||
        !matchPattern(op.getOperand(), m_Constant(&splatConst)))
      return failure();
    return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(
        rewriter, op,
        DenseElementsAttr::get(resultType,
                               splatConst.getSplatValue<Attribute>()));
  }
};

//===----------------------------------------------------------------------===//
// LogicalRightShiftOp
//===----------------------------------------------------------------------===//

/// Fold trivial `stablehlo.logical_shift_right` when the shift has a greater
/// width than the element type.
struct RewriteTrivialLogicalRightShiftPattern
    : public StablehloExtFoldOpPattern<stablehlo::ShiftRightLogicalOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;
  LogicalResult matchAndRewrite(stablehlo::ShiftRightLogicalOp op,
                                PatternRewriter &rewriter) const override {
    TensorType resultType = op.getType();

    // Make sure we rule out index type, since 'getElementTypeBitWidth' will
    // fail in that case.
    if (resultType.isIndex() || !resultType.hasStaticShape())
      return failure();

    int64_t bitWidth = resultType.getElementTypeBitWidth();
    ElementsAttr attr;

    // Try to match a constant shift amount.
    if (!matchPattern(op.getRhs(), m_Constant(&attr)) || !attr.isSplat())
      return failure();

    int64_t shiftAmount = attr.getSplatValue<APInt>().getSExtValue();
    if (shiftAmount < bitWidth)
      return failure();
    return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(
        rewriter, op, rewriter.getZeroAttr(resultType));
  }
};

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

/// This pattern checks if a scatter is a  "canonical scatter-nd"-like operation
/// that completely overwrites the source tensor. In this case, all the
/// input/source tensors can be replaced by the updates tensors.
///
/// Note that this pattern only detects when there is a single slice
/// being inserted at the zero index, and the slice completely overwrites the
/// source tensor. It does not yet detect when there are multiple slices that
/// insert into 'iota' indices, which would also be valid for replacement
/// by the updates tensors.
struct SimplifyTrivialScatter
    : public StablehloExtFoldOpPattern<stablehlo::ScatterOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;
  LogicalResult matchAndRewrite(stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    // Check whether the indices are uniformly zero.
    if (SplatElementsAttr attr;
        !matchPattern(op.getScatterIndices(), m_Constant(&attr)) ||
        !attr.getSplatValue<APInt>().isZero())
      return rewriter.notifyMatchFailure(op, "indices not splat zero");

    if (!stablehlo_ext::isCanonicalScatterNd(op))
      return rewriter.notifyMatchFailure(op, "not a canonical scatter nd op");

    // Check whether there is a single slice update. This occurs if the shape of
    // the inserted slice is the same as the input and the number of slices
    // inserted is one.
    stablehlo::ScatterDimensionNumbersAttr dimNums =
        op.getScatterDimensionNumbersAttr();
    RankedTensorType inputType =
        cast<RankedTensorType>(op.getInputs().front().getType());
    RankedTensorType updateType =
        cast<RankedTensorType>(op.getUpdates().front().getType());

    // Build the implicit inserted slice shape.
    SmallVector<int64_t> implicitSliceShape(inputType.getRank(), 0);
    for (int64_t i : dimNums.getInsertedWindowDims())
      implicitSliceShape[i] = 1;

    unsigned updateWindowDim = 0;
    ArrayRef<int64_t> updateWindowDims = dimNums.getUpdateWindowDims();
    for (int64_t i = 0;
         i < inputType.getRank() && updateWindowDim < updateWindowDims.size();
         ++i) {
      if (implicitSliceShape[i] == 0)
        implicitSliceShape[i] =
            updateType.getDimSize(updateWindowDims[updateWindowDim++]);
    }

    if (implicitSliceShape != inputType.getShape())
      return rewriter.notifyMatchFailure(
          op, "implicit update slice not correct shape");

    if (!llvm::all_of(updateType.getShape().drop_back(
                          dimNums.getUpdateWindowDims().size()),
                      [](int64_t x) { return x == 1; }))
      return rewriter.notifyMatchFailure(op, "more than one update slice");

    // Fast path for when no reshape is necessary.
    if (updateType == inputType) {
      rewriter.replaceOp(op, op.getUpdates());
      return success();
    }

    if (!updateType.hasStaticShape() || !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "update|input have dynamic shape");

    SmallVector<Value> replacements(op.getUpdates());
    for (Value &update : replacements)
      update = rewriter.create<stablehlo::ReshapeOp>(update.getLoc(), inputType,
                                                     update);
    rewriter.replaceOp(op, replacements);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Misc Patterns
//===----------------------------------------------------------------------===//

/// This patterns inserts a tensor.cast between a returned Value if the Value's
/// tensor shape does not match the corresponding function result type. This can
/// happen if the upstream StableHlo aggressive simplification pass does not
/// properly insert casts where it is supposed to. This is a relatively common
/// mistake in the upstream pass which will result in a verification error at
/// the end of the pass if left unhandled. Therefore, we insert this pattern to
/// automatically insert required casts if possible until upstream figures out
/// how to avoid this mistake from occurring.
struct FixInvalidReturnWorkaround
    : public StablehloExtFoldOpPattern<func::ReturnOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    FunctionType funcType =
        op->getParentOfType<func::FuncOp>().getFunctionType();
    bool changed = false;
    SmallVector<Value> newOperands(op.getOperands());
    for (auto [idx, value] : llvm::enumerate(op.getOperands())) {
      if (funcType.getResult(idx) == value.getType())
        continue;
      // If it is not a tensor type, then something else has gone horribly
      // wrong, we can't fix it.
      auto tensorType = dyn_cast<RankedTensorType>(value.getType());
      auto desiredType = dyn_cast<RankedTensorType>(funcType.getResult(idx));
      if (!tensorType || !desiredType)
        continue;

      // Check for cast compatibility. If not cast compatible, then something
      // else has gone horribly wrong, we can't fix it.
      if (!tensor::CastOp::areCastCompatible(tensorType, desiredType))
        continue;

      auto castOp =
          rewriter.create<tensor::CastOp>(value.getLoc(), desiredType, value);
      newOperands[idx] = castOp;
      changed = true;
    }
    if (!changed)
      return failure();
    rewriter.modifyOpInPlace(
        op, [&]() { op.getOperandsMutable().assign(newOperands); });
    return success(changed);
  }
};

/// Fold `stablehlo.op(..., tensor.cast(x)... )` to `stablehlo.op(..., x, ...)`
/// if the cast is a generalizing cast (it is removing some static dims of the
/// type of  `x` and replacing them with dynamic dimensions).
struct AbsorbTensorCastProducer : public RewritePattern {
  AbsorbTensorCastProducer(MLIRContext *ctx, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag{}, benefit, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa_and_present<stablehlo::StablehloDialect>(op->getDialect()) ||
        // Composite op types cannot be refined in-place.
        isa<stablehlo::CompositeOp>(op))
      return failure();

    // For each operand, try to absorb the cast operation. For most StableHLO
    // ops, this is legal, but for some operations that have additional
    // constraints, the legality of this depends on which operand is being
    // refined.
    auto hasGeneralizingCast = [](OpOperand &operand) -> tensor::CastOp {
      if (!canUpdateTypeWithoutCast(operand))
        return nullptr;
      Value value = operand.get();
      // Not all stablehlo operands are tensors -- some can have types like
      // 'tuple' or special quantized types.
      auto rtt = dyn_cast<RankedTensorType>(value.getType());
      if (!rtt)
        return nullptr;
      auto castOp = value.getDefiningOp<tensor::CastOp>();
      if (!castOp)
        return nullptr;
      auto operandType =
          dyn_cast<RankedTensorType>(castOp.getOperand().getType());
      if (castOp && operandType &&
          tensorrt::isTargetRefinementOfSource(rtt.getShape(),
                                               operandType.getShape()))
        return castOp;
      return nullptr;
    };
    bool changed = false;
    SmallVector<Value> newInputs;
    for (OpOperand &v : op->getOpOperands()) {
      auto castOp = hasGeneralizingCast(v);
      changed |= castOp != nullptr;
      newInputs.push_back(castOp ? castOp.getOperand() : v.get());
    }
    if (!changed)
      return failure();
    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(newInputs); });
    return success();
  }
};

/// Pattern: broadcast_in_dim(splat, _) -> constant(splat)
/// TODO: This pattern is reproduced from upstream because we have no way of
/// selectively including it but excluding other patterns.
struct FoldBroadcastInDimSplatPattern final
    : StablehloExtFoldOpPattern<mlir::stablehlo::BroadcastInDimOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    TypedValue<RankedTensorType> operand = op.getOperand();

    if (SplatElementsAttr cstAttr;
        matchPattern(operand, m_Constant(&cstAttr))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, SplatElementsAttr::get(op.getType(),
                                     cstAttr.getSplatValue<Attribute>()));
      return success();
    }
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// These patterns are reproduced from upstream Stablehlo
// "aggressive folder" patterns. They can be removed once we fix upstream and
// can directly use those patterns.
//
//===----------------------------------------------------------------------===//

namespace {
/// Fold `stablehlo.iota` only if the result type has integer type and is very
/// small.
struct EvalIotaOpPattern : public StablehloExtFoldOpPattern<IotaOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;
  LogicalResult matchAndRewrite(IotaOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType resultType = op.getType();
    int64_t numElems = resultType.getNumElements();
    if (numElems > sizeLimit)
      return rewriter.notifyMatchFailure(op, "too many elements to fold");

    auto elementType = resultType.getElementType();

    if (!elementType.isInteger())
      return rewriter.notifyMatchFailure(op, "expected integer result type");

    auto outputSize = resultType.getNumElements();
    auto resultBitWidth = elementType.getIntOrFloatBitWidth();
    int64_t dimension = op.getIotaDimension();

    if (outputSize == 0) {
      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, DenseIntElementsAttr::get(resultType, ArrayRef<APInt>{}));
      return success();
    }

    llvm::SmallVector<APInt> values;
    values.reserve(outputSize);

    int64_t sequences = 1;
    int64_t sequenceMax = resultType.getDimSize(dimension);
    int64_t elementRepetitions = 1;
    for (int64_t i = 0; i < resultType.getRank(); i++) {
      sequences *= i < dimension ? resultType.getDimSize(i) : 1;
      elementRepetitions *= i > dimension ? resultType.getDimSize(i) : 1;
    }

    for (int64_t i = 0; i < sequences; ++i) {
      for (int64_t value = 0; value < sequenceMax; ++value) {
        for (int64_t k = 0; k < elementRepetitions; ++k) {
          values.push_back(APInt(resultBitWidth, value));
        }
      }
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseIntElementsAttr::get(resultType, values));
    return success();
  }
};

struct FoldConcatenateOpPattern final
    : StablehloExtFoldOpPattern<mlir::stablehlo::ConcatenateOp> {
  using StablehloExtFoldOpPattern::StablehloExtFoldOpPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType type = op.getType();
    if (!type.hasStaticShape() || type.getNumElements() > this->sizeLimit)
      return failure();

    // Fold concatenate when all inputs are constants.
    OperandRange inputs = op.getInputs();
    SmallVector<ElementsAttr> constants(inputs.size());
    for (auto [input, constant] : llvm::zip_equal(inputs, constants)) {
      if (!matchPattern(input, m_Constant(&constant)))
        return failure();
    }

    uint64_t dim = op.getDimension();
    ArrayRef<int64_t> shape = type.getShape();
    int64_t topSize = std::accumulate(shape.begin(), shape.begin() + dim,
                                      int64_t{1}, std::multiplies<>{});

    SmallVector<Attribute> newElems;
    newElems.reserve(type.getNumElements());

    for (int64_t i = 0; i != topSize; ++i) {
      for (ElementsAttr attr : constants) {
        size_t bottomSize = attr.getNumElements() / topSize;
        auto begin = attr.value_begin<Attribute>() + (i * bottomSize);
        newElems.append(begin, begin + bottomSize);
      }
    }

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), newElems));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public Functions and Pass Implementation
//===----------------------------------------------------------------------===//

void stablehlo_ext::populateStableHloAbsorbTensorCastPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AbsorbTensorCastProducer>(patterns.getContext());
}

void stablehlo_ext::populateTargetIndependentSimplificationPatterns(
    RewritePatternSet &patterns, int64_t sizeLimit, PatternBenefit benefit) {
  MLIRContext *ctx = patterns.getContext();
  // clang-format off
  patterns.insert<
      CombineConsecutiveTranspose,
      ConcatSingleSegment,
      ConstFoldCompare,
      ConstFoldConvert,
      ConstFoldDiv,
      ConstFoldFloor,
      ConstFoldGatherOnSplat,
      ConstFoldReshape,
      ConstFoldStablehloSlice,
      ConstFoldSub,
      ConstFoldTranspose,
      EliminateCascadedConverts,
      EvalBinaryOpPattern<stablehlo::AddOp, std::plus<>>,
      EvalBinaryOpPattern<stablehlo::MulOp, std::multiplies<>>,
      EvalBinaryOpPattern<stablehlo::SubtractOp, std::minus<>>,
      EvalIotaOpPattern,
      FixInvalidReturnWorkaround,
      FoldAndOp,
      FoldBroadcastInDimSplatPattern,
      FoldConcatenateOpPattern,
      FoldOrOp,
      RewriteTrivialLogicalRightShiftPattern,
      RsqrtFolder,
      SimplifyReshapeBroadcastInDimReshape,
      SimplifyTrivialMinOrTrivalMax<MaxOp>,
      SimplifyTrivialMinOrTrivalMax<MinOp>,
      SimplifyTrivialScatter,
      SimplifyTrivialSlice,
      SqrtOpFolder
    >(sizeLimit, ctx, benefit);
  // clang-format on
  populateStableHloAbsorbTensorCastPatterns(patterns);
  stablehlo::populateStablehloCanonicalizationPatterns(
      ctx, &patterns, benefit.getBenefit() - 1);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
}

namespace {
class ConstantFoldingPass
    : public stablehlo_ext::impl::ConstantFoldingPassBase<ConstantFoldingPass> {
public:
  using Base::Base;

  std::shared_ptr<FrozenRewritePatternSet> patterns;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patterns_(ctx);
    stablehlo_ext::populateTargetIndependentSimplificationPatterns(
        patterns_, constantFoldSizeLimit);
    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(patterns_));
    return success();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    GreedyRewriteConfig config{};
    config.useTopDownTraversal = true;
    if (failed(applyPatternsGreedily(op, *patterns, config))) {
      emitError(op->getLoc())
          << "failed to apply patterns in " << getArgument();
      ;
      return signalPassFailure();
    }
  }
};
} // namespace
