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
// Misc Patterns
//===----------------------------------------------------------------------===//

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

    // TODO: Support more iota folding, but doing so currently causes OOMs,
    // so this pattern needs to be enabled more carefully.
    if (outputSize != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected output size to be 1, but got: " +
                  std::to_string(outputSize));
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

void stablehlo_ext::populateTargetIndependentSimplificationPatterns(
    RewritePatternSet &patterns, int64_t sizeLimit,
    const stablehlo::StablehloAggressiveFolderPassOptions &folderOptions,
    PatternBenefit benefit) {
  MLIRContext *ctx = patterns.getContext();
  // clang-format off
  patterns.insert<
      ConstFoldCompare,
      ConstFoldConvert,
      ConstFoldDiv,
      ConstFoldFloor,
      ConstFoldGatherOnSplat,
      ConstFoldReshape,
      ConstFoldStablehloSlice,
      ConstFoldSub,
      ConstFoldTranspose,
      EvalBinaryOpPattern<stablehlo::AddOp, std::plus<>>,
      EvalBinaryOpPattern<stablehlo::MulOp, std::multiplies<>>,
      EvalBinaryOpPattern<stablehlo::SubtractOp, std::minus<>>,
      EvalIotaOpPattern,
      FoldAndOp,
      FoldBroadcastInDimSplatPattern,
      FoldConcatenateOpPattern,
      FoldOrOp,
      RewriteTrivialLogicalRightShiftPattern,
      RsqrtFolder,
      SqrtOpFolder
    >(sizeLimit, ctx, benefit);
  // clang-format on
  populateStableHloAbsorbTensorCastPatterns(patterns);
  stablehlo::populateStablehloCanonicalizationPatterns(
      ctx, &patterns, benefit.getBenefit() - 1);

  stablehlo::populateStablehloAggressiveFolderPatterns(ctx, &patterns,
                                                       folderOptions);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);

  stablehlo::StablehloAggressiveSimplificationPassOptions simplificationOptions{
      /*foldOpElementLimit=*/folderOptions.foldOpElementLimit,
  };
  stablehlo_ext::populateStableHloExtSimplificationsPatterns(
      patterns, simplificationOptions, benefit);
}

namespace {
class ConstantFoldingPass
    : public stablehlo_ext::impl::ConstantFoldingPassBase<ConstantFoldingPass> {
public:
  using Base::Base;

  std::shared_ptr<FrozenRewritePatternSet> patterns;

  std::shared_ptr<stablehlo::StablehloAggressiveFolderPassOptions>
      folderOptions;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patterns_(ctx);
    folderOptions =
        std::make_shared<stablehlo::StablehloAggressiveFolderPassOptions>();
    folderOptions->optimizeFloat = false;
    folderOptions->foldOpElementLimit = constantFoldSizeLimit;
    folderOptions->assumeNoUndeclaredSideEffects = false;
    stablehlo_ext::populateTargetIndependentSimplificationPatterns(
        patterns_, constantFoldSizeLimit, *folderOptions, PatternBenefit(1));

    patterns = std::make_shared<FrozenRewritePatternSet>(std::move(patterns_));
    return success();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    auto config = GreedyRewriteConfig().setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(op, *patterns, config))) {
      emitError(op->getLoc())
          << "failed to apply patterns in " << getArgument();
      ;
      return signalPassFailure();
    }
  }
};
} // namespace
