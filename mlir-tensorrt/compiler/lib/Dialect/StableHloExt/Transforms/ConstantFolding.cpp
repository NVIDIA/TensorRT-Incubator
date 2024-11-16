//===- ConstantFolding.cpp  -----------------------------------------------===//
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
/// Implementation of the `stablehlo-ext-constant-folding` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Utils/ConstantFoldUtils.h"
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir-tensorrt-dialect/Utils/StaticValueUtils.h"
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Patterns.h"
#include "mlir-tensorrt/Dialect/StableHloExt/Utils/Utils.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace stablehlo_ext {
#define GEN_PASS_DEF_CONSTANTFOLDINGPASS
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Passes.h.inc"
} // namespace stablehlo_ext
} // namespace mlir

using namespace mlir;
using namespace mlir::stablehlo;
using namespace mlir::stablehlo_ext;

/// Specifies the builtin limit on the number of elements in a result beyond
/// which we do not fold (to prevent long compilation time and excess memory
/// usage in the resulting program). This may not apply in certain cases such as
/// when folding operations on splat constants.
///
/// The current values  was chosen to match the corresponding limit in MLIR-MHLO
/// folders. It skews heavily toward making compilation time fast since a float
/// constant with 65536 elements would only take 256kB. However, our cases,
/// often times it is desireable to fold more aggressively (e.g. so that the
/// constant can be baked into a TRT engine). In the future we will have
/// different levels to reflect this tradeoff.
constexpr int64_t kFoldOpEltLimit = 65536;

template <typename AttrType>
static bool exceedsSizeLimit(AttrType attr) {
  return !attr.isSplat() && attr.getNumElements() > kFoldOpEltLimit;
}

template <typename AttrType, typename... AttrTypes>
static bool exceedsSizeLimit(AttrType attr, AttrTypes... other) {
  return exceedsSizeLimit(attr) || exceedsSizeLimit(other...);
}

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

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

namespace {
/// Perform folding of `stablehlo.transpose(stablehlo.constant)`.
struct ConstFoldTranspose : OpRewritePattern<stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputType =
        dyn_cast<RankedTensorType>(op.getOperand().getType());
    RankedTensorType resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!inputType || !resultType)
      return failure();

    // Fold the input to a constant if possible, otherwise return.
    ElementsAttr inputConst;
    if (!matchPattern(op.getOperand(), m_Constant(&inputConst)) ||
        exceedsSizeLimit(inputConst))
      return failure();

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
struct CombineConsecutiveTranspose : OpRewritePattern<stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
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
// BroadcastInDimOp
//===----------------------------------------------------------------------===//

/// Rewrite a `stablehlo.broadcast_in_dim` operation to a reshape when possible.
/// The upstream version of this is overly conservative.
struct BroadcastInDimOpCanon final
    : OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType type = op.getType();

    TypedValue<RankedTensorType> operand = op.getOperand();
    RankedTensorType operandTy = operand.getType();

    ArrayRef<int64_t> dims = op.getBroadcastDimensions();

    if (dims.empty() || !type.hasStaticShape() || !operandTy.hasStaticShape() ||
        type.getNumElements() != operandTy.getNumElements())
      return failure();

    // Check that dims are an increasing sequence. Check that
    // no broadcasting occurs along these dimensions.
    int64_t lastDimIdx = dims.front();
    for (int64_t dimIdx : dims.drop_front()) {
      if (dimIdx < lastDimIdx)
        return failure();
      lastDimIdx = dimIdx;
    }

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, type, operand);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MinOp and MaxOp
//===----------------------------------------------------------------------===//

/// Simplify `MinOp` if the operands are identical.
template <typename OpType>
struct SimplifyTrivialMinOrTrivalMax final : OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

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
struct ConstFoldReshape : public OpRewritePattern<stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto constOp = op.getOperand().getDefiningOp<stablehlo::ConstantOp>();
    if (!constOp)
      return failure();
    auto attr = mlir::constantFoldReshape(op.getType(), constOp.getValueAttr());
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
    : public OpRewritePattern<stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
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
struct ConstFoldConvert : public OpRewritePattern<stablehlo::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    ElementsAttr operandValue{};
    if (!matchPattern(op.getOperand(), m_Constant(&operandValue)) ||
        exceedsSizeLimit(operandValue))
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
    : public OpRewritePattern<stablehlo::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;
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
struct SqrtOpFolder : public OpRewritePattern<stablehlo::SqrtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SqrtOp op,
                                PatternRewriter &rewriter) const override {
    // This op can accept Float and Complex types. We only handle float here.
    FloatType elementType = dyn_cast<FloatType>(op.getType().getElementType());
    if (!elementType)
      return failure();

    // Check for constant operand.
    ElementsAttr inpAttr{};
    if (!matchPattern(op.getOperand(), m_Constant(&inpAttr)) ||
        exceedsSizeLimit(inpAttr))
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
struct RsqrtFolder : public OpRewritePattern<stablehlo::RsqrtOp> {
  using OpRewritePattern::OpRewritePattern;
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
        exceedsSizeLimit(attr))
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

template <typename Convert>
static ElementsAttr compareFolder(RankedTensorType resultType,
                                  ElementsAttr lhsAttr, ElementsAttr rhsAttr) {
  DenseElementsAttr lhs = dyn_cast<DenseElementsAttr>(lhsAttr);
  DenseElementsAttr rhs = dyn_cast<DenseElementsAttr>(rhsAttr);
  if (!lhs || !rhs)
    return nullptr;
  assert(lhs.getType() == rhs.getType() && "expected equal type lhs/rhs");

  if (!isa<FloatType>(lhs.getType().getElementType()))
    return nullptr;

  if (lhs.isSplat() && rhs.isSplat())
    return DenseElementsAttr::get(
        resultType,
        Convert()(lhs.getSplatValue<APFloat>(), rhs.getSplatValue<APFloat>()));

  SmallVector<bool> values;
  values.reserve(lhs.getNumElements());
  for (auto [lVal, rVal] :
       llvm::zip(lhs.getValues<APFloat>(), rhs.getValues<APFloat>()))
    values.push_back(Convert()(lVal, rVal));
  return DenseElementsAttr::get(cast<ShapedType>(resultType), values);
}

namespace {
struct ConstFoldCompare : public OpRewritePattern<stablehlo::CompareOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompareOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<FloatType>(op.getLhs().getType().getElementType()) ||
        !isa<FloatType>(op.getRhs().getType().getElementType()))
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "lhs and rhs should be float");
    if (!op.getType().hasStaticShape())

      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "result type must be static");

    ElementsAttr lhsAttr{};
    if (!matchPattern(op.getLhs(), m_Constant(&lhsAttr)))
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "lhs needs to be a constant");
    ElementsAttr rhsAttr{};
    if (!matchPattern(op.getRhs(), m_Constant(&rhsAttr)))
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "rhs needs to be a constant");

    ComparisonDirection direction = op.getComparisonDirection();

    if (exceedsSizeLimit(lhsAttr, rhsAttr))
      return failure();

// Upstream StableHLO `StablehloAggresiveSimplification` pass has folders for
// integer type.
#define COMPARE_FOLDER(comparison, Func)                                       \
  if (direction == comparison) {                                               \
    ElementsAttr resultFloat =                                                 \
        compareFolder<Func<APFloat>>(op.getType(), lhsAttr, rhsAttr);          \
    if (!resultFloat)                                                          \
      return failure();                                                        \
    return replaceOpWithNewOpAndMaybeCast<stablehlo::ConstantOp>(rewriter, op, \
                                                                 resultFloat); \
  }

    COMPARE_FOLDER(ComparisonDirection::EQ, std::equal_to);
    COMPARE_FOLDER(ComparisonDirection::NE, std::not_equal_to);
    COMPARE_FOLDER(ComparisonDirection::LT, std::less);
    COMPARE_FOLDER(ComparisonDirection::LE, std::less_equal);
    COMPARE_FOLDER(ComparisonDirection::GT, std::greater);
    COMPARE_FOLDER(ComparisonDirection::GE, std::greater_equal);
#undef COMPARE_FOLDER
    return success();
  }
};

/// Perform constant folding for 'stablehlo.div'. Note that this only handles
/// floating-point element types since the upstream folder handles integer
/// element types.
struct ConstFoldDiv : public OpRewritePattern<stablehlo::DivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DivOp op,
                                PatternRewriter &rewriter) const override {
    DenseFPElementsAttr lhsAttr{}, rhsAttr{};
    if (!matchPattern(op.getLhs(), m_Constant(&lhsAttr)) ||
        !matchPattern(op.getRhs(), m_Constant(&rhsAttr)) ||
        exceedsSizeLimit(lhsAttr, rhsAttr))
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
struct ConstFoldFloor : public OpRewritePattern<stablehlo::FloorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::FloorOp op,
                                PatternRewriter &rewriter) const override {
    DenseFPElementsAttr operandAttr{};
    if (!matchPattern(op.getOperand(), m_Constant(&operandAttr)) ||
        exceedsSizeLimit(operandAttr))
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
struct ConstFoldSub : public OpRewritePattern<stablehlo::SubtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SubtractOp op,
                                PatternRewriter &rewriter) const override {
    DenseFPElementsAttr lhsAttr{}, rhsAttr{};
    if (!matchPattern(op.getLhs(), m_Constant(&lhsAttr)) ||
        !matchPattern(op.getRhs(), m_Constant(&rhsAttr)) ||
        exceedsSizeLimit(lhsAttr, rhsAttr))
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
// IotaOp
//===----------------------------------------------------------------------===//

/// Rewrites Iota operation across multiple dimensions to a single dimension
/// Iota on 0th dimension, followed by BroadcastInDim.
struct CanonicalizeIotaToUnitRank : OpRewritePattern<IotaOp> {
  using OpRewritePattern<IotaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IotaOp iotaOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType resultType = iotaOp.getType();
    if (resultType.getRank() < 2)
      return rewriter.notifyMatchFailure(iotaOp->getLoc(),
                                         "rank needs to be >= 2");
    uint64_t iotaDimension = iotaOp.getIotaDimension();
    RankedTensorType newIotaType = RankedTensorType::get(
        {resultType.getDimSize(iotaDimension)}, resultType.getElementType());
    Value newIotaOp = rewriter.create<IotaOp>(iotaOp->getLoc(), newIotaType,
                                              /*iota_dimension=*/0);
    SmallVector<int64_t> broadcastDims{static_cast<int64_t>(iotaDimension)};
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(iotaOp, resultType, newIotaOp,
                                                  broadcastDims);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

struct FoldOrOp : public OpRewritePattern<stablehlo::OrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::OrOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType resultType = op.getType();
    if (!resultType.hasStaticShape() ||
        resultType.getNumElements() > kFoldOpEltLimit)
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

struct FoldAndOp : public OpRewritePattern<stablehlo::AndOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::AndOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType resultType = op.getType();
    if (!resultType.hasStaticShape() ||
        resultType.getNumElements() > kFoldOpEltLimit)
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
struct ConstFoldStablehloSlice : public OpRewritePattern<stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

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
        exceedsSizeLimit(operandValue))
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
struct SimplifyTrivialSlice : public OpRewritePattern<stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

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

/// Drop any segments of `stablehlo.concatenate`that are empty.
struct ConcatDropEmptySegments
    : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto isNotEmpty = [](Value value) {
      auto rtt = cast<RankedTensorType>(value.getType());
      return !rtt.hasStaticShape() || rtt.getNumElements() > 0;
    };
    if (llvm::all_of(op.getInputs(), isNotEmpty))
      return failure();
    auto newInputs =
        llvm::to_vector(llvm::make_filter_range(op.getInputs(), isNotEmpty));
    rewriter.modifyOpInPlace(
        op, [&]() { op.getInputsMutable().assign(newInputs); });
    return success();
  }
};

/// If there is only one operand, just replace with itself.
struct ConcatSingleSegment : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

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
struct ConstFoldGatherOnSplat : public OpRewritePattern<stablehlo::GatherOp> {
  using OpRewritePattern::OpRewritePattern;
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
    : public OpRewritePattern<stablehlo::ShiftRightLogicalOp> {
  using OpRewritePattern::OpRewritePattern;
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

/// This patterns inserts a tensor.cast between a returned Value if the Value's
/// tensor shape does not match the corresponding function result type. This can
/// happen if the upstream StableHlo aggressive simplification pass does not
/// properly insert casts where it is supposed to. This is a relatively common
/// mistake in the upstream pass which will result in a verification error at
/// the end of the pass if left unhandled. Therefore, we insert this pattern to
/// automatically insert required casts if possible until upstream figures out
/// how to avoid this mistake from occurring.
struct FixInvalidReturnWorkaround : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

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
      auto rtt = cast<RankedTensorType>(value.getType());
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
} // namespace

/// Populates patterns that are temporarily reproduced here from upstream
/// commits we have not yet integrated.
static void populateFutureUpstreamPatterns(RewritePatternSet &patterns);

void stablehlo_ext::populateStableHloAbsorbTensorCastPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AbsorbTensorCastProducer>(patterns.getContext());
}

namespace {
class ConstantFoldingPass
    : public stablehlo_ext::impl::ConstantFoldingPassBase<ConstantFoldingPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    // clang-format off
    patterns.insert<
        BroadcastInDimOpCanon,
        CanonicalizeIotaToUnitRank,
        CombineConsecutiveTranspose,
        ConcatDropEmptySegments,
        ConcatSingleSegment,
        ConstFoldCompare,
        ConstFoldConvert,
        ConstFoldDiv,
        ConstFoldFloor,
        ConstFoldGatherOnSplat,
        ConstFoldReshape,
        ConstFoldSub,
        ConstFoldStablehloSlice,
        ConstFoldTranspose,
        EliminateCascadedConverts,
        FixInvalidReturnWorkaround,
        FoldAndOp,
        FoldOrOp,
        RewriteTrivialLogicalRightShiftPattern,
        RsqrtFolder,
        SimplifyReshapeBroadcastInDimReshape,
        SimplifyTrivialMinOrTrivalMax<MaxOp>,
        SimplifyTrivialMinOrTrivalMax<MinOp>,
        SimplifyTrivialSlice,
        SqrtOpFolder
      >(ctx);
    // clang-format on
    populateFutureUpstreamPatterns(patterns);
    populateStableHloAbsorbTensorCastPatterns(patterns);
    stablehlo::populateStablehloCanonicalizationPatterns(ctx, &patterns);
    tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);

    GreedyRewriteConfig config{};
    config.useTopDownTraversal = true;
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
      emitError(op->getLoc())
          << "failed to apply patterns in " << getArgument();
      ;
      return signalPassFailure();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
/// The  patterns below this point are reproduced from
/// https://github.com/openxla/stablehlo/commit/5d15ab064f165cc6773ef4ba949ac083ae8e1fea,
/// which is in upstream, but our current pinned StableHlo commit is not there
/// yet. The patterns can be removed in the next StableHLO upgrade.
///
//===----------------------------------------------------------------------===//

///
/// In cases where a concat is fed into a slice, it
/// is possible the concat can be simplified or bypassed. This checks which
/// inputs to the concat are used by the slice, either reducing the number of
/// concatenated values or entirely removes the concat. Pattern:
/// slice(concat(X,Y,Z,...),...) -> concat(slice(X),slice(Y),slice(Z))
struct SimplifySliceOfConcat : public OpRewritePattern<SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp slice,
                                PatternRewriter &rewriter) const override {
    RankedTensorType resultTy = slice.getType();
    if (!resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(slice, "result shape not static");

    auto concat = slice.getOperand().getDefiningOp<ConcatenateOp>();
    if (!concat)
      return rewriter.notifyMatchFailure(slice, "slice input not concat");

    RankedTensorType concatType = concat.getType();
    uint64_t dimension = concat.getDimension();

    ArrayRef<int64_t> start = slice.getStartIndices();
    ArrayRef<int64_t> limit = slice.getLimitIndices();

    int64_t sliceStart = start[dimension];
    int64_t sliceLimit = limit[dimension];

    // We need to determine what inputs from the concat affect the slice, and
    // how the bounds of the slice need to be updated for the minimally required
    // inputs.
    int64_t runningSize = 0;
    int64_t frontOffset = concatType.getShape()[dimension];

    auto subsetStart = concat.operand_end();
    auto subsetEnd = concat.operand_end();
    for (auto it = concat.operand_begin(); it < concat.operand_end(); ++it) {
      Value input = *it;
      auto inputTy = cast<RankedTensorType>(input.getType());
      if (inputTy.isDynamicDim(dimension))
        return rewriter.notifyMatchFailure(
            slice, "concat input has dynamic dimension");

      int64_t dimSize = inputTy.getShape()[dimension];

      // If this position is in the slice its the start of the subset and we
      // need to update the start and limit values.
      if (runningSize + dimSize > sliceStart &&
          subsetStart == concat.operand_end()) {
        subsetStart = it;
        frontOffset = runningSize;
      }

      // Determine the last required offset.
      if (runningSize < sliceLimit) {
        subsetEnd = it + 1;
      }

      runningSize += dimSize;
    }

    auto subsetSize = subsetEnd - subsetStart;
    // We need all inputs so no optimization.
    if (subsetSize == concat.getNumOperands())
      return rewriter.notifyMatchFailure(slice,
                                         "slice needs all concat inputs");

    // If there's nothing to slice that means the output is an empty tensor and
    // there is dead code. We do nothing here and rely on other passes to clean
    // this up.
    if (subsetSize == 0)
      return rewriter.notifyMatchFailure(slice, "slice is empty");

    if (subsetSize > 1 && !concat.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(slice,
                                         "slice is not the only concat user");

    auto concatRange = OperandRange(subsetStart, subsetEnd);
    auto newConcat = rewriter.create<ConcatenateOp>(
        concat.getLoc(), concatRange, concat.getDimension());

    SmallVector<int64_t> newStart(start);
    SmallVector<int64_t> newLimit(limit);
    newStart[dimension] -= frontOffset;
    newLimit[dimension] -= frontOffset;

    rewriter.replaceOpWithNewOp<SliceOp>(
        slice, newConcat, rewriter.getDenseI64ArrayAttr(newStart),
        rewriter.getDenseI64ArrayAttr(newLimit), slice.getStrides());
    return success();
  }
};

/// Flatten sequential concatenations as long as the parent concatenation either
/// has a single use or is <= 32 elements.
class SimplifyConcatOfConcatPattern
    : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto getFlattenedOperands = [&](const Value &val) -> ValueRange {
      auto definingOp = dyn_cast_or_null<ConcatenateOp>(val.getDefiningOp());
      if (!definingOp || definingOp.getDimension() != op.getDimension())
        return val;
      if (definingOp->hasOneUse())
        return definingOp.getInputs();
      if (!definingOp.getType().hasStaticShape())
        return val;
      if (definingOp.getType().getNumElements() <= 32)
        return definingOp.getInputs();
      return val;
    };

    bool needToFlatten = false;
    int operandCount = 0;
    for (Value val : op.getInputs()) {
      ValueRange result = getFlattenedOperands(val);
      if (result.size() != 1 || result[0] != val)
        needToFlatten = true;
      operandCount += result.size();
    }
    if (!needToFlatten)
      return rewriter.notifyMatchFailure(op, "no need to flatten");

    llvm::SmallVector<Value, 6> newOperands;
    newOperands.reserve(operandCount);
    for (Value operand : op.getInputs())
      llvm::append_range(newOperands, getFlattenedOperands(operand));

    rewriter.modifyOpInPlace(op, [&] { op->setOperands(newOperands); });
    return success();
  }
};

void populateFutureUpstreamPatterns(RewritePatternSet &patterns) {
  patterns.add<SimplifySliceOfConcat, SimplifyConcatOfConcatPattern>(
      patterns.getContext());
}
