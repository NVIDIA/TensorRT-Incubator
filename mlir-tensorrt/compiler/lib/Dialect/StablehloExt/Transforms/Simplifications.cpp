//===- Simplifications.cpp ------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Patterns.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/GatherScatterUtils.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_STABLEHLOEXTSIMPLIFICATIONSPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

using namespace mlir;
using namespace mlir::stablehlo_ext;
using namespace mlir::stablehlo;

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

  if (originalTypes.size() != newTypes.size())
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
    if (!tensor::CastOp::areCastCompatible(v.getType(), originalType))
      return failure();
    finalReplacements.push_back(
        rewriter.create<tensor::CastOp>(originalOp->getLoc(), originalType, v));
  }
  rewriter.replaceOp(originalOp, finalReplacements);
  return success();
}

namespace {
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
// DotGeneralOp
//===----------------------------------------------------------------------===//

/// Rewrite `stablehlo.dot_general` to `stablehlo.multiply` when there are no
/// contracting dimensions. This is a pure pointwise multiply after
/// appropriately transposing operands to make batching dims leading and
/// broadcasting both operands to the dot_general result shape.
///
/// If the result element type has higher precision than the operand element
/// type, insert `stablehlo.convert` of both operands to the result element type
/// before broadcasting and multiplying.
struct DotGeneralNoContractionToMul
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  static Value createStaticI32DimTensor(PatternRewriter &rewriter, Location loc,
                                        int64_t value) {
    auto ty = RankedTensorType::get({1}, rewriter.getI32Type());
    auto attr = DenseIntElementsAttr::get(ty, {static_cast<int32_t>(value)});
    return rewriter.create<stablehlo::ConstantOp>(loc, ty, attr);
  }

  static Value createOutputDimensionsTensor(PatternRewriter &rewriter,
                                            Location loc,
                                            RankedTensorType resultType,
                                            int64_t numBatchDims, Value lhs,
                                            Value rhs) {
    int64_t resultRank = resultType.getRank();
    auto vecTy = RankedTensorType::get({resultRank}, rewriter.getI32Type());
    if (resultRank == 1 && !resultType.isDynamicDim(0))
      return createStaticI32DimTensor(rewriter, loc, resultType.getDimSize(0));

    auto lhsTy = cast<RankedTensorType>(lhs.getType());
    auto rhsTy = cast<RankedTensorType>(rhs.getType());
    int64_t lhsRank = lhsTy.getRank();
    int64_t rhsRank = rhsTy.getRank();
    (void)rhsRank;

    SmallVector<Value> dimVecPieces;
    dimVecPieces.reserve(resultRank);
    for (int64_t outDim = 0; outDim < resultRank; ++outDim) {
      Value dim0d;
      if (!resultType.isDynamicDim(outDim)) {
        dim0d = createStaticI32DimTensor(rewriter, loc,
                                         resultType.getDimSize(outDim));
        dimVecPieces.push_back(dim0d);
        continue;
      }

      // Determine which operand provides this dimension size.
      Value source = lhs;
      int64_t sourceDim = outDim;
      if (outDim >= lhsRank) {
        source = rhs;
        sourceDim = numBatchDims + (outDim - lhsRank);
      }

      // get_dimension_size returns tensor<i32> (0D). Reshape to tensor<1xi32>.
      RankedTensorType i32ScalarTensorType =
          RankedTensorType::get({}, rewriter.getI32Type());
      Value getDim = rewriter.create<stablehlo::GetDimensionSizeOp>(
          loc, i32ScalarTensorType, source, sourceDim);
      dim0d = rewriter.create<stablehlo::ReshapeOp>(
          loc, i32ScalarTensorType.clone({1}), getDim);
      dimVecPieces.push_back(dim0d);
    }

    if (resultRank == 1)
      return dimVecPieces.front();

    return rewriter.create<stablehlo::ConcatenateOp>(loc, vecTy, dimVecPieces,
                                                     /*dimension=*/0);
  }

  static Value createBroadcastToResult(PatternRewriter &rewriter, Location loc,
                                       RankedTensorType resultType,
                                       Value outputDimensions, Value operand,
                                       ArrayRef<int64_t> broadcastDims) {
    if (cast<RankedTensorType>(operand.getType()) == resultType)
      return operand;

    // `stablehlo.broadcast_in_dim` requires static result shape. Use
    // `stablehlo.dynamic_broadcast_in_dim` otherwise.
    if (resultType.hasStaticShape())
      return rewriter.create<stablehlo::BroadcastInDimOp>(
          loc, resultType, operand, broadcastDims);

    assert(outputDimensions &&
           "expected output_dimensions for dynamic broadcast");

    // In this dot_general(no contraction) -> broadcast+mul lowering, any
    // "expansion" happens only by inserting new result dimensions. All operand
    // dimensions are mapped (via broadcast_dimensions) to result dimensions and
    // preserve their extent. Therefore, all operand dimensions are known
    // non-expanding.
    RankedTensorType operandType = cast<RankedTensorType>(operand.getType());
    SmallVector<int64_t> nonExpanding =
        llvm::to_vector(llvm::seq<int64_t>(0, operandType.getRank()));

    auto bcastDimsAttr = rewriter.getDenseI64ArrayAttr(broadcastDims);
    DenseI64ArrayAttr expandingAttr;
    DenseI64ArrayAttr nonExpandingAttr =
        nonExpanding.empty() ? DenseI64ArrayAttr()
                             : rewriter.getDenseI64ArrayAttr(nonExpanding);

    return rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
        loc, resultType, operand, outputDimensions, bcastDimsAttr,
        expandingAttr, nonExpandingAttr);
  }

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    stablehlo::DotDimensionNumbersAttr dimNums = op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsContractDims = dimNums.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractDims = dimNums.getRhsContractingDimensions();
    if (!lhsContractDims.empty() || !rhsContractDims.empty())
      return failure();

    RankedTensorType lhsType = op.getLhs().getType();
    RankedTensorType rhsType = op.getRhs().getType();
    RankedTensorType resultType = op.getType();
    Type lhsETy = lhsType.getElementType();
    Type rhsETy = rhsType.getElementType();
    Type resultETy = resultType.getElementType();
    if (lhsETy != rhsETy)
      return rewriter.notifyMatchFailure(op,
                                         "operand element types do not match");

    bool needsPromotion = lhsETy != resultETy;
    if (needsPromotion) {
      // Only handle promotion to a "wider" element type.
      const DataLayout &dl = DataLayout::closest(op);
      if (dl.getTypeSize(resultETy) <= dl.getTypeSize(lhsETy))
        return rewriter.notifyMatchFailure(
            op, "result element type is not wider than operands");
    }

    ArrayRef<int64_t> lhsBatchDims = dimNums.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchDims = dimNums.getRhsBatchingDimensions();
    if (lhsBatchDims.size() != rhsBatchDims.size())
      return rewriter.notifyMatchFailure(op, "batch dims size mismatch");
    int64_t numBatchDims = lhsBatchDims.size();
    if (numBatchDims > lhsType.getRank() || numBatchDims > rhsType.getRank())
      return rewriter.notifyMatchFailure(op, "invalid batch dims");

    // With no contracting dims, the result rank is:
    //   rank(result) = rank(lhs) + rank(rhs) - numBatchDims
    if (resultType.getRank() != lhsType.getRank() + rhsType.getRank() -
                                    static_cast<int64_t>(numBatchDims))
      return rewriter.notifyMatchFailure(op, "unexpected result rank");

    // Build a permutation that makes batching dims contiguous and leading while
    // preserving the relative order of non-batch dims.
    auto getPerm = [](ArrayRef<int64_t> batchDims,
                      int64_t rank) -> std::optional<SmallVector<int64_t>> {
      llvm::SmallSetVector<int64_t, 8> perm(batchDims.begin(), batchDims.end());
      for (int64_t dim = 0; dim < rank; ++dim)
        if (!perm.contains(dim))
          perm.insert(dim);
      if (llvm::equal(perm, llvm::seq<int64_t>(0, rank)))
        return std::nullopt;
      return llvm::to_vector(llvm::iterator_range(perm.begin(), perm.end()));
    };

    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    if (auto lhsPerm = getPerm(lhsBatchDims, lhsType.getRank()))
      lhs = rewriter.create<stablehlo::TransposeOp>(loc, lhs, *lhsPerm);
    if (auto rhsPerm = getPerm(rhsBatchDims, rhsType.getRank()))
      rhs = rewriter.create<stablehlo::TransposeOp>(loc, rhs, *rhsPerm);

    // If needed, promote both operands to the result element type before
    // broadcasting and multiplying.
    if (needsPromotion) {
      auto lhsPromotedType =
          cast<RankedTensorType>(lhs.getType()).clone(resultETy);
      auto rhsPromotedType =
          cast<RankedTensorType>(rhs.getType()).clone(resultETy);
      lhs = rewriter.create<stablehlo::ConvertOp>(loc, lhsPromotedType, lhs);
      rhs = rewriter.create<stablehlo::ConvertOp>(loc, rhsPromotedType, rhs);
    }

    auto lhsRank = cast<RankedTensorType>(lhs.getType()).getRank();
    auto rhsRank = cast<RankedTensorType>(rhs.getType()).getRank();

    // Broadcast both operands to the dot_general result shape. After transpose,
    // the batch dims are leading:
    // - LHS dims map to output dims [0 .. lhsRank)
    // - RHS batch dims map to [0 .. numBatchDims) and RHS remaining dims map to
    //   output dims [lhsRank .. resultRank)
    SmallVector<int64_t> lhsBroadcastDims =
        llvm::to_vector(llvm::seq<int64_t>(0, lhsRank));
    SmallVector<int64_t> rhsBroadcastDims;
    rhsBroadcastDims.reserve(rhsRank);
    llvm::append_range(rhsBroadcastDims, llvm::seq<int64_t>(0, numBatchDims));
    for (int64_t i = 0, e = rhsRank - numBatchDims; i < e; ++i)
      rhsBroadcastDims.push_back(lhsRank + i);

    // Only compute dynamic output_dimensions if needed (dynamic result AND
    // at least one operand actually needs broadcasting).
    Value outputDimensions;
    if (!resultType.hasStaticShape() &&
        (cast<RankedTensorType>(lhs.getType()) != resultType ||
         cast<RankedTensorType>(rhs.getType()) != resultType)) {
      outputDimensions = createOutputDimensionsTensor(rewriter, loc, resultType,
                                                      numBatchDims, lhs, rhs);
    }

    lhs = createBroadcastToResult(rewriter, loc, resultType, outputDimensions,
                                  lhs, lhsBroadcastDims);
    rhs = createBroadcastToResult(rewriter, loc, resultType, outputDimensions,
                                  rhs, rhsBroadcastDims);

    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, resultType, lhs, rhs);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MinOp and MaxOp
//===----------------------------------------------------------------------===//

/// Simplify `MinOp` if the operands are identical.
template <typename OpType>
struct SimplifyTrivialMinOrTrivalMax : OpRewritePattern<OpType> {
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
// ConcatenateOp
// Note that the constant folding routine is upstream in StableHlo
// simplification patterns.
//===----------------------------------------------------------------------===//

/// If there is only one operand, just replace with itself.
/// TODO: This only differs from the equivalent upstream pattern in that it
/// inserts a cast if the types differ.
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
// SliceOp
//===----------------------------------------------------------------------===//

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
struct SimplifyTrivialScatter : public OpRewritePattern<stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;
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
} // namespace

void stablehlo_ext::populateStablehloDotGeneralToMultiplyPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DotGeneralNoContractionToMul>(patterns.getContext());
}

void stablehlo_ext::populateStableHloExtSimplificationsPatterns(
    RewritePatternSet &patterns,
    const stablehlo::StablehloAggressiveSimplificationPassOptions &options,
    PatternBenefit benefit) {
  // clang-format off
  patterns.add<
    CombineConsecutiveTranspose,
    ConcatSingleSegment,
    EliminateCascadedConverts,
    SimplifyReshapeBroadcastInDimReshape,
    SimplifyTrivialMinOrTrivalMax<MaxOp>,
    SimplifyTrivialMinOrTrivalMax<MinOp>,
    SimplifyTrivialScatter,
    SimplifyTrivialSlice
  >(patterns.getContext(), benefit);
  // clang-format on

  mlir::stablehlo_ext::populateStableHloAbsorbTensorCastPatterns(patterns);
  mlir::stablehlo::populateStablehloCanonicalizationPatterns(
      patterns.getContext(), &patterns, options, benefit);
  stablehlo_ext::populateStablehloDotGeneralToMultiplyPatterns(patterns);
}

namespace {

class StablehloExtSimplificationsPass
    : public stablehlo_ext::impl::StablehloExtSimplificationsPassBase<
          StablehloExtSimplificationsPass> {
public:
  using Base::Base;

  std::shared_ptr<stablehlo::StablehloAggressiveSimplificationPassOptions>
      options;

  LogicalResult initialize(MLIRContext *ctx) override {
    options = std::make_shared<
        stablehlo::StablehloAggressiveSimplificationPassOptions>();
    options->foldOpElementLimit = foldOpElementLimit;
    return success();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    stablehlo_ext::populateStableHloExtSimplificationsPatterns(patterns,
                                                               *options);
    auto config = GreedyRewriteConfig();
    if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
      emitError(op->getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
