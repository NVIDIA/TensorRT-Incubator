//===- ReductionConversions.cpp -------------------------------------------===//
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
///
/// Implementation of pass to convert StableHLO reduction and contraction ops to
/// TensorRT dialect ops.
///
//===----------------------------------------------------------------------===//
#include "Matchers.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir-tensorrt/Conversion/Patterns.h"
#include "mlir-tensorrt/Conversion/StablehloToTensorRT/StablehloToTensorRT.h"
#include "mlir-tensorrt/Conversion/TensorRTCommon/ConvertToTensorRTCommon.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "stablehlo-to-tensorrt"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;
using mlir::tensorrt::TensorValue;

/// Drop the unit dimension at `dimToDrop` from each of `values`.
static SmallVector<Value>
createRankReducedResults(TensorRTConversionPatternRewriter &rewriter,
                         Location loc, ResultRange values, int64_t dimToDrop,
                         int64_t trtMajorVersion) {
  assert(!values.empty());
  SmallVector<Value> result;
  result.reserve(values.size());
  for (Value v : values) {
    auto inputType = dyn_cast<RankedTensorType>(v.getType());
    assert((!inputType || inputType.getDimSize(dimToDrop) == 1) &&
           "expected value to have unit dim to drop");
    auto rtt = RankedTensorType::Builder(inputType);
    rtt.dropDim(dimToDrop);
    Value collapsed =
        rewriter.checkAndCreate<tensorrt::CollapseRankOp>(loc, Type(rtt), v);
    result.push_back(collapsed);
  }
  return result;
}

template <typename TensorRTOpType, stablehlo::ComparisonDirection dir>
static LogicalResult matchAndReplaceStablehloArgMinMax(
    stablehlo::ReduceOp op, TensorRTConversionPatternRewriter &rewriter,
    Value operand, ArrayRef<int64_t> reductionDims, int64_t trtMajorVersion) {
  if (!matchPattern(op,
                    matchers::detail::StablehloArgMinMaxReduceMatcher<dir>()))
    return failure();
  auto argMinOrMaxOp = rewriter.checkAndCreate<TensorRTOpType>(
      op.getLoc(),
      /*input=*/operand, /*axis=*/reductionDims.front());
  // Rank reduce the results.
  if (!argMinOrMaxOp)
    return failure();
  SmallVector<Value> replacements = createRankReducedResults(
      rewriter, op.getLoc(), argMinOrMaxOp.getResults(), reductionDims.front(),
      trtMajorVersion);
  rewriter.replaceOp(op, replacements);
  return success();
}

/// Given a stablehlo reduction operation, convert to a `tensorrt.reduce`
/// operation if it is a simple reduction (e.g. sum, mul, max/min) that be
/// converted 1-1. Caller must do the replacement, this just creates the new
/// operation and returns the new value.
static FailureOr<Value>
convertSimpleReductions(TensorRTConversionPatternRewriter &rewriter,
                        stablehlo::ReduceOp op, ArrayRef<int64_t> reductionDim,
                        Value input, Value init, int64_t trtMajorVersion) {
  // TODO: verify the init is the neutral value based on the op below.
  if (!matchPattern(init, m_Constant()))
    return failure();

  Block *reduceBody = &op.getBody().front();
  auto termOp = cast<stablehlo::ReturnOp>(reduceBody->getTerminator());
  if (termOp->getNumOperands() != 1 || reduceBody->getNumArguments() != 2)
    return failure();

  Location loc = op.getLoc();
  Value retValue = termOp.getOperands()[0];
  auto bbLhs = matchers::m_Val(reduceBody->getArgument(0));
  auto bbRhs = matchers::m_Val(reduceBody->getArgument(1));

  tensorrt::ReduceOperation reductionOp;
  if (matchPattern(retValue, m_Op<stablehlo::AddOp>(bbLhs, bbRhs)))
    reductionOp = tensorrt::ReduceOperation::kSUM;
  else if (matchPattern(retValue, m_Op<stablehlo::MulOp>(bbLhs, bbRhs)))
    reductionOp = tensorrt::ReduceOperation::kPROD;
  else if (matchPattern(retValue, m_Op<stablehlo::MinOp>(bbLhs, bbRhs)))
    reductionOp = tensorrt::ReduceOperation::kMIN;
  else if (matchPattern(retValue, m_Op<stablehlo::MaxOp>(bbLhs, bbRhs)))
    reductionOp = tensorrt::ReduceOperation::kMAX;
  else
    return failure();

  auto reduceOp = rewriter.checkAndCreate<tensorrt::ReduceOp>(
      loc, op.getType(0), input,
      /*reduceDims=*/
      reductionDim,
      /*keepdims=*/false, reductionOp);
  if (!reduceOp)
    return failure();
  return reduceOp.getResult();
}

static FailureOr<Value> convertBooleanReductions(RewriterBase &rewriter,
                                                 stablehlo::ReduceOp op,
                                                 ArrayRef<int64_t> reductionDim,
                                                 Value input, Value init) {
  Location loc = op.getLoc();
  // Create an int32 tensor types equivalent to the boolean tensor types.
  auto originalInputType = cast<RankedTensorType>(input.getType());
  auto originalResultType = cast<RankedTensorType>(op->getResultTypes()[0]);
  if (!originalResultType.getElementType().isInteger(1) ||
      !originalInputType.getElementType().isInteger(1))
    return failure();

  RankedTensorType integerInputType =
      RankedTensorType::Builder(originalInputType)
          .setElementType(rewriter.getI32Type());
  RankedTensorType integerResultType =
      RankedTensorType::Builder(originalResultType)
          .setElementType(rewriter.getI32Type());

  // Create the new reduction type.
  Block *reduceBody = &op.getBody().front();
  auto termOp = cast<stablehlo::ReturnOp>(reduceBody->getTerminator());
  if (termOp->getNumOperands() != 1 || reduceBody->getNumArguments() != 2)
    return failure();
  Value retValue = termOp.getOperands()[0];
  auto bbLhs = matchers::m_Val(reduceBody->getArgument(0));
  auto bbRhs = matchers::m_Val(reduceBody->getArgument(1));
  tensorrt::ReduceOperation reductionOpType;
  if (matchPattern(retValue, m_Op<stablehlo::OrOp>(bbLhs, bbRhs)))
    reductionOpType = tensorrt::ReduceOperation::kSUM;
  else if (matchPattern(retValue, m_Op<stablehlo::AndOp>(bbLhs, bbRhs)))
    reductionOpType = tensorrt::ReduceOperation::kPROD;
  else
    return failure();

  // Cast i1 to i32.
  Value i32Input =
      rewriter.create<tensorrt::IdentityOp>(loc, integerInputType, input);

  auto reduceOp = rewriter.create<tensorrt::ReduceOp>(
      loc, integerResultType, i32Input,
      /*reduceDims=*/SmallVector<int64_t>{reductionDim},
      /*keepdims=*/false, reductionOpType);
  // Cast i32 to i1.
  return rewriter
      .create<tensorrt::IdentityOp>(loc, originalResultType,
                                    reduceOp.getResult())
      .getResult();
}

/// Returns true if `dims` is a contiguous sequence of integers starting from 0.
static bool isSequence(ArrayRef<int64_t> dims) {
  return llvm::equal(dims, llvm::seq<int64_t>(0, dims.size()));
}

namespace {

// Converts a `stablehlo.reduce` operation to a `tensorrt.reduce` operation.
struct ConvertReduceOp
    : public ConvertHloOpToTensorRTPattern<stablehlo::ReduceOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorRTConversionPatternRewriter trtRewriter(rewriter,
                                                  targetTrtMajorVersion);

    Value operand = adaptor.getInputs().front();
    SmallVector<int64_t> reductionDims = llvm::to_vector(op.getDimensions());
    // Try to match and handle the ArgMin/ArgMax cases.
    if (succeeded(matchAndReplaceStablehloArgMinMax<
                  tensorrt::ArgMaxOp, stablehlo::ComparisonDirection::GE>(
            op, trtRewriter, operand, reductionDims, targetTrtMajorVersion)))
      return success();
    if (succeeded(matchAndReplaceStablehloArgMinMax<
                  tensorrt::ArgMinOp, stablehlo::ComparisonDirection::LE>(
            op, trtRewriter, operand, reductionDims, targetTrtMajorVersion)))
      return success();

    // Try to match the simpler reductions across a single input.
    if (op.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "number of reduction inputs not 1");
    Value init = adaptor.getInitValues().front();

    FailureOr<Value> replacement =
        convertBooleanReductions(rewriter, op, reductionDims, operand, init);
    if (succeeded(replacement)) {
      trtRewriter.replaceOp(op, *replacement);
      return success();
    }

    replacement = convertSimpleReductions(trtRewriter, op, reductionDims,
                                          operand, init, targetTrtMajorVersion);
    if (failed(replacement))
      return rewriter.notifyMatchFailure(
          op, "could not do simple reduction transform");
    trtRewriter.replaceOp(op, *replacement);
    return success();
  }
};

/// Convert `stablehlo.dot` to `tensorrt.matrix_multiply`.
/// TODO: clean since `dot` op is removed from stable hlo in the favor of
/// `dot_general`.
struct ConvertDot : public ConvertHloOpToTensorRTPattern<stablehlo::DotOp> {
  using ConvertHloOpToTensorRTPattern<
      stablehlo::DotOp>::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorRTConversionPatternRewriter trtRewriter(rewriter,
                                                  targetTrtMajorVersion);

    TensorType resultType = op.getType();
    tensorrt::MatrixOperation qualifierLhs = tensorrt::MatrixOperation::kNONE;
    tensorrt::MatrixOperation qualifierRhs = tensorrt::MatrixOperation::kNONE;
    auto lhsType = cast<TensorType>(adaptor.getLhs().getType());
    auto rhsType = cast<TensorType>(adaptor.getRhs().getType());
    if (lhsType.getRank() == 1)
      qualifierLhs = tensorrt::MatrixOperation::kVECTOR;
    if (rhsType.getRank() == 1)
      qualifierRhs = tensorrt::MatrixOperation::kVECTOR;

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto replacement = trtRewriter.checkAndCreate<tensorrt::MatrixMultiplyOp>(
        op->getLoc(), resultType, lhs, rhs, qualifierLhs, qualifierRhs);
    if (!replacement)
      return failure();

    return replaceWithCast(trtRewriter, op, replacement.getResult());
  }
};

struct EinsumHelper {

  EinsumHelper(stablehlo::DotGeneralOp op)
      : dimNums(op.getDotDimensionNumbers()), op(op) {}

  FailureOr<std::string> getEquation() {
    const int64_t lhsRank = op.getLhs().getType().getRank();
    const int64_t rhsRank = op.getRhs().getType().getRank();
    FailureOr<std::string> batchLetters = getBatchDimLetters();
    FailureOr<std::string> contractionLetters = getContractionDimLetters();
    FailureOr<std::string> lhsResultDimLetters =
        getResultDimLetters(dimNums.getLhsBatchingDimensions(),
                            dimNums.getLhsContractingDimensions(), lhsRank);
    FailureOr<std::string> rhsResultDimLetters =
        getResultDimLetters(dimNums.getRhsBatchingDimensions(),
                            dimNums.getRhsContractingDimensions(), rhsRank);
    if (failed(batchLetters) || failed(contractionLetters) ||
        failed(lhsResultDimLetters) || failed(rhsResultDimLetters))
      return failure();

    std::string equation;
    emitOperandTerms(lhsRank, *batchLetters, *contractionLetters,
                     *lhsResultDimLetters, dimNums.getLhsBatchingDimensions(),
                     dimNums.getLhsContractingDimensions(), equation);
    equation += ",";
    emitOperandTerms(rhsRank, *batchLetters, *contractionLetters,
                     *rhsResultDimLetters, dimNums.getRhsBatchingDimensions(),
                     dimNums.getRhsContractingDimensions(), equation);
    equation += "->";
    equation += *batchLetters + *lhsResultDimLetters + *rhsResultDimLetters;
    return equation;
  }

private:
  stablehlo::DotDimensionNumbersAttr dimNums;
  stablehlo::DotGeneralOp op;

  static constexpr StringRef kTermPool = "abcdefghijklmnopqrstuvwxyz";

  LogicalResult appendTerm(std::string &result) {
    if (termPos >= kTermPool.size())
      return failure();
    result += kTermPool[termPos++];
    return success();
  }

  FailureOr<std::string> getBatchDimLetters() {
    std::string result = "";
    ArrayRef<int64_t> lhsBatchDims = dimNums.getLhsBatchingDimensions();
    for (int64_t _ : lhsBatchDims) {
      if (failed(appendTerm(result)))
        return failure();
    }
    return result;
  }

  FailureOr<std::string> getContractionDimLetters() {
    std::string result = "";
    ArrayRef<int64_t> rhsContractingDims =
        dimNums.getRhsContractingDimensions();
    for (int64_t _ : rhsContractingDims) {
      if (failed(appendTerm(result)))
        return failure();
    }
    return result;
  }

  FailureOr<std::string> getResultDimLetters(ArrayRef<int64_t> batchDims,
                                             ArrayRef<int64_t> contractionDims,
                                             int64_t rank) {
    std::string result;
    for (int64_t dim : llvm::seq<int64_t>(0, rank)) {
      if (llvm::is_contained(batchDims, dim) ||
          llvm::is_contained(contractionDims, dim))
        continue;
      if (failed(appendTerm(result)))
        return failure();
    }
    return result;
  }

  void emitOperandTerms(int64_t rank, StringRef batchDimTerms,
                        StringRef contractionDimTerms, StringRef resultDimTerms,
                        ArrayRef<int64_t> batchDims,
                        ArrayRef<int64_t> contractionDims,
                        std::string &result) {
    for (int64_t idx : llvm::seq<int64_t>(0, rank)) {
      if (llvm::is_contained(batchDims, idx)) {
        result += batchDimTerms.front();
        batchDimTerms = batchDimTerms.drop_front();
        continue;
      }
      if (llvm::is_contained(contractionDims, idx)) {
        result += contractionDimTerms.front();
        contractionDimTerms = contractionDimTerms.drop_front();
        continue;
      }
      result += resultDimTerms.front();
      resultDimTerms = resultDimTerms.drop_front();
    }
    assert(batchDimTerms.empty() && "expected all batch dim terms to be used");
    assert(contractionDimTerms.empty() &&
           "expected all contraction dim terms to be used");
    assert(resultDimTerms.empty() &&
           "expected all result dim terms to be used");
  }

  std::string batchDimLetters;
  unsigned termPos = 0;
};

/// Convert `stablehlo.dot_general` to `tensorrt.einsum`.
struct ConvertDotGeneralToEinsum
    : public ConvertHloOpToTensorRTPattern<stablehlo::DotGeneralOp> {
  using ConvertHloOpToTensorRTPattern<
      stablehlo::DotGeneralOp>::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::DotGeneralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorRTConversionPatternRewriter trtRewriter(rewriter,
                                                  targetTrtMajorVersion);

    TensorType resultType = op.getType();
    // Determine the TRT equivalent qualifier.
    auto lhs = cast<TensorValue>(adaptor.getLhs());
    auto rhs = cast<TensorValue>(adaptor.getRhs());
    TensorType lhsType = lhs.getType();

    if (lhsType.getElementType().isInteger(32))
      return failure();

    // 'stablehlo.dot_general' allows for promotion of the result element
    // type. We treat this as equivalent to compute/accumulator element type
    // being equal to the result type. In TensorRT, we have limited control
    // over the accumulator element type, but you're supposed to be able to
    // specify it using cast operaitons on the operands.
    Type computeElementType = resultType.getElementType();
    if (computeElementType != lhsType.getElementType()) {
      FailureOr<TensorValue> castedLhs =
          this->castTensor(trtRewriter, computeElementType, lhs);
      FailureOr<TensorValue> castedRhs =
          this->castTensor(trtRewriter, computeElementType, rhs);
      if (failed(castedLhs) || failed(castedRhs))
        return failure();
      lhs = std::move(*castedLhs);
      rhs = std::move(*castedRhs);
    }

    EinsumHelper helper(op);
    FailureOr<std::string> equation = helper.getEquation();
    if (failed(equation))
      return failure();

    tensorrt::EinsumOp replacement =
        trtRewriter.checkAndCreate<tensorrt::EinsumOp>(
            op.getLoc(), resultType, ValueRange{lhs, rhs},
            trtRewriter.getStringAttr(*equation));
    if (!replacement)
      return failure();
    return replaceWithCast(trtRewriter, op, replacement.getResult());
  }
};

/// Convert `stablehlo.dot_general` to `tensorrt.matrix_multiply`.
struct ConvertDotGeneralToMatrixMultiply
    : public ConvertHloOpToTensorRTPattern<stablehlo::DotGeneralOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;

  LogicalResult
  matchAndRewrite(stablehlo::DotGeneralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorRTConversionPatternRewriter trtRewriter(rewriter,
                                                  targetTrtMajorVersion);

    stablehlo::DotDimensionNumbersAttr dimNums = op.getDotDimensionNumbers();
    const int64_t numContractionDims =
        dimNums.getRhsContractingDimensions().size();

    // The batching dimensions must form a contiguous sequence [0, ....,
    // NumBatchingDims-1].
    if (!isSequence(dimNums.getLhsBatchingDimensions()) ||
        !isSequence(dimNums.getRhsBatchingDimensions()))
      return failure();

    RankedTensorType resultType = op.getType();

    // Determine the TRT equivalent qualifier.
    tensorrt::MatrixOperation qualifierLhs = tensorrt::MatrixOperation::kNONE;
    tensorrt::MatrixOperation qualifierRhs = tensorrt::MatrixOperation::kNONE;
    auto lhs = cast<TensorValue>(adaptor.getLhs());
    auto rhs = cast<TensorValue>(adaptor.getRhs());
    RankedTensorType lhsType = lhs.getType();
    RankedTensorType rhsType = rhs.getType();

    if (lhsType.getElementType().isInteger(32))
      return failure();

    // 'stablehlo.dot_general' allows for promotion of the result element type.
    // We treat this as equivalent to compute/accumulator element type being
    // equal to the result type. In TensorRT, we have limited control over the
    // accumulator element type, but you're supposed to be able to specify it
    // using cast operaitons on the operands.
    Type computeElementType = resultType.getElementType();
    if (computeElementType != lhsType.getElementType()) {
      FailureOr<TensorValue> castedLhs =
          this->castTensor(trtRewriter, computeElementType, lhs);
      FailureOr<TensorValue> castedRhs =
          this->castTensor(trtRewriter, computeElementType, rhs);
      if (failed(castedLhs) || failed(castedRhs))
        return failure();
      lhs = std::move(*castedLhs);
      rhs = std::move(*castedRhs);
    }

    // We don't handle multiple contraction dims.
    if (numContractionDims != 1)
      return failure();

    // We don't handle multiple outer product dimensions.
    int64_t numBatchDims =
        static_cast<int64_t>(dimNums.getLhsBatchingDimensions().size());
    if (rhsType.getRank() > numBatchDims + numContractionDims + 1 ||
        lhsType.getRank() > numBatchDims + numContractionDims + 1)
      return failure();

    if (lhsType.getRank() == numBatchDims + numContractionDims + 1) {
      if (dimNums.getLhsContractingDimensions().front() ==
          lhsType.getRank() - 1)
        qualifierLhs = tensorrt::MatrixOperation::kNONE;
      else if (dimNums.getLhsContractingDimensions().front() ==
               lhsType.getRank() - 2)
        qualifierLhs = tensorrt::MatrixOperation::kTRANSPOSE;
      else
        return failure();
      // No explicit outer product dimension
    } else if (lhsType.getRank() == numBatchDims + numContractionDims) {
      qualifierLhs = tensorrt::MatrixOperation::kVECTOR;
    } else {
      return failure();
    }

    if (rhsType.getRank() == numBatchDims + numContractionDims + 1) {
      if (dimNums.getRhsContractingDimensions().front() ==
          rhsType.getRank() - 1)
        qualifierRhs = tensorrt::MatrixOperation::kTRANSPOSE;
      else if (dimNums.getRhsContractingDimensions().front() ==
               rhsType.getRank() - 2)
        qualifierRhs = tensorrt::MatrixOperation::kNONE;
      else
        return failure();
    } else if (rhsType.getRank() == numBatchDims + numContractionDims) {
      qualifierRhs = tensorrt::MatrixOperation::kVECTOR;
    } else {
      return failure();
    }
    auto replacement = trtRewriter.checkAndCreate<tensorrt::MatrixMultiplyOp>(
        op->getLoc(), resultType, lhs, rhs, qualifierLhs, qualifierRhs);
    if (!replacement)
      return failure();
    return replaceWithCast(trtRewriter, op, replacement.getResult());
  }
};

} // namespace

void mlir::populateStablehloReductionAndContractionToTensorRtConversionPattern(
    TensorRTTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, PatternBenefit dotToEinsumBenefit) {
  // clang-format off
  patterns.add<
    ConvertDot,
    ConvertDotGeneralToMatrixMultiply,
    ConvertReduceOp
  >(typeConverter, patterns.getContext(), benefit);
  patterns.add<
    ConvertDotGeneralToEinsum
  >(typeConverter, patterns.getContext(), dotToEinsumBenefit);
  // clang-format on
}
