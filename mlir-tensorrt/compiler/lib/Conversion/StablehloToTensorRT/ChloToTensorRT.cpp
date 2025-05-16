//===- ChloOps.cpp --------------------------------------------------------===//
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
/// Converters for Chlo ops that can map directly to TensorRT.
///
//===----------------------------------------------------------------------===//
#include "Matchers.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Conversion/StablehloToTensorRT/StablehloToTensorRT.h"
#include "stablehlo/dialect/ChloOps.h"

#define DEBUG_TYPE "chlo-to-tensorrt"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {
/// Convert `chlo.erf` to `tensorrt.unary`. This can be done in a 1-1 manner.
struct ConvertChloErfToTensorRT
    : public ConvertHloOpToTensorRTPattern<chlo::ErfOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(chlo::ErfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorRTConversionPatternRewriter trtRewriter(rewriter,
                                                  targetTrtMajorVersion);
    Location loc = op->getLoc();

    auto operand = adaptor.getOperand();
    auto operandType = cast<RankedTensorType>(operand.getType());
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();
    if (operandType.getRank() == 0) {
      RankedTensorType newShape =
          RankedTensorType::get({1}, operandType.getElementType());
      auto expOperand = trtRewriter.checkAndCreate<tensorrt::ExpandRankOp>(
          loc, newShape, operand);
      if (!expOperand)
        return failure();
      operand = expOperand;
    }
    auto unaryOp = trtRewriter.checkAndCreate<tensorrt::UnaryOp>(
        op.getLoc(), operand.getType(), operand,
        tensorrt::UnaryOperation::kERF);
    if (!unaryOp)
      return failure();
    TypedValue<RankedTensorType> result = unaryOp.getResult();
    // Cast back if required.
    if (result.getType().getElementType() != op.getType().getElementType()) {
      auto castedResult =
          castTensor(trtRewriter, op.getType().getElementType(), result);
      if (failed(castedResult))
        return failure();
      result = *castedResult;
    }
    // collapse rank if required
    if (result.getType().getRank() != op.getType().getRank()) {
      auto collapsedResult =
          trtRewriter.checkAndCreate<tensorrt::CollapseRankOp>(
              loc, op.getType(), result);
      if (!collapsedResult)
        return failure();
      result = collapsedResult;
    }
    trtRewriter.replaceOp(op, result);
    return success();
  }
};

// Convert `chlo.top_k` to `tensorrt.top_k`.
struct ConvertChloTopKOpToTensorRT
    : public ConvertHloOpToTensorRTPattern<chlo::TopKOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(chlo::TopKOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorRTConversionPatternRewriter trtRewriter(rewriter,
                                                  targetTrtMajorVersion);

    auto operand = adaptor.getOperand();
    RankedTensorType operandType = cast<RankedTensorType>(operand.getType());

    int64_t rank = operandType.getRank();
    uint64_t axis = static_cast<uint64_t>(rank) - 1;
    uint64_t k = op.getK();

    // The value of k must be <= 3840, as of TRT 8.6.1.
    // Refer to
    // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_top_k_layer.html
    if (k > 3840)
      return rewriter.notifyMatchFailure(
          op, "k exceeds the maximum supported by TensorRT");
    RankedTensorType inputType =
        cast<RankedTensorType>(op.getOperand().getType());
    if (inputType.getRank() == 1) {
      // TensorRT doesn't support 1D top k. We expand rank to meet TensorRT
      // need.
      RankedTensorType newShape = RankedTensorType::get(
          {inputType.getShape().front(), 1}, inputType.getElementType());
      auto expandRankOp = trtRewriter.checkAndCreate<tensorrt::ExpandRankOp>(
          op->getLoc(), newShape, operand);
      if (!expandRankOp)
        return failure();
      operand = expandRankOp;
    }

    auto topkOp = trtRewriter.checkAndCreate<tensorrt::TopKOp>(
        op->getLoc(), operand, k, axis, tensorrt::TopKOperation::kMAX);
    if (!topkOp)
      return failure();

    if (topkOp.getValues().getType() != op.getValues().getType()) {
      // Rank expansion happened before, to support 1D input. Collapse ranks
      // back.
      auto collapsedValues =
          trtRewriter.checkAndCreate<tensorrt::CollapseRankOp>(
              op->getLoc(), op.getValues().getType(), topkOp.getValues());
      if (!collapsedValues)
        return failure();
      auto collapsedIndices =
          trtRewriter.checkAndCreate<tensorrt::CollapseRankOp>(
              op->getLoc(), op.getIndices().getType(), topkOp.getIndices());
      if (!collapsedIndices)
        return failure();
      trtRewriter.replaceOp(op, ValueRange{collapsedValues, collapsedIndices});
      return success();
    }
    trtRewriter.replaceOp(op, topkOp);
    return success();
  }
};
} // namespace

void mlir::populateChloToTensorRtLegalityAndPatterns(
    TensorRTTypeConverter &typeConverter, ConversionTarget &target,
    RewritePatternSet &patterns) {
  target.addIllegalOp<chlo::ErfOp>();
  patterns.add<ConvertChloErfToTensorRT, ConvertChloTopKOpToTensorRT>(
      typeConverter, patterns.getContext());
}
