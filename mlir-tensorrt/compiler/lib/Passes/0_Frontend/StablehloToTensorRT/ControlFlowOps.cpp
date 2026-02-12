//===- ControlFlowOps.cpp -- ----------------------------------------------===//
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
/// Implementation of patterns for converting `stablehlo` control flow
/// operations to TensorRT dialect operations.
///
//===----------------------------------------------------------------------===//
#include "ControlFlowOps.h"
#include "Matchers.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

//// Given an `stablehloRegion` from a Stable HLO operation, inlien the region
/// into the tensorrt region and replace the terminator with a `tensorrt.yield`
/// operation.
static void inlineStablehloRegionIntoTensorRtRegion(PatternRewriter &rewriter,
                                                    Region &stablehloRegion,
                                                    Region &tensorrtRegion) {
  if (!tensorrtRegion.empty())
    rewriter.eraseBlock(&tensorrtRegion.back());
  rewriter.inlineRegionBefore(stablehloRegion, tensorrtRegion,
                              tensorrtRegion.end());
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&tensorrtRegion.back());
  auto *terminator = tensorrtRegion.back().getTerminator();
  rewriter.replaceOpWithNewOp<tensorrt::YieldOp>(terminator,
                                                 terminator->getOperands());
}

namespace {

/// Convert `stablehlo.while` into `tensorrt.while`.
struct ConvertWhileOp
    : public ConvertHloOpToTensorRTPattern<stablehlo::WhileOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes)))
      return failure();

    auto whileOp = rewriter.create<tensorrt::WhileOp>(op.getLoc(), resultTypes,
                                                      adaptor.getOperand());
    rewriter.inlineRegionBefore(op.getCond(), whileOp.getCondRegion(),
                                whileOp.getCondRegion().end());
    rewriter.setInsertionPointToEnd(&whileOp.getCondRegion().front());
    Operation *terminator = whileOp.getCondRegion().front().getTerminator();
    rewriter.replaceOpWithNewOp<tensorrt::ConditionOp>(
        terminator, terminator->getOperand(0),
        whileOp.getWhileConditionLoopCarriedDeps());

    // Remove an existing block, then move the region over.
    Region &tensorrtBody = whileOp.getBodyRegion();
    Region &stablehloBody = op.getBody();
    if (!tensorrtBody.empty())
      rewriter.eraseBlock(&tensorrtBody.back());
    rewriter.inlineRegionBefore(stablehloBody, tensorrtBody,
                                tensorrtBody.end());
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&tensorrtBody.back());
    Operation *bodyTerm = tensorrtBody.back().getTerminator();
    rewriter.replaceOpWithNewOp<tensorrt::YieldOp>(bodyTerm,
                                                   bodyTerm->getOperands());
    rewriter.replaceOp(op, whileOp.getResults());
    return success();
  }
};

/// Convert `stablehlo.case` to `tensorrt.if` operation.
struct ConvertCaseOp : public ConvertHloOpToTensorRTPattern<stablehlo::CaseOp> {
  using ConvertHloOpToTensorRTPattern::ConvertHloOpToTensorRTPattern;
  LogicalResult
  matchAndRewrite(stablehlo::CaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // If there is only one region, then just inline the body immediately before
    // the case op.
    if (op.getBranches().size() == 1) {
      Block &caseBody = op.getBranches().front().front();
      Operation *term = caseBody.getTerminator();
      ValueRange results = term->getOperands();
      rewriter.eraseOp(term);
      rewriter.inlineBlockBefore(&caseBody, op.getOperation());
      rewriter.replaceOp(op, results);
      return success();
    }

    // JAX code commonly produces code that looks like
    // ```
    // %i32 = stablehlo.convert %bool to %i32
    // %result = stablehlo.case(%i32) {.... } {....}
    // ```
    // so we handle this case separately from the general case
    // because we can produce more concise IR.
    // TODO: move this to a preperatory rewrite.
    auto isCastFromBool = [](Operation *op) {
      if (!isa_and_nonnull<tensorrt::IdentityOp, stablehlo::ConvertOp>(op))
        return false;
      RankedTensorType producerType =
          cast<RankedTensorType>(op->getOperand(0).getType());
      return isa_and_nonnull<tensorrt::IdentityOp, stablehlo::ConvertOp>(op) &&
             producerType.getElementType().isInteger(1) &&
             producerType.getNumElements() == 1;
    };
    if (isCastFromBool(adaptor.getIndex().getDefiningOp()) &&
        op->getNumRegions() == 2) {
      Value index = adaptor.getIndex().getDefiningOp()->getOperand(0);
      auto trtIfOp = rewriter.create<tensorrt::IfOp>(
          op.getLoc(), op->getResultTypes(), index);
      // The index value of "0" corresponds to the first source branch; but this
      // corresponds to "False" region of an if op.
      inlineStablehloRegionIntoTensorRtRegion(rewriter, op.getBranches()[0],
                                              trtIfOp.getFalseRegion());
      inlineStablehloRegionIntoTensorRtRegion(rewriter, op.getBranches()[1],
                                              trtIfOp.getTrueRegion());
      rewriter.replaceOp(op, trtIfOp->getResults());
      return success();
    }

    // Otherwise, if there are only two regions, we can still convert to a
    // single `if` statement.
    if (op->getNumRegions() == 2) {
      Value index = adaptor.getIndex();
      Value zero = rewriter.create<tensorrt::ConstantOp>(
          op.getLoc(),
          cast<ElementsAttr>(rewriter.getZeroAttr(index.getType())));
      Value condition = rewriter.create<tensorrt::ElementWiseOp>(
          op.getLoc(), index, zero, tensorrt::ElementWiseOperation::kEQUAL);
      auto trtIfOp = rewriter.create<tensorrt::IfOp>(
          op.getLoc(), op->getResultTypes(), condition);
      inlineStablehloRegionIntoTensorRtRegion(rewriter, op.getBranches()[0],
                                              trtIfOp.getTrueRegion());
      inlineStablehloRegionIntoTensorRtRegion(rewriter, op.getBranches()[1],
                                              trtIfOp.getFalseRegion());
      rewriter.replaceOp(op, trtIfOp->getResults());
      return success();
    }

    // TODO: handle the other general cases. We need to create nested if ops.
    return failure();
  }
};
} // namespace

void mlir::populateStablehloControlFlowToTensorRtPatterns(
    TensorRTTypeConverter &typeConverter, RewritePatternSet &patterns,
    bool convertLoops, bool convertConditionals) {
  if (convertLoops)
    patterns.insert<ConvertWhileOp>(typeConverter, patterns.getContext());
  if (convertConditionals)
    patterns.insert<ConvertCaseOp>(typeConverter, patterns.getContext());
}
