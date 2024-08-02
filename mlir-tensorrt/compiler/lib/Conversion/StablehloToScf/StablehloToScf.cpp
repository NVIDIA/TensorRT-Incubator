//===- StablehloToScf.cpp ---------------------------------------*- C++ -*-===//
//
// Logic for this pass is adapted from MHLO upstream `legalize-control-flow`
// pass at
// https://github.com/openxla/xla/blob/main/xla/mlir_hlo/mhlo/transforms/legalize_control_flow/legalize_control_flow.cc//
// This project has the Apache License v2.0 license. Check
// https://github.com/openxla/xla/blob/main/LICENSE for the license
// information.
//
// Changes are copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of a pass to convert stablehlo control flow ops to scf ops.
///
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSTABLEHLOTOSCFPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// Inlines stablehlo op region into scf region. Replaces stablehlo `ReturnOp`
/// with scf `YieldOp`.
static void inlineStablehloRegionIntoSCFRegion(PatternRewriter &rewriter,
                                               Region &stablehlo, Region &scf) {
  // Remove an existing block, if any.
  if (!scf.empty())
    rewriter.eraseBlock(&scf.back());
  // Move the region over.
  rewriter.inlineRegionBefore(stablehlo, scf, scf.end());
  // Replace stablehlo `ReturnOp`with scf `YieldOp`.
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&scf.back());
  auto *terminator = scf.back().getTerminator();
  rewriter.replaceOpWithNewOp<scf::YieldOp>(terminator,
                                            terminator->getOperands());
}

/// Extracts a scalar from tensor with a single element.
static Value extractScalarFromTensorValue(OpBuilder &b, Value tensor) {
  Location loc = tensor.getLoc();
  // If ranked tensor, first collapse shape.
  if (tensor.getType().cast<RankedTensorType>().getRank() != 0)
    tensor = b.create<tensor::CollapseShapeOp>(
        loc, tensor, SmallVector<ReassociationIndices>());

  return b.create<tensor::ExtractOp>(loc, tensor, ValueRange());
}

namespace {
/// Convert `stablehlo.while` op to `scf.while` op.
/// Region mapping is as follows from stablehlo while to scf while,
/// `cond` -> `before`
/// `body` -> `after`
struct ConvertWhileOp : public OpConversionPattern<stablehlo::WhileOp> {
  using OpConversionPattern<stablehlo::WhileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stablehlo::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto scfWhile = rewriter.create<scf::WhileOp>(
        op.getLoc(), op->getResultTypes(), op->getOperands());
    // Move `cond` region of stablehlo to `before` region of scf while op.
    rewriter.inlineRegionBefore(op.getCond(), scfWhile.getBefore(),
                                scfWhile.getBefore().end());
    // Replace `stablehlo.return` with `scf.yield`. Return type of
    // `stablehlo.return` is a tensor but `scf.condition` op condition input
    // needs to be a scalar `i1`. Thus, we need to extract the scalar.
    rewriter.setInsertionPointToEnd(&scfWhile.getBefore().front());
    auto stablehloConditionReturnOp =
        cast<stablehlo::ReturnOp>(scfWhile.getBefore().front().back());
    auto scalarI1 = extractScalarFromTensorValue(
        rewriter, stablehloConditionReturnOp->getOperand(0));
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        stablehloConditionReturnOp, scalarI1, scfWhile.getBeforeArguments());
    // Move `body` region of stablehlo to `after` region of scf while op.
    inlineStablehloRegionIntoSCFRegion(rewriter, op.getBody(),
                                       scfWhile.getAfter());
    rewriter.replaceOp(op, scfWhile);
    return success();
  }
};

/// Convert `stablehlo.if` to `scf.if`.
/// Region mapping is as follows from stablehlo if to scf if,
/// `true_branch` -> `thenRegion`
/// `false_branch` -> `elseRegion`
struct ConvertIfOp : public OpConversionPattern<stablehlo::IfOp> {
  using OpConversionPattern<stablehlo::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stablehlo::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto scfIf = rewriter.create<scf::IfOp>(
        op->getLoc(), op->getResultTypes(),
        extractScalarFromTensorValue(rewriter, op.getPred()));
    // Inline stablhlo `true_branch` region
    inlineStablehloRegionIntoSCFRegion(rewriter, op.getTrueBranch(),
                                       scfIf.getThenRegion());
    // Inline stablehlo `false_branch` region
    inlineStablehloRegionIntoSCFRegion(rewriter, op.getFalseBranch(),
                                       scfIf.getElseRegion());
    rewriter.replaceOp(op, scfIf);
    return success();
  }
};
} // namespace

/// Recursively create if/else ops to handle each branch in the `CaseOp`.
/// If `index` of case op is -1, it indicates last branch. To meet this
/// requirements, last two branches part of a single `if` op.
static scf::IfOp createNestedCases(int currentIdx, stablehlo::CaseOp op,
                                   stablehlo::CaseOp::Adaptor adaptor,
                                   PatternRewriter &outerBuilder) {
  Location loc = op.getLoc();
  Value idxValue = adaptor.getIndex();
  auto finalIdx = op.getBranches().size() - 2;

  // Determine if the current index matches the case index.
  auto scalarType = idxValue.getType();
  auto shapedType = scalarType.cast<ShapedType>();
  auto constAttr = DenseElementsAttr::get(
      shapedType,
      {outerBuilder.getI32IntegerAttr(currentIdx).cast<mlir::Attribute>()});
  Value currentIdxVal = outerBuilder.create<stablehlo::ConstantOp>(
      loc, idxValue.getType(), constAttr);

  auto scfIf = outerBuilder.create<scf::IfOp>(
      loc, op.getResultTypes(),
      extractScalarFromTensorValue(outerBuilder,
                                   outerBuilder.create<stablehlo::CompareOp>(
                                       loc, idxValue, currentIdxVal,
                                       stablehlo::ComparisonDirection::EQ)),
      /*withElseRegion=*/true);
  inlineStablehloRegionIntoSCFRegion(outerBuilder, op.getBranches()[currentIdx],
                                     scfIf.getThenRegion());
  int nextIdx = currentIdx + 1;
  // Don't recurse for the final default block.
  if (currentIdx == static_cast<int64_t>(finalIdx)) {
    inlineStablehloRegionIntoSCFRegion(outerBuilder, op.getBranches()[nextIdx],
                                       scfIf.getElseRegion());
  } else {
    PatternRewriter::InsertionGuard guard(outerBuilder);
    outerBuilder.setInsertionPointToEnd(&scfIf.getElseRegion().back());
    auto innerIf = createNestedCases(nextIdx, op, adaptor, outerBuilder);
    outerBuilder.create<scf::YieldOp>(op.getLoc(), innerIf.getResults());
  }
  return scfIf;
}

namespace {
/// Rewrites `stablehlo.case` to a nested `scf.if` ops.
struct ConvertCaseOp : public OpConversionPattern<stablehlo::CaseOp> {
  using OpConversionPattern<stablehlo::CaseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stablehlo::CaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Inline the op if there is only a default block.
    if (op.getBranches().size() == 1) {
      Block &block = op.getBranches().front().front();
      auto results = block.getTerminator()->getOperands();
      // Remove the stablehlo.return terminator, then inline the block.
      rewriter.eraseOp(block.getTerminator());
      rewriter.inlineBlockBefore(/*source=*/&block, /*dest=*/op.getOperation(),
                                 /*argValues=*/{});
      rewriter.replaceOp(op, results);
      return success();
    }

    // Begin recursion with case 0.
    rewriter.replaceOp(
        op, createNestedCases(0, op, adaptor, rewriter).getResults());
    return success();
  }
};

struct StablehloToScfPass
    : public impl::ConvertStablehloToScfPassBase<StablehloToScfPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertWhileOp, ConvertIfOp, ConvertCaseOp>(&getContext());
    mlir::ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target
        .addIllegalOp<stablehlo::IfOp, stablehlo::WhileOp, stablehlo::CaseOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to apply patterns in " << getArgument();
      signalPassFailure();
    }
  }
};
} // namespace