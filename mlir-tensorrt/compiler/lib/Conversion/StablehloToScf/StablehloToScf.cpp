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

#include "mlir-tensorrt/Transforms/Transforms.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "stablehlo/conversions/linalg/transforms/MapStablehloToScalarOp.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

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
  RankedTensorType rtt = cast<RankedTensorType>(tensor.getType());
  SmallVector<Value> zeros(rtt.getRank(),
                           b.create<arith::ConstantIndexOp>(loc, 0));
  return b.create<tensor::ExtractOp>(loc, tensor, zeros);
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
  auto shapedType = cast<ShapedType>(scalarType);
  auto constAttr = DenseElementsAttr::get(
      shapedType,
      {cast<mlir::Attribute>(outerBuilder.getI32IntegerAttr(currentIdx))});
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
} // namespace

//===----------------------------------------------------------------------===//
// Code after this point is not part of the original MHLO pass.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// These patterns are meant to perform canonicalization and uplift of
// scf.while to scf.for after the conversion from stablehlo to scf.
//===----------------------------------------------------------------------===//

/// Scalarize a `stablehlo.compare` op.
static Value scalarizeStablehloCompareOp(stablehlo::CompareOp op,
                                         PatternRewriter &rewriter) {
  auto scalarOperands = llvm::map_to_vector(op.getOperands(), [&](Value v) {
    return extractScalarFromTensorValue(rewriter, v);
  });
  return stablehlo::StablehloOpToStdScalarOp::mapOp<stablehlo::CompareOp>(
      op, op.getType().getElementType(), scalarOperands, &rewriter);
}

namespace {

/// Scalarize a `stablehlo.compare` op used by a `tensor.extract` op.
struct ScalarizeStablehloCompareUsedByExtractPattern
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto compareOp = op.getTensor().getDefiningOp<stablehlo::CompareOp>();
    if (!compareOp || !compareOp.getType().hasStaticShape() ||
        compareOp.getType().getNumElements() != 1 ||
        !compareOp.getType().getElementType().isSignlessIntOrIndex())
      return failure();
    rewriter.setInsertionPoint(compareOp);
    Value scalarCompare = scalarizeStablehloCompareOp(compareOp, rewriter);
    rewriter.replaceOp(op, scalarCompare);
    return success();
  }
};
} // namespace

static bool isScalarizable(Type type) {
  if (auto rtt = dyn_cast<RankedTensorType>(type))
    return rtt.hasStaticShape() && rtt.getNumElements() == 1;
  return false;
}

static FailureOr<Value> convertToScalar(Operation *op,
                                        PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  RankedTensorType rtt = cast<RankedTensorType>(op->getResult(0).getType());
  SmallVector<Value> scalarOperands;
  for (Value operand : op->getOperands())
    scalarOperands.push_back(extractScalarFromTensorValue(rewriter, operand));
  return llvm::TypeSwitch<Operation *, FailureOr<Value>>(op)
      .Case<
          // clang-format off
          stablehlo::AbsOp,
          stablehlo::AddOp,
          stablehlo::AndOp,
          stablehlo::Atan2Op,
          stablehlo::BitcastConvertOp,
          stablehlo::CbrtOp,
          stablehlo::CeilOp,
          stablehlo::ClampOp,
          stablehlo::ClzOp,
          stablehlo::CompareOp,
          stablehlo::ComplexOp,
          stablehlo::ConvertOp,
          stablehlo::CosineOp,
          stablehlo::DivOp,
          stablehlo::ExpOp,
          stablehlo::Expm1Op,
          stablehlo::FloorOp,
          stablehlo::ImagOp,
          stablehlo::IsFiniteOp,
          stablehlo::Log1pOp,
          stablehlo::LogOp,
          stablehlo::LogisticOp,
          stablehlo::MaxOp,
          stablehlo::MinOp,
          stablehlo::MulOp,
          stablehlo::NegOp,
          stablehlo::NotOp,
          stablehlo::OrOp,
          stablehlo::PopulationCountOp,
          stablehlo::PowOp,
          stablehlo::RealOp,
          stablehlo::ReducePrecisionOp,
          stablehlo::RemOp,
          stablehlo::RoundNearestEvenOp,
          stablehlo::RoundOp,
          stablehlo::RsqrtOp,
          stablehlo::SelectOp,
          stablehlo::ShiftLeftOp,
          stablehlo::ShiftRightArithmeticOp,
          stablehlo::ShiftRightLogicalOp,
          stablehlo::SignOp,
          stablehlo::SineOp,
          stablehlo::SqrtOp,
          stablehlo::SubtractOp,
          stablehlo::TanhOp,
          stablehlo::XorOp
          // clang-format on
          >([&](auto op) -> FailureOr<Value> {
        return stablehlo::StablehloOpToStdScalarOp::mapOp(
            op, rtt.getElementType(), scalarOperands, &rewriter);
      })
      .Default([](auto op) -> FailureOr<Value> { return failure(); });
}

namespace {
/// Scalarize operations which feed into the condition argument of
/// `scf.condition`.
struct ScalarizeWhileConditionProducers
    : public OpRewritePattern<scf::ConditionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ConditionOp op,
                                PatternRewriter &rewriter) const override {
    auto scfWhile = op->getParentOfType<scf::WhileOp>();
    if (!scfWhile || scfWhile.getBefore() != op->getParentRegion())
      return rewriter.notifyMatchFailure(
          op, "op is not in the before region of a scf.while op");

    Region &beforeRegion = scfWhile.getBefore();
    BackwardSliceOptions options{};
    options.inclusive = false;
    options.omitUsesFromAbove = true;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *op) {
      return beforeRegion.isAncestor(op->getParentRegion()) &&
             (llvm::isa_and_present<stablehlo::StablehloDialect>(
                  op->getDialect()) ||
              llvm::isa<tensor::ExtractOp>(op));
    };

    SetVector<Operation *> producers;
    getBackwardSlice(op.getCondition(), &producers, options);

    bool changed = false;
    for (Operation *producer : producers) {
      if (!isa_and_present<stablehlo::StablehloDialect>(
              producer->getDialect()) ||
          !producer->hasTrait<OpTrait::Elementwise>() ||
          producer->getNumResults() != 1)
        continue;
      if (!isScalarizable(producer->getResult(0).getType()))
        continue;
      if (!llvm::all_of(producer->getOperandTypes(), isScalarizable))
        continue;
      FailureOr<Value> scalarized = convertToScalar(producer, rewriter);
      if (failed(scalarized))
        continue;
      rewriter.setInsertionPointAfterValue(*scalarized);
      rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
          producer, producer->getResult(0).getType(), scalarized.value());
      changed = true;
    }
    return success(changed);
  }
};
} // namespace

/// Check if the add op is a valid induction variable increment.
static bool matchInductionVariableIncrement(stablehlo::AddOp op,
                                            scf::WhileOp parentWhile) {
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  if (matchPattern(lhs, m_Constant()) || matchPattern(rhs, m_Constant()))
    return true;
  Region *whileRegion = parentWhile->getParentRegion();
  return lhs.getParentRegion()->isAncestor(whileRegion) ||
         rhs.getParentRegion()->isAncestor(whileRegion);
}

namespace {
/// Scalarize any `stablehlo.add` operations in the 'after' region of
/// a scf.while op.
struct ScalarizeStablehloAddOp : public OpRewritePattern<stablehlo::AddOp> {
  using OpRewritePattern<stablehlo::AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::AddOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "op has more than one use, cannot scalarize");
    auto extractUser = dyn_cast<tensor::ExtractOp>(*op->user_begin());
    if (!extractUser || !extractUser->hasOneUse() ||
        !isa<scf::YieldOp>(*extractUser->user_begin()))
      return rewriter.notifyMatchFailure(
          op, "op result is not extracted and yielded from region");

    auto scfWhile = extractUser->getParentOfType<scf::WhileOp>();
    if (!scfWhile || scfWhile.getAfter() != op->getParentRegion())
      return rewriter.notifyMatchFailure(
          op, "op is not in the after region of a scf.while op");

    // One operand must be a constant or defined above in order to be
    // considered as the loop step.
    if (!matchInductionVariableIncrement(op, scfWhile))
      return rewriter.notifyMatchFailure(
          op, "op is not a valid induction variable increment");

    // Find a block argument that has been scalarized.
    auto findBlockArgument = [](Value v) -> BlockArgument {
      Value source{};
      if (matchPattern(v,
                       m_Op<tensor::FromElementsOp>(matchers::m_Any(&source))))
        return dyn_cast<BlockArgument>(source);
      return {};
    };
    BlockArgument arg = findBlockArgument(op.getLhs());
    if (!arg)
      arg = findBlockArgument(op.getRhs());
    if (!arg || arg.getParentRegion() != scfWhile.getAfter())
      return rewriter.notifyMatchFailure(
          op, "could not find block argument in after region");

    // Check that the corresponding block argument in the `before` region feeds
    // into a comparison.
    Region &before = scfWhile.getBefore();
    if (arg.getArgNumber() >= before.getNumArguments() ||
        before.getArgument(arg.getArgNumber()).getType() != arg.getType())
      return rewriter.notifyMatchFailure(
          op, "could not find block argument in before region");
    auto beforeArg = before.getArgument(arg.getArgNumber());
    if (!llvm::all_of(beforeArg.getUsers(),
                      llvm::IsaPred<scf::ConditionOp, arith::CmpIOp>))
      return rewriter.notifyMatchFailure(
          op, "block argument is not consumed by a comparison op");

    // Check that the before region has a block argument in the same position
    // and is consumed by a comparison op.
    RankedTensorType rtt = op.getType();
    Type elementType = rtt.getElementType();
    if (!rtt.hasStaticShape() || rtt.getNumElements() != 1 ||
        !elementType.isSignlessIntOrIndex())
      return rewriter.notifyMatchFailure(op, "op is not a scalar add op");

    auto scalarOperands = llvm::map_to_vector(op.getOperands(), [&](Value v) {
      return extractScalarFromTensorValue(rewriter, v);
    });

    auto scalarAdd =
        stablehlo::StablehloOpToStdScalarOp::mapOp<stablehlo::AddOp>(
            op, elementType, scalarOperands, &rewriter);
    auto fromElements =
        rewriter.create<tensor::FromElementsOp>(op.getLoc(), rtt, scalarAdd);
    rewriter.replaceOp(op, fromElements);
    return success();
  }
};
} // namespace

/// This is used by the SCF while detensorization patterns to determine whether
/// a block argument of the 'before' region should be scalarized. We want to
/// scalarize the block argument corresponding to the induction variable of the
/// for loop. It will have a user like `stablehlo.compare` or `tensor.extract`.
static bool shouldScalarizeWhileBeforeArg(BlockArgument arg, Value initOperand,
                                          Value yieldOperand) {
  return cast<RankedTensorType>(arg.getType())
             .getElementType()
             .isSignlessIntOrIndex() &&
         llvm::count_if(arg.getUsers(),
                        llvm::IsaPred<stablehlo::CompareOp, arith::CmpIOp,
                                      tensor::ExtractOp>) >= 1;
}

/// This is used by the SCF while detensorization patterns to determine whether
/// a block argument of the 'after' region should be scalarized. We want to
/// scalarize the block argument corresponding to the induction variable of the
/// for loop. It will have a user like `stablehlo.add` or `tensor.extract`.
static bool shouldScalarizeWhileAfterArg(BlockArgument arg, Value condOperand,
                                         Value result) {
  RankedTensorType rtt = cast<RankedTensorType>(arg.getType());
  auto whileOp = arg.getParentRegion()->getParentOfType<scf::WhileOp>();
  Region &before = whileOp.getBefore();
  if (before.getNumArguments() <= arg.getArgNumber() ||
      before.getArgument(arg.getArgNumber()).getType() !=
          rtt.getElementType() ||
      !llvm::all_of(before.getArgument(arg.getArgNumber()).getUsers(),
                    llvm::IsaPred<arith::CmpIOp, tensor::FromElementsOp>))
    return false;

  auto condProducer = condOperand.getDefiningOp<tensor::FromElementsOp>();
  if (!condProducer || condProducer.getElements().size() != 1 ||
      !isa<BlockArgument>(condProducer.getElements().front()))
    return false;

  return rtt.getElementType().isSignlessIntOrIndex() &&
         llvm::count_if(arg.getUsers(),
                        llvm::IsaPred<stablehlo::AddOp, arith::AddIOp,
                                      tensor::ExtractOp>) >= 1;
}

/// Populates the patterns to uplift scf.while to scf.for. This requires
/// detensorization as well as the upstream uplift patterns.
static LogicalResult applyWhileToForUpliftPatterns(Operation *op) {
  RewritePatternSet patterns(op->getContext());
  scf::populateUpliftWhileToForPatterns(patterns);
  scf::WhileOp::getCanonicalizationPatterns(patterns, op->getContext());
  scf::IfOp::getCanonicalizationPatterns(patterns, op->getContext());
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::FromElementsOp::getCanonicalizationPatterns(patterns,
                                                      patterns.getContext());
  tensor::ExtractOp::getCanonicalizationPatterns(patterns,
                                                 patterns.getContext());
  populateSCFDetensorizeWhilePatterns(patterns, shouldScalarizeWhileBeforeArg,
                                      shouldScalarizeWhileAfterArg,
                                      /*benefit=*/10);
  patterns.add<ScalarizeStablehloAddOp,
               ScalarizeStablehloCompareUsedByExtractPattern,
               ScalarizeWhileConditionProducers>(op->getContext());
  return applyPatternsGreedily(op, std::move(patterns));
}

namespace {
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
      return signalPassFailure();
    }
    if (failed(applyWhileToForUpliftPatterns(getOperation()))) {
      emitError(getOperation()->getLoc())
          << "failed to apply while-to-for uplift patterns in "
          << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace