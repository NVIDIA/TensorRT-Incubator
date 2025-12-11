//===- StablehloToLinalg.cpp ----------------------------------------------===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/conversions/linalg/transforms/Rewriters.h"
#include "stablehlo/conversions/linalg/transforms/TypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
#define GEN_PASS_DEF_STABLEHLOTOLINALGPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// Strip algorithm attribute from dot_general when safe for Linalg conversion.
// Remove algorithm attribute from DotGeneralOp if it represents a standard
// configuration that doesn't require special handling. This includes:
// 1. Standard precision matching actual types
// 2. Reduced precision computations (bf16/tf32) on f32 inputs
// 3. Mixed precision with valid accumulation types
static LogicalResult stripDotGeneralAlgorithm(stablehlo::DotGeneralOp op) {
  auto algorithm = op.getAlgorithm();
  if (!algorithm)
    return failure();

  // Extract element types using modern cast API
  auto lhsElemType = op.getLhs().getType().getElementType();
  auto rhsElemType = op.getRhs().getType().getElementType();
  auto resultElemType = op.getType().getElementType();

  // Get algorithm precision types
  Type lhsPrecision = algorithm->getLhsPrecisionType();
  Type rhsPrecision = algorithm->getRhsPrecisionType();
  Type accType = algorithm->getAccumulationType();

  // Helper to check if types form a valid reduced precision pattern
  auto isValidReducedPrecision = [](Type precision, Type accumulator) {
    return (precision.isF16() || precision.isBF16() || precision.isTF32()) &&
           accumulator.isF32();
  };

  // Check for valid precision configurations:
  // 1. Standard: precision matches input types, accumulator matches output
  // 2. Reduced: f32 inputs computed at lower precision (f16/bf16/tf32)
  // 3. Mixed: lower precision inputs accumulated to higher precision

  bool validPrecisionConfig = false;

  // Both sides must use the same precision
  if (lhsPrecision == rhsPrecision) {
    // Standard case: precision matches input types
    if (lhsPrecision == lhsElemType && rhsPrecision == rhsElemType) {
      validPrecisionConfig =
          accType == resultElemType || // Accumulator matches result
          isValidReducedPrecision(lhsPrecision,
                                  accType); // Or valid reduced pattern
    }
    // Reduced precision on f32 inputs
    else if (lhsElemType.isF32() && rhsElemType.isF32() &&
             resultElemType.isF32()) {
      validPrecisionConfig = isValidReducedPrecision(lhsPrecision, accType);
    }
    // Mixed precision: f16 inputs -> f32 result
    else if (lhsElemType.isF16() && rhsElemType.isF16() &&
             lhsPrecision.isF16() && resultElemType.isF32()) {
      validPrecisionConfig = accType.isF32();
    }
  }

  if (!validPrecisionConfig)
    return failure();

  if (algorithm->getLhsComponentCount() != 1 ||
      algorithm->getRhsComponentCount() != 1)
    return failure();

  // Create new op without algorithm attribute
  OpBuilder builder(op);
  auto newOp = builder.create<stablehlo::DotGeneralOp>(
      op.getLoc(), op.getType(), op.getLhs(), op.getRhs(),
      op.getDotDimensionNumbers(), op.getPrecisionConfig().value_or(nullptr),
      /*algorithm=*/nullptr);

  op->replaceAllUsesWith(newOp);
  op.erase();

  return success();
}

/// Match indexing map '-DN + const' and return the dimension in 'dim'.
static bool matchReverseIndexingExpr(AffineExpr expr, unsigned &dim,
                                     int64_t &offset) {
  // First try to match: (-1 * dim) + offset
  if (auto addExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (addExpr.getKind() == mlir::AffineExprKind::Add) {
      auto mulExpr = dyn_cast<AffineBinaryOpExpr>(addExpr.getLHS());
      if (!mulExpr || mulExpr.getKind() != mlir::AffineExprKind::Mul)
        return false;

      // Match the negative one.
      auto constExpr = dyn_cast<AffineConstantExpr>(mulExpr.getRHS());
      if (!constExpr || constExpr.getValue() != -1)
        return false;

      // Match dimension expr
      auto dimExpr = dyn_cast<AffineDimExpr>(mulExpr.getLHS());
      if (!dimExpr)
        return false;

      // Match the constant of the add.
      auto constOffset = dyn_cast<AffineConstantExpr>(addExpr.getRHS());
      if (!constOffset)
        return false;

      dim = dimExpr.getPosition();
      offset = constOffset.getValue();
      return true;
    }
  }

  // Also try to match just: (-1 * dim) with implicit offset 0
  if (auto mulExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (mulExpr.getKind() == mlir::AffineExprKind::Mul) {
      // Match the negative one.
      auto constExpr = dyn_cast<AffineConstantExpr>(mulExpr.getRHS());
      if (!constExpr || constExpr.getValue() != -1)
        return false;

      // Match dimension expr
      auto dimExpr = dyn_cast<AffineDimExpr>(mulExpr.getLHS());
      if (!dimExpr)
        return false;

      dim = dimExpr.getPosition();
      offset = 0; // Implicit offset is 0 for plain -d0
      return true;
    }
  }

  return false;
}

/// Matching indexing map where each result is either a dimension iterator or a
/// reversal of an iterator.
static bool matchReverseIndexingMap(OpOperand *inputOperand, AffineMap map,
                                    ArrayRef<int64_t> staticLoopRanges) {
  if (map.getNumResults() != staticLoopRanges.size())
    return false;

  SmallVector<AffineExpr> newResults;
  bool hasReverse{false};
  for (auto [idx, expr] : llvm::enumerate(map.getResults())) {
    unsigned dim{0};
    int64_t offset{0};
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
      continue;

    if (!matchReverseIndexingExpr(expr, dim, offset))
      return false;
    if (staticLoopRanges[dim] != offset + 1)
      return false;
    hasReverse = true;
  }
  return hasReverse;
}

namespace {

/// Rewrite 'linalg.generic' that have an iteration map containing results like
/// '-d0 + dimSize-1'. These are currently generated by e.g. stablehlo.reverse,
/// but the upstream linalg transformations have poor support for correctly
/// handling the maps. Instead, we can drop the input and use `linalg.index` to
/// gather the required value. See:
/// https://github.com/llvm/llvm-project/issues/113021.
struct FixupReverseLinalgGenericPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> staticLoopRanges = op.getStaticLoopRanges();
    SmallVector<std::tuple<Value, AffineMap, unsigned>> operandsToReplace;

    for (auto [idx, operand] : llvm::enumerate(op.getDpsInputOperands())) {
      if (op.isScalar(operand))
        continue;
      AffineMap indexingMap = op.getMatchingIndexingMap(operand);
      if (!matchReverseIndexingMap(operand, indexingMap, staticLoopRanges))
        continue;
      BlockArgument matchingArg = op.getRegionInputArgs()[idx];
      if (matchingArg.use_empty())
        continue;
      operandsToReplace.push_back({operand->get(), indexingMap, idx});
    }

    if (operandsToReplace.empty())
      return failure();

    rewriter.setInsertionPointToStart(op.getBlock());
    SmallVector<OpFoldResult> point;
    for (auto idx : llvm::seq<unsigned>(0, op.getNumLoops()))
      point.push_back(
          rewriter.create<linalg::IndexOp>(op.getLoc(), idx).getResult());

    for (auto [operand, indexingMap, idx] : operandsToReplace) {
      BlockArgument matchingArg = op.getRegionInputArgs()[idx];
      // Materialize the extract.
      SmallVector<Value> coords =
          llvm::map_to_vector(affine::makeComposedFoldedMultiResultAffineApply(
                                  rewriter, op.getLoc(), indexingMap, point),
                              [&](OpFoldResult result) {
                                return mlir::getValueOrCreateConstantIndexOp(
                                    rewriter, op.getLoc(), result);
                              });
      Value extracted =
          rewriter.create<tensor::ExtractOp>(op.getLoc(), operand, coords);
      rewriter.replaceAllUsesWith(matchingArg, extracted);
      rewriter.finalizeOpModification(op);
    }
    return success();
  }
};

struct ConvertStablehloGetDimSizePattern
    : public OpConversionPattern<stablehlo::GetDimensionSizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(stablehlo::GetDimensionSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value dim =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), op.getDimension());
    Value dimSize =
        rewriter.create<tensor::DimOp>(op.getLoc(), adaptor.getOperand(), dim);
    // Cast to target element type.
    Type targetElementType = op.getType().getElementType();
    if (targetElementType != dimSize.getType()) {
      if (targetElementType.isSignlessInteger())
        dimSize = rewriter.create<arith::IndexCastUIOp>(
            op.getLoc(), targetElementType, dimSize);
      else
        return rewriter.notifyMatchFailure(op,
                                           "unsupported target element type");
    }
    auto fromElements = rewriter.create<tensor::FromElementsOp>(
        op.getLoc(), op.getType(), dimSize);
    rewriter.replaceOp(op, fromElements.getResult());
    return success();
  }
};

class StablehloToLinalgPass
    : public impl::StablehloToLinalgPassBase<StablehloToLinalgPass> {
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override {
    target = std::make_shared<ConversionTarget>(*context);
    target->addLegalDialect<
        bufferization::BufferizationDialect, arith::ArithDialect,
        complex::ComplexDialect, linalg::LinalgDialect, math::MathDialect,
        tensor::TensorDialect, scf::SCFDialect, shape::ShapeDialect>();
    target->addLegalOp<UnrealizedConversionCastOp>();
    target->addDynamicallyLegalDialect<stablehlo::StablehloDialect>(
        [](Operation *op) -> std::optional<bool> {
          if (isa<stablehlo::ScatterOp>(op))
            return std::nullopt;

          if (isa<stablehlo::CustomCallOp>(op))
            return std::nullopt;

          // Check if parent is op like `stablehlo.reduce`.
          Operation *parent = op->getParentOp();
          while (parent) {
            if (isa<FunctionOpInterface>(parent))
              return false;
            if (isa_and_present<stablehlo::StablehloDialect>(
                    parent->getDialect()))
              return true;
            parent = parent->getParentOp();
          }
          return false;
        });

    patterns = [&] {
      RewritePatternSet patterns_(context);
      populateStablehloToLinalgConversionPatterns(context, converter,
                                                  &patterns_,
                                                  /*enablePrimitiveOps=*/false,
                                                  /*enableSparseOps=*/false);
      patterns_.add<ConvertStablehloGetDimSizePattern>(context);
      return patterns_;
    }();

    cleanupPatterns = [&] {
      RewritePatternSet cleanupPatterns_(context);
      cleanupPatterns_.add<FixupReverseLinalgGenericPattern>(context);
      linalg::populateEraseUnusedOperandsAndResultsPatterns(cleanupPatterns_);
      return cleanupPatterns_;
    }();

    return success();
  }

  void runOnOperation() override {
    // First, strip algorithm attributes from dot_general ops
    getOperation()->walk([&](stablehlo::DotGeneralOp dotOp) {
      // Attempt to strip the algorithm attribute; failures are silently ignored
      // as they indicate the op should keep its algorithm attribute
      (void)stripDotGeneralAlgorithm(dotOp);
    });

    // Then apply the conversion patterns
    if (failed(applyPartialConversion(getOperation(), *target, patterns))) {
      emitError(getOperation()->getLoc(), "failed to apply conversion in ")
          << getArgument();
      return signalPassFailure();
    }

    if (failed(applyPatternsGreedily(getOperation(), cleanupPatterns))) {
      emitError(getOperation()->getLoc(), "failed to apply conversion in ")
          << getArgument();
      return signalPassFailure();
    }
  }

private:
  std::shared_ptr<ConversionTarget> target;
  FrozenRewritePatternSet patterns;
  FrozenRewritePatternSet cleanupPatterns;
  stablehlo::LinalgTypeConverter converter;
};
} // namespace
