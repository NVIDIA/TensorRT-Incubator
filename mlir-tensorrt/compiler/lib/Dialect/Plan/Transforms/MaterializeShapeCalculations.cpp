//===- MaterializeShapeCalculations.cpp -----------------------------------===//
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
/// Implementation of the `plan-materialize-shape-calculations` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Analysis/TensorKindAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/Analysis/BoundsAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/conversions/linalg/transforms/MapStablehloToScalarOp.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "plan-materialize-shape-calculations"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "] "

namespace mlir::plan {
#define GEN_PASS_DEF_MATERIALIZESHAPECALCULATIONSPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

/// Create the `plan.with_shape` op that operates on `v`. This may require
/// materialization of `tensor.dim` operations.
static WithShapeOp createWithShapeOp(RewriterBase &rewriter, Location loc,
                                     Value v) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfterValue(v);
  RankedTensorType rtt = cast<RankedTensorType>(v.getType());
  SmallVector<Value> dims;
  dims.reserve(rtt.getRank());
  for (int64_t i = 0, e = rtt.getRank(); i < e; i++) {
    auto dimOp = rewriter.create<tensor::DimOp>(
        loc, v, rewriter.create<arith::ConstantIndexOp>(loc, i));
    dims.push_back(dimOp);
  }
  return rewriter.create<plan::WithShapeOp>(loc, v.getType(), v, dims);
}

/// If `v` is dynamically shaped, replace uses with `plan.with_shape` of `v`.
static void addWithShapeOpAndUpdateUses(RewriterBase &rewriter, Value v) {
  // If it is statically shaped or if all our users are already
  // 'plan.with_shape' or 'tensor.dim', then no need to do anything.
  RankedTensorType rtt = llvm::dyn_cast<RankedTensorType>(v.getType());
  if (!rtt || rtt.hasStaticShape() ||
      llvm::all_of(v.getUsers(), [](Operation *user) {
        return isa<WithShapeOp, tensor::DimOp>(user);
      }))
    return;

  // Create the `plan.with_shape` op.
  auto withOp = createWithShapeOp(rewriter, v.getLoc(), v);
  rewriter.replaceUsesWithIf(v, withOp.getResult(), [&](OpOperand &use) {
    return !isa<WithShapeOp, tensor::DimOp>(use.getOwner());
  });
}

static WithValuesOp
createWithValuesOp(RewriterBase &rewriter, Location loc, Value v,
                   llvm::SmallPtrSet<Operation *, 8> &newUsers) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfterValue(v);
  RankedTensorType rtt = cast<RankedTensorType>(v.getType());
  assert(rtt.hasStaticShape() && "expected statically shaped operand");
  SmallVector<Value> elements;
  elements.reserve(rtt.getNumElements());
  SmallVector<int64_t> basis = mlir::computeSuffixProduct(rtt.getShape());
  for (int64_t i = 0, e = rtt.getNumElements(); i < e; i++) {
    SmallVector<Value> coord = llvm::map_to_vector(
        mlir::delinearize(i, basis), [&](int64_t val) -> Value {
          return rewriter.create<arith::ConstantIndexOp>(loc, val);
        });
    auto extractOp = rewriter.create<tensor::ExtractOp>(loc, v, coord);
    newUsers.insert(extractOp);
    elements.push_back(extractOp);
  }
  return rewriter.create<plan::WithValuesOp>(loc, v.getType(), v, elements);
}

/// For each tensor-typed value, create a `plan.with_shape` operation and
/// materialize the `tensor.dim` operations to form the shape. The below rewrite
/// patterns can then be run in order to create an independent chain of scalar
/// operations (`arith` ops, `tensort.extract`, etc.) that materialize the shape
/// for each dynamic tensor value.
///
/// These chains can then be easily clustered and factored out either into
/// independent shape calculation functions or left in-place.
static LogicalResult addWithShapeOps(RewriterBase &rewriter,
                                     DataFlowSolver &solver, Operation *op) {
  OpBuilder::InsertionGuard g(rewriter);
  WalkResult result = op->walk([&](Operation *op) {
    // Don't insert redundant `plan.with_shape` or `plan.with_values` ops and
    // no need to bother with constants.
    if (isa<WithShapeOp, WithValuesOp>(op) ||
        op->hasTrait<OpTrait::ConstantLike>())
      return WalkResult::skip();

    if (auto func = dyn_cast<FunctionOpInterface>(op)) {
      for (BlockArgument arg : func.getArguments())
        addWithShapeOpAndUpdateUses(rewriter, arg);
    }

    if (!op->hasTrait<OpTrait::IsTerminator>() &&
        !isa_and_nonnull<stablehlo::StablehloDialect, scf::SCFDialect,
                         tensor::TensorDialect,
                         bufferization::BufferizationDialect>(
            op->getDialect()) &&
        !isa<ReifyRankedShapedTypeOpInterface>(op))
      return WalkResult::advance();

    rewriter.setInsertionPoint(op);
    for (OpOperand &v : op->getOpOperands()) {
      // If this may be a shape tensor/host tensor, then create the
      // `with_values` op.
      if (TensorValueBoundsAnalysis::shouldAnalyzeValueBounds(v.get()) &&
          !matchPattern(v.get(), m_Constant())) {
        const auto *kind = solver.lookupState<TensorKindLattice>(v.get());
        if (!kind || kind->getValue().isUninitialized() ||
            !kind->getValue().isHostVisible())
          continue;
        llvm::SmallPtrSet<Operation *, 8> newUsers;
        WithValuesOp withOp =
            createWithValuesOp(rewriter, op->getLoc(), v.get(), newUsers);
        newUsers.insert(withOp);
        rewriter.replaceUsesWithIf(
            v.get(), withOp.getResult(),
            [&](OpOperand &use) { return !newUsers.contains(use.getOwner()); });
        continue;
      }
    }

    rewriter.setInsertionPointAfter(op);
    for (Value result : op->getResults())
      addWithShapeOpAndUpdateUses(rewriter, result);

    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

namespace {

struct SimplifyExtractOfShapeOf : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto shapeOfOp = op.getTensor().getDefiningOp<shape::ShapeOfOp>();
    if (!shapeOfOp)
      return failure();

    rewriter.replaceOpWithNewOp<tensor::DimOp>(op, shapeOfOp.getOperand(),
                                               op.getIndices().front());
    return success();
  }
};

} // namespace

/// For each value in `operands`, extract the scalar value located at `indices`.
static SmallVector<Value> getScalarOperands(RewriterBase &rewriter,
                                            Location loc, OperandRange operands,
                                            ValueRange indices) {
  SmallVector<Value> result;
  for (Value operand : operands) {
    result.push_back(rewriter.create<tensor::ExtractOp>(loc, operand, indices));
  }
  return result;
}

/// Given a `producer` operation (StableHLO elementwise-mappable
/// operation) and a index into the result tensor (`indices`), produce an
/// equivalent scalar operation by extracting the relevant scalars from the
/// operands and creating equivalent "standard" dialect operations (e.g.
/// upstream arith ops). Exactly which operations are materialized is determined
/// by the `StablehloOpToStdScalar` utilities upstream.
/// Note that these routines from StableHLO-to-Linalg produce a particular
/// behavior for signed integer division/remainder functions. Division by 0
/// produces "-1", for example.
/// TODO: decide how we want to handle overflow behavior
/// See also https://github.com/openxla/stablehlo/issues/1157.
static FailureOr<Value> mapOpToScalar(RewriterBase &rewriter,
                                      Operation *producer, ValueRange indices) {

  auto mapStablehloToStdScalar = [&](auto op) -> FailureOr<Value> {
    return stablehlo::StablehloOpToStdScalarOp::mapOp(
        op, mlir::getElementTypeOrSelf(op.getType()),
        getScalarOperands(rewriter, op.getLoc(), producer->getOperands(),
                          indices),
        &rewriter);
  };

  return llvm::TypeSwitch<Operation *, FailureOr<Value>>(producer)
      .Case([&](stablehlo::ShiftRightLogicalOp op) {
        return mapStablehloToStdScalar(op);
      })
      .Case([&](stablehlo::SignOp op) { return mapStablehloToStdScalar(op); })
      .Case(
          [&](stablehlo::SubtractOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::XorOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::AbsOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::AddOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::AndOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::BitcastConvertOp op) {
        return mapStablehloToStdScalar(op);
      })
      .Case([&](stablehlo::ClampOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::ClzOp op) { return mapStablehloToStdScalar(op); })
      .Case(
          [&](stablehlo::CompareOp op) { return mapStablehloToStdScalar(op); })
      .Case(
          [&](stablehlo::ConvertOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::DivOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::MaxOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::MinOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::MulOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::NegOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::NotOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::OrOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::PopulationCountOp op) {
        return mapStablehloToStdScalar(op);
      })
      .Case([&](stablehlo::RemOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::SelectOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::ShiftLeftOp op) {
        return mapStablehloToStdScalar(op);
      })
      .Case([&](stablehlo::ShiftRightArithmeticOp op) {
        return mapStablehloToStdScalar(op);
      })
      .Case([&](stablehlo::ShiftRightLogicalOp op) {
        return mapStablehloToStdScalar(op);
      })
      .Case([&](stablehlo::SignOp op) { return mapStablehloToStdScalar(op); })
      .Case(
          [&](stablehlo::SubtractOp op) { return mapStablehloToStdScalar(op); })
      .Case([&](stablehlo::XorOp op) { return mapStablehloToStdScalar(op); })

      .Default([&](Operation *op) { return failure(); });
}

namespace {

/// Simplify `tensor.extract( stablehlo.[elementwise op] )` into
/// `arith.[elementwise|unary](tensor.extract(operands)...)`.
struct SimplifyExtractOfEwise : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    Operation *producer = op.getTensor().getDefiningOp();
    if (!producer)
      return failure();
    if (!producer->hasTrait<OpTrait::Elementwise>() &&
        !isa<stablehlo::SelectOp>(producer))
      return failure();

    FailureOr<Value> scalar =
        mapOpToScalar(rewriter, producer, op.getIndices());
    if (failed(scalar))
      return rewriter.notifyMatchFailure(
          op, "failed to map stablehlo op to scalar arithmetic op");
    rewriter.replaceOp(op, *scalar);
    return success();
  }
};

/// Replace `tensor.extract(stablehlo.concatenate(...))` with the appropriate
/// extraction from one of the concatenation operands.
struct SimplifyExtractOfConcat : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    auto concatOp = op.getTensor().getDefiningOp<stablehlo::ConcatenateOp>();
    if (!concatOp)
      return failure();

    int64_t dimension = concatOp.getDimension();

    std::optional<SmallVector<int64_t>> coords =
        getConstantIntValues(getAsOpFoldResult(op.getIndices()));
    if (!coords)
      return failure();

    // Find the segment index in which the linear index belongs.
    Value operand{};
    for (Value v : concatOp.getOperands()) {
      auto operandType = cast<RankedTensorType>(v.getType());
      if (operandType.getDimSize(dimension) <= (*coords)[dimension]) {
        (*coords)[dimension] -= operandType.getDimSize(dimension);
        continue;
      }
      operand = v;
      break;
    }

    // Find linear offset within in the operand shape.
    assert(operand && "expected valid Value");
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
        op, operand, llvm::map_to_vector(*coords, [&](int64_t c) -> Value {
          return rewriter.create<arith::ConstantIndexOp>(op->getLoc(), c);
        }));

    return success();
  }
};

/// Replaces `tensor.extract(tensor.reshape)` with an extraction from the
/// reshape's operand.
struct SimplifyExtractOfReshape : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    auto reshapeOp = op.getTensor().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshapeOp)
      return failure();

    std::optional<SmallVector<int64_t>> coords =
        getConstantIntValues(getAsOpFoldResult(op.getIndices()));
    if (!coords)
      return failure();

    // Get lienar coords.
    SmallVector<int64_t> resultBasis =
        mlir::computeSuffixProduct(reshapeOp.getType().getShape());
    SmallVector<int64_t> operandBasis =
        mlir::computeSuffixProduct(reshapeOp.getOperand().getType().getShape());

    int64_t lienarIndex = mlir::linearize(*coords, resultBasis);
    SmallVector<int64_t> operandCoords =
        mlir::delinearize(lienarIndex, operandBasis);

    // Find linear offset within in the operand shape.
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
        op, reshapeOp.getOperand(),
        llvm::map_to_vector(operandCoords, [&](int64_t c) -> Value {
          return rewriter.create<arith::ConstantIndexOp>(op->getLoc(), c);
        }));

    return success();
  }
};

/// Replaces `tensor.dim(stablehlo.composite(x))` with
/// `tensor.dim(x)` in the case where the attribute 'is_pointwise' is present.
/// The attribute may be added in cases where stablehlo.composite was created at
/// the frontend or by a separate compiler pass.
struct ResolveDimOfCompositeOp : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const override {
    auto compositeOp = op.getSource().getDefiningOp<stablehlo::CompositeOp>();
    if (!compositeOp ||
        !compositeOp.getCompositeAttributesAttr().contains("is_pointwise"))
      return failure();
    rewriter.replaceOpWithNewOp<tensor::DimOp>(
        op, compositeOp->getOperands().front(), op.getIndex());
    return success();
  }
};

/// Replaces `tensor.extract(stablehlo.get_dimension_size(...))` with a
/// `tensor.dim` operation.
struct SimplifyExtractOfDimSize : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto dimOp = op.getTensor().getDefiningOp<stablehlo::GetDimensionSizeOp>();
    if (!dimOp)
      return failure();

    // Find linear offset within in the operand shape.
    Value dimExtent =
        rewriter.create<tensor::DimOp>(op.getLoc(), dimOp.getOperand(),
                                       rewriter.create<arith::ConstantIndexOp>(
                                           op->getLoc(), dimOp.getDimension()));

    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, op.getType(),
                                                    dimExtent);

    return success();
  }
};

/// Rewrites `tensor.extract(stablehlo.slice(x, ...), coords...)` with
/// `tensor.extract(x, adjusted_coords...)` assuming that `coords...` of
/// the `tensor.extract` are constants.
struct SimplifyExtractOfSlice : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto producer = op.getTensor().getDefiningOp<stablehlo::SliceOp>();
    if (!producer)
      return failure();

    /// Retrieve constant extract coordinates.
    SmallVector<APInt> indices;
    indices.reserve(op.getIndices().size());
    if (!llvm::all_of(op.getIndices(), [&](Value v) {
          return matchPattern(v, m_ConstantInt(&indices.emplace_back()));
        }))
      return failure();

    /// Adjust the coordinates based on the slice.
    ArrayRef<int64_t> start = producer.getStartIndices();
    ArrayRef<int64_t> stride = producer.getStrides();
    SmallVector<Value> newIndices;
    newIndices.reserve(indices.size());
    for (auto [extractCoord, startCoord, strideCoord] :
         llvm::zip_equal(indices, start, stride)) {
      extractCoord *= strideCoord;
      extractCoord += startCoord;
      newIndices.push_back(rewriter.create<arith::ConstantIndexOp>(
          op.getLoc(), extractCoord.getSExtValue()));
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.getTensorMutable().assign(producer.getOperand());
      op.getIndicesMutable().assign(newIndices);
    });
    return success();
  }
};

/// Replaces `tensor.dim(plan.with_shape(x, shape), dimNum)` with the
/// appropriate shape scalar (if dimNum is static) or with `tensor.dim(x,
/// dimNum)` if `dimNum` is dynamic.
struct SimplifyDimOfWithShapeOp : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const override {
    auto withShapeOp = op.getSource().getDefiningOp<plan::WithShapeOp>();
    if (!withShapeOp)
      return failure();

    OpFoldResult dim = op.getDimension();
    if (std::optional<int64_t> staticDim = getConstantIntValue(dim)) {
      rewriter.replaceOp(op, withShapeOp.getShape()[*staticDim]);
      return success();
    }
    rewriter.replaceOpWithNewOp<tensor::DimOp>(op, withShapeOp.getOperand(),
                                               op.getIndex());
    return success();
  }
};

/// Replaces `tensor.extract(plan.with_values(x, shape), coords...)` with
/// `tensor.extract(x, cooords...)`.
struct SimplifyExtractOfWithValuesRewrite
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto withValuesOp = op.getTensor().getDefiningOp<plan::WithValuesOp>();
    if (!withValuesOp)
      return failure();
    rewriter.modifyOpInPlace(
        op, [&]() { op.getTensorMutable().assign(withValuesOp.getOperand()); });
    return success();
  }
};

/// Simplifies `tensor.extract` from `tensor.cast` as long as the cast
/// operation is simply changing dimensions from known-to-unknown (or vice
/// versa) or changing the encoding attribute.
struct SimplifyExtractOfCast : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto castOp = op.getTensor().getDefiningOp<tensor::CastOp>();
    if (!castOp || !isa<RankedTensorType>(castOp.getType()) ||
        !isa<RankedTensorType>(castOp.getSource().getType()))
      return failure();
    rewriter.modifyOpInPlace(
        op, [&]() { op.getTensorMutable().assign(castOp.getSource()); });
    return success();
  }
};

/// Replaces redundant `arith.maxsi` operations with a single op. For example,
/// replaces `arith.maxsi(arith.maxsi(x, y), (y|x)) with `arith.maxsi(x, y)`.
struct SimplifyRedundantMaxSI : public OpRewritePattern<arith::MaxSIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MaxSIOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsProducer = op.getLhs().getDefiningOp<arith::MaxSIOp>();
    if (!lhsProducer)
      return failure();

    if (!llvm::is_contained(lhsProducer->getOperands(), op.getRhs()))
      return failure();

    rewriter.replaceOp(op, lhsProducer);
    return success();
  }
};

/// Rewrite `tensor.dim` of an operation that implements the
/// InferShapedTypeOpInterface. This is modified from the upstream version since
/// the StableHlo op interface implementations and the equivalent pattern
/// upstream don't seem to agree on the necessity of the shape tensors having
/// type `index`.
struct ResolveDimOfInferShapedTypePattern
    : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto shapedTypeOp =
        dimOp.getSource().getDefiningOp<InferShapedTypeOpInterface>();
    if (!shapedTypeOp)
      return failure();

    std::optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex)
      return failure();

    SmallVector<Value> reifiedResultShapes;
    if (failed(shapedTypeOp.reifyReturnTypeShapes(
            rewriter, shapedTypeOp->getOperands(), reifiedResultShapes)))
      return failure();

    assert(reifiedResultShapes.size() == shapedTypeOp->getNumResults() &&
           "expected equal number of result types");
    Value resultShape = reifiedResultShapes[cast<OpResult>(dimOp.getSource())
                                                .getResultNumber()];
    auto resultShapeType = dyn_cast<RankedTensorType>(resultShape.getType());
    if (!resultShapeType ||
        !isa<IntegerType, IndexType>(resultShapeType.getElementType()))
      return failure();

    Value extracted = rewriter.create<tensor::ExtractOp>(
        dimOp.getLoc(), resultShape, dimOp.getIndex());
    if (extracted.getType().isIndex()) {
      rewriter.replaceOp(dimOp, extracted);
      return success();
    }
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
        dimOp, rewriter.getIndexType(), extracted);
    return success();
  }
};

} // namespace

/// Populate canonicalization patterns for all op types listed as template
/// parameters.
template <typename... Ops>
static void addCanonicalizationPatterns(RewritePatternSet &patterns) {
  (Ops::getCanonicalizationPatterns(patterns, patterns.getContext()), ...);
}

namespace {
class MaterializeShapeCalculationsPass
    : public plan::impl::MaterializeShapeCalculationsPassBase<
          MaterializeShapeCalculationsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<TensorKindAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(op))) {
      emitError(op->getLoc()) << "failed to run TensorKindAnalysis";
      return signalPassFailure();
    }

    if (failed(addWithShapeOps(rewriter, solver, op))) {
      emitError(op->getLoc())
          << "failed to add shape reification operations in " << getArgument();
      return signalPassFailure();
    }

    LLVM_DEBUG(DBGS() << "after adding shape reification operations:\n"
                      << *op << "\n");

    // Apply patterns that propagate the scalar IR chain associated with
    // `plan.(with_values|with_shape)` operations.
    FrozenRewritePatternSet patterns = [&] {
      RewritePatternSet patterns_(ctx);
      memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns_);
      // clang-format off
      addCanonicalizationPatterns<
        arith::AndIOp,
        arith::AddIOp,
        arith::CmpIOp,
        arith::DivSIOp,
        arith::ExtSIOp,
        arith::IndexCastOp,
        arith::MaxSIOp,
        arith::MulIOp,
        arith::OrIOp,
        arith::RemSIOp,
        arith::SelectOp,
        arith::SubIOp,
        arith::TruncIOp,
        arith::XOrIOp,
        plan::WithShapeOp,
        plan::WithValuesOp,
        tensor::ExtractOp,
        tensor::FromElementsOp
      >(patterns_);
      patterns_.add<
        ResolveDimOfInferShapedTypePattern,
        ResolveDimOfCompositeOp,
        SimplifyDimOfWithShapeOp,
        SimplifyExtractOfCast,
        SimplifyExtractOfConcat,
        SimplifyExtractOfDimSize,
        SimplifyExtractOfEwise,
        SimplifyExtractOfReshape,
        SimplifyExtractOfShapeOf,
        SimplifyExtractOfSlice,
        SimplifyExtractOfWithValuesRewrite,
        SimplifyRedundantMaxSI
      >(ctx);
      // clang-format on
      return patterns_;
    }();

    auto applySimplifications = [&]() -> LogicalResult {
      if (failed(applyPatternsAndFoldGreedily(op, patterns)))
        return emitError(op->getLoc())
               << "failed to run patterns in " << getArgument();
      return success();
    };

    constexpr unsigned kSimplificationRounds = 2;
    for (unsigned i = 0; i < kSimplificationRounds; i++) {
      if (failed(applySimplifications()))
        return signalPassFailure();
      DominanceInfo domInfo;
      mlir::eliminateCommonSubExpressions(rewriter, domInfo, getOperation());
    }
  }
};

} // namespace
