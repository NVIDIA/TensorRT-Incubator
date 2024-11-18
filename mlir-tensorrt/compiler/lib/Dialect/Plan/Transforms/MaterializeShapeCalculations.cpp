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
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Dialect/Plan/Analysis/BoundsAnalysis.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Patterns.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "stablehlo/transforms/Passes.h"
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
  return rewriter.create<plan::WithShapeOp>(loc, v, dims);
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
  return rewriter.create<plan::WithValuesOp>(loc, v, elements);
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

    auto reshapeOp = op.getTensor().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshapeOp)
      return failure();

    if (!reshapeOp.getOperand().getType().hasStaticShape())
      return failure();

    std::optional<SmallVector<int64_t>> coords =
        getConstantIntValues(getAsOpFoldResult(op.getIndices()));
    if (!coords)
      return failure();

    SmallVector<int64_t> resultBasis =
        mlir::computeSuffixProduct(reshapeOp.getType().getShape());
    SmallVector<int64_t> operandBasis =
        mlir::computeSuffixProduct(reshapeOp.getOperand().getType().getShape());

    int64_t linearIndex = mlir::linearize(*coords, resultBasis);
    SmallVector<int64_t> operandCoords =
        mlir::delinearize(linearIndex, operandBasis);

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
        op, reshapeOp.getOperand(),
        llvm::map_to_vector(operandCoords, [&](int64_t c) -> Value {
          return rewriter.create<arith::ConstantIndexOp>(op->getLoc(), c);
        }));

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

/// Rewrite `tensor.dim` of `stablehlo.composite` by inlining a copy of the
/// region (note that the inlining is purely used for shape calculation so
/// we expect it to be elided).
struct ResolveDimOfCompositeGeneric : public OpRewritePattern<tensor::DimOp> {
  ResolveDimOfCompositeGeneric(MLIRContext *ctx,
                               SymbolTableCollection &symbolTable)
      : OpRewritePattern(ctx), symbolTable(symbolTable) {}

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto compositeOp =
        dimOp.getSource().getDefiningOp<stablehlo::CompositeOp>();
    if (!compositeOp)
      return failure();

    std::optional<int64_t> dimIndex = dimOp.getConstantIndex();
    if (!dimIndex)
      return failure();

    // Get the callable.
    auto decomp = symbolTable.lookupSymbolIn<func::FuncOp>(
        dimOp->getParentOfType<ModuleOp>(),
        SymbolRefAttr::get(rewriter.getContext(),
                           compositeOp.getDecomposition()));
    if (!decomp)
      return failure();

    // Clone in the composite body.
    Region &body = decomp.getBody();
    if (!body.hasOneBlock())
      return failure();

    IRMapping mapping;
    mapping.map(body.getArguments(), compositeOp->getOperands());
    for (Operation &op : body.front().without_terminator()) {
      Operation *clone = rewriter.clone(op, mapping);
      mapping.map(op.getResults(), clone->getResults());
    }

    // Replace the dim op.
    rewriter.modifyOpInPlace(dimOp, [&]() {
      OpOperand &operand = dimOp.getSourceMutable();
      operand.assign(mapping.lookup(body.front().getTerminator()->getOperand(
          operand.getOperandNumber())));
    });

    return success();
  }
  SymbolTableCollection &symbolTable;
};

///===----------------------------------------------------------------------===//
// IntegerRange-based Optimization Patterns
//
// The below patterns use our modified IntegerRangeAnalysis
// (`ShapeIntegerRangeAnalysis`) to perform optimizations. The
// ShapeIntegerRangeAnalysis can sometimes infer a narrower range for
// operations like `tensor.dim` due to e.g. special function arg attributes
// that encode the bounds information.
//===----------------------------------------------------------------------===//

/// Replaces `tensor.dim` with `arith.constant` if the range of the `tensor.dim`
/// was narrowed to a constant value.
struct ResolveTrivialDimOpPattern : public OpRewritePattern<tensor::DimOp> {
  ResolveTrivialDimOpPattern(MLIRContext *context, DataFlowSolver &s)
      : OpRewritePattern(context), solver(s) {}

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const override {

    auto *maybeResultRange =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(op.getResult());
    if (!maybeResultRange || maybeResultRange->getValue().isUninitialized())
      return failure();
    const ConstantIntRanges &resultRange =
        maybeResultRange->getValue().getValue();
    const APInt &min = resultRange.umin();
    const APInt &max = resultRange.umax();
    if (min.getZExtValue() != max.getZExtValue())
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, max.getZExtValue());
    return success();
  }

private:
  DataFlowSolver &solver;
};

/// Replaces `plan.with_values` with `arith.constant` if the values are
/// determined to be constant.
struct SimplifyConstantWithValuesPattern
    : public OpRewritePattern<plan::WithValuesOp> {
  SimplifyConstantWithValuesPattern(MLIRContext *context, DataFlowSolver &s)
      : OpRewritePattern(context), solver(s) {}

  LogicalResult matchAndRewrite(plan::WithValuesOp op,
                                PatternRewriter &rewriter) const override {
    auto *maybeResultRange =
        solver.lookupState<TensorValueBoundsLattice>(op.getResult());
    if (!maybeResultRange || maybeResultRange->getValue().isUninitialized())
      return failure();

    std::optional<DenseElementsAttr> constVal =
        maybeResultRange->getValue().getConstantValues(op.getType());
    if (!constVal)
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, *constVal);
    return success();
  }

private:
  DataFlowSolver &solver;
};

/// This rewrite listener handles updating the solver state as the IR is updated
/// in the pattern-based rewrite driver.
class DataFlowListener : public RewriterBase::Listener {
public:
  DataFlowListener(DataFlowSolver &s) : s(s) {}

protected:
  void notifyOperationErased(Operation *op) override {
    s.eraseState(op);
    for (Value res : op->getResults())
      s.eraseState(res);
  }

  DataFlowSolver &s;
};
} // namespace

/// Run the integer range analysis and perform rewrites based on the results.
static LogicalResult applyIntegerRangeBasedOptimizations(Operation *op) {
  // Simplify based on integer range analysis.
  MLIRContext *ctx = op->getContext();
  DataFlowSolver solver;
  SymbolTableCollection symbolTable;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<ShapeIntegerRangeAnalysis>();
  solver.load<TensorValueBoundsAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return failure();

  DataFlowListener listener(solver);
  RewritePatternSet patterns(ctx);
  patterns.add<ResolveTrivialDimOpPattern, SimplifyConstantWithValuesPattern>(
      ctx, solver);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    return failure();

  return success();
}

/// Populate canonicalization patterns for all op types listed as template
/// parameters.
template <typename... Ops>
static void addCanonicalizationPatterns(RewritePatternSet &patterns) {
  (Ops::getCanonicalizationPatterns(patterns, patterns.getContext()), ...);
}

/// Convert 'tensorrt' dialect arg/result bounds attribute into 'plan' bounds
/// attribute.
static Attribute convertArgOrResultAttr(OpBuilder &b, Type type,
                                        tensorrt::ShapeProfileAttr trtAttr,
                                        bool isValueBounds) {
  MLIRContext *ctx = b.getContext();
  if (isValueBounds) {
    Type elementType = mlir::getElementTypeOrSelf(type);
    assert(elementType.isIntOrIndex() && "expected int or index element type");
    SmallVector<int64_t> boundsShape;
    if (auto shapedType = dyn_cast<ShapedType>(type))
      boundsShape = llvm::to_vector(shapedType.getShape());
    auto boundsValueType = RankedTensorType::get(boundsShape, elementType);
    auto convertI64ArrayToDenseElements = [&](ArrayRef<int64_t> i64Vals) {
      return DenseElementsAttr::get(
          boundsValueType,
          llvm::map_to_vector(i64Vals, [&](int64_t i64Val) -> Attribute {
            return b.getIntegerAttr(elementType, i64Val);
          }));
    };
    return plan::BoundsAttr::get(
        ctx, BoundsKind::Value, DenseI64ArrayAttr{}, DenseI64ArrayAttr{},
        convertI64ArrayToDenseElements(trtAttr.getMin()),
        convertI64ArrayToDenseElements(trtAttr.getMax()));
  }
  return plan::BoundsAttr::get(ctx, plan::BoundsKind::Shape, trtAttr.getMin(),
                               trtAttr.getMax());
}

static void convertArgAndResultAttrs(OpBuilder &b, func::FuncOp op) {
  StringRef tensorrtShapeBoundsAttrName =
      mlir::tensorrt::TensorRTDialect::getShapeProfileArgAttrName();
  StringRef tensorrtValueBoundsAttrName =
      mlir::tensorrt::TensorRTDialect::getShapeTensorValueBoundsArgAttrName();

  StringRef planShapeBoundsAttrName =
      mlir::plan::PlanDialect::getShapeBoundsAttrName();
  StringRef planValueBoundsAttrName =
      mlir::plan::PlanDialect::getValueBoundsAttrName();

  for (unsigned idx = 0; idx < op.getNumArguments(); idx++) {
    Type type = op.getArgumentTypes()[idx];
    if (auto attr = op.getArgAttrOfType<tensorrt::ShapeProfileAttr>(
            idx, tensorrtShapeBoundsAttrName)) {
      op.removeArgAttr(idx, tensorrtShapeBoundsAttrName);
      op.setArgAttr(idx, planShapeBoundsAttrName,
                    convertArgOrResultAttr(b, type, attr, false));
    }
    if (auto attr = op.getArgAttrOfType<tensorrt::ShapeProfileAttr>(
            idx, tensorrtValueBoundsAttrName)) {
      op.removeArgAttr(idx, tensorrtValueBoundsAttrName);
      op.setArgAttr(idx, planValueBoundsAttrName,
                    convertArgOrResultAttr(b, type, attr, true));
    }
  }
  for (unsigned idx = 0; idx < op.getNumResults(); idx++) {
    Type type = op.getResultTypes()[idx];
    if (auto attr = op.getResultAttrOfType<tensorrt::ShapeProfileAttr>(
            idx, tensorrtShapeBoundsAttrName)) {
      op.removeArgAttr(idx, tensorrtShapeBoundsAttrName);
      op.setResultAttr(idx, planShapeBoundsAttrName,
                       convertArgOrResultAttr(b, type, attr, false));
    }
    if (auto attr = op.getResultAttrOfType<tensorrt::ShapeProfileAttr>(
            idx, tensorrtValueBoundsAttrName)) {
      op.removeArgAttr(idx, tensorrtValueBoundsAttrName);
      op.setResultAttr(idx, planValueBoundsAttrName,
                       convertArgOrResultAttr(b, type, attr, true));
    }
  }
}

//===----------------------------------------------------------------------===//
// MaterializeShapeCalculationsPass
//===----------------------------------------------------------------------===//

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

    /// Convert `tensorrt` dialect bounds func arg/result attributes.
    /// TODO: should this be moved to a dedicated pass?
    op->walk([&](func::FuncOp func) {
      convertArgAndResultAttrs(rewriter, func);
      return WalkResult::skip();
    });

    // Run TensorKindAnalysis and populate the `plan.with_shape|with_values`
    // operations.
    {
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
            << "failed to add shape reification operations in "
            << getArgument();
        return signalPassFailure();
      }
    }

    LLVM_DEBUG(DBGS() << "after adding shape reification operations:\n"
                      << *op << "\n");

    // Apply patterns that propagate the scalar IR chain associated with
    // `plan.(with_values|with_shape)` operations.
    FrozenRewritePatternSet patterns = [&] {
      RewritePatternSet patterns_(ctx);
      memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns_);
      stablehlo_ext::populateStableHloAbsorbTensorCastPatterns(patterns_);
      stablehlo::populateStablehloCanonicalizeDynamismPatterns(&patterns_, ctx);
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
      patterns_.add<
        ResolveDimOfCompositeGeneric
      >(ctx, symbolTable);
      // clang-format on
      return patterns_;
    }();

    auto applySimplificationPatterns = [&]() -> LogicalResult {
      if (failed(applyPatternsAndFoldGreedily(op, patterns)))
        return emitError(op->getLoc())
               << "failed to run patterns in " << getArgument();
      return success();
    };

    constexpr unsigned kSimplificationRounds = 2;
    for (unsigned i = 0; i < kSimplificationRounds; i++) {
      if (failed(applySimplificationPatterns()))
        return signalPassFailure();
      DominanceInfo domInfo;
      mlir::eliminateCommonSubExpressions(rewriter, domInfo, getOperation());

      // Simplify based on integer range analysis.
      if (failed(applyIntegerRangeBasedOptimizations(op)))
        return signalPassFailure();
    }
  }
};

} // namespace
