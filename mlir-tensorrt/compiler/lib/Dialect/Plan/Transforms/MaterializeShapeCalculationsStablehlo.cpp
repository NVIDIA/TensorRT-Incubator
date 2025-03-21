//===- MaterializeShapeCalculationsStablehlo.cpp --------------------------===//
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
#ifdef MLIR_TRT_ENABLE_HLO
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/MaterializeShapeCalculations.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Patterns.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "stablehlo/conversions/linalg/transforms/MapStablehloToScalarOp.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::plan;

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

    if (!reshapeOp.getOperand().getType().hasStaticShape())
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

void plan::populateMaterializeShapeCalculationsStablehloPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTable) {
  stablehlo_ext::populateStableHloAbsorbTensorCastPatterns(patterns);
  stablehlo::populateStablehloCanonicalizeDynamismPatterns(
      &patterns, patterns.getContext());
  patterns.add<SimplifyExtractOfConcat, SimplifyExtractOfDimSize,
               SimplifyExtractOfEwise, SimplifyExtractOfReshape,
               SimplifyExtractOfSlice>(patterns.getContext());
  patterns.add<ResolveDimOfCompositeGeneric>(patterns.getContext(),
                                             symbolTable);
}
#endif // MLIR_TRT_ENABLE_HLO
