//===- RefineTypes.cpp ----------------------------------------------------===//
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
/// Implementation of `plan-refine-types`.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/Utils/ShapeUtils.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StableHloExt/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/StablehloRefineShapes.h"

namespace mlir::plan {
#define GEN_PASS_DEF_PLANREFINETYPESPASS
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"
} // namespace mlir::plan

using namespace mlir;
using namespace mlir::plan;

/// If `toUpdate`'s type can be refined in-place (e.g. because all users are
/// StableHlo ops), then do so. Otherwise, update the type and replace all
/// existing uses with a `tensor.cast` back to the original type.
static void updateTypeInPlaceAndMaybeInsertCast(RewriterBase &rewriter,
                                                Value toUpdate,
                                                RankedTensorType newType) {

  Type oldType = toUpdate.getType();
  rewriter.modifyOpInPlace(toUpdate.getDefiningOp(),
                           [&]() { toUpdate.setType(newType); });

  // If all the users are StableHLO or TensorRT ops, then they all allow
  // in-place update of operand types.
  auto isTensorRTOp = [](Operation *op) {
    return llvm::isa<tensorrt::TensorRTDialect>(op->getDialect());
  };
  if (stablehlo::canUpdateTypeWithoutCast(toUpdate, isTensorRTOp))
    return;

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfterValue(toUpdate);
  auto castOp =
      rewriter.create<tensor::CastOp>(toUpdate.getLoc(), oldType, toUpdate);
  rewriter.replaceAllUsesExcept(toUpdate, castOp, castOp);
}

/// Replace `original` with `replacement` if the types match. Otherwise, insert
/// a `tensor.cast` of `v` if the types are "cast compatible", meaning that one
/// type is a generalization of the other. Otherwise, return failure.
static LogicalResult
maybeCastAndReplace(RewriterBase &rewriter,
                    TypedValue<RankedTensorType> original,
                    TypedValue<RankedTensorType> replacement) {
  if (original.getType() == replacement.getType()) {
    rewriter.replaceAllUsesWith(original, replacement);
    return success();
  }

  // We should not be handling element type differences here. If this occurs,
  // something has gone wrong. Types should have matching ranks and one shape
  // should be a generalization of the other.
  RankedTensorType originalType = original.getType();
  RankedTensorType newType = replacement.getType();
  if ((originalType.getElementType() != newType.getElementType()) ||
      (!tensorrt::isTargetRefinementOfSource(originalType.getShape(),
                                             newType.getShape()) &&
       !tensorrt::isTargetRefinementOfSource(newType.getShape(),
                                             originalType.getShape())))
    return failure();

  auto castOp = rewriter.create<tensor::CastOp>(original.getLoc(), originalType,
                                                replacement);
  rewriter.replaceAllUsesWith(original, castOp);
  return success();
}

/// Given a set of scalar integer shape components `dimVals` that describe the
/// shape of the TensorType, return a refined tensor type (e.g. if some of the
/// shape components are constants). If the tensor type can not be refined, then
/// returns nullopt.
static std::optional<SmallVector<int64_t>>
getRefinedShape(ValueRange dimVals, RankedTensorType originalType) {
  // Create a new shape and try to refine it.
  SmallVector<int64_t> newShape(originalType.getShape());
  bool didRefine = false;
  for (unsigned i = 0; i < newShape.size(); i++) {
    // Can't refine a dim that's already static.
    if (!ShapedType::isDynamic(newShape[i]))
      continue;
    APInt dim{};
    if (!matchPattern(dimVals[i], m_ConstantInt(&dim)))
      continue;
    didRefine = true;
    newShape[i] = dim.getSExtValue();
  }

  if (!didRefine)
    return {};
  return newShape;
}

namespace {

struct RefineDynamicBroadcast
    : public OpRewritePattern<stablehlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto withOp = op.getOutputDimensions().getDefiningOp<WithValuesOp>();
    if (!withOp)
      return failure();

    // Create a new shape and try to refine it.
    std::optional<SmallVector<int64_t>> newShape =
        getRefinedShape(withOp.getElements(), op.getType());
    if (!newShape)
      return failure();

    updateTypeInPlaceAndMaybeInsertCast(rewriter, op.getResult(),
                                        op.getType().clone(*newShape));
    return success();
  }
};

/// Simplifies a pattern
/// `stablehlo.dynamic_bcast_in_dim(plan.with_shape(x, coords...),
///   plan.with_values(shape, vals...))`
/// if `vals...` and `coords...` are the same integer scalar values and
/// the broadcast dimensions are non-permuting. In that
/// case, we know that provably the shape of the input does not change and
/// no transposition occurs, so it must be a trivial identity.
struct SimplifyIdentityDynamicBroadcast
    : public OpRewritePattern<stablehlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto withValuesOp = op.getOutputDimensions().getDefiningOp<WithValuesOp>();
    if (!withValuesOp)
      return failure();
    auto withShapeOp = op.getOperand().getDefiningOp<WithShapeOp>();
    if (!withShapeOp)
      return failure();

    if (withShapeOp.getShape() != withValuesOp.getElements() ||
        !llvm::equal(op.getBroadcastDimensions(),
                     llvm::seq<int64_t>(0, op.getType().getRank())))
      return rewriter.notifyMatchFailure(op,
                                         "not provably an identity broadcast");
    return maybeCastAndReplace(rewriter, op, withShapeOp.getOperand());
  }
};

struct RefineDynamicIota : public OpRewritePattern<stablehlo::DynamicIotaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::DynamicIotaOp op,
                                PatternRewriter &rewriter) const override {
    auto withOp = op.getOutputShape().getDefiningOp<WithValuesOp>();
    if (!withOp)
      return failure();

    // Create a new shape and try to refine it.
    std::optional<SmallVector<int64_t>> newShape =
        getRefinedShape(withOp.getElements(), op.getType());
    if (!newShape)
      return failure();

    updateTypeInPlaceAndMaybeInsertCast(rewriter, op.getResult(),
                                        op.getType().clone(*newShape));
    return success();
  }
};

/// Upstream StableHLO pattern uses `builtin.unrealized_cast` while we prefer to
/// use `tensor.cast` due to better verification. Other passes may create such
/// `tensor.cast`, this pattern absorbs them into function return and refines
/// the function type.
struct AbsorbCastsIntoFuncReturnPattern
    : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    bool needsUpdate = false;
    SmallVector<Value> newOperands(op->getOperands());
    for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
      auto castOp = dyn_cast_or_null<tensor::CastOp>(operand.getDefiningOp());
      if (!castOp || !isa<RankedTensorType>(castOp.getType()) ||
          !isa<RankedTensorType>(castOp.getSource().getType()) ||
          !tensorrt::isTargetRefinementOfSource(
              castOp.getType().getShape(),
              castOp.getSource().getType().getShape()))
        continue;
      newOperands[i] = castOp.getSource();
      needsUpdate = true;
    }
    if (!needsUpdate)
      return failure();

    rewriter.modifyOpInPlace(
        op, [&]() { op.getOperandsMutable().assign(newOperands); });

    // If the type of the enclosing `func.func` needs an update, we simply
    // call setType. We can afford this simplicity because our algorithm
    // currently supports only one function per module.

    auto func = cast<func::FuncOp>(op->getParentOp());
    rewriter.modifyOpInPlace(func, [&]() {
      func.setType(rewriter.getFunctionType(func.getArgumentTypes(),
                                            TypeRange(newOperands)));
    });
    return success();
  }
};

struct WithShapeAbsorbCastPattern : public OpRewritePattern<WithShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(WithShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto castOp = op.getOperand().getDefiningOp<tensor::CastOp>();
    if (!castOp || !isa<RankedTensorType>(castOp.getSource().getType()) ||
        !tensorrt::isTargetRefinementOfSource(
            castOp.getType().getShape(),
            castOp.getSource().getType().getShape()))
      return failure();

    if (stablehlo::canUpdateTypeWithoutCast(op.getResult())) {
      rewriter.modifyOpInPlace(op, [&]() {
        op.getOperandMutable().assign(castOp.getSource());
        op.getResult().setType(
            cast<RankedTensorType>(castOp.getSource().getType()));
      });
      return success();
    }

    auto newWithShapeOp = rewriter.create<WithShapeOp>(
        op.getLoc(), castOp.getOperand(), op.getShape());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                                newWithShapeOp);
    return success();
  }
};

/// Given a pattern `plan.with_shape(stablehlo_op, dims...)`, if inspection of
/// `dims` yields an opportunity to refine the type of `with_shape`, then
/// `stablehlo_op` can also be refined. The refinements are made (and casts are
/// inserted if required).
struct StableHloRefineTypeFromWithShapeGeneric
    : public OpRewritePattern<WithShapeOp> {
  using OpRewritePattern<WithShapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WithShapeOp withOp,
                                PatternRewriter &rewriter) const override {
    auto producer = withOp.getOperand().getDefiningOp();
    if (!producer || !producer->hasOneUse() ||
        !isa<stablehlo::StablehloDialect>(producer->getDialect()))
      return failure();

    // Create a new shape and try to refine it.
    std::optional<SmallVector<int64_t>> newShape =
        getRefinedShape(withOp.getShape(), withOp.getOperand().getType());
    if (!newShape)
      return failure();

    // Update type of the producer.
    updateTypeInPlaceAndMaybeInsertCast(
        rewriter, withOp.getOperand(),
        withOp.getOperand().getType().clone(*newShape));

    // Update type of the WithShapeOp.
    updateTypeInPlaceAndMaybeInsertCast(rewriter, withOp.getResult(),
                                        withOp.getType().clone(*newShape));
    return success();
  }
};

/// Given a pattern `plan.with_shape(tensorrt_op, dims...)`, if inspection of
/// `dims` yields an opportunity to refine the type of `with_shape`, then
/// `tensorrt_op` can also be refined. The refinements are made (and casts are
/// inserted if required).
struct TensorRTRefineTypeFromWithShapeGeneric
    : public OpRewritePattern<WithShapeOp> {
  using OpRewritePattern<WithShapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WithShapeOp withOp,
                                PatternRewriter &rewriter) const override {
    auto producer = withOp.getOperand().getDefiningOp();
    if (!producer || !producer->hasOneUse() ||
        !isa<tensorrt::TensorRTDialect>(producer->getDialect()))
      return failure();

    // Create a new shape and try to refine it.
    std::optional<SmallVector<int64_t>> newShape =
        getRefinedShape(withOp.getShape(), withOp.getOperand().getType());
    if (!newShape)
      return failure();

    // Update type of the producer.
    updateTypeInPlaceAndMaybeInsertCast(
        rewriter, withOp.getOperand(),
        withOp.getOperand().getType().clone(*newShape));

    // Update type of the WithShapeOp.
    updateTypeInPlaceAndMaybeInsertCast(rewriter, withOp.getResult(),
                                        withOp.getType().clone(*newShape));
    return success();
  }
};

class PlanRefineTypesPass
    : public plan::impl::PlanRefineTypesPassBase<PlanRefineTypesPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    func::FuncOp funcTarget =
        stablehlo::getStablehloRefineShapesTarget(getOperation());
    if (!funcTarget)
      return;

    GreedyRewriteConfig config{};
    config.useTopDownTraversal = true;

    // clang-format off
    patterns.add<
        AbsorbCastsIntoFuncReturnPattern,
        RefineDynamicBroadcast,
        RefineDynamicIota,
        SimplifyIdentityDynamicBroadcast,
        StableHloRefineTypeFromWithShapeGeneric,
        WithShapeAbsorbCastPattern,
        TensorRTRefineTypeFromWithShapeGeneric
      >(ctx);
    // clang-format on
    stablehlo::populateStablehloRefineShapesPatterns(&patterns, ctx);
    stablehlo::populateStablehloCanonicalizationPatterns(ctx, &patterns);
    if (failed(applyPatternsAndFoldGreedily(funcTarget, std::move(patterns),
                                            config))) {
      emitError(funcTarget.getLoc())
          << "failed to apply patterns in " << getArgument() << "\n";
      return signalPassFailure();
    }
  }
};
} // namespace
