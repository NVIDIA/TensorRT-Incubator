//===- ExpandOps.cpp --------------------------------------------*- c++ -*-===//
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
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tensorrt {
#define GEN_PASS_DEF_EXPANDOPSPASS
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.h.inc"
} // namespace tensorrt
} // namespace mlir

using namespace mlir;
using namespace mlir::tensorrt;

namespace {

/// Rewrite `tensorrt.expand_rank`/`tensorrt.collapse_rank` into a
/// `tensorrt.shuffle` operation.
template <typename OpType>
struct RewriteRankReshapeOpToShuffle : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    // `tensorrt.expand_rank`/`tensorrt_collapse` rank ops support ranked
    // tensors of any time. However, shuffle supports only
    // RankedTensorOf<[TensorRT_I8, I32, F16, F32]>. Thus, when op is of type
    // `I1`, it needs to be supported via `I32` conversion
    TensorValue operand = op.getInput();
    RankedTensorType resultType =
        cast<RankedTensorType>(op.getResult().getType());
    if (operand.getType().getElementType().isInteger(1)) {
      RankedTensorType i32OperandType =
          RankedTensorType::Builder(cast<RankedTensorType>(operand.getType()))
              .setElementType(rewriter.getI32Type());
      resultType = RankedTensorType::Builder(resultType)
                       .setElementType(rewriter.getI32Type());
      operand =
          rewriter.create<IdentityOp>(op.getLoc(), i32OperandType, operand);
    }

    auto maybeCastBack = [&](TensorValue tensor) {
      if (tensor.getType() == op.getType())
        return tensor;
      return rewriter.create<IdentityOp>(op.getLoc(), op.getType(), tensor)
          .getResult();
    };

    // The ops `expand_rank`/`collapse_rank` do not have a "zero is placeholder"
    // semantic or option.
    rewriter.replaceOp(op, maybeCastBack(rewriter.create<ShuffleOp>(
                               op.getLoc(), resultType,
                               /*input=*/operand,
                               /*zero_is_placeholder=*/false)));
    return success();
  }
};

/// Rewrite `tensorrt.reshape` to `tensorrt.shuffle`.
struct RewriteReshapeOpToShuffle : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Shuffle cannot handle I1 tensors, so cast to i32 and back if required.
    TensorValue operand = op.getInput();
    RankedTensorType shuffleResultType = cast<RankedTensorType>(op.getType());
    if (operand.getType().getElementType().isInteger(1)) {
      RankedTensorType i32TensorType =
          RankedTensorType::Builder(cast<RankedTensorType>(operand.getType()))
              .setElementType(rewriter.getI32Type());
      shuffleResultType = RankedTensorType::Builder(shuffleResultType)
                              .setElementType(rewriter.getI32Type());
      operand =
          rewriter.create<IdentityOp>(op.getLoc(), i32TensorType, operand);
    }

    auto maybeCastBack = [&](TensorValue tensor) {
      if (tensor.getType() == op.getType())
        return tensor;
      return rewriter.create<IdentityOp>(op.getLoc(), op.getType(), tensor)
          .getResult();
    };

    // Handle the dynamic case.
    if (op.getShape()) {
      rewriter.replaceOp(op, maybeCastBack(rewriter.create<ShuffleOp>(
                                 op.getLoc(), shuffleResultType,
                                 /*input=*/operand,
                                 /*dynamic_reshape=*/op.getShape(),
                                 /*zero_is_placeholder=*/false)));
      return success();
    }

    rewriter.replaceOp(op, maybeCastBack(rewriter.create<ShuffleOp>(
                               op.getLoc(), shuffleResultType,
                               /*input=*/operand,
                               /*zero_is_placeholder=*/false)));
    return failure();
  }
};

template <typename OpType, TopKOperation topKType>
struct ArgMinMaxToTopK : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    TypedValue<RankedTensorType> operand = op.getInput();
    RankedTensorType argMinMaxInputType =
        cast<RankedTensorType>(op.getInput().getType());

    auto maybeCastBack = [&](TensorValue tensor) {
      if (tensor.getType() == op.getType(0))
        return tensor;
      return rewriter.create<IdentityOp>(op.getLoc(), op.getType(0), tensor)
          .getResult();
    };

    // TensorRT TopK op accepts tensors of 2 or more dimension in only FP32
    // and FP16.
    if (isa<IntegerType>(argMinMaxInputType.getElementType())) {
      RankedTensorType f32CastType =
          RankedTensorType::Builder(argMinMaxInputType)
              .setElementType(rewriter.getF32Type());
      operand = rewriter.create<IdentityOp>(op.getLoc(), f32CastType, operand);
    }

    if (argMinMaxInputType.getRank() == 1) {
      RankedTensorType expandResultType =
          RankedTensorType::get({1, argMinMaxInputType.getShape()[0]},
                                operand.getType().getElementType());
      operand =
          rewriter.create<ExpandRankOp>(op.getLoc(), expandResultType, operand);
      int64_t topKAxis{1};
      TopKOp topK = rewriter.create<TopKOp>(op.getLoc(), operand,
                                            /*k=*/1,
                                            /*axis=*/topKAxis,
                                            /*topKOperation=*/topKType);
      RankedTensorType collapsedValueResultType =
          RankedTensorType::get({1}, operand.getType().getElementType());
      RankedTensorType collapsedIndicesResultType =
          RankedTensorType::get({1}, rewriter.getIntegerType(32));
      CollapseRankOp collapsedValues = rewriter.create<CollapseRankOp>(
          op.getLoc(), collapsedValueResultType, topK.getResult(0));
      CollapseRankOp collapsedIndices = rewriter.create<CollapseRankOp>(
          op.getLoc(), collapsedIndicesResultType, topK.getResult(1));
      rewriter.replaceOp(op,
                         {maybeCastBack(collapsedValues), collapsedIndices});
      return success();
    }
    TopKOp topK = rewriter.create<TopKOp>(op.getLoc(), operand,
                                          /*k=*/1,
                                          /*axis=*/op.getAxis(),
                                          /*topKOperation=*/topKType);
    rewriter.replaceOp(op,
                       {maybeCastBack(topK.getValues()), topK.getResult(1)});
    return success();
  }
};

/// Lower transpose to shuffle op.
struct TransposeToShuffle : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    AffineMap perm = op.getPermutation();
    assert(perm.isPermutation() && "expected perm to be a permutation");
    auto identitySequence =
        llvm::to_vector(llvm::seq<int64_t>(0, op.getType().getRank()));
    auto transposeSequence = perm.compose(identitySequence);
    rewriter.replaceOpWithNewOp<ShuffleOp>(op, op.getType(), op.getInput(),
                                           /*dynamic_reshape=*/Value(),
                                           /*firstTranspose=*/transposeSequence,
                                           /*reshapeDimensions=*/
                                           DenseI64ArrayAttr(),
                                           /*secondTranspose=*/identitySequence,
                                           /*zero_is_placeholder=*/false);
    return success();
  }
};
} // namespace

/// Create transpose + expand_rank on the input of a `tensorrt.broadcast` so
/// that the result has the same rank as the `tensorrt.broadcast` result and the
/// equivalent `broadcastDims` will preserve the ordering of the input dims.
static TensorValue reshapeBroadcastInput(OpBuilder &b, Location loc,
                                         BroadcastOp op) {
  // Determine the ordering of the dimensions after the broadcast map.
  assert(op.getBroadcastDimsPermutation().isIdentity() &&
         "expected identity broadcast permutation");
  TensorValue input = op.getInput();
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  // If input is a scalar, just reshape to all 1's.
  if (input.getType().getRank() == 0)
    return b.create<ExpandRankOp>(
        loc, resultType.clone(SmallVector<int64_t>(resultType.getRank(), 1)),
        input);

  if (input.getType().hasStaticShape()) {
    SmallVector<int64_t> expandedShape;
    unsigned inputIdx = 0;
    ArrayRef<int64_t> broadcastDims = op.getBroadcastDims();
    for (unsigned i = 0, e = resultType.getRank(); i < e; i++) {
      if (inputIdx < input.getType().getRank() &&
          i == broadcastDims[inputIdx]) {
        expandedShape.push_back(input.getType().getDimSize(inputIdx++));
        continue;
      }
      expandedShape.push_back(1);
    }
    Type expandedRankType = resultType.clone(expandedShape);
    return b.create<ExpandRankOp>(loc, expandedRankType, input);
  }

  // For dynamic cases, we need to assemble the shape and use
  // `tensorrt.reshape`.
  SmallVector<Value> shapeValueScalars;
  SmallVector<int64_t> typeShape;
  unsigned inputIdx = 0;
  ArrayRef<int64_t> broadcastDims = op.getBroadcastDims();
  auto inputShape = b.create<ShapeOp>(loc, input);
  Value c1 =
      b.create<ConstantOp>(loc, cast<ElementsAttr>(b.getI32TensorAttr({1})));
  for (int32_t i = 0, e = resultType.getRank(); i < e; i++) {
    if (inputIdx < input.getType().getRank() && i == broadcastDims[inputIdx]) {
      typeShape.push_back(input.getType().getDimSize(inputIdx));
      shapeValueScalars.push_back(b.create<SliceOp>(
          loc, inputShape, ArrayRef<int32_t>{static_cast<int32_t>(inputIdx)},
          ArrayRef<int32_t>{1}, ArrayRef<int32_t>{1}));
      inputIdx++;
      continue;
    }
    typeShape.push_back(1);
    shapeValueScalars.push_back(c1);
  }
  Value newShapeValue =
      b.create<ConcatenationOp>(loc, shapeValueScalars, /*dim=*/0);
  return b.create<ReshapeOp>(loc, input.getType().clone(typeShape), input,
                             newShapeValue);
}

namespace {

struct BroadcastRemoveTranspose : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    AffineMap broadcastPerm = op.getBroadcastDimsPermutation();
    if (broadcastPerm.isIdentity())
      return failure();
    TensorValue input = op.getInput();
    auto transposeOp = rewriter.create<tensorrt::TransposeOp>(
        op.getLoc(), input, broadcastPerm);
    SmallVector<int64_t> reorderedBroadcastDims =
        applyPermutationMap(broadcastPerm, op.getBroadcastDims());
    rewriter.replaceOpWithNewOp<tensorrt::BroadcastOp>(
        op, op.getType(), transposeOp.getResult(), op.getShape(),
        reorderedBroadcastDims);
    return success();
  }
};

/// Convert specific broadcast operations to slice (copy) operations.
struct TileLikeBroadcastToSlice : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    TensorValue input = op.getInput();
    TensorType resultType = op.getType();

    if (!op.getBroadcastDimsPermutation().isIdentity())
      return failure();

    // The ranks must be equal and the permutation of the indices caused by the
    // broadcast must be equal. Otherwise, insert a shuffle and/or reshape.
    if (input.getType().getRank() != resultType.getRank())
      input = reshapeBroadcastInput(rewriter, op.getLoc(), op);
    TensorType inputType = input.getType();

    // Create staticStart and staticStride attributes
    auto staticStart = rewriter.getDenseI32ArrayAttr(
        SmallVector<int32_t>(inputType.getRank(), 0));
    auto staticStride = rewriter.getDenseI32ArrayAttr(
        SmallVector<int32_t>(inputType.getRank(), 1));

    if (resultType.hasStaticShape()) {
      // For static shapes, use DenseI32ArrayAttr for size
      SmallVector<int32_t> size_vec =
          llvm::map_to_vector(resultType.getShape(), [](int64_t x) {
            return static_cast<int32_t>(x);
          });
      auto staticSize = rewriter.getDenseI32ArrayAttr(size_vec);

      rewriter.replaceOpWithNewOp<SliceOp>(
          op, resultType, input, /*fill=*/Value(), /*start=*/Value(),
          /*size=*/Value(), /*stride=*/Value(), staticStart, staticSize,
          staticStride, tensorrt::SliceMode::kWRAP);
    } else {
      // For dynamic shapes, use the original op's shape operand
      rewriter.replaceOpWithNewOp<SliceOp>(
          op, resultType, input, /*fill=*/Value(), /*start=*/Value(),
          op.getShape(), /*stride=*/Value(), staticStart,
          /*staticSize=*/nullptr, staticStride, tensorrt::SliceMode::kWRAP);
    }
    return success();
  }
};

/// Pass to expand extension operations into more fundamental operations.
class ExpandOpsPass : public tensorrt::impl::ExpandOpsPassBase<ExpandOpsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ArgMinMaxToTopK<ArgMinOp, TopKOperation::kMIN>,
                 ArgMinMaxToTopK<ArgMaxOp, TopKOperation::kMAX>,
                 TransposeToShuffle, RewriteReshapeOpToShuffle,
                 RewriteRankReshapeOpToShuffle<ExpandRankOp>,
                 RewriteRankReshapeOpToShuffle<CollapseRankOp>,
                 TileLikeBroadcastToSlice, BroadcastRemoveTranspose>(
        &getContext());

    // Add limited canonicalization patterns for ops that could be created.
    ShuffleOp::getCanonicalizationPatterns(patterns, &getContext());

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
