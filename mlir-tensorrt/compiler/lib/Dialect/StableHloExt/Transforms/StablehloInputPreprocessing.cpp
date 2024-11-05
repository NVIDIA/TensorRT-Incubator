//===- StablehloInputPreprocessing.cpp  -----------------------------------===//
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
/// Implements a pass that applies various patterns to StableHLO IR to prepare
/// it for conversion to the TensorRT dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Patterns.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_STABLEHLOINPUTPREPROCESSINGPASS
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

using namespace mlir;
using namespace mlir::stablehlo;

namespace {
/// Fold trivial `stablehlo.logical_shift_right` when the shift has a greater
/// width than the element type.
struct StablehloRewriteTrivialLogicalRightShift
    : public OpRewritePattern<stablehlo::ShiftRightLogicalOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::ShiftRightLogicalOp op,
                                PatternRewriter &rewriter) const override {
    TensorType resultType = op.getType();
    int64_t bitWidth = resultType.getElementTypeBitWidth();
    ElementsAttr attr;
    // Try to match a constant shift amount.
    if (matchPattern(op.getRhs(), m_Constant(&attr))) {
      if (!attr.isSplat())
        return failure();
      int64_t shiftAmount = attr.getSplatValue<APInt>().getSExtValue();
      if (shiftAmount < bitWidth)
        return failure();
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, rewriter.getZeroAttr(resultType));
      return success();
    }
    return failure();
  }
};
} // namespace

static Value makeSplatF32TensorConstantLike(OpBuilder &b, Location loc,
                                            float constant, Value val) {
  auto rtt = cast<RankedTensorType>(val.getType());
  return b.create<stablehlo::ConstantOp>(loc,
                                         DenseElementsAttr::get(rtt, constant));
}

static Value makeSplatTensorInfConstantLike(OpBuilder &b, Location loc,
                                            Value val, bool isNegInf) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return makeSplatF32TensorConstantLike(
      b, loc,
      APFloat::getInf(ty.getFloatSemantics(), isNegInf).convertToFloat(), val);
}

/// Reproduced from `chlo-legalize-to-stablehlo` from here:
/// https://github.com/openxla/stablehlo/blob/c3f456500f0f2e96fdb4a98fde4fbe48b4d624b8/stablehlo/transforms/ChloLegalizeToStablehlo.cpp
/// (Apache 2.0 License:
/// https://github.com/openxla/stablehlo/blob/c3f456500f0f2e96fdb4a98fde4fbe48b4d624b8/LICENSE)
/// TODO: remove this once we modify upstream to expose this lowering pattern in
/// a public header so we don't have to copy-paste like this.
static Value erfInv32Stablehlo(RewriterBase &b, Location loc, ValueRange args) {
  constexpr int kDegree = 9;
  constexpr std::array<float, 9> wLessThan5Constants = {
      2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f,
      -4.39150654e-06f, 0.00021858087f,  -0.00125372503f,
      -0.00417768164f,  0.246640727f,    1.50140941f};
  constexpr std::array<float, 9> wGreaterThan5Constants = {
      -0.000200214257f, 0.000100950558f, 0.00134934322f,
      -0.00367342844f,  0.00573950773f,  -0.0076224613f,
      0.00943887047f,   1.00167406f,     2.83297682f};

  Value x = args[0];
  // Compute logarithm of (1+arg) using log1p(arg) which is more precise than
  // log(1+arg) when arg is close to zero. For more details, see
  // https://en.cppreference.com/w/cpp/numeric/math/log1p
  Value minusXSquared =
      b.create<stablehlo::MulOp>(loc, x, b.create<stablehlo::NegOp>(loc, x));
  Value w = b.create<stablehlo::NegOp>(
      loc, b.create<stablehlo::Log1pOp>(loc, minusXSquared));

  Value lt = b.create<stablehlo::CompareOp>(
      loc, w, makeSplatF32TensorConstantLike(b, loc, 5.0, x),
      stablehlo::ComparisonDirection::LT);
  auto coefficient = [&](int i) {
    return b.create<stablehlo::SelectOp>(
        loc, lt,
        makeSplatF32TensorConstantLike(b, loc, wLessThan5Constants[i], x),
        makeSplatF32TensorConstantLike(b, loc, wGreaterThan5Constants[i], x));
  };
  w = b.create<stablehlo::SelectOp>(
      loc, lt,
      b.create<stablehlo::SubtractOp>(
          loc, w, makeSplatF32TensorConstantLike(b, loc, 2.5, x)),
      b.create<stablehlo::SubtractOp>(
          loc, b.create<stablehlo::SqrtOp>(loc, w),
          makeSplatF32TensorConstantLike(b, loc, 3.0, x)));
  Value p = coefficient(0);
  for (int i = 1; i < kDegree; ++i) {
    p = b.create<stablehlo::AddOp>(loc, coefficient(i),
                                   b.create<stablehlo::MulOp>(loc, p, w));
  }

  // Result modulo edge cases.
  Value result = b.create<stablehlo::MulOp>(loc, p, x);

  // Handle edge cases, namely erfinv(+/-1) = +/-inf.  (The above computation is
  // indeterminate, and can give nan or -/+inf.)
  return b.create<stablehlo::SelectOp>(
      loc,
      b.create<stablehlo::CompareOp>(
          loc, b.create<stablehlo::AbsOp>(loc, x),
          makeSplatF32TensorConstantLike(b, loc, 1, x),
          stablehlo::ComparisonDirection::EQ),
      b.create<stablehlo::MulOp>(
          loc, x, makeSplatTensorInfConstantLike(b, loc, x, false)),
      result);
}

namespace {

struct ConvertErfInvOpToStablehlo final : OpRewritePattern<chlo::ErfInvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(chlo::ErfInvOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type elementType = op.getResult().getType().getElementType();
    if (!isa<FloatType>(elementType) ||
        elementType.getIntOrFloatBitWidth() > 32)
      return failure();

    Value operand = op.getOperand();
    if (!elementType.isF32())
      operand = rewriter.create<stablehlo::ConvertOp>(loc, operand,
                                                      rewriter.getF32Type());

    Value result = erfInv32Stablehlo(rewriter, loc, operand);
    if (result.getType() != op.getResult().getType())
      result = rewriter.create<stablehlo::ConvertOp>(
          loc, result, op.getResult().getType().getElementType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

class StablehloInputPreprocessing
    : public mlir::stablehlo_ext::impl::StablehloInputPreprocessingPassBase<
          StablehloInputPreprocessing> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<StablehloRewriteTrivialLogicalRightShift,
                    ConvertErfInvOpToStablehlo>(ctx);
    stablehlo_ext::populateCanonicalizeStablehloConvolutionPatterns(patterns);
    stablehlo_ext::populateCanonicalizeStablehloScatterPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      emitError(op->getLoc()) << "failed to run patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};

} // namespace
