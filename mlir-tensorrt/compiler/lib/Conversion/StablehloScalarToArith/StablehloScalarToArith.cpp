//===- StablehloScalarToArith.cpp --------------------------------*- C++-*-===//
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
/// Implementation of pass to convert stablehlo ops with scalar operands to
/// arith dialect ops.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/StablehloScalarToArith/StablehloScalarToArith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "stablehlo/conversions/linalg/transforms/MapStablehloToScalarOp.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSTABLEHLOSCALARTOARITHPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
template <typename OpTy>
struct StablehloScalarToArith : public OneToNOpConversionPattern<OpTy> {
  using OneToNOpConversionPattern<OpTy>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OneToNOpConversionPattern<OpTy>::OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    // This pattern is only valid for specific operations that have a single
    // result value.
    assert(op->getNumResults() == 1 && "expected single-result operation");

    ShapedType resultTy = cast<ShapedType>(op->getResultTypes().front());
    SmallVector<Value> replacements;
    Type targetType = adaptor.getResultMapping().getConvertedTypes(0).front();
    for (unsigned i = 0, e = resultTy.getNumElements(); i < e; i++) {
      SmallVector<Value, 4> operands;
      for (ValueRange range : adaptor.getOperands()) {
        assert((range.size() == 1 || range.size() == e) &&
               "expected remapped values to be of size 1 or equal to "
               "number of elements");
        operands.push_back(range.size() > 1 ? range[i] : range[0]);
      }
      Value scalarResult = stablehlo::StablehloOpToStdScalarOp::mapOp(
          op, targetType, operands, &rewriter);
      if (!scalarResult)
        return failure();
      replacements.push_back(scalarResult);
    }
    rewriter.replaceOp(op, replacements, adaptor.getResultMapping());
    return success();
  }
};

/// Rewrite `stablehlo.broadcast_in_dim` of a scalar to just forward scalar.
struct StablehloRewriteBroadcastInDim
    : public OneToNOpConversionPattern<stablehlo::BroadcastInDimOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult
  matchAndRewrite(stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    TensorType inputType = op.getOperand().getType();
    TensorType resultType = op.getType();
    if (inputType.getNumElements() != 1 || resultType.getRank() != 1)
      return failure();
    SmallVector<Value> replacements(
        adaptor.getResultMapping().getConvertedTypes().size(),
        adaptor.getOperand().front());
    rewriter.replaceOp(op, replacements, adaptor.getResultMapping());
    return success();
  }
};

/// Convert `stablehlo.reshape`. A reshape does not change the underlying
/// 'layout' of a tensor, and therefore we do not need to change the ordering of
/// the scalars here. It becomes a no-op.
struct StablehloRewriteReshapeScalar
    : public OneToNOpConversionPattern<stablehlo::ReshapeOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult
  matchAndRewrite(stablehlo::ReshapeOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getFlatOperands(),
                       adaptor.getResultMapping());
    return success();
  }
};

/// Rewrite `stablehlo.concatenate` of 1D tensors to just forward the scalars.
/// TODO: expand support to rank > 1.
struct StablehloRewriteConcat
    : public OneToNOpConversionPattern<stablehlo::ConcatenateOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult
  matchAndRewrite(stablehlo::ConcatenateOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    if (!llvm::all_of(op->getOperandTypes(), [](Type t) {
          return cast<RankedTensorType>(t).getRank() == 1;
        }))
      return failure();
    rewriter.replaceOp(op, adaptor.getFlatOperands(),
                       adaptor.getResultMapping());
    return success();
  }
};
} // namespace

/// Return the equivalent attribute for the integer `idx` represented as a
/// scalar `type`.
static Attribute getScalarValue(RewriterBase &rewriter, Type type,
                                int64_t idx) {
  if (isa<FloatType>(type))
    return rewriter.getFloatAttr(type, static_cast<double>(idx));
  if (isa<IndexType>(type))
    return rewriter.getIndexAttr(idx);
  if (auto integerType = dyn_cast<IntegerType>(type))
    return rewriter.getIntegerAttr(
        type, APInt(cast<IntegerType>(type).getWidth(), idx));
  return {};
}

namespace {

/// Rewrite `stablehlo.iota` to a list of constants.
struct StablehloRewriteIota
    : public OneToNOpConversionPattern<stablehlo::IotaOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult
  matchAndRewrite(stablehlo::IotaOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    TensorType type = op.getType();
    if (type.getRank() != 1)
      return failure();
    SmallVector<Value> replacements;
    for (unsigned i = 0; i < type.getNumElements(); i++) {
      replacements.push_back(rewriter.create<arith::ConstantOp>(
          op.getLoc(), type.getElementType(),
          cast<TypedAttr>(getScalarValue(rewriter, type.getElementType(), i))));
    }
    rewriter.replaceOp(op, replacements, adaptor.getResultMapping());
    return success();
  }
};

// Rewrite `stablehlo.slice` to a list of scalars.
struct StablehloRewriteSlice
    : public OneToNOpConversionPattern<stablehlo::SliceOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;
  LogicalResult
  matchAndRewrite(stablehlo::SliceOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    TensorType inputType = op.getOperand().getType();

    if (inputType.getRank() != 1)
      return failure();
    int64_t offset = op.getStartIndices().front();
    int64_t stride = op.getStrides().front();
    int64_t limit = op.getLimitIndices().front();
    SmallVector<Value> replacements;
    for (int64_t idx = offset; idx < limit; idx += stride) {
      replacements.push_back(adaptor.getOperand()[idx]);
    }
    rewriter.replaceOp(op, replacements, adaptor.getResultMapping());
    return success();
  }
};
} // namespace

/// Clone the `body` of a stablehlo reduction operation using the given `args`
/// (scalars) as replacements for the block arguments. The scalars that should
/// be yielded from the body are returned.
static FailureOr<SmallVector<Value>>
cloneStablehloScalarReduceBlock(RewriterBase &rewriter, Block *body,
                                ValueRange args) {
  IRMapping mapping;
  for (auto [blockArg, replacement] :
       llvm::zip_equal(body->getArguments(), args))
    mapping.map(blockArg, replacement);
  for (Operation &bodyOp : body->without_terminator()) {
    // ValueRange operands = bodyOp.getOperands();
    // If a conversion cast was added, we can try to just skip over it.
    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(bodyOp)) {
      assert(castOp->getOperands().size() == 1 && castOp.getNumResults() == 1 &&
             "expected case to map from scalar value to scalar tensor");
      Value remapped = mapping.lookup(castOp->getOperand(0));
      if (remapped.getType() == castOp.getResult(0).getType()) {
        mapping.map(castOp.getResult(0), remapped);
        continue;
      }
    }

    // If not a Stablehlo op, then some conversion has alrady occurred. Clone as
    // normal.
    if (!isa<stablehlo::StablehloDialect>(bodyOp.getDialect())) {
      rewriter.clone(bodyOp, mapping);
      continue;
    }

    // Otherwise, we need to remap arguments manually. Just return failure here
    // and we can re-try when the body is converted.
    return failure();
  }

  // Get the yielded values. We expect each yielded value to be defined by a
  // cast from calar type to tensor scalar. Search back past the cast to get the
  // scalar value to yield.
  SmallVector<Value> yieldedValues =
      llvm::map_to_vector(body->getTerminator()->getOperands(), [&](Value v) {
        v = mapping.lookup(v);
        auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>();
        assert(castOp && castOp->getNumOperands() == 1 &&
               "expected scalar-to-tensor cast");
        return castOp.getOperand(0);
      });
  return yieldedValues;
}

namespace {
/// Convert `stablehlo.reduce` to operate on scalars.
/// TODO: expand support to rank > 1.
struct StablehloRewriteReduce
    : public OneToNOpConversionPattern<stablehlo::ReduceOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(stablehlo::ReduceOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1)
      return failure();
    RankedTensorType inputType =
        dyn_cast<RankedTensorType>(op.getInputs().front().getType());

    if (!inputType || inputType.getRank() != 1)
      return failure();

    ValueRange inputScalars = adaptor.getInputs().front();
    ValueRange initScalars = adaptor.getInitValues().front();
    assert(initScalars.size() == 1 && "expected a single init scalar");

    // Serialize the reduction.
    Value accum = initScalars.front();
    // const TypeConverter *converter = getTypeConverter();
    for (int64_t i = 0; i < inputType.getNumElements(); i++) {
      FailureOr<SmallVector<Value>> yieldedResults =
          cloneStablehloScalarReduceBlock(rewriter, &op.getBody().front(),
                                          {inputScalars[i], accum});
      if (failed(yieldedResults))
        return failure();
      assert(yieldedResults->size() == 1 && "expected single-scalar yield");
      accum = yieldedResults->front();
    }
    rewriter.replaceOp(op, accum, adaptor.getResultMapping());
    return success();
  }
};
} // namespace

/// Extract scalar from `src` at the given linear index (interpreted using
/// canonical "row major" strides).
static Value extractScalarFromTensor(OpBuilder &rewriter, Location loc,
                                     Value src, int64_t linearIndex) {
  auto vt = dyn_cast<RankedTensorType>(src.getType());
  assert(vt && "expected `src` to have RankedTensorType");

  auto getIndexConst = [&](int64_t idx) -> Value {
    return rewriter.create<arith::ConstantIndexOp>(loc, idx);
  };

  // If src is a splat constant, then just create a scalar constant.
  DenseElementsAttr splatAttr;
  if (matchPattern(src, m_Constant<DenseElementsAttr>(&splatAttr)) &&
      splatAttr.isSplat())
    return rewriter.create<arith::ConstantOp>(
        loc, vt.getElementType(), splatAttr.getSplatValue<TypedAttr>());

  // Fast path when src is linear.
  if (vt.getRank() <= 1) {
    assert(linearIndex < vt.getNumElements() && "linear index out of bounds");
    return rewriter.create<tensor::ExtractOp>(
        loc, src, SmallVector<Value>(vt.getRank(), getIndexConst(linearIndex)));
  }
  // Delinearize the index in row-major order.
  SmallVector<int64_t> indices =
      mlir::delinearize(linearIndex, computeSuffixProduct(vt.getShape()));
  return rewriter.create<tensor::ExtractOp>(
      loc, src, llvm::map_to_vector(indices, getIndexConst));
}

bool mlir::stablehlo_ext::isScalarizableType(Type t) {
  auto rtt = dyn_cast<RankedTensorType>(t);
  if (!rtt || !rtt.hasStaticShape())
    return false;
  return rtt && rtt.hasStaticShape() && rtt.getNumElements() <= 4;
}

/// Return a 1-to-N type converter for scalarizing Tensor types to unpacked
/// scalar types.
OneToNTypeConverter stablehlo_ext::getScalarizationTypeConverter() {
  // Add a type converter, target and source materialization to convert
  // `tensor<1xdtype>` to `dtype` and back.
  OneToNTypeConverter typeConverter;
  typeConverter.addConversion([](Type t) -> std::optional<Type> { return t; });
  typeConverter.addConversion([&](Type t, SmallVectorImpl<Type> &result)
                                  -> std::optional<LogicalResult> {
    auto rtt = dyn_cast<RankedTensorType>(t);
    if (!rtt)
      return std::nullopt;
    if (!stablehlo_ext::isScalarizableType(rtt))
      return std::nullopt;
    Type elType = rtt.getElementType();
    if (isa<IntegerType>(elType) && !elType.isSignlessInteger())
      elType =
          IntegerType::get(elType.getContext(), elType.getIntOrFloatBitWidth());
    result.append(rtt.getNumElements(), elType);
    return success();
  });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, TypeRange resultTypes, Value input,
         Location loc) -> std::optional<SmallVector<Value>> {
        if (!isScalarizableType(input.getType()))
          return std::nullopt;
        RankedTensorType intermediateTensorType =
            cast<RankedTensorType>(input.getType());
        Type elType = intermediateTensorType.getElementType();
        if (isa<IntegerType>(elType) && !elType.isSignlessInteger()) {
          input =
              builder
                  .create<UnrealizedConversionCastOp>(
                      loc,
                      intermediateTensorType.clone(IntegerType::get(
                          elType.getContext(), elType.getIntOrFloatBitWidth())),
                      input)
                  .getResult(0);
        }

        SmallVector<Value> scalars;
        for (unsigned i = 0; i < resultTypes.size(); i++)
          scalars.push_back(extractScalarFromTensor(builder, loc, input, i));
        return scalars;
      });
  typeConverter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    RankedTensorType intermediateTensorType =
        cast<RankedTensorType>(resultType);

    // If we are converting back to a sign-full type, then make sure we
    // create the 'from_elements' type using the signless type.
    Type elType = intermediateTensorType.getElementType();
    if (isa<IntegerType>(elType) && !elType.isSignlessInteger()) {
      intermediateTensorType = intermediateTensorType.clone(IntegerType::get(
          elType.getContext(), elType.getIntOrFloatBitWidth()));
    }
    Value fromElements = builder.create<tensor::FromElementsOp>(
        loc, intermediateTensorType, inputs);
    if (fromElements.getType() == resultType)
      return fromElements;
    return builder
        .create<UnrealizedConversionCastOp>(loc, resultType, fromElements)
        .getResult(0);
  });
  return typeConverter;
}

namespace {
struct StablehloScalarToArithPass
    : public impl::ConvertStablehloScalarToArithPassBase<
          StablehloScalarToArithPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Add a type converter, target and source materialization to convert
    // `tensor<1xdtype>` to `dtype` and back.
    OneToNTypeConverter typeConverter =
        stablehlo_ext::getScalarizationTypeConverter();

    // Populate conversion rewrite patterns.
    RewritePatternSet patterns(ctx);
    /// TODO: stablehlo does not define CopyOp and TanOp
    patterns.add<StablehloScalarToArith<stablehlo::AbsOp>,
                 StablehloScalarToArith<stablehlo::AddOp>,
                 StablehloScalarToArith<stablehlo::AndOp>,
                 StablehloScalarToArith<stablehlo::Atan2Op>,
                 StablehloScalarToArith<stablehlo::BitcastConvertOp>,
                 StablehloScalarToArith<stablehlo::CbrtOp>,
                 StablehloScalarToArith<stablehlo::CeilOp>,
                 StablehloScalarToArith<stablehlo::ClampOp>,
                 StablehloScalarToArith<stablehlo::ClzOp>,
                 StablehloScalarToArith<stablehlo::CompareOp>,
                 StablehloScalarToArith<stablehlo::ComplexOp>,
                 StablehloScalarToArith<stablehlo::ConvertOp>,
                 StablehloScalarToArith<stablehlo::CosineOp>,
                 StablehloScalarToArith<stablehlo::DivOp>,
                 StablehloScalarToArith<stablehlo::ExpOp>,
                 StablehloScalarToArith<stablehlo::Expm1Op>,
                 StablehloScalarToArith<stablehlo::FloorOp>,
                 StablehloScalarToArith<stablehlo::ImagOp>,
                 StablehloScalarToArith<stablehlo::IsFiniteOp>,
                 StablehloScalarToArith<stablehlo::Log1pOp>,
                 StablehloScalarToArith<stablehlo::LogOp>,
                 StablehloScalarToArith<stablehlo::LogisticOp>,
                 StablehloScalarToArith<stablehlo::MaxOp>,
                 StablehloScalarToArith<stablehlo::MinOp>,
                 StablehloScalarToArith<stablehlo::MulOp>,
                 StablehloScalarToArith<stablehlo::NegOp>,
                 StablehloScalarToArith<stablehlo::NotOp>,
                 StablehloScalarToArith<stablehlo::OrOp>,
                 StablehloScalarToArith<stablehlo::PopulationCountOp>,
                 StablehloScalarToArith<stablehlo::PowOp>,
                 StablehloScalarToArith<stablehlo::RealOp>,
                 StablehloScalarToArith<stablehlo::ReducePrecisionOp>,
                 StablehloScalarToArith<stablehlo::RemOp>,
                 StablehloScalarToArith<stablehlo::RoundNearestEvenOp>,
                 StablehloScalarToArith<stablehlo::RoundOp>,
                 StablehloScalarToArith<stablehlo::RsqrtOp>,
                 StablehloScalarToArith<stablehlo::SelectOp>,
                 StablehloScalarToArith<stablehlo::ShiftLeftOp>,
                 StablehloScalarToArith<stablehlo::ShiftRightArithmeticOp>,
                 StablehloScalarToArith<stablehlo::ShiftRightLogicalOp>,
                 StablehloScalarToArith<stablehlo::SignOp>,
                 StablehloScalarToArith<stablehlo::SineOp>,
                 StablehloScalarToArith<stablehlo::SqrtOp>,
                 StablehloScalarToArith<stablehlo::SubtractOp>,
                 StablehloScalarToArith<stablehlo::TanhOp>,
                 StablehloScalarToArith<stablehlo::XorOp>>(
        typeConverter, patterns.getContext(), PatternBenefit(1));
    patterns.add<StablehloRewriteBroadcastInDim, StablehloRewriteReshapeScalar,
                 StablehloRewriteConcat, StablehloRewriteIota,
                 StablehloRewriteSlice, StablehloRewriteReduce>(
        typeConverter, patterns.getContext(), PatternBenefit(1));
    // Run the conversion.
    if (failed(applyPartialOneToNConversion(getOperation(), typeConverter,
                                            std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
