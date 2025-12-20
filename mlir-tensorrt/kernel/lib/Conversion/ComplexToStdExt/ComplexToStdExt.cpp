//===- ComplexToStdExt.cpp ------------------------------------------------===//
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
///
/// This pass converts complex operations to arith operations and then
/// applies additional patterns to completely eliminate complex types
/// via integer bitcast.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-kernel/Conversion/Passes.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTCOMPLEXTOSTANDARDEXT
#include "mlir-kernel/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// Pack real and imaginary f32 components into i64 value
static Value packComplexComponents(OpBuilder &rewriter, Location loc,
                                   Value realFloat, Value imagFloat) {
  assert(realFloat.getType() == imagFloat.getType() &&
         "real and imaginary float types must match");
  Type componentIntType =
      rewriter.getIntegerType(imagFloat.getType().getIntOrFloatBitWidth());
  Type packedIntType =
      rewriter.getIntegerType(realFloat.getType().getIntOrFloatBitWidth() * 2);

  // Cast float components to integer using bitcast
  Value realInt =
      rewriter.create<arith::BitcastOp>(loc, componentIntType, realFloat);
  Value imagInt =
      rewriter.create<arith::BitcastOp>(loc, componentIntType, imagFloat);

  // Extend to packed width.
  Value realPacked =
      rewriter.create<arith::ExtUIOp>(loc, packedIntType, realInt);
  Value imagPacked =
      rewriter.create<arith::ExtUIOp>(loc, packedIntType, imagInt);

  // Shift imaginary part to upper bits.
  Value shiftAmount = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(
               packedIntType, realFloat.getType().getIntOrFloatBitWidth()));
  Value imagPackedShifted =
      rewriter.create<arith::ShLIOp>(loc, imagPacked, shiftAmount);

  // Combine real and imaginary parts using OR
  return rewriter.create<arith::OrIOp>(loc, realPacked, imagPackedShifted);
}

/// Pack real and imaginary float attributes into an integer attribute.
static IntegerAttr packComplexComponents(OpBuilder &rewriter, Location loc,
                                         FloatAttr realFloat,
                                         FloatAttr imagFloat) {
  assert(realFloat.getType() == imagFloat.getType() &&
         "real and imaginary float types must match");
  Type componentIntType =
      rewriter.getIntegerType(imagFloat.getType().getIntOrFloatBitWidth());
  Type packedIntType =
      rewriter.getIntegerType(realFloat.getType().getIntOrFloatBitWidth() * 2);

  // Cast float components to integer using bitcast
  APInt realInt = realFloat.getValue().bitcastToAPInt();
  APInt imagInt = imagFloat.getValue().bitcastToAPInt();

  // Extend to packed width.
  APInt realPacked = realInt.zextOrTrunc(packedIntType.getIntOrFloatBitWidth());
  APInt imagPacked = imagInt.zextOrTrunc(packedIntType.getIntOrFloatBitWidth());

  // Shift imaginary part to upper bits.
  APInt imagPackedShifted =
      imagPacked.shl(componentIntType.getIntOrFloatBitWidth());

  // Combine real and imaginary parts using OR.
  return IntegerAttr::get(packedIntType, realPacked | imagPackedShifted);
}

namespace {

/// Convert complex.im|complex.re to extract imaginary/real part from packed
/// i64.
template <typename OpType>
struct ConvertComplexImReToArith : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op,
                  typename OpConversionPattern<OpType>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = adaptor.getComplex();
    Type integerType = operand.getType();
    if (!integerType.isInteger(64) && !integerType.isInteger(128))
      return failure();
    Type halfIntegerType =
        rewriter.getIntegerType(integerType.getIntOrFloatBitWidth() / 2);
    Type floatType = halfIntegerType == rewriter.getI32Type()
                         ? rewriter.getF32Type()
                         : rewriter.getF64Type();
    Value realBits =
        rewriter.create<arith::TruncIOp>(loc, halfIntegerType, operand);

    if constexpr (std::is_same_v<OpType, complex::ReOp>) {
      rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, floatType, realBits);
      return success();
    }

    if constexpr (std::is_same_v<OpType, complex::ImOp>) {
      Value shiftAmount = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(
                   integerType, integerType.getIntOrFloatBitWidth() / 2));
      Value shiftedInput =
          rewriter.create<arith::ShRUIOp>(loc, operand, shiftAmount);
      Value imagBits =
          rewriter.create<arith::TruncIOp>(loc, halfIntegerType, shiftedInput);
      rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, floatType, imagBits);
      return success();
    }

    return failure();
  }
};

/// Convert complex.constant to packed i64 representation
struct ConvertComplexConstantToArith
    : public OpConversionPattern<complex::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the complex constant value
    auto complexAttr = mlir::cast<ArrayAttr>(op.getValue());
    auto realAttr = mlir::cast<FloatAttr>(complexAttr[0]);
    auto imagAttr = mlir::cast<FloatAttr>(complexAttr[1]);

    // Create constant operations for real and imaginary parts
    Value realPart = rewriter.create<arith::ConstantOp>(loc, realAttr);
    Value imagPart = rewriter.create<arith::ConstantOp>(loc, imagAttr);

    // Pack the real and imaginary parts into a single i64 value
    Value packedResult =
        packComplexComponents(rewriter, loc, realPart, imagPart);
    rewriter.replaceOp(op, packedResult);
    return success();
  }
};

/// Convert complex.create to pack the real and imaginary parts into a single
/// integer value.
struct CreateComplexOpConverter
    : public OpConversionPattern<complex::CreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::CreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, packComplexComponents(rewriter, op.getLoc(),
                                                 adaptor.getReal(),
                                                 adaptor.getImaginary()));
    return success();
  }
};

/// Convert `memref.global` if it has an illegal type attribute.
struct MemRefGlobalConverterPattern
    : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType convertedType = llvm::dyn_cast_if_present<MemRefType>(
        getTypeConverter()->convertType(op.getType()));
    if (!convertedType)
      return failure();

    // Convert the initial value to the new type.
    auto initialValue =
        llvm::dyn_cast_if_present<ElementsAttr>(op.getInitialValueAttr());
    if (!initialValue || !isa<RankedTensorType>(initialValue.getType()))
      return failure();

    auto convertedInitialValueType =
        cast<RankedTensorType>(initialValue.getType())
            .clone(convertedType.getElementType());

    // Unfortunately, we have to repack here since we are converting to an
    // integer; a simple bitcast is not possible since MLIR treats the complex
    // elements as ArrayAttr.
    if (auto splatAttr = dyn_cast<SplatElementsAttr>(initialValue)) {
      auto splatValue = splatAttr.getSplatValue<ArrayAttr>();
      assert(splatValue.size() == 2 && "complex splat must have 2 elements");
      auto realValue = cast<FloatAttr>(splatValue[0]);
      auto imagValue = cast<FloatAttr>(splatValue[1]);
      auto packedValue =
          packComplexComponents(rewriter, op.getLoc(), realValue, imagValue);
      initialValue =
          SplatElementsAttr::get(convertedInitialValueType, packedValue);
    } else {
      // Fallback to element-wise conversion.
      if (convertedInitialValueType.getElementType().isInteger(64)) {
        SmallVector<int64_t> packedValues;
        for (auto element : initialValue.getValues<ArrayAttr>()) {
          auto realValue = cast<FloatAttr>(element[0]);
          auto imagValue = cast<FloatAttr>(element[1]);
          auto packedValue = packComplexComponents(rewriter, op.getLoc(),
                                                   realValue, imagValue);
          packedValues.push_back(packedValue.getInt());
        }
        initialValue =
            DenseIntElementsAttr::get(convertedInitialValueType, packedValues);
      } else {
        SmallVector<APInt> packedValues;
        for (auto element : initialValue.getValues<ArrayAttr>()) {
          auto realValue = cast<FloatAttr>(element[0]);
          auto imagValue = cast<FloatAttr>(element[1]);
          auto packedValue = packComplexComponents(rewriter, op.getLoc(),
                                                   realValue, imagValue);
          packedValues.push_back(packedValue.getValue());
        }
        initialValue =
            DenseIntElementsAttr::get(convertedInitialValueType, packedValues);
      }
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.setType(convertedType);
      op.setInitialValueAttr(initialValue);
    });
    return success();
  }
};

// Generic pattern that performs complex -> integer of same bit width
// type conversion.
class GenericTypeConverter : public ConversionPattern {
public:
  GenericTypeConverter(const llvm::StringSet<> &convertOpGenerically,
                       TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context),
        convertOpGenerically(convertOpGenerically) {}

  /// Specifies whether an operation can be converted using the
  /// GenericTypeConverter.
  static bool isSupportedOp(Operation *op,
                            const llvm::StringSet<> &convertOpGenerically) {
    if (convertOpGenerically.contains(op->getName().getStringRef()))
      return true;
    mlir::Dialect *dialect = op->getDialect();
    if (isa_and_nonnull<memref::MemRefDialect>(dialect))
      return isa<memref::AllocaOp, memref::AllocOp, memref::CastOp,
                 memref::CollapseShapeOp, memref::CopyOp, memref::DeallocOp,
                 memref::DimOp, memref::ExpandShapeOp,
                 memref::ExtractAlignedPointerAsIndexOp,
                 memref::ExtractStridedMetadataOp, memref::GetGlobalOp,
                 memref::LoadOp, memref::MemorySpaceCastOp, memref::ReallocOp,
                 memref::ReinterpretCastOp, memref::ReshapeOp, memref::StoreOp,
                 memref::SubViewOp, memref::TransposeOp, memref::ViewOp>(op);
    if (isa_and_nonnull<tensor::TensorDialect>(dialect))
      return isa<tensor::CollapseShapeOp, tensor::DimOp, tensor::EmptyOp,
                 tensor::ExpandShapeOp, tensor::ExtractOp,
                 tensor::ExtractSliceOp, tensor::GenerateOp, tensor::InsertOp,
                 tensor::InsertSliceOp, tensor::PadOp,
                 tensor::ParallelInsertSliceOp, tensor::RankOp,
                 tensor::YieldOp>(op);
    if (isa_and_nonnull<bufferization::BufferizationDialect>(dialect))
      return isa<bufferization::AllocTensorOp, bufferization::DeallocTensorOp,
                 bufferization::ToBufferOp, bufferization::ToTensorOp>(op);
    if (isa_and_nonnull<linalg::LinalgDialect>(dialect))
      return isa<linalg::LinalgOp, linalg::YieldOp>(op);
    if (isa_and_nonnull<arith::ArithDialect>(dialect))
      return isa<arith::SelectOp>(op);
    return false;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isSupportedOp(op, convertOpGenerically))
      return failure();

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();

    auto *newOp = Operation::create(
        op->getLoc(), op->getName(), resultTypes, operands, op->getAttrs(),
        op->getPropertiesStorage(), op->getSuccessors(), op->getNumRegions());
    for (auto regions : llvm::zip(op->getRegions(), newOp->getRegions())) {
      Region &before = std::get<0>(regions);
      Region &parent = std::get<1>(regions);
      rewriter.inlineRegionBefore(before, parent, parent.end());
      if (failed(rewriter.convertRegionTypes(&parent, *typeConverter)))
        return failure();
    }
    rewriter.insert(newOp);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

  const llvm::StringSet<> &convertOpGenerically;
};
} // namespace

/// Returns true if the operation is nested in a `gpu.module` operation.
static bool isNestedInGPUModule(Operation *op) {
  return op->getParentOfType<gpu::GPUModuleOp>();
}

namespace {
class ConvertComplexToStandardExt
    : public mlir::impl::ConvertComplexToStandardExtBase<
          ConvertComplexToStandardExt> {
public:
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Convert complex dialect operations to arith. These patterns are
    // defined upstream and use `complex.create`/`complex.re`/`complex.im`
    // to convert complex operations to more primitive scalar arith.
    {
      RewritePatternSet patterns(op->getContext());
      populateComplexToStandardConversionPatterns(
          patterns, complex::ComplexRangeFlags::none);

      ConversionTarget target(*op->getContext());
      target.addLegalDialect<arith::ArithDialect, math::MathDialect>();
      target.addLegalOp<complex::CreateOp, complex::ImOp, complex::ReOp>();
      if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
        op->emitError(
            "failed to convert complex operations to arith operations");
        return signalPassFailure();
      }
    }

    // Completely eliminate complex operations, even in tensor element types,
    // linal operations, control flow operations, etc.
    TypeConverter typeConverter;
    typeConverter.addConversion(
        [&](Type type) -> std::optional<Type> { return type; });
    typeConverter.addConversion([&](ComplexType type) -> std::optional<Type> {
      Type elementType = type.getElementType();
      if (elementType.isF32())
        return IntegerType::get(op->getContext(), 64);
      if (elementType.isF64())
        return IntegerType::get(op->getContext(), 128);
      return std::nullopt;
    });
    typeConverter.addConversion([&](ShapedType type) -> std::optional<Type> {
      if (!isa<ComplexType>(type.getElementType()))
        return std::nullopt;
      std::optional<Type> elementType =
          typeConverter.convertType(type.getElementType());
      if (!elementType)
        return std::nullopt;
      return type.clone(*elementType);
    });

    // Helper function to materialize conversion between types.
    // Used for both source and target materialization.
    auto materializeConversion = [](OpBuilder &builder, Type resultType,
                                    ValueRange inputs, Location loc) -> Value {
      if (inputs.size() == 1) {
        Value input = inputs[0];
        Type inputType = input.getType();
        if (isa<ShapedType>(inputType) && isa<ShapedType>(resultType)) {
          if (executor::BufferBitcastOp::areCastCompatible(
                  TypeRange(inputType), TypeRange(resultType))) {
            return builder
                .create<executor::BufferBitcastOp>(loc, resultType, input)
                .getResult();
          }
        }
      }
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };

    typeConverter.addTargetMaterialization(materializeConversion);
    typeConverter.addSourceMaterialization(materializeConversion);

    llvm::StringSet<> convertOpGenericallySet;
    for (StringRef opName : convertOpGenerically)
      convertOpGenericallySet.insert(opName);

    ConversionTarget target(*op->getContext());
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (convertOpGenericallySet.contains(op->getName().getStringRef()))
        return typeConverter.isLegal(op);
      if (!isNestedInGPUModule(op))
        return true;
      if (isa_and_nonnull<complex::ComplexDialect>(op->getDialect()))
        return false;
      if (auto globalOp = dyn_cast<memref::GlobalOp>(op))
        return typeConverter.isLegal(globalOp.getType());
      if (GenericTypeConverter::isSupportedOp(op, convertOpGenericallySet))
        return typeConverter.isLegal(op);
      if (auto funcOp = dyn_cast<func::FuncOp>(op))
        return typeConverter.isSignatureLegal(funcOp.getFunctionType());
      if (isa<func::CallOp, func::ReturnOp>(op))
        return typeConverter.isLegal(op);
      return true;
    });

    RewritePatternSet patterns(op->getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);
    // clang-format off
    patterns.add<
      ConvertComplexConstantToArith,
      ConvertComplexImReToArith<complex::ImOp>,
      ConvertComplexImReToArith<complex::ReOp>,
      CreateComplexOpConverter,
      MemRefGlobalConverterPattern
    >(typeConverter, patterns.getContext());
    // clang-format on
    patterns.add<GenericTypeConverter>(convertOpGenericallySet, typeConverter,
                                       patterns.getContext());

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      op->emitError("failed to convert complex operations to arith operations");
      return signalPassFailure();
    }
  }
};

} // namespace
