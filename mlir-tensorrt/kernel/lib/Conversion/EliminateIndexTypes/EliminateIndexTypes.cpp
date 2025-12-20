//===- EliminateIndexTypes.cpp -------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION &
// AFFILIATES.
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
/// A pass that replaces all uses of the `index` type with a fixed width
/// signless integer in both scalar and shaped types. The pass is intended for
/// device code (e.g., code nested under `gpu.module`) but is applied
/// generically; legality is defined via a `TypeConverter`.
///
//===----------------------------------------------------------------------===//

#include "mlir-kernel/Conversion/Passes.h" // IWYU pragma: keep NOLINT
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"

namespace mlir {
#define GEN_PASS_DEF_ELIMINATEINDEXTYPES
#include "mlir-kernel/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Replace an `arith.constant` that produces an index (or shaped index) with a
/// constant of the converted integer type.
class ConvertArithConstant : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = getTypeConverter();
    Type convertedType = typeConverter->convertType(op.getType());
    if (!convertedType)
      return failure();

    Attribute value = op.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      if (!isa<IntegerType>(convertedType))
        return failure();
      unsigned width = convertedType.getIntOrFloatBitWidth();
      value = rewriter.getIntegerAttr(convertedType,
                                      intAttr.getValue().sextOrTrunc(width));
    } else if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(value)) {
      auto convertedShaped = dyn_cast<ShapedType>(convertedType);
      if (!convertedShaped)
        return failure();
      Type elementType = convertedShaped.getElementType();
      if (!isa<IntegerType>(elementType))
        return failure();
      unsigned width = elementType.getIntOrFloatBitWidth();
      value = denseAttr.mapValues(elementType, [width](const APInt &v) {
        return v.sextOrTrunc(width);
      });
    } else {
      return failure();
    }

    auto typed = cast<TypedAttr>(value);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, typed.getType(), typed);
    return success();
  }
};

/// Convert `arith.index_cast` to the appropriate extend/truncate when the
/// target index type differs in bitwidth. If the bitwidth is unchanged, the
/// cast is folded away.
class ConvertIndexCast : public OpConversionPattern<arith::IndexCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type targetType = getTypeConverter()->convertType(op.getType());
    if (!targetType)
      return failure();

    Type sourceType = adaptor.getIn().getType();
    if (targetType == sourceType) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }

    if (!targetType.isIntOrIndex() || !sourceType.isIntOrIndex())
      return failure();

    unsigned targetBits = targetType.getIntOrFloatBitWidth();
    unsigned sourceBits = sourceType.getIntOrFloatBitWidth();
    if (targetBits > sourceBits) {
      rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, targetType,
                                                  adaptor.getIn());
      return success();
    }
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, targetType,
                                                 adaptor.getIn());
    return success();
  }
};

/// Convert `arith.index_castui` similarly but with zero-extend semantics.
class ConvertIndexCastUI : public OpConversionPattern<arith::IndexCastUIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type targetType = getTypeConverter()->convertType(op.getType());
    if (!targetType)
      return failure();

    Type sourceType = adaptor.getIn().getType();
    if (targetType == sourceType) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }

    if (!targetType.isIntOrIndex() || !sourceType.isIntOrIndex())
      return failure();

    unsigned targetBits = targetType.getIntOrFloatBitWidth();
    unsigned sourceBits = sourceType.getIntOrFloatBitWidth();
    if (targetBits > sourceBits) {
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, targetType,
                                                  adaptor.getIn());
      return success();
    }
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, targetType,
                                                 adaptor.getIn());
    return success();
  }
};

/// Generic conversion pattern for `arith` operations.
class GenericArithConverter : public ConversionPattern {
public:
  GenericArithConverter(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isa_and_nonnull<arith::ArithDialect>(op->getDialect()))
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
};

/// Ensure the induction variable for `scf.for` uses the converted integer type
/// instead of `index`. The default SCF structural conversions only check result
/// types for legality, so explicitly rebuild the loop when the IV type is
/// illegal.
class ConvertForInductionVar : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = getTypeConverter();

    // Only trigger if the induction variable type is illegal.
    if (typeConverter->isLegal(op.getInductionVar().getType()))
      return failure();

    // Convert the loop body argument types (induction variable + iter args).
    if (failed(rewriter.convertRegionTypes(&op.getRegion(), *typeConverter)))
      return failure();

    scf::ForOp newOp = rewriter.create<scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs(), /*bodyBuilder=*/nullptr,
        op.getUnsignedCmp());
    newOp->setAttrs(op->getAttrs());

    // Drop the automatically created empty block and inline the converted body.
    rewriter.eraseBlock(newOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class EliminateIndexTypes
    : public mlir::impl::EliminateIndexTypesBase<EliminateIndexTypes> {
public:
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    IntegerType indexIntType = IntegerType::get(ctx, indexBitwidth);

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion(
        [&](IndexType) { return static_cast<Type>(indexIntType); });
    typeConverter.addConversion([&](ShapedType type) -> std::optional<Type> {
      if (!isa<IndexType>(type.getElementType()))
        return std::nullopt;
      return type.clone(indexIntType);
    });
    typeConverter.addConversion(
        [&](FunctionType funcTy) -> std::optional<Type> {
          SmallVector<Type> inputs, results;
          if (failed(typeConverter.convertTypes(funcTy.getInputs(), inputs)) ||
              failed(typeConverter.convertTypes(funcTy.getResults(), results)))
            return std::nullopt;
          return FunctionType::get(funcTy.getContext(), inputs, results);
        });

    auto materializeCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) -> Value {
      if (inputs.size() != 1)
        return {};
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    typeConverter.addTargetMaterialization(materializeCast);
    typeConverter.addSourceMaterialization(materializeCast);

    ConversionTarget target(*ctx);
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<scf::IfOp, scf::IndexSwitchOp>(
        [&](Operation *op) { return typeConverter.isLegal(op->getResults()); });
    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      return typeConverter.isLegal(op.getInductionVar().getType()) &&
             typeConverter.isLegal(op.getResultTypes());
    });
    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
      // We only have conversions for a subset of ops that use scf.yield
      // terminators.
      if (!isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::IndexSwitchOp>(
              op->getParentOp()))
        return true;
      return typeConverter.isLegal(op.getOperands());
    });
    target.addDynamicallyLegalOp<scf::WhileOp, scf::ConditionOp>(
        [&](Operation *op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalDialect<arith::ArithDialect>(
        [&](Operation *op) { return typeConverter.isLegal(op); });
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        if (auto functionType =
                dyn_cast<FunctionType>(funcOp.getFunctionType())) {
          return typeConverter.isSignatureLegal(functionType);
        }
      }
      return typeConverter.isLegal(op);
    });

    RewritePatternSet patterns(ctx);
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                        typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    scf::populateSCFStructuralTypeConversions(typeConverter, patterns);
    mlir::populateBranchOpInterfaceTypeConversionPattern(patterns,
                                                         typeConverter);
    patterns.add<ConvertArithConstant, ConvertIndexCast, ConvertIndexCastUI>(
        typeConverter, ctx);
    patterns.add<ConvertForInductionVar>(typeConverter, ctx);
    patterns.add<GenericArithConverter>(typeConverter, ctx);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      op->emitError() << "failed to eliminate index types";
      signalPassFailure();
    }
  }
};
} // namespace
