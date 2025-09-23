//===- PlanToExecutor.cpp -------------------------------------------------===//
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
/// Implementation of the `convert-plan-to-executor` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTPLANTOEXECUTORPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

class GenericStructuralConverter : public ConversionPattern {
public:
  GenericStructuralConverter(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (isa<RegionBranchOpInterface>(op) ||
        isa<FunctionOpInterface, arith::ConstantOp, memref::GlobalOp>(op))
      return failure();

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();
    auto *newOp = Operation::create(
        op->getLoc(), op->getName(), resultTypes, operands, op->getAttrs(),
        op->getPropertiesStorage(), op->getSuccessors(), op->getNumRegions());
    for (auto [before, parent] :
         llvm::zip(op->getRegions(), newOp->getRegions())) {
      rewriter.inlineRegionBefore(before, parent, parent.end());
      if (failed(rewriter.convertRegionTypes(&parent, *typeConverter)))
        return failure();
    }
    rewriter.insert(newOp);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

/// Rewrite `memref.global` if it has an illegal type attribute.
struct MemRefGlobalConverterPattern
    : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType convertedType = llvm::dyn_cast_if_present<MemRefType>(
        getTypeConverter()->convertType(op.getType()));
    if (!convertedType || convertedType == op.getType())
      return failure();
    rewriter.modifyOpInPlace(op, [&]() { op.setType(convertedType); });
    return success();
  }
};

/// Rewrite `executor.global` if it has an illegal type attribute.
struct ExecutorGlobalConverterPattern
    : public OpConversionPattern<executor::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(executor::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType convertedType = llvm::dyn_cast_if_present<MemRefType>(
        getTypeConverter()->convertType(op.getType()));
    if (!convertedType || convertedType == op.getType())
      return failure();
    rewriter.modifyOpInPlace(op, [&]() { op.setType(convertedType); });
    return success();
  }
};

/// Rewrite `arith.constant` so that the encodings are properly converted.
struct ConstantOpConverter : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast_or_null<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultType)
      return failure();

    TypedAttr constVal = op.getValue();
    if (constVal.getType() != resultType) {
      auto elements = dyn_cast<DenseElementsAttr>(constVal);
      if (!elements)
        return failure();

      auto rtt = dyn_cast<RankedTensorType>(elements.getType());
      if (!rtt)
        return failure();

      // The types should only differ by the encoding.
      if (rtt.getShape() != resultType.getShape() ||
          rtt.getElementType() != resultType.getElementType())
        return failure();

      constVal = DenseElementsAttr::getFromRawBuffer(resultType,
                                                     elements.getRawData());
    }

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, constVal);
    return success();
  }
};
} // namespace

/// Convert 'plan' dialect or 'tensorrt' dialect bounds into 'executor' bounds
/// attributes.
static Attribute convertArgOrResultAttr(OpBuilder &b, Attribute attr) {
  MLIRContext *ctx = attr.getContext();
  if (auto planAttr = dyn_cast<plan::BoundsAttr>(attr)) {
    if (planAttr.isShapeBound())
      return executor::DimensionBoundsAttr::get(ctx, planAttr.getMinShape(),
                                                planAttr.getMaxShape());
    if (planAttr.isValueBound())
      return executor::ValueBoundsAttr::get(ctx, planAttr.getMinValues(),
                                            planAttr.getMaxValues());
  }
  return attr;
}

/// Convert 'plan' dialect arg|result attributes into 'executor' dialect
/// attributes for all function arg attrs and res attrs.
static void convertArgAndResultAttrs(OpBuilder &b, func::FuncOp op) {
  StringRef executorShapeBoundsAttrName =
      mlir::executor::ExecutorDialect::getShapeBoundsAttrName();
  StringRef executorValueBoundsAttrName =
      mlir::executor::ExecutorDialect::getValueBoundsAttrName();

  StringRef planShapeBoundsAttrName =
      mlir::plan::PlanDialect::kShapeBoundsAttrName;
  StringRef planValueBoundsAttrName =
      mlir::plan::PlanDialect::kValueBoundsAttrName;

  for (unsigned idx = 0; idx < op.getNumArguments(); idx++) {
    if (auto attr = op.getArgAttr(idx, planShapeBoundsAttrName)) {
      op.removeArgAttr(idx, planShapeBoundsAttrName);
      op.setArgAttr(idx, executorShapeBoundsAttrName,
                    convertArgOrResultAttr(b, attr));
    }
    if (auto attr = op.getArgAttr(idx, planValueBoundsAttrName)) {
      op.removeArgAttr(idx, planValueBoundsAttrName);
      op.setArgAttr(idx, executorValueBoundsAttrName,
                    convertArgOrResultAttr(b, attr));
    }

    if (auto attr = op.getArgAttr(idx, plan::PlanDialect::kResultArgAttrName)) {
      op.removeArgAttr(idx, plan::PlanDialect::kResultArgAttrName);
      op.setArgAttr(idx, executor::ExecutorDialect::kResultArgAttrName, attr);
    }
    if (auto attr =
            op.getArgAttr(idx, plan::PlanDialect::kDonationArgAttrName)) {
      op.setArgAttr(idx, executor::ExecutorDialect::kDonatedArgAttrName, attr);
      op.removeArgAttr(idx, plan::PlanDialect::kDonationArgAttrName);
    }
  }
  for (unsigned idx = 0; idx < op.getNumResults(); idx++) {
    if (auto attr = op.getResultAttr(idx, planShapeBoundsAttrName)) {
      op.removeResultAttr(idx, b.getStringAttr(planShapeBoundsAttrName));
      op.setResultAttr(idx, executorShapeBoundsAttrName,
                       convertArgOrResultAttr(b, attr));
    }
    if (auto attr = op.getResultAttr(idx, planValueBoundsAttrName)) {
      op.removeResultAttr(idx, b.getStringAttr(planValueBoundsAttrName));
      op.setResultAttr(idx, executorValueBoundsAttrName,
                       convertArgOrResultAttr(b, attr));
    }
  }
}

/// Returns true if the given `op` is considered as legal for plan-to-executor
/// conversion.
static bool isLegalOp(Operation *op, const TypeConverter &typeConverter) {
  auto isLegalType = [&](Type t) { return typeConverter.isLegal(t); };
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    return llvm::all_of(funcOp.getArgumentTypes(), isLegalType) &&
           llvm::all_of(funcOp.getResultTypes(), isLegalType) &&
           llvm::all_of(funcOp.getFunctionBody().getArgumentTypes(),
                        isLegalType);
  }

  for (Region &region : op->getRegions()) {
    if (!llvm::all_of(region.getArgumentTypes(), isLegalType))
      return false;
  }

  return llvm::all_of(op->getOperandTypes(), isLegalType) &&
         llvm::all_of(op->getResultTypes(), isLegalType);
}

namespace {
class PlanToExecutorPass
    : public impl::ConvertPlanToExecutorPassBase<PlanToExecutorPass> {
  using Base::Base;

  void runOnOperation() override {

    TypeConverter typeConverter;
    typeConverter.addConversion(
        [](Type t) -> std::optional<Type> { return t; });

    typeConverter.addConversion(
        [&typeConverter](RankedTensorType rtt) -> std::optional<Type> {
          if (!rtt.getEncoding())
            return rtt;
          std::optional<Attribute> convertedEncoding =
              typeConverter.convertTypeAttribute(rtt, rtt.getEncoding());
          if (!convertedEncoding)
            return rtt;
          return RankedTensorType::get(rtt.getShape(), rtt.getElementType(),
                                       *convertedEncoding);
        });
    typeConverter.addConversion(
        [&typeConverter](BaseMemRefType type) -> std::optional<Type> {
          if (!type.getMemorySpace())
            return type;
          std::optional<Attribute> convertedEncoding =
              typeConverter.convertTypeAttribute(type, type.getMemorySpace());
          if (!convertedEncoding)
            return type;
          if (auto rankedMemref = dyn_cast<MemRefType>(type))
            return MemRefType::get(
                rankedMemref.getShape(), rankedMemref.getElementType(),
                rankedMemref.getLayout(), *convertedEncoding);
          return UnrankedMemRefType::get(type.getElementType(),
                                         *convertedEncoding);
        });

    typeConverter.addTypeAttributeConversion(
        [](ShapedType type, plan::MemorySpaceAttr attr)
            -> TypeConverter::AttributeConversionResult {
          auto getSpace = [&](executor::MemoryType x) {
            return executor::MemoryTypeAttr::get(type.getContext(), x);
          };
          switch (attr.getValue()) {
          case plan::MemorySpace::device:
            return getSpace(executor::MemoryType::device);
          case plan::MemorySpace::host:
            return getSpace(executor::MemoryType::host);
          case plan::MemorySpace::host_pinned:
            return getSpace(executor::MemoryType::host_pinned);
          case plan::MemorySpace::unified:
            return getSpace(executor::MemoryType::unified);
          case plan::MemorySpace::unknown:
            return Attribute{};
          }
          return TypeConverter::AttributeConversionResult::abort();
        });

    typeConverter.addSourceMaterialization([&](OpBuilder &builder,
                                               Type resultType,
                                               ValueRange inputs,
                                               Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
    typeConverter.addTargetMaterialization([&](OpBuilder &builder,
                                               Type resultType,
                                               ValueRange inputs,
                                               Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    ConversionTarget target(getContext());
    target.addIllegalDialect<plan::PlanDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&typeConverter](func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<memref::GlobalOp>(
        [&typeConverter](memref::GlobalOp op) {
          return typeConverter.isLegal(op.getType());
        });
    target.addDynamicallyLegalOp<executor::GlobalOp>(
        [&typeConverter](executor::GlobalOp op) {
          return typeConverter.isLegal(op.getType());
        });
    target.addLegalOp<gpu::GPUModuleOp>();
    target.markUnknownOpDynamicallyLegal([&typeConverter](Operation *op) {
      if (!isLegalOp(op, typeConverter))
        return false;

      return mlir::isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             mlir::isLegalForBranchOpInterfaceTypeConversionPattern(
                 op, typeConverter) ||
             mlir::isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });
    target.markOpRecursivelyLegal<gpu::GPUModuleOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<GenericStructuralConverter, ConstantOpConverter,
                 MemRefGlobalConverterPattern, ExecutorGlobalConverterPattern>(
        typeConverter, &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    ModuleOp module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(module->getLoc())
          << "failed to apply plan-to-executor conversions";
      return signalPassFailure();
    }

    // Convert the result/argument function bounds attributes.
    OpBuilder builder(&getContext());
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      convertArgAndResultAttrs(builder, func);

      // TODO: should we just create metadata here?
      // Update the shape function attribute to executor dialect attribute.
      if (auto shapeFuncSym = func->getAttrOfType<SymbolRefAttr>(
              plan::PlanDialect::kShapeFuncAttrName)) {
        func->removeAttr(plan::PlanDialect::kShapeFuncAttrName);
        func->setAttr(executor::ExecutorDialect::kShapeFuncAttrName,
                      shapeFuncSym);
      }
    }
  }
};
} // namespace
