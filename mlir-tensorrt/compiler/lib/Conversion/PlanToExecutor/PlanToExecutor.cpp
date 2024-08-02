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
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
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

static constexpr llvm::StringRef kShapeBoundsAttrName =
    "tensorrt.shape_profile";
static constexpr llvm::StringRef kValueBoundsAttrName = "tensorrt.value_bounds";

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

/// Rewrite `arith.constant` so that the encodings are properly converted.
struct ConstantOpConverter : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()
                          ->convertType(op.getType())
                          .dyn_cast_or_null<RankedTensorType>();
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
static Attribute convertArgOrResultAttr(OpBuilder &b, Attribute attr,
                                        llvm::StringRef name) {
  MLIRContext *ctx = attr.getContext();
  if (auto planAttr = dyn_cast<plan::BoundsAttr>(attr)) {
    if (planAttr.isShapeBound())
      return executor::DimensionBoundsAttr::get(ctx, planAttr.getMinShape(),
                                                planAttr.getMaxShape());
    if (planAttr.isValueBound())
      return executor::ValueBoundsAttr::get(ctx, planAttr.getMinValues(),
                                            planAttr.getMaxValues());
  }
  if (auto trtAttr = dyn_cast<tensorrt::ShapeProfileAttr>(attr)) {
    if (name == kValueBoundsAttrName)
      return executor::ValueBoundsAttr::get(
          ctx, b.getI64TensorAttr(trtAttr.getMin()),
          b.getI64TensorAttr(trtAttr.getMax()));
    if (name == kShapeBoundsAttrName)
      return executor::DimensionBoundsAttr::get(
          ctx, b.getDenseI64ArrayAttr(trtAttr.getMin()),
          b.getDenseI64ArrayAttr(trtAttr.getMax()));
  }
  return attr;
}

/// Convert 'plan' dialect or 'tensorrt' dialect bounds into 'executor' bounds
/// attributes for all function arg attrs and res attrs.
static void convertArgAndResultAttrs(OpBuilder &b, func::FuncOp op) {
  for (unsigned idx = 0; idx < op.getNumArguments(); idx++) {
    if (auto attr = op.getArgAttr(idx, kShapeBoundsAttrName))
      op.setArgAttr(idx, kShapeBoundsAttrName,
                    convertArgOrResultAttr(b, attr, kShapeBoundsAttrName));
    if (auto attr = op.getArgAttr(idx, kValueBoundsAttrName))
      op.setArgAttr(idx, kValueBoundsAttrName,
                    convertArgOrResultAttr(b, attr, kValueBoundsAttrName));

    if (auto attr = op.getArgAttr(idx, plan::PlanDialect::kResultArgAttrName)) {
      op.removeArgAttr(idx, plan::PlanDialect::kResultArgAttrName);
      op.setArgAttr(idx, executor::ExecutorDialect::kResultArgAttrName, attr);
    }
  }
  for (unsigned idx = 0; idx < op.getNumResults(); idx++) {
    if (auto attr = op.getResultAttr(idx, kShapeBoundsAttrName))
      op.setResultAttr(idx, kShapeBoundsAttrName,
                       convertArgOrResultAttr(b, attr, kShapeBoundsAttrName));
    if (auto attr = op.getResultAttr(idx, kValueBoundsAttrName))
      op.setResultAttr(idx, kValueBoundsAttrName,
                       convertArgOrResultAttr(b, attr, kValueBoundsAttrName));
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
        [](RankedTensorType type, plan::MemorySpaceAttr attr) -> Attribute {
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
          llvm_unreachable("unknown plan::MemorySpace enumeration value");
        });
    typeConverter.addTypeAttributeConversion(
        [](MemRefType type, plan::MemorySpaceAttr attr) -> Attribute {
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
          llvm_unreachable("unknown plan::MemorySpace enumeration value");
        });

    ConversionTarget target(getContext());
    target.addLegalDialect<executor::ExecutorDialect>();
    target.addIllegalDialect<plan::PlanDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&typeConverter](func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<memref::GlobalOp>(
        [&typeConverter](memref::GlobalOp op) {
          return typeConverter.isLegal(op.getType());
        });
    target.markUnknownOpDynamicallyLegal([&typeConverter](Operation *op) {
      if (!isLegalOp(op, typeConverter))
        return false;

      if (op->hasTrait<OpTrait::ReturnLike>())
        return typeConverter.isLegal(op->getOperandTypes());

      if (isa<BranchOpInterface>(op)) {
        return mlir::isLegalForBranchOpInterfaceTypeConversionPattern(
            op, typeConverter);
      }
      if (isa<func::ReturnOp>(op)) {
        return isLegalForReturnOpTypeConversionPattern(op, typeConverter);
      }

      for (Region &region : op->getRegions()) {
        if (!typeConverter.isLegal(region.getArgumentTypes()))
          return false;
      }
      return true;
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<GenericStructuralConverter, ConstantOpConverter,
                 MemRefGlobalConverterPattern>(typeConverter, &getContext());
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
