//===- SpecialFloatsTypeConversion.cpp ------------------------------------===//
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
/// Definition of the `kernel-special-floats-type-conversion` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::kernel {
#define GEN_PASS_DEF_KERNELSPECIALFLOATSTYPECONVERSIONPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace mlir::kernel

using namespace mlir;
using namespace mlir::kernel;

/// Returns MLIR type from string type name.
/// Only `special` type parsing is provided.
static std::optional<FloatType> parseSpecialType(MLIRContext *ctx,
                                                 StringRef name) {
  Builder b(ctx);
  return llvm::StringSwitch<std::optional<FloatType>>(name)
      .Case("f4E2M1FN", b.getType<Float4E2M1FNType>())
      .Case("f8E4M3FN", b.getType<Float8E4M3FNType>())
      .Default(std::nullopt);
}

/// We process only selected operations, as mentioned below.
static bool shouldProcessOp(Operation *op) {
  // scf, cf dialect ops should be processed.
  if (isa_and_nonnull<scf::SCFDialect, cf::ControlFlowDialect,
                      func::FuncDialect>(op->getDialect()))
    return true;
  // memref.load, memref.store and memref.reinterpret_cast ops should be
  // processed.
  if (isa<memref::LoadOp, memref::StoreOp, memref::ReinterpretCastOp>(op))
    return true;
  return false;
}

namespace {

class SpecialFloatsTypeConverter : public TypeConverter {
public:
  SpecialFloatsTypeConverter(ArrayRef<Type> specialTypes) {
    addConversion([specialTypes](Type type) -> std::optional<Type> {
      if (llvm::is_contained(specialTypes, type))
        return IntegerType::get(type.getContext(),
                                type.getIntOrFloatBitWidth());
      return type;
    });
    addConversion([specialTypes](MemRefType type) -> std::optional<Type> {
      if (llvm::is_contained(specialTypes, type.getElementType()))
        return type.clone(
            IntegerType::get(type.getContext(), type.getElementTypeBitWidth()));
      return type;
    });
    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return Value();
      return builder.create<arith::BitcastOp>(loc, resultType, inputs.front())
          .getOut();
    });
    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return Value();
      return builder.create<arith::BitcastOp>(loc, resultType, inputs.front())
          .getOut();
    });
  }
};

// Generic pattern that performs `special float -> integer of same bit width`
// type conversion.
class GenericFloatToIntegerTypeConverter : public ConversionPattern {
public:
  GenericFloatToIntegerTypeConverter(TypeConverter &typeConverter,
                                     MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!shouldProcessOp(op))
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

class KernelSpecialFloatsTypeConversionPass
    : public kernel::impl::KernelSpecialFloatsTypeConversionPassBase<
          KernelSpecialFloatsTypeConversionPass> {
  using Base::Base;
  void runOnOperation() override {
    gpu::GPUModuleOp gpuModuleOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Parse `specialTypes` flag and get MLIR types.
    SmallVector<Type> parsedSpecialTypes;

    // If no special types list is provided, `f8E4M3FN` and `f4E2M1FN` are added
    // by default.
    if (specialTypes.empty()) {
      specialTypes.push_back("f4E2M1FN");
      specialTypes.push_back("f8E4M3FN");
    }
    for (StringRef specialTypeStr : specialTypes) {
      std::optional<FloatType> maybeSpecialType =
          parseSpecialType(&getContext(), specialTypeStr);
      if (!maybeSpecialType) {
        emitError(UnknownLoc::get(&getContext()),
                  "could not parse input string type '" + specialTypeStr +
                      "' to a known floating-point type");
        return signalPassFailure();
      }
      parsedSpecialTypes.push_back(*maybeSpecialType);
    }

    SpecialFloatsTypeConverter typeConverter(parsedSpecialTypes);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect>();
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });

    RewritePatternSet patterns(ctx);
    patterns.add<GenericFloatToIntegerTypeConverter>(typeConverter, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    if (failed(applyFullConversion(gpuModuleOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
