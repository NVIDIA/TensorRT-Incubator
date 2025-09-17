//===- EmulateUnsupportedFloats.cpp ---------------------------------------===//
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
/// Implementation of the `executor-emulate-unsupported-floats` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace executor {
#define GEN_PASS_DEF_EXECUTOREMULATEUNSUPPORTEDFLOATSPASS
#include "mlir-executor/Executor/Transforms/Passes.h.inc"
} // namespace executor
} // namespace mlir

using namespace mlir;
using namespace mlir::executor;

/// Returns MLIR type from string type name.
/// Other float types are not added here because they are supported by executor
/// runtime. In future, other quantized types should be added here.
static std::optional<FloatType> parseUnsupportedType(MLIRContext *ctx,
                                                     StringRef name) {
  Builder b(ctx);
  return llvm::StringSwitch<std::optional<FloatType>>(name)
      .Case("f4E2M1FN", b.getType<Float4E2M1FNType>())
      .Default(std::nullopt);
}

namespace {
/// Pattern to convert executor constant with unsupported type to fp32 constant.
struct ExecutorConstantConverter
    : public OpConversionPattern<executor::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(executor::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();

    // Create f16 constant.
    if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
      rewriter.replaceOpWithNewOp<executor::ConstantOp>(
          op, resultType,
          FloatAttr::get(resultType, floatAttr.getValueAsDouble()));
      return success();
    }

    return failure();
  }
};

/// Generic pattern that rewrites any op by rewriting its operands and
/// results.
struct GenericEmulationConversion : public ConversionPattern {
public:
  GenericEmulationConversion(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isa_and_present<executor::ExecutorDialect>(op->getDialect()))
      return success();

    if (typeConverter->isLegal(op))
      return success();

    // Ops with attribute needs to be handled specially. As of now, only
    // `executor.constant` is handled. In future, remaining ops from the list
    // below needs to be handled.
    if (isa<executor::ConstantOp, executor::DataSegmentOp, executor::GlobalOp,
            executor::FuncOp, executor::CallOp, executor::GetOffsetOp,
            executor::AllocaOp>(op))
      return failure();

    // Region ops are not handled
    if (op->getNumRegions() != 0)
      return failure();

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();

    auto *newOp = Operation::create(
        op->getLoc(), op->getName(), resultTypes, operands, op->getAttrs(),
        op->getPropertiesStorage(), op->getSuccessors(), op->getRegions());
    rewriter.insert(newOp);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

class ExecutorEmulateUnsupportedFloatsPass
    : public executor::impl::ExecutorEmulateUnsupportedFloatsPassBase<
          ExecutorEmulateUnsupportedFloatsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    // Parse `unsupportedSourceTypes` flag and get MLIR types.
    SmallVector<Type> typesToEmulate;

    // If no unsupported types list is provided, at least `f4E2M1FN` which we
    // know is not supported by executor runtime.
    if (unsupportedSourceTypes.empty())
      unsupportedSourceTypes.push_back("f4E2M1FN");

    for (StringRef unsupportedTypeStr : unsupportedSourceTypes) {
      std::optional<FloatType> maybeUnsupportedType =
          parseUnsupportedType(&getContext(), unsupportedTypeStr);
      if (!maybeUnsupportedType) {
        emitError(getOperation()->getLoc(),
                  "could not parse input string type '" + unsupportedTypeStr +
                      "' to a known floating-point type");
        return signalPassFailure();
      }
      typesToEmulate.push_back(*maybeUnsupportedType);
    }

    // We always emulate operations in `f16` type.
    Type targetType = Float16Type::get(&getContext());

    TypeConverter typeConverter;
    typeConverter.addConversion([&](Type t) -> std::optional<Type> {
      // If `t` is in `typesToEmulate`, return target type.
      if (llvm::is_contained(typesToEmulate, t))
        return targetType;
      return t;
    });
    typeConverter.addTargetMaterialization(
        [](OpBuilder &b, Type target, ValueRange inputs, Location loc) {
          if (inputs.size() != 1)
            return Value();
          return b.create<executor::ExtfOp>(loc, target, inputs.front())
              .getResult();
        });
    typeConverter.addSourceMaterialization(
        [](OpBuilder &b, Type source, ValueRange inputs, Location loc) {
          if (inputs.size() != 1)
            return Value();
          return b.create<executor::TruncfOp>(loc, source, inputs.front())
              .getResult();
        });

    ConversionTarget target(getContext());
    // Don't try to legalize functions and other ops that don't need expansion.
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    target.addDynamicallyLegalDialect<executor::ExecutorDialect>(
        [&](Operation *op) -> std::optional<bool> {
          // executor.extf f4E2M1FN -> f16 is legal
          if (auto extf = dyn_cast_or_null<executor::ExtfOp>(op)) {
            return isa<Float4E2M1FNType>(extf.getOperand().getType()) &&
                   extf.getResult().getType().isF16();
          }
          // executor.truncf f16 -> f4E2M1FN is legal
          if (auto extf = dyn_cast_or_null<executor::TruncfOp>(op)) {
            return isa<Float4E2M1FNType>(extf.getResult().getType()) &&
                   extf.getOperand().getType().isF16();
          }
          return typeConverter.isLegal(op);
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<GenericEmulationConversion, ExecutorConstantConverter>(
        typeConverter, &getContext());
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      emitError(getOperation()->getLoc(),
                "Failed to emulate unsupported types by the executor dialect.");
      return signalPassFailure();
    }
  }
};
} // namespace
