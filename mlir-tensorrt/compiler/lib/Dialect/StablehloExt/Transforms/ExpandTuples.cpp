//===- CanonicalizeGather.cpp  --------------------------------------------===//
//
// The expand tuples pass logic is inspired from the XLA project
// `xla/mlir_hlo/mhlo/transforms/expand_hlo_tuples/expand_hlo_tuples.cc`
// and has the original license: Apache License v2.0. See
// https://github.com/openxla/xla/blob/main/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
// Changes are copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of the `stablehlo-ext-expand-tuples` pass.
///
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo_ext {
#define GEN_PASS_DEF_EXPANDTUPLESPASS
#define GEN_PASS_DECL_EXPANDTUPLESPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace stablehlo_ext
} // namespace mlir

using namespace mlir;
using namespace mlir::stablehlo;

namespace {

// This pass assumes the function to be expanded has no callees, to be specific,
// the function is more like the main function.
class ExpandTuplesPass
    : public stablehlo_ext::impl::ExpandTuplesPassBase<ExpandTuplesPass> {
public:
  using Base::Base;
  void convertMaybeNestedTuples(OpBuilder &builder, Location loc, Value input,
                                SmallVector<Value> &r) {
    for (unsigned i = 0; i < cast<TupleType>(input.getType()).size(); i++) {
      Value tupleElement =
          builder.create<stablehlo::GetTupleElementOp>(loc, input, i)
              .getResult();
      if (isa<TupleType>(tupleElement.getType()))
        convertMaybeNestedTuples(builder, loc, tupleElement, r);
      else
        r.push_back(tupleElement);
    }
  }

  void runOnOperation() override {
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });
    typeConverter.addConversion(
        [](TupleType tupleType, SmallVectorImpl<Type> &types) {
          tupleType.getFlattenedTypes(types);
          return success();
        });

    typeConverter.addSourceMaterialization([](OpBuilder &builder, Type type,
                                              ValueRange inputs,
                                              Location loc) -> Value {
      if (!isa<TupleType>(type))
        return Value();
      return builder.create<stablehlo::TupleOp>(loc, type, inputs);
    });

    typeConverter.addTargetMaterialization(
        [this](OpBuilder &builder, TypeRange resultTypes, ValueRange inputs,
               Location loc) -> SmallVector<Value> {
          if (inputs.size() != 1)
            return {};
          Value input = inputs.front();
          auto inputType = dyn_cast<TupleType>(input.getType());
          if (!inputType)
            return {};
          SmallVector<Value> r;
          convertMaybeNestedTuples(builder, loc, input, r);
          return r;
        });

    RewritePatternSet patterns(&getContext());
    mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    // Run conversion.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {

      emitError(getOperation()->getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
