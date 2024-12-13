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
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo_ext {
#define GEN_PASS_DEF_EXPANDTUPLESPASS
#define GEN_PASS_DECL_EXPANDTUPLESPASS
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Passes.h.inc"
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
    OneToNTypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });
    typeConverter.addConversion(
        [](TupleType tupleType, SmallVectorImpl<Type> &types) {
          tupleType.getFlattenedTypes(types);
          return success();
        });

    typeConverter.addArgumentMaterialization([](OpBuilder &builder, Type type,
                                                ValueRange inputs,
                                                Location loc) -> Value {
      if (!isa<TupleType>(type))
        return Value();
      return builder.create<stablehlo::TupleOp>(loc, type, inputs);
    });

    typeConverter.addSourceMaterialization([](OpBuilder &builder, Type type,
                                              ValueRange inputs,
                                              Location loc) -> Value {
      if (!isa<TupleType>(type))
        return Value();
      return builder.create<stablehlo::TupleOp>(loc, type, inputs);
    });

    typeConverter.addTargetMaterialization(
        [this](OpBuilder &builder, TypeRange resultTypes, Value input,
               Location loc) -> std::optional<SmallVector<Value>> {
          auto inputType = dyn_cast<TupleType>(input.getType());
          if (!inputType)
            return std::nullopt;
          SmallVector<Value> r;
          convertMaybeNestedTuples(builder, loc, input, r);
          return r;
        });

    RewritePatternSet patterns(&getContext());
    mlir::populateFuncTypeConversionPatterns(typeConverter, patterns);
    // Run conversion.
    if (failed(applyPartialOneToNConversion(getOperation(), typeConverter,
                                            std::move(patterns)))) {

      emitError(getOperation()->getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }
};
} // namespace
