//===- LowerSpecialCustomCalls.cpp ----------------------------------------===//
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
/// Lower special custom call operations.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::stablehlo;

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_LOWERSPECIALCUSTOMCALLS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

namespace {
/// Lower `stablehlo.custom_call` directly to a CHLO operation (OpType) if
/// possible. The `name` parameter specifes the `custom_call` target that we
/// are looking for.
template <typename OpType>
struct LowerCustomCallPattern
    : public OpConversionPattern<stablehlo::CustomCallOp> {
  using OpConversionPattern::OpConversionPattern;

  LowerCustomCallPattern(MLIRContext *ctx, StringRef name,
                         PatternBenefit benefit = 1)
      : OpConversionPattern(ctx, benefit), name(name.str()) {}
  std::string name;

  LogicalResult
  matchAndRewrite(stablehlo::CustomCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != name)
      return failure();

    auto extraAttrs = llvm::dyn_cast_if_present<DictionaryAttr>(
        op->getDiscardableAttr("mhlo.attributes"));
    rewriter.replaceOpWithNewOp<OpType>(
        op, op.getResultTypes(), op.getOperands(), extraAttrs.getValue());
    return success();
  }
};
} // namespace

static bool isNoopShardingOp(stablehlo::CustomCallOp op) {
  return op.getCallTargetName() == "Sharding" && op->getNumOperands() == 1 &&
         op->getNumResults() == 1 &&
         op.getResultTypes() == op->getOperandTypes();
}

namespace {
/// Lower `stablehlo.custom_call` that represents a sharding annotation. These
/// are sometimes inserted by JAX, even if running the model with a single GPU,
/// if the computation device placement wasn't explicitly set. In single GPU
/// mode, we these should be no-ops, so we just eliminate them.
struct LowerNoopShardingPatterns
    : public OpConversionPattern<stablehlo::CustomCallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stablehlo::CustomCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isNoopShardingOp(op))
      return failure();
    auto sharding = op->getAttrOfType<StringAttr>("mhlo.sharding");
    if (!sharding || sharding.strref() != "{replicated}")
      return failure();
    rewriter.replaceOp(op, op->getOperand(0));
    return success();
  }
};
} // namespace

namespace {
class LowerSpecialCustomCallsPass
    : public stablehlo_ext::impl::LowerSpecialCustomCallsBase<
          LowerSpecialCustomCallsPass> {
  using Base::Base;

  std::vector<std::string> illegalCallNames = {"mhlo.erf", "mhlo.tan",
                                               "mhlo.topk", "Sharding"};

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerCustomCallPattern<chlo::ErfOp>>(ctx, "mhlo.erf");
    patterns.add<LowerCustomCallPattern<chlo::TanOp>>(ctx, "mhlo.tan");
    patterns.add<LowerCustomCallPattern<chlo::TopKOp>>(ctx, "mhlo.topk");
    patterns.add<LowerNoopShardingPatterns>(ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<chlo::ChloDialect>();
    target.addDynamicallyLegalOp<stablehlo::CustomCallOp>(
        [&](stablehlo::CustomCallOp op) {
          return !llvm::is_contained(illegalCallNames, op.getCallTargetName());
        });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
