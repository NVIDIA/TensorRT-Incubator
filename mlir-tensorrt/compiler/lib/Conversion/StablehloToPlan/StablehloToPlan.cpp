//===- StablehloToPlan.cpp ------------------------------------------------===//
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
/// Implementation of the `convert-stablehlo-to-plan` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSTABLEHLOTOPLANPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir
using namespace mlir;

namespace {

/// Convert `stablehlo.optimization_barrier` to `plan.optimization_barrier`.
/// TODO: Currently this does not support ops that have `!stablehlo.token`
/// typed operands.
struct OptimizationBarrierPattern
    : public OpRewritePattern<stablehlo::OptimizationBarrierOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::OptimizationBarrierOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: support stablehlo.token type conversions.
    if (!llvm::all_of(op.getOperandTypes(),
                      [](Type t) { return isa<RankedTensorType>(t); }))
      return failure();
    rewriter.replaceOpWithNewOp<plan::OptimizationBarrierOp>(
        op, op->getOperandTypes(), op.getOperands());
    return success();
  }
};

struct ConvertStablehloToPlanPass
    : public impl::ConvertStablehloToPlanPassBase<ConvertStablehloToPlanPass> {
  using Base::Base;

  std::shared_ptr<FrozenRewritePatternSet> patterns;

  LogicalResult initialize(MLIRContext *context) override {
    patterns = std::make_shared<FrozenRewritePatternSet>([&] {
      RewritePatternSet patterns_(context);
      patterns_.add<OptimizationBarrierPattern>(context);
      return patterns_;
    }());
    return success();
  }

  void runOnOperation() override {
    walkAndApplyPatterns(getOperation(), *patterns);
  }
};
} // namespace
