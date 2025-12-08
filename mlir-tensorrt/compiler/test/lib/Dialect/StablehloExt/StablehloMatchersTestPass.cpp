//===- StablehloMatchersTestPass.cpp -------------------------------------===//
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
/// This pass is used to test the StablehloMatchers utility.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/Utils/StablehloMatchers.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::stablehlo;

namespace {
struct TestRaiseToSoftmax : public OpRewritePattern<stablehlo::DivOp> {
  using OpRewritePattern<stablehlo::DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DivOp op,
                                PatternRewriter &rewriter) const override {

    if (op->hasAttr("__matched__softmax__")) {
      return failure();
    }
    Value deducedSoftmaxInp;
    int64_t softmaxAxis;
    if (matchPattern(op.getOperation(), stablehlo::m_StableHLOSoftmaxMatcher(
                                            deducedSoftmaxInp, softmaxAxis))) {
      op->setAttr("__matched__softmax__", UnitAttr::get(op->getContext()));
      return success();
    }
    op->setAttr("__not__softmax__", UnitAttr::get(op->getContext()));
    return failure();
  }
};

/// This pass is used to raise the input IR to multiple recognized MHA patterns.
/// Eg: for stablehlo: stablehlo.dot_general -> tensorrt.softmax ->
/// stablehlo.dot_general
///     for tensorrt: tensorrt.einsum -> tensorrt.softmax -> tensorrt.einsum
/// In both of these cases, it is possible that tensorrt.softmax can be broken
/// down to more basic ops like subtract, exponential and divide.
class TestStablehloMatchersPass
    : public mlir::PassWrapper<TestStablehloMatchersPass,
                               mlir::OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestStablehloMatchersPass)

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet mhaPatterns(ctx);
    mhaPatterns.add<TestRaiseToSoftmax>(mhaPatterns.getContext());
    if (failed(applyPatternsGreedily(op, std::move(mhaPatterns)))) {
      emitError(op->getLoc()) << "failed to convert patterns from "
                                 "stablehlo to tensorrt. ";
      return signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "test-mtrt-stablehlo-matchers";
  }
};
} // namespace

namespace mlir::stablehlo_ext {
void registerTestStablehloMatchersPass();
}

void mlir::stablehlo_ext::registerTestStablehloMatchersPass() {
  mlir::PassRegistration<TestStablehloMatchersPass>();
}
