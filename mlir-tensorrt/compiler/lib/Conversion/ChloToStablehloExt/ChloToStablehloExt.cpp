//===- ChloToStablehloExt.cpp ---------------------------------------------===//
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
/// Convert certain CHLO ops to stablehlo ops. We only need this instantiation
/// of the upstream pass since we need to selectively preserve certain CHLO ops
/// like 'top k'.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTCHLOTOSTABLEHLOEXTPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct ChloToStablehloExtPass
    : public impl::ConvertChloToStableHloExtPassBase<ChloToStablehloExtPass> {
public:
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override {
    target = std::make_shared<ConversionTarget>(*context);

    target->addDynamicallyLegalDialect<chlo::ChloDialect>([&](Operation *op) {
      if (isa<chlo::ErfOp>(op))
        return preserveErf.getValue();
      if (isa<chlo::TopKOp>(op))
        return preserveTopK.getValue();
      return false;
    });

    target->markUnknownOpDynamicallyLegal([](Operation *op) { return true; });

    RewritePatternSet patterns_(context);
    stablehlo::populateChloToStablehloPatterns(context, &patterns_);
    patterns = std::move(patterns_);

    return success();
  }

  void runOnOperation() override {
    if (failed(applyPartialConversion(getOperation(), *target, patterns))) {
      emitError(getOperation()->getLoc())
          << "failed to apply patterns in " << getArgument();
      signalPassFailure();
    }
  }

private:
  std::shared_ptr<mlir::ConversionTarget> target;
  FrozenRewritePatternSet patterns;
};
} // namespace