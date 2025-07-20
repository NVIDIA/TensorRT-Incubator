//===- TargetSpecificOptimizations.cpp ------------------------------------===//
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
/// Implementation of "stablehlo-ext-canonicalize".
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Patterns.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

namespace mlir::stablehlo_ext {
#define GEN_PASS_DEF_TARGETSPECIFICOPTIMIZATIONSPASS
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h.inc"
} // namespace mlir::stablehlo_ext

using namespace mlir;
using namespace mlir::stablehlo_ext;

FailureOr<TargetSpecificCanonicalizationOptions>
TargetSpecificCanonicalizationOptions::parse(
    llvm::ArrayRef<std::string> disabled) {
  SmallVector<llvm::StringRef> optimizations = {
      "convolution", "gather", "scatter", "dot-general", "gather-to-slice",
  };

  TargetSpecificCanonicalizationOptions o{};
  for (StringRef arg : optimizations) {
    bool enable = !llvm::is_contained(disabled, arg);
    if (arg == "convolution")
      o.enableConvolutionCanonicalization = enable;
    else if (arg == "gather")
      o.enableGatherCanonicalization = enable;
    else if (arg == "scatter")
      o.enableScatterCanonicalization = enable;
    else if (arg == "dot-general")
      o.enableDotGeneralCanonicalization = enable;
    else if (arg == "gather-to-slice")
      o.enableGatherToSlice = enable;
    else
      return failure();
  }
  return o;
}

namespace {

class TargetSpecificOptimizationsPass
    : public stablehlo_ext::impl::TargetSpecificOptimizationsPassBase<
          TargetSpecificOptimizationsPass> {
public:
  using Base::Base;

  TargetSpecificOptimizationsPass(TargetSpecificCanonicalizationOptions options,
                                  int64_t constantFoldSizeLimit)
      : Base(TargetSpecificOptimizationsPassOptions{{}, constantFoldSizeLimit}),
        options(std::move(options)) {}

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patterns_(ctx);

    if (!options) {
      FailureOr<TargetSpecificCanonicalizationOptions> parsed =
          TargetSpecificCanonicalizationOptions::parse(disablePatterns);
      if (failed(parsed))
        return failure();
      options = std::move(parsed);
    }

    if (options->enableConvolutionCanonicalization)
      stablehlo_ext::populateCanonicalizeStablehloConvolutionPatterns(
          patterns_);
    if (options->enableScatterCanonicalization)
      stablehlo_ext::populateCanonicalizeStablehloScatterPatterns(patterns_);
    if (options->enableGatherCanonicalization)
      stablehlo_ext::populateCanonicalizeStablehloGatherPatterns(patterns_);
    if (options->enableDotGeneralCanonicalization)
      stablehlo_ext::populateCanonicalizeStablehloDotGeneralPatterns(patterns_);
    if (options->enableGatherToSlice)
      stablehlo_ext::populateGatherToSlicePatterns(patterns_);

    stablehlo_ext::populateTargetIndependentSimplificationPatterns(
        patterns_, constantFoldSizeLimit);

    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    if (failed(applyPatternsGreedily(op, patterns))) {
      emitError(op->getLoc())
          << "failed to apply patterns in " << getArgument();
      return signalPassFailure();
    }
  }

  std::optional<TargetSpecificCanonicalizationOptions> options{};
  FrozenRewritePatternSet patterns{nullptr};
};

} // namespace

std::unique_ptr<mlir::Pass>
stablehlo_ext::createTargetSpecificOptimizationsPass(
    TargetSpecificCanonicalizationOptions options,
    int64_t constantFoldSizeLimit) {
  return std::make_unique<TargetSpecificOptimizationsPass>(
      options, constantFoldSizeLimit);
}