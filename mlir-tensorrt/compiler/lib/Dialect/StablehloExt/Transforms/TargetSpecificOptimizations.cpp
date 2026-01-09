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
  TargetSpecificCanonicalizationOptions o{};
  for (llvm::StringRef arg : disabled) {
    o.disable(arg);
  }
  return o;
}

void TargetSpecificCanonicalizationOptions::disable(
    llvm::StringRef patternSetMnemonic) {
  if (patternSetMnemonic == "convolution")
    enableConvolutionCanonicalization = false;
  else if (patternSetMnemonic == "gather")
    enableGatherCanonicalization = false;
  else if (patternSetMnemonic == "scatter")
    enableScatterCanonicalization = false;
  else if (patternSetMnemonic == "dot-general")
    enableDotGeneralCanonicalization = false;
  else if (patternSetMnemonic == "gather-to-slice")
    enableGatherToSlice = false;
  else
    llvm::report_fatal_error("Invalid stablehlo target-specific optimization "
                             "pattern set mnemonic: " +
                             patternSetMnemonic);
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

    folderOptions =
        std::make_shared<stablehlo::StablehloAggressiveFolderPassOptions>();
    folderOptions->optimizeFloat = false;
    folderOptions->foldOpElementLimit = constantFoldSizeLimit;
    folderOptions->assumeNoUndeclaredSideEffects = false;

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
        patterns_, constantFoldSizeLimit, *folderOptions, PatternBenefit(1));

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
  std::shared_ptr<stablehlo::StablehloAggressiveFolderPassOptions>
      folderOptions;
};

} // namespace

std::unique_ptr<mlir::Pass>
stablehlo_ext::createTargetSpecificOptimizationsPass(
    TargetSpecificCanonicalizationOptions options,
    int64_t constantFoldSizeLimit) {
  return std::make_unique<TargetSpecificOptimizationsPass>(
      options, constantFoldSizeLimit);
}
