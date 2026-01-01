//===- StablehloInputPipeline.h ------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Declarations for StableHLO input pipelines.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_INPUTPIPELINES_STABLEHLOINPUTPIPELINE
#define MLIR_TENSORRT_COMPILER_INPUTPIPELINES_STABLEHLOINPUTPIPELINE

#include "mlir-tensorrt-common/Support/Options.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

namespace mlir {
class OpPassManager;
}

namespace mtrt::compiler {

/// Options for processing StableHLO input IR.
struct StablehloInputOptions : public mlir::OptionsGroup {
  using mlir::OptionsGroup::OptionsGroup;

  static llvm::cl::OptionCategory category;

  Option<bool> legalizeControlFlowToSCF{
      this->ctx, "stablehlo-input-legalize-to-scf",
      llvm::cl::desc("lower StableHLO control flow ops to SCF"),
      llvm::cl::init(true), llvm::cl::cat(category)};

  /// Whether to preserve 'chlo.erf' ops or lower them to 'stablehlo' ops.
  /// By default, we preserve since it has a 1-1 correspondence with a TensorRT
  /// op.
  Option<bool> preserveChloErf{
      this->ctx, "stablehlo-input-preserve-chlo-erf",
      llvm::cl::desc("don't lower chlo.erf to stablehlo"), llvm::cl::init(true),
      llvm::cl::cat(category)};

  /// Whether to preserve 'chlo.top_k' ops or lower them to 'stablehlo' ops.
  /// By default, we preserve since it has a 1-1 correspondence with a TensorRT
  /// op.
  Option<bool> preserveChloTopK{
      this->ctx, "stablehlo-input-preserve-chlo-topk",
      llvm::cl::desc("don't lower chlo.top_k to stablehlo"),
      llvm::cl::init(true), llvm::cl::cat(category)};

  /// Whether to disable running the inliner.
  Option<bool> disableInliner{
      this->ctx, "stablehlo-input-disable-inliner",
      llvm::cl::desc(
          "Whether to disable running the inliner as part of the pipeline"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  /// Options for target-specific optimizations.
  ListOption<std::string> disableTargetSpecificOptimizationPatterns{
      this->ctx, "stablehlo-input-disable-patterns",
      llvm::cl::desc("List StableHLO target-specific optimization pattern sets "
                     "to disable."),
      llvm::cl::callback([this](const std::string &value) {
        this->targetSpecificOptions.disable(value);
      }),
      llvm::cl::cat(category)};

  /// The computation size limit for constant folding.
  Option<int64_t> constantFoldSizeLimit{
      this->ctx, "stablehlo-input-fold-limit",
      llvm::cl::desc("The computation size limit for constant folding"),
      llvm::cl::init(128), llvm::cl::cat(category)};

  /// The cost threshold for unrolling for loops. Loops with a cost <= the
  /// threshold will be unrolled.
  Option<uint64_t> unrollThreshold{
      this->ctx, "stablehlo-input-unroll-threshold",
      llvm::cl::desc("The cost threshold for unrolling loops in the StableHLO "
                     "input pipeline"),
      llvm::cl::init(100), llvm::cl::cat(category)};

  mlir::stablehlo_ext::TargetSpecificCanonicalizationOptions
      targetSpecificOptions{};
};

/// Construct a pipeline for preprocessing StableHLO IR to convert it into the
/// canonical form. Some passes in this pipeline transforms ops to simplify
/// TensorRT conversion. Argument `addConstantFoldingPasses` is a callable that
/// adds stablehlo constant folding passes.
void buildStablehloInputPipeline(
    mlir::OpPassManager &pm, const StablehloInputOptions &opts,
    std::function<void(mlir::OpPassManager &pm,
                       const StablehloInputOptions &opts)>
        &&addConstantFoldingPasses);

/// Register stablehlo input pipelines.
void registerStableHloInputPipelines();

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_INPUTPIPELINES_STABLEHLOINPUTPIPELINE
