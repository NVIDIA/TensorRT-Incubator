//===- StableHloInputPipelines.cpp ----------------------------------------===//
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
#include "mlir-tensorrt/Compiler/StablehloToExecutable/Passes.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"

using namespace mlir;
using namespace mlirtrt;
using namespace mlirtrt::compiler;

static void buildStableHloSimplificationPipeline(
    OpPassManager &pm,
    const mlir::ConvertChloToStableHloExtPassOptions &chloToStablehloOptions) {
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloLegalizeCompositeToCallPass());
  pm.addPass(createInlinerPass());
  // Some match-and-raise patterns should be performed before canonicalization,
  // since the pattern is based on specific frontend patterns (e.g. JAX).
  pm.addPass(stablehlo_ext::createExpandTuplesPass());
  pm.addPass(stablehlo_ext::createCanonicalizeShapesPass());
  pm.addPass(stablehlo_ext::createStablehloRaiseQDQPass());
  pm.addPass(stablehlo_ext::createConstantFoldingPass());
  pm.addPass(stablehlo_ext::createGatherToSlicePass());
  pm.addPass(stablehlo_ext::createCanonicalizeShapesPass());

  // We don't do the CHLO legalization until this point since we want to wait
  // until after `canonicalize-shapes` has run at least once. This reduces the
  // likelihood of generating `shape` dialect ops.
  pm.addPass(mlir::createConvertChloToStableHloExtPass(chloToStablehloOptions));

  pm.addPass(stablehlo_ext::createCanonicalizeDotGeneralPass());
  pm.addPass(stablehlo_ext::createConstantFoldingPass());
  pm.addPass(stablehlo_ext::createCanonicalizeShapesPass());
  pm.addNestedPass<func::FuncOp>(
      stablehlo_ext::createCanonicalizeScatterPass());
  pm.addNestedPass<func::FuncOp>(stablehlo_ext::createCanonicalizeGatherPass());
  pm.addPass(stablehlo_ext::createConstantFoldingPass());
  pm.addPass(stablehlo_ext::createCanonicalizeShapesPass());
}

void mlirtrt::compiler::buildStablehloPreProcessingPipeline(
    OpPassManager &pm, const StableHloInputOptions &opts) {

  // `stablehlo-ext-lower-special-custom-calls`:
  // Lower `stablehlo.custom_call` that have special meanings.
  pm.addPass(stablehlo_ext::createLowerSpecialCustomCalls());

  // `stabelhlo-legalize-composite-to-call`:
  // Substitute `stablehlo.composite` with its implementation where possible.
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloLegalizeCompositeToCallPass());

  // Run the inliner. Must be sequenced after lowering composites to
  // call.
  if (!opts.disableInliner)
    pm.addPass(createInlinerPass());

  // `stablehlo-ext-expand-tuples`:
  // - Eliminate `stablehlo.tuple` types and related ops through flattening.
  // - May change function types.
  pm.addPass(stablehlo_ext::createExpandTuplesPass());

  // `stablehlo-ext-raise-qdq`:
  // - Some match-and-raise patterns for Q/DQ that
  //   should be performed before canonicalization
  //   since the pattern is based on specific frontend patterns (e.g. JAX).
  pm.addNestedPass<func::FuncOp>(stablehlo_ext::createStablehloRaiseQDQPass());

  // `stablehlo-ext-constant-folding`:
  // Constant fold on functions.
  pm.addNestedPass<func::FuncOp>(stablehlo_ext::createConstantFoldingPass());

  // `stablehlo-ext-canonicalize-shapes`:
  // - Fixed point interation of dynamic pipeline.
  // - May change function types.
  pm.addPass(stablehlo_ext::createCanonicalizeShapesPass());

  //===----------------------------------------------------------------------===//
  // Non-folding simplifications.
  //===----------------------------------------------------------------------===//

  // `convert-stablehlo-to-scf`:
  if (opts.legalizeControlFlowToSCF)
    pm.addNestedPass<func::FuncOp>(mlir::createConvertStablehloToScfPass());

  // `convert-chlo-to-stablehlo-ext`:
  // We don't do the CHLO legalization until this point since we want to wait
  // until after `canonicalize-shapes` has run at least once. This reduces the
  // likelihood of generating `shape` dialect ops.
  pm.addPass(mlir::createConvertChloToStableHloExtPass(
      ConvertChloToStableHloExtPassOptions{
          /*preserveErf=*/opts.preserveChloErf,
          /*preserveTopK=*/opts.preserveChloTopK,
      }));

  // `stablehlo-ext-gather-to-slice`:
  // TODO: move this upstream
  // Convert `stablehlo.gather` to `stablehlo.slice` where possible.
  pm.addNestedPass<func::FuncOp>(
      stablehlo_ext::createTargetSpecificOptimizationsPass(
          opts.targetSpecificOptions, opts.constantFoldSizeLimit));

  // `stablehlo-ext-canonicalize-shapes`:
  // - Fixed point iteration
  pm.addPass(stablehlo_ext::createCanonicalizeShapesPass());

  pm.addPass(stablehlo::createStablehloConvertToSignlessPass());

  // `cse`:
  pm.addPass(createCSEPass());

  // `canonicalize`:
  pm.addPass(createCanonicalizerPass());
}

namespace {
struct StableHloInputPipelineOptions
    : public PassPipelineOptions<StableHloInputPipelineOptions> {
  Option<bool> legalizeControlFlowToSCF{
      *this, "legalize-control-flow-to-scf",
      llvm::cl::desc("lower StableHLO control flow ops to SCF"),
      llvm::cl::init(false)};

  Option<bool> preserveChloErf{
      *this, "preserve-chlo-erf",
      llvm::cl::desc("don't lower chlo.erf to stablehlo"),
      llvm::cl::init(true)};
  Option<bool> preserveChloTopK{
      *this, "preserve-chlo-topk",
      llvm::cl::desc("don't lower chlo.top_k to stablehlo"),
      llvm::cl::init(true)};
  Option<bool> disableInliner{
      *this, "disable-inliner",
      llvm::cl::desc(
          "Whether to disable running the inliner as part of the pipeline"),
      llvm::cl::init(false)};

  Option<int64_t> constantFoldSizeLimit{
      *this, "constant-fold-size-limit",
      llvm::cl::desc("The computation size limit for constant folding"),
      llvm::cl::init(65536)};

  ListOption<std::string> targetSpecificOptimizationsPatterns{
      *this, "target-specific-optimizations-patterns",
      llvm::cl::desc("Comma-separated list of target-specific optimization "
                     "pattern sets to enable. "
                     "Available pattern sets: dot-general, gather, scatter, "
                     "convolution, gather-to-slice, all"),
      llvm::cl::list_init<std::string>({"all"})};
};
} // namespace

void mlirtrt::compiler::registerStableHloInputPipelines() {
  PassPipelineRegistration<StableHloInputPipelineOptions>(
      "stablehlo-preprocessing-pipeline",
      "Apply StableHlo input processing pipeline to prepare for "
      "TensorRT conversion",
      [](OpPassManager &pm, const StableHloInputPipelineOptions &opts) {
        StableHloInputOptions inputOpts;
        inputOpts.legalizeControlFlowToSCF = opts.legalizeControlFlowToSCF;
        inputOpts.preserveChloErf = opts.preserveChloErf;
        inputOpts.preserveChloTopK = opts.preserveChloTopK;
        inputOpts.disableInliner = opts.disableInliner;

        FailureOr<stablehlo_ext::TargetSpecificCanonicalizationOptions> parsed =
            stablehlo_ext::TargetSpecificCanonicalizationOptions::parse(
                opts.targetSpecificOptimizationsPatterns);
        if (failed(parsed))
          llvm::report_fatal_error("Invalid target-specific Stablehlo "
                                   "optimization pattern set names.");
        inputOpts.targetSpecificOptions = std::move(*parsed);
        inputOpts.constantFoldSizeLimit = opts.constantFoldSizeLimit;

        buildStablehloPreProcessingPipeline(pm, inputOpts);
      });

  PassPipelineRegistration<StableHloInputPipelineOptions>(
      "stablehlo-simplification-pipeline",
      "Apply StableHLO simplification passes",
      [](OpPassManager &pm, const StableHloInputPipelineOptions &opts) {
        buildStableHloSimplificationPipeline(
            pm, {opts.preserveChloErf, opts.preserveChloTopK});
      });
}
