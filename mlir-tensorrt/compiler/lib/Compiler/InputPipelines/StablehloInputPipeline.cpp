//===- StableHloInputPipeline.cpp ----------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
#include "mlir-tensorrt/Compiler/InputPipelines/StablehloInputPipeline.h"
#include "mlir-tensorrt-common/Utils/PassManagerUtils.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"

using namespace mlir;
using namespace mtrt;
using namespace mtrt::compiler;

llvm::cl::OptionCategory StablehloInputOptions::category = {
    "MLIR-TensorRT StableHLO Input Options", ""};

void mtrt::compiler::buildStablehloInputPipeline(
    OpPassManager &pm, const StablehloInputOptions &opts,
    std::function<void(mlir::OpPassManager &pm,
                       const StablehloInputOptions &opts)>
        &&addConstantFoldingPasses) {

  // `stablehlo-ext-lower-special-custom-calls`:
  // Lower `stablehlo.custom_call` that have special meanings.
  pm.addPass(stablehlo_ext::createLowerSpecialCustomCalls());
  pm.addPass(stablehlo_ext::createStablehloLowerCheckCustomCallsPass());

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

  // `convert-stablehlo-to-plan`:
  // - Convert `stablehlo.optimization_barrier` to `plan.optimization_barrier`.
  pm.addPass(createConvertStablehloToPlanPass());

  addNestedPasses<func::FuncOp>(pm, [&opts](OpPassManager &pm) {
    // `convert-stablehlo-to-scf`:
    if (opts.legalizeControlFlowToSCF) {
      pm.addPass(mlir::createConvertStablehloToScfPass());
      pm.addPass(mtrt::createSCFFloatStrengthReducePass());
      pm.addPass(mtrt::createSCFUnrollPass(
          mtrt::SCFUnrollPassOptions{opts.unrollThreshold}));
      pm.addPass(mtrt::createSCFForCyclicIterArgsPass());
    }

    // `stablehlo-canonicalize-dynamism`:
    // - Canonicalize dynamic shape op variants.
    // - This is run prior to any folding because it can catch correctness
    //   issues in the input IR which may be masked by folding.
    pm.addPass(stablehlo::createStablehloCanonicalizeDynamismPass());
  });

  // `stablehlo-ext-constant-folding`:
  // Constant fold on functions.
  addConstantFoldingPasses(pm, opts);

  // `stablehlo-ext-refine-shapes{interprocedural=false}`:
  // - Refine shapes of operations without trying to refine function
  //   types.
  // - We run this prior to `stablehlo-ext-canonicalize-shapes`
  //   with `interprocedural=false` since upstream has an issue
  //   with `stablehlo-refine-shapes` convergence failure in its
  //   use of the greedy rewrite driver.
  pm.addPass(
      stablehlo_ext::createRefineShapesPass({/*interprocedural=*/false}));

  // `stablehlo-ext-canonicalize-shapes`:
  // - Fixed-point iteration of dynamic pipeline of
  //   `stablehlo-canonicalize-dynamism` and `stablehlo-ext-refine-shapes`.
  // - May change function types.
  pm.addPass(stablehlo_ext::createCanonicalizeShapesPass({
      /*maxIterations=*/6,
      /*interprocedural=*/true,
  }));

  //===----------------------------------------------------------------------===//
  // Non-folding simplifications.
  //===----------------------------------------------------------------------===//

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
  // - Fixed-point iteration
  pm.addPass(stablehlo_ext::createCanonicalizeShapesPass());

  pm.addPass(stablehlo::createStablehloConvertToSignlessPass());

  // `cse`:
  pm.addPass(createCSEPass());

  // `canonicalize`:
  pm.addPass(createCanonicalizerPass());
}

namespace {
/// We don't have a way to automatically convert OptionsGroup or CLOptionScope
/// objects to `mlir::PassPipelineOptions`, so for now we manually redefine
/// the CLI-only pipeline options here.
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
      llvm::cl::init(10)};

  ListOption<std::string> disableTargetSpecificOptimizationPatterns{
      *this, "disable-target-specific-optimization-patterns",
      llvm::cl::desc("Comma-separated list of target-specific optimization "
                     "pattern sets to disable. "
                     "Available pattern sets: dot-general, gather, scatter, "
                     "convolution, gather-to-slice"),
      llvm::cl::list_init<std::string>({})};

  Option<int64_t> unrollThreshold{
      *this, "unroll-threshold",
      llvm::cl::desc("The cost threshold for unrolling loops."),
      llvm::cl::init(100)};
};
} // namespace

void mtrt::compiler::registerStableHloInputPipelines() {
  PassPipelineRegistration<StableHloInputPipelineOptions>(
      "stablehlo-input-pipeline",
      "Apply StableHlo input processing pipeline to prepare for "
      "TensorRT conversion",
      [](OpPassManager &pm, const StableHloInputPipelineOptions &opts) {
        LocalScopedOptionsGroup<StablehloInputOptions> inputOpts;
        inputOpts.get().legalizeControlFlowToSCF.setValue(
            opts.legalizeControlFlowToSCF);
        inputOpts.get().preserveChloErf.setValue(opts.preserveChloErf);
        inputOpts.get().preserveChloTopK.setValue(opts.preserveChloTopK);
        inputOpts.get().disableInliner.setValue(opts.disableInliner);
        inputOpts.get().unrollThreshold.setValue(opts.unrollThreshold);
        inputOpts.get().disableTargetSpecificOptimizationPatterns.assign(
            opts.disableTargetSpecificOptimizationPatterns);
        inputOpts.get().constantFoldSizeLimit.setValue(
            opts.constantFoldSizeLimit);
        buildStablehloInputPipeline(
            pm, inputOpts,
            [](mlir::OpPassManager &pm, const StablehloInputOptions &opts) {
              pm.addNestedPass<func::FuncOp>(
                  stablehlo_ext::createConstantFoldingPass(
                      stablehlo_ext::ConstantFoldingPassOptions{
                          opts.constantFoldSizeLimit}));
            });
      });
}
