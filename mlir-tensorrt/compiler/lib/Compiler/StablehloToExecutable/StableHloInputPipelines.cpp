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
#include "mlir-tensorrt/Dialect/StableHloExt/Transforms/Passes.h"
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
  if (!opts.disableInliner)
    pm.addPass(createInlinerPass());
  pm.addPass(stablehlo_ext::createLowerSpecialCustomCalls());

  // Simplify StableHLO graph
  buildStableHloSimplificationPipeline(
      pm, ConvertChloToStableHloExtPassOptions{
              /*preserveErf=*/opts.preserveChloErf,
              /*preserveTopK=*/opts.preserveChloTopK,
          });
  pm.addPass(createCSEPass());
  pm.addPass(stablehlo_ext::createCanonicalizeConvolutionPass());
  if (opts.legalizeControlFlowToSCF)
    pm.addPass(mlir::createConvertStablehloToScfPass());
  pm.addPass(createCSEPass());
  pm.addPass(stablehlo_ext::createConstantFoldingPass());
  pm.addPass(stablehlo_ext::createCanonicalizeShapesPass());
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
