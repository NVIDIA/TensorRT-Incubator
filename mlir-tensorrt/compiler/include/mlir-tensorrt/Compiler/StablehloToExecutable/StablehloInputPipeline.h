//===- Passes.h ----------------------------------------------===//
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
/// Declarations for opt tool pipeline command-line registration for pipelines
/// related to "stablehlo-to-executable".
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_STABLEHLOINPUTPIPELINE
#define MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_STABLEHLOINPUTPIPELINE

#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"

namespace mlir {
class OpPassManager;
}

namespace mtrt::compiler {

/// Options for processing StableHLO input IR.
struct StableHloInputOptions {
  /// Whether to lower Stablehlo control flow ops to SCF dialect ops.
  bool legalizeControlFlowToSCF = false;

  /// Whether to preserve 'chlo.erf' ops or lower them to 'stablehlo' ops.
  /// By default, we preserve since it has a 1-1 correspondence with a TensorRT
  /// op.
  bool preserveChloErf = true;

  /// Whether to preserve 'chlo.top_k' ops or lower them to 'stablehlo' ops.
  /// By default, we preserve since it has a 1-1 correspondence with a TensorRT
  /// op.
  bool preserveChloTopK = true;

  /// Whether to disable running the inliner.
  bool disableInliner = false;

  /// Options for target-specific optimizations.
  mlir::stablehlo_ext::TargetSpecificCanonicalizationOptions
      targetSpecificOptions{};

  /// The computation size limit for constant folding.
  int64_t constantFoldSizeLimit = 10;

  /// The cost threshold for unrolling for loops. Loops with a cost <= the
  /// threshold will be unrolled.
  uint64_t unrollThreshold = 100;
};

/// Construct a pipeline for preprocessing StableHLO IR to convert it into the
/// canonical form. Some passes in this pipeline transforms ops to simplify
/// TensorRT conversion. Argument `addConstantFoldingPasses` is a callable that
/// adds stablehlo constant folding passes.
void buildStablehloPreProcessingPipeline(
    mlir::OpPassManager &pm, const StableHloInputOptions &opts,
    std::function<void(mlir::OpPassManager &pm,
                       const StableHloInputOptions &opts)>
        &&addConstantFoldingPasses);

/// Register stablehlo input pipelines.
void registerStableHloInputPipelines();

} // namespace mtrt::compiler

#endif // MLIR_TENSORRT_COMPILER_STABLEHLOTOEXECUTABLE_STABLEHLOINPUTPIPELINE
