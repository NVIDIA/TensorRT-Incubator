//===- StableHloInputPipelines.h --------------------------------*- C++ -*-===//
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
/// Declarations for pipelines that process Stablehlo input IR.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_PIPELINES_STABLEHLOINPUTPIPELINES_H
#define MLIR_TENSORRT_PIPELINES_STABLEHLOINPUTPIPELINES_H

namespace mlir {

class OpPassManager;

/// Options for processing StableHLO input IR.
struct StableHloInputOptions {
  /// Whether to lower Stablehlo control flow ops to SCF dialect ops.
  bool legalizeControlFlowToSCF = false;
  /// Whether to lower chlo.erf into primitive stablehlo operations.
  bool legalizeChloErfToStablehlo = false;
  /// Whether to disable running the inliner.
  bool disableInliner = false;
  /// Whether to lower chlo to stablehlo.
  bool convertChloToStablehlo = false;
};

/// Construct a pipeline for preprocessing StableHLO IR to convert it into the
/// canonical form. Some passes in this pipeline transforms ops to simplify
/// TensorRT conversion.
void buildStablehloPreProcessingPipeline(OpPassManager &pm,
                                         const StableHloInputOptions &opts);

/// Register stablehlo input pipelines.
void registerStableHloInputPipelines();

} // namespace mlir

#endif // MLIR_TENSORRT_PIPELINES_STABLEHLOINPUTPIPELINES_H
