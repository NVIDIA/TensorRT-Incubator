//===- Passes.h -------------------------------------------------*- C++ -*-===//
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
/// Plan dialect pass declarations
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_H
#define MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_H

#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include <memory>
#include <mlir/Pass/Pass.h>

//===----------------------------------------------------------------------===//
// Add Tablegen'd pass declarations and registration methods.
//===----------------------------------------------------------------------===//
namespace mlir {
namespace plan {

struct ClusterTargetOption;

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"

/// Creates the segmentation pipeline for StableHLO input. This pass pipeline
/// performs shape materialization, clustering, and outlining.
void buildPlanSegmentationPipeline(
    OpPassManager &pm, const plan::StablehloClusteringPassOptions &opts);

/// Options for running `one-shot-bufferize` for the `plan-bufferize` pass.
struct ExecutorBufferizationOptions
    : public bufferization::OneShotBufferizationOptions {
  explicit ExecutorBufferizationOptions(
      ModuleOp targetOp,
      plan::MemorySpace defaultMemorySpace = plan::MemorySpace::device);

  plan::MemorySpace defaultExecutorMemorySpace;
};

/// Run one-shot-bufferization on the given module. After bufferization, verify
/// that loads/stores directly `device` memory space do not occur.
/// NOTE: For development purposes, you can consider
/// using `MemorySpace::unified` as the default memory space. This will lead to
/// using CUDA unified memory, which is less efficient than a correct
/// host/device space management but more efficient than naively inserting
/// Host-Device or Device-Host copies everywhere where device load/store occurs.
LogicalResult
executorOneShotModuleBufferize(ModuleOp targetOp,
                               const ExecutorBufferizationOptions &options);

/// Build a pipeline (targeting ModuleOp) for bufferization.
void buildPlanBufferizationPipeline(OpPassManager &pm);

/// Build a post-bufferization pipeline that performs optimizations on memrefs.
void buildPlanBufferOptimizationPipeline(OpPassManager &pm);

/// Register PassPipelines associated with the Plan dialect.
void registerPlanDialectPipelines();

} // namespace plan
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_H
