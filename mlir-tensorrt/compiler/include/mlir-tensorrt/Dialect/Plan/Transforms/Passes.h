//===- Passes.h -------------------------------------------------*- C++ -*-===//
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
/// Plan dialect pass declarations
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_H
#define MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_H

#include "mlir-tensorrt-dialect/Target/TranslateToTensorRT.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include <memory>
#include <mlir/Pass/Pass.h>

//===----------------------------------------------------------------------===//
// Add Tablegen'd pass declarations and registration methods.
//===----------------------------------------------------------------------===//
namespace mlir {
namespace plan {

namespace detail {
/// Shorthand adaptor for declaring LLVM CL options for the InputKind
/// enum used by different Plan segmentation passes.
inline llvm::cl::ValuesClass createInputKindClOptions() {
  return ::llvm::cl::values(
      clEnumValN(InputKind::Stablehlo, "stablehlo", "StableHLO IR"),
      clEnumValN(InputKind::TensorRT, "tensorrt", "TensorRT IR"),
      clEnumValN(InputKind::Linalg, "linalg", "Linalg IR"));
}
} // namespace detail

struct ClusterTargetOption;

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h.inc"

/// Creates the segmentation pipeline for StableHLO input. This pass pipeline
/// performs shape materialization, clustering, and outlining.
void buildPlanSegmentationPipeline(OpPassManager &pm,
                                   const plan::ClusteringPassOptions &opts);

struct PlanBufferizationOptions {
  /// Force entrypoint functions to return allocations rather than trying to use
  /// destination-passing (DPS) style.
  bool forceEntrypointsReturnAllocs{false};

  /// In the buffer deallocation transformation, calls to private functions use
  /// a default behavior where the private function may not assume ownership of
  /// memref arguments and the ownership of any returned memrefs is assumed by
  /// the caller. Setting this option to 'true' allows the transformation to
  /// instead rewrite private functions and their call sites to add additional
  /// `i1` arguments and results to pass ownership information across the
  /// function boundary.
  bool deallocationPrivateFuncDynamicOwnership{false};

  /// Enable buffer hoisting out of loops.
  bool enableBufferLoopHoisting{true};

  /// Enable buffer hoisting.
  bool enableBufferHoisting{true};

  /// Enable promotion of host buffers to pinned memory using heuristics.
  bool enablePinnedMemoryPromotion{true};
};

/// Build a complete bufferization pipeline, which includes: bufferization,
/// optimizations, and buffer deallocation.
void buildPlanBufferizationPipeline(OpPassManager &pm,
                                    const PlanBufferizationOptions &options);

/// Register PassPipelines associated with the Plan dialect.
void registerPlanDialectPipelines();

/// Construct a pass that outlines constant foldable subgraphs.
std::unique_ptr<Pass> createOutlineConstantFoldableSubgraphsPass(
    std::function<bool(Operation *)> skipClustering = {});

} // namespace plan
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_PLAN_TRANSFORMS_PASSES_H
