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

#include "mlir-tensorrt-common/Support/Options.h"
#include "mlir-tensorrt/Dialect/Plan/IR/PlanEnums.h"
#include "mlir/Pass/Pass.h"

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

struct PlanClusteringOptions : public mlir::OptionsGroup {
  using OptionsGroup::OptionsGroup;

  static llvm::cl::OptionCategory category;

  /// This option feeds into the `plan-create-closed-regions` pass to determine
  /// whether to prefer use of `plan.alloc_cluster` over `plan.dps_cluster` (if
  /// the backend supports it).
  ///
  /// Note that this *does not* affect the calling convention of entrypoints,
  /// which is controlled by the `forceEntrypointsReturnAllocs` option.
  Option<bool> preferAllocCallingConvention{
      this->ctx, "clustering-prefer-alloc-cconv", llvm::cl::init(false),
      llvm::cl::desc("Prefer using callee-allocating calling conventions"
                     " (callee allocates buffer results) over DPS when "
                     "outlining clusters."),
      llvm::cl::cat(category)};

  Option<bool> disableShapeFuncCreation{
      this->ctx,
      "clustering-disable-shape-func-creation",
      llvm::cl::init(false),
      llvm::cl::desc("Disable creation of shape functions when clustering."),
      llvm::cl::cat(category),
      llvm::cl::Hidden};
};

/// Creates the segmentation pipeline for StableHLO input. This pass pipeline
/// performs shape materialization, clustering, and outlining.
void buildPlanSegmentationPipeline(OpPassManager &pm, int abiVersion,
                                   plan::InputKind inputKind,
                                   bool entrypointUsesAllocCConv,
                                   llvm::StringRef entrypoint,
                                   const plan::PlanClusteringOptions &opts);

struct PlanBufferizationOptions {
  /// Force entrypoint functions to return allocations rather than trying to use
  /// destination-passing (DPS) style.
  bool forceEntrypointsReturnAllocs{false};

  /// Function argument attribute `plan.aliasing_output`, if present, hints
  /// argument is donated. However, it is not guaranteed that donated argument
  /// is bufferized to the same buffer as that of result it is donated for,
  /// after `-plan-module-bufferize` pass. If this flag is set to true, pass
  /// `-plan-confirm-argument-donation` fails if true donation doesn't happen.
  /// By default, this flag is set to false.
  bool failOnDonationArgumentRejection{false};

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
