//===- ModuleBufferization.h ------------------------------------*- C++ -*-===//
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
/// Declarations for the `plan-module-bufferize` pass.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Utils/ModuleUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

namespace mlir::plan {

/// Contains a copy of information from bufferization upstream
/// FuncAnalysisState. This is necessary to persist this information past the
/// point where the associated AnalysisState is live.
struct FuncAnalysisStateInfo {
  using FuncOp = func::FuncOp;
  using FuncOpAnalysisState = bufferization::func_ext::FuncOpAnalysisState;

  FuncAnalysisStateInfo(
      const bufferization::func_ext::FuncAnalysisState &info) {
    equivalentFuncArgs = info.equivalentFuncArgs;
    aliasingReturnVals = info.aliasingReturnVals;
    readBbArgs = info.readBbArgs;
    writtenBbArgs = info.writtenBbArgs;
    analyzedFuncOps = info.analyzedFuncOps;
  }

  /// Append this information to the FuncAnalysisState in `state`.
  void appendToState(bufferization::OneShotAnalysisState &state) const;

  // Note: Function arguments and/or function return values may disappear during
  // bufferization. Functions and their CallOps are analyzed and bufferized
  // separately. To ensure that a CallOp analysis/bufferization can access an
  // already bufferized function's analysis results, we store bbArg/return value
  // indices instead of BlockArguments/OpOperand pointers.

  /// A set of block argument indices.
  using BbArgIndexSet = DenseSet<int64_t>;

  /// A mapping of indices to indices.
  using IndexMapping = DenseMap<int64_t, int64_t>;

  /// A mapping of indices to a list of indices.
  using IndexToIndexListMapping = DenseMap<int64_t, SmallVector<int64_t>>;

  /// A mapping of ReturnOp OpOperand indices to equivalent FuncOp BBArg
  /// indices.
  DenseMap<FuncOp, IndexMapping> equivalentFuncArgs;

  /// A mapping of FuncOp BBArg indices to aliasing ReturnOp OpOperand indices.
  DenseMap<FuncOp, IndexToIndexListMapping> aliasingReturnVals;

  /// A set of all read BlockArguments of FuncOps.
  DenseMap<FuncOp, BbArgIndexSet> readBbArgs;

  /// A set of all written-to BlockArguments of FuncOps.
  DenseMap<FuncOp, BbArgIndexSet> writtenBbArgs;

  /// Keep track of which FuncOps are fully analyzed or currently being
  /// analyzed.
  DenseMap<FuncOp, FuncOpAnalysisState> analyzedFuncOps;
};

/// A map from Module-like operations to a copy of their final FuncAnalysisState
/// information.
using ModuleFuncAnalysisCache =
    llvm::DenseMap<Operation *, FuncAnalysisStateInfo>;

/// Analyze a single module op.
LogicalResult
analyzeOneModuleOp(ModuleLikeOp moduleOp,
                   bufferization::OneShotAnalysisState &state,
                   bufferization::BufferizationStatistics *statistics);

/// Rewrite `memref.load|store` that operate on non host-visible memory spaces.
/// Required host memory allocations and copying to/from device memory are
/// inserted.
LogicalResult
fixupHostModule(ModuleLikeOp module,
                const bufferization::OneShotBufferizationOptions &options);

/// Setup a new OneShotAnalysisState. For each module op in `funcInfo` where
/// `moduleOp` is a parent, also populate the function state information into
/// the new analysis state.
void setupAnalysisStateForModule(ModuleLikeOp moduleOp,
                                 const ModuleFuncAnalysisCache &funcInfo,
                                 bufferization::OneShotAnalysisState &newState);

/// Append FuncAnalysisState information to the ModuleFuncAnalysisCache.
void appendAnalysisResultsToCache(
    ModuleLikeOp op, ModuleFuncAnalysisCache &cache,
    const bufferization::OneShotAnalysisState &state);

} // namespace mlir::plan
