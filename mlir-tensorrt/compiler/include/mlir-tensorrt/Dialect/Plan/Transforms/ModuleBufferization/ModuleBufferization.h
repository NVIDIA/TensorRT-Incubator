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
#include "mlir-tensorrt/Utils/ModuleUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

namespace mlir::plan {

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

} // namespace mlir::plan