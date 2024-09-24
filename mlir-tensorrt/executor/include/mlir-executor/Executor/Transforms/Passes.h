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
/// Declarations for Executor dialect passes.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_TRANSFORMS_PASSES_H
#define MLIR_TENSORRT_DIALECT_EXECUTOR_TRANSFORMS_PASSES_H

#include "mlir-executor/Utils/ModuleLikePass.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncDialect;
}

namespace executor {

struct ConvertStdToExecutorPassOptions;

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir-executor/Executor/Transforms/Passes.h.inc"

/// Builds a pipeline from lowering Executor-compatible IR to the final form
/// prior to target translation.
void buildExecutorLoweringPipeline(
    OpPassManager &pm,
    const ConvertStdToExecutorPassOptions &stdToExecutorOpts);

/// Register Executor-related pass pipelines.
void registerExecutorPassPipelines();

} // namespace executor
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_TRANSFORMS_PASSES_H
