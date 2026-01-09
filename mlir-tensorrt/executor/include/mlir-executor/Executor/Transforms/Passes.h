//===- Passes.h -------------------------------------------------*- C++ -*-===//
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
///
/// Declarations for Executor dialect passes.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_TRANSFORMS_PASSES_H
#define MLIR_TENSORRT_DIALECT_EXECUTOR_TRANSFORMS_PASSES_H

#include "mlir-tensorrt-common/Utils/ModuleLikePass.h" // IWYU pragma: keep
#include "mlir/Pass/Pass.h"                            // IWYU pragma: keep

namespace mlir {
class TypeConverter;
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
/// Additional type conversions can be applied during the `std-to-executor` and
/// `executor-to-executor` passes by supplying the
/// `populateAdditionalTypeConversions` callback. This should be used to convert
/// additional opaque types that are defined in custom dialects within function,
/// control-flow, and global operations that have not yet had type conversions
/// applied.
void buildExecutorLoweringPipeline(
    OpPassManager &pm, const ConvertStdToExecutorPassOptions &stdToExecutorOpts,
    const std::function<void(TypeConverter &)>
        &populateAdditionalTypeConversions = {});

/// Register Executor-related pass pipelines.
void registerExecutorPassPipelines();

} // namespace executor
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_TRANSFORMS_PASSES_H
