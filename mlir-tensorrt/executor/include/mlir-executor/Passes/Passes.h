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
/// Declarations for all Executor dialect passes (conversion and transforms).
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_PASSES_PASSES_H
#define MLIR_EXECUTOR_PASSES_PASSES_H

#include "mlir-tensorrt-common/Utils/ModuleLikePass.h" // IWYU pragma: keep
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class RewritePatternSet;
class ConversionTarget;
class DataLayout;
class TypeConverter;
namespace func {
class FuncDialect;
}
} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen'd pass declarations and registration
//===----------------------------------------------------------------------===//
namespace mlir::executor {
class ExecutorTypeConverter;
struct ConvertStdToExecutorPassOptions;

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir-executor/Passes/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Conversion API
//===----------------------------------------------------------------------===//

/// Populate memref-to-executor patterns.
void populateMemRefToExecutorPatterns(RewritePatternSet &patterns,
                                      ExecutorTypeConverter &typeConverter,
                                      bool allowUncheckedMemrefCastConversion);

void populateLinalgToExecutorPatterns(RewritePatternSet &patterns,
                                      ExecutorTypeConverter &typeConverter);

/// Populate arith-to-executor patterns.
void populateArithToExecutorPatterns(RewritePatternSet &patterns,
                                     ExecutorTypeConverter &typeConverter);

void populateFuncToExecutorPatterns(RewritePatternSet &patterns,
                                    ExecutorTypeConverter &typeConverter);

void populateControlFlowToExecutorPatterns(
    RewritePatternSet &patterns, ExecutorTypeConverter &typeConverter);

/// Populate the target such that any Executor dialect is legal if its types are
/// legal.
void populateExecutorDialectLegality(ExecutorTypeConverter &typeConverter,
                                     ConversionTarget &target);

/// Populate patterns that do Executor-to-Executor conversions for the given
/// type converter.
void populateExecutorStructuralConversionPatternsAndLegality(
    RewritePatternSet &patterns, ExecutorTypeConverter &converter,
    ConversionTarget &target);

/// Create a `std-to-executor` pass with additional type conversions added via
/// callback.
std::unique_ptr<Pass> createConvertStdToExecutorPass(
    const ConvertStdToExecutorPassOptions &stdToExecutorOpts,
    const std::function<void(TypeConverter &)>
        &populateAdditionalTypeConversions);

/// Create a `executor-to-executor` pass with additional type conversions added
/// via callback.
std::unique_ptr<Pass> createConvertExecutorToExecutorPass(
    const ConvertExecutorToExecutorPassOptions &executorToExecutorOpts,
    const std::function<void(TypeConverter &)>
        &populateAdditionalTypeConversions);

//===----------------------------------------------------------------------===//
// Pipeline API
//===----------------------------------------------------------------------===//

/// Builds a pipeline from lowering Executor-compatible IR to the final form
/// prior to target translation.
void buildExecutorLoweringPipeline(
    OpPassManager &pm, const ConvertStdToExecutorPassOptions &stdToExecutorOpts,
    const std::function<void(TypeConverter &)>
        &populateAdditionalTypeConversions = {});

/// Register Executor-related pass pipelines.
void registerExecutorPassPipelines();

} // namespace mlir::executor

#endif // MLIR_EXECUTOR_PASSES_PASSES_H
