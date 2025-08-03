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
#ifndef INCLUDE_MLIR_EXECUTOR_CONVERSION_PASSES
#define INCLUDE_MLIR_EXECUTOR_CONVERSION_PASSES

#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlir {
class RewritePatternSet;
class ConversionTarget;
class DataLayout;
class TypeConverter;
} // namespace mlir

//===----------------------------------------------------------------------===//
// Add Tablegen'd pass declarations and registration methods.
//===----------------------------------------------------------------------===//
namespace mlir::executor {
class ExecutorTypeConverter;

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir-executor/Conversion/Passes.h.inc"

/// Populate memref-to-executor patterns.
void populateMemRefToExecutorPatterns(RewritePatternSet &patterns,
                                      ExecutorTypeConverter &typeConverter,
                                      bool allowUncheckedMemrefCastConversion);

void populateLinalgToExecutorPatterns(RewritePatternSet &patterns,
                                      ExecutorTypeConverter &typeConverter);

/// Populate arith-to-executor patterns.
void populateArithToExecutorPatterns(RewritePatternSet &patterns,
                                     ExecutorTypeConverter &typeConverter);
// Populate func-to-executor patterns.
void populateFuncToExecutorPatterns(RewritePatternSet &patterns,
                                    ExecutorTypeConverter &typeConverter);
// Populate cf-to-executor patterns.
void populateControlFlowToExecutorPatterns(
    RewritePatternSet &patterns, ExecutorTypeConverter &typeConverter);

/// Populate the target such that any Executor dialect is legal if its types are
/// legal.
void populateExecutorDialectLegality(ExecutorTypeConverter &typeConverter,
                                     ConversionTarget &target);

/// Populate patterns that do Executor-to-Executor conversions for the given
/// type converter. This includes structural type conversion patterns that
/// are needed to lower things like `executor.call`, `executor.func` and so
/// on.
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

} // namespace mlir::executor

#endif // INCLUDE_MLIR_EXECUTOR_CONVERSION_PASSES
