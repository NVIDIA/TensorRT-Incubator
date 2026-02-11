//===- Transforms.h ---------------------------------------------*- C++ -*-===//
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
/// Generic MLIR transformations that do not depend on any MLIR-TensorRT
/// dialects.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_TRANSFORMS_TRANSFORMS_H
#define MLIR_TENSORRT_TRANSFORMS_TRANSFORMS_H

#include "mlir/IR/Value.h"
#include <functional>
namespace mlir {
class ModuleOp;
class RewriterBase;
class RewritePatternSet;
class PatternBenefit;

/// Remove any operations nested below `op` that have the "IsolatedFromAbove"
/// and "SymbolTable" attribute.
void dropNestedModules(RewriterBase &rewriter, ModuleOp op);

using ShouldScalarizeWhileBeforeArgFunc =
    std::function<bool(BlockArgument, Value initOperand, Value yieldOperand)>;
using ShouldScalarizeWhileAfterArgFunc =
    std::function<bool(BlockArgument, Value condOperand, Value result)>;

/// Populates the patterns to detensorize scf.while ops. The provided functions
/// are used to control whether the arguments in each region are a candidate for
/// scalarization. They will currently only receive arguments that are tensor
/// types with a single element.
void populateSCFDetensorizeWhilePatterns(
    RewritePatternSet &patterns,
    ShouldScalarizeWhileBeforeArgFunc shouldScalarizeBeforeArg,
    ShouldScalarizeWhileAfterArgFunc shouldScalarizeAfterArg,
    PatternBenefit benefit);

} // namespace mlir

#endif // MLIR_TENSORRT_TRANSFORMS_TRANSFORMS_H
