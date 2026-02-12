//===- SCFDetensorize.h ---------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_TRANSFORMS_SCFDETENSORIZE_SCFDETENSORIZE_H
#define MLIR_TENSORRT_TRANSFORMS_SCFDETENSORIZE_SCFDETENSORIZE_H

#include "mlir/IR/Value.h"
#include <functional>

namespace mlir {
class ModuleOp;
class RewriterBase;
class RewritePatternSet;
class PatternBenefit;

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

#endif // MLIR_TENSORRT_TRANSFORMS_SCFDETENSORIZE_SCFDETENSORIZE_H
