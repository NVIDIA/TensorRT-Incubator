//===- Transforms.h ---------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_TRANSFORMS_DROPNESTEDMODULES_DROPNESTEDMODULES_H
#define MLIR_TENSORRT_TRANSFORMS_DROPNESTEDMODULES_DROPNESTEDMODULES_H

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

} // namespace mlir

#endif // MLIR_TENSORRT_TRANSFORMS_DROPNESTEDMODULES_DROPNESTEDMODULES_H
