//===- AliasAnalysis.h ----------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// Extended alias analysis with support for executor-specific constructs.
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_ANALYSIS_ALIASANALYSIS
#define MLIR_TENSORRT_ANALYSIS_ALIASANALYSIS

#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"

namespace mlir {

/// LocalAliasAnalysis extended to support "executor.restrict" attribute and
/// executor.abi.recv operations.
///
/// Two values are considered non-aliasing if:
/// 1. Both are function arguments and at least one has "executor.restrict", OR
/// 2. Both are produced by executor.abi.recv from different function arguments,
/// OR
/// 3. One is a function argument and the other comes from executor.abi.recv
///    with a different source argument.
class RestrictAwareAliasAnalysis : public LocalAliasAnalysis {
protected:
  AliasResult aliasImpl(Value lhs, Value rhs) override;
};

/// Helper to create an AliasAnalysis instance with RestrictAwareAliasAnalysis.
AliasAnalysis createRestrictAwareAliasAnalysis(Operation *op);

} // namespace mlir

#endif // MLIR_TENSORRT_ANALYSIS_ALIASANALYSIS
