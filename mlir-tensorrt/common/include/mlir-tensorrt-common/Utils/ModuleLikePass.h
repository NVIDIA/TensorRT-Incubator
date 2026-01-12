//===- ModuleLikePass.h --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Defines a pass with static scheduling filter that only allows it to be run
/// on "module-like" operations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_UTILS_MODULELIKEPASS
#define MLIR_TENSORRT_COMMON_UTILS_MODULELIKEPASS

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Pass to transform an operation that is "module like", meaning:
/// - it is isolated from above
/// - it has SymbolTable trait
/// - it has single-region with a single-block.
///
/// Module-like passes must not:
///   - modify any other operations within the parent region, as other threads
///     may be manipulating them concurrently.
///   - modify any state within the parent operation, this includes adding
///     additional operations.
///
/// Derived interface passes are expected to provide the following:
///   - A 'void runOnOperation()' method.
///   - A 'StringRef getName() const' method.
///   - A 'std::unique_ptr<Pass> clonePass() const' method.
///   - Check additional dynamic scheduling requirements using
///     `checkIsModuleLike` in the `runOnOperation` method.
class ModuleLikePass : public OperationPass<> {
protected:
  using OperationPass::OperationPass;

  /// Indicate if the current pass can be scheduled on the given operation type.
  /// For an InterfacePass, this checks if the operation implements the given
  /// interface.
  bool canScheduleOn(RegisteredOperationName opName) const final;

  /// Used in pass `runOnOperation` implementation to ensure that the op meets
  /// requirements that can't be checked in 'canScheduleOn'.
  static LogicalResult checkIsModuleLike(Operation *op);
};
} // namespace mlir

#endif // MLIR_TENSORRT_COMMON_UTILS_MODULELIKEPASS
