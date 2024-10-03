//===- ModulePass.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Defines a pass with static scheduling filter that only allows it to be run
/// on "module-like" operations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_UTILS_MODULEPASS
#define MLIR_EXECUTOR_UTILS_MODULEPASS

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
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
  bool canScheduleOn(RegisteredOperationName opName) const final {
    return opName.hasTrait<OpTrait::IsIsolatedFromAbove>() &&
           opName.hasTrait<OpTrait::SymbolTable>();
  }

  /// Used in pass `runOnOperation` implementation to ensure that the op meets
  /// requirements that can't be checked in 'canScheduleOn'.
  static LogicalResult checkIsModuleLike(Operation *op) {
    if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock())
      return emitError(op->getLoc())
             << "expected a module-like operation with a single region "
                "containing a single block";
    return success();
  }
};
} // namespace mlir

#endif // MLIR_EXECUTOR_UTILS_MODULEPASS
