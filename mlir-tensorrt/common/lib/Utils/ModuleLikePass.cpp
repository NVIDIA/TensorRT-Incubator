//===- ModuleLikePass.cpp ------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Utils/ModuleLikePass.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

/// Indicate if the current pass can be scheduled on the given operation type.
/// For an InterfacePass, this checks if the operation implements the given
/// interface.
bool ModuleLikePass::canScheduleOn(RegisteredOperationName opName) const {
  return opName.hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         opName.hasTrait<OpTrait::SymbolTable>();
}

/// Used in pass `runOnOperation` implementation to ensure that the op meets
/// requirements that can't be checked in 'canScheduleOn'.
LogicalResult ModuleLikePass::checkIsModuleLike(Operation *op) {
  if (op->getNumRegions() != 1 || !op->getRegion(0).hasOneBlock())
    return emitError(op->getLoc())
           << "expected a module-like operation with a single region "
              "containing a single block";
  return success();
}
