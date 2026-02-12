//===- ExpandOps.h -------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===--------------------------------------------------------------------===//
///
/// Declarations for the `executor-expand-ops` pass.
///
//===--------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_EXECUTOR_TRANSFORMS_EXPANDOPS
#define MLIR_EXECUTOR_EXECUTOR_TRANSFORMS_EXPANDOPS

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir {
namespace executor {

/// Calculate the offset for a multi-dimensional indexing operation.
/// This function computes the byte offset for accessing an element of type
/// `elemType` using the provided `indices`. The result is an integer value
/// of type `resultType`.
///
/// \param rewriter The rewriter to use for creating operations
/// \param layout The data layout to use for size/alignment calculations
/// \param loc The location to use for created operations
/// \param elemType The type of the element being indexed
/// \param resultType The integer type for the result offset value
/// \param indices The indices for the offset calculation
/// \param checkPrecisionLoss If true, insert runtime checks for precision loss
/// \return The computed offset value, or failure if the calculation is invalid
FailureOr<Value> calculateOffset(RewriterBase &rewriter,
                                 const DataLayout &layout, Location loc,
                                 Type elemType, Type resultType,
                                 ArrayRef<OpFoldResult> indices,
                                 bool checkPrecisionLoss = false);

} // namespace executor
} // namespace mlir

#endif // MLIR_EXECUTOR_EXECUTOR_TRANSFORMS_EXPANDOPS
