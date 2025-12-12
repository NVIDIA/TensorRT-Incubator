//===- ScatterUtils.h -----------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Utility functions for operations similar to `stablehlo.scatter`.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_UTILS_SCATTERUTILS
#define MLIR_KERNEL_UTILS_SCATTERUTILS

#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::kernel {

/// Verify an operation which is similar to `stablehlo.scatter`, except for the
///  following differences:
///  - The `updateComputation` uses scalar types, not 0-d tensor types.
///  - Type promotion is not allowed.
///  - Quant types are not allowed.
LogicalResult verifyStablehloLikeScatterOp(
    std::optional<Location> location, ValueRange inputs, Value scatterIndices,
    ValueRange updates, ArrayRef<int64_t> updateWindowDims,
    ArrayRef<int64_t> insertedWindowDims, ArrayRef<int64_t> inputBatchingDims,
    ArrayRef<int64_t> scatterIndicesBatchingDims,
    ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim,
    Region &updateComputation);

} // namespace mlir::kernel

#endif // MLIR_KERNEL_UTILS_SCATTERUTILS
