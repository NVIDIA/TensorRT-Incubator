//===- RegionUtils.h --------------------------------------------*- C++ -*-===//
//
// This file contains code modified from upstream MLIR project.
// The original code is licensed under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
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
/// This file contains the declarations for the `RegionUtils` class.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_UTILS_REGIONUTILS
#define MLIR_TENSORRT_COMMON_UTILS_REGIONUTILS

#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class RewriterBase;

/// This function is adapted from the upstream MLIR function
/// "makeRegionIsolatedFromAbove" here:
/// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Transforms/Utils/RegionUtils.cpp.
/// Original code licensed under the Apache License v2.0 with LLVM Exceptions.
///
/// The modifications are:
/// - Add a callback to allow reordering the captured values.
/// - Fix an issue with the original implementation where uses in nested regions
///   were not correctly replaced.
///
/// Create a closed region that is "isolated from above". For each value used in
/// the region which is defined above that region, the value is either return in
/// the result vector (to become the new arguments which should be passed to the
/// region) or the producer is cloned into the region. The action is controlled
/// by the callback `cloneOperationIntoRegion`. This procedure is applied
/// recursively to cloned regions.
///
/// \param builder The builder to use to clone the operations.
/// \param region The region to create a closed region from.
/// \param cloneOperationIntoRegion A function to determine if the producer of
/// the value should be cloned into the region.
/// \param reorderCapturedValues A function to reorder the captured values
/// before creating the new region arguments.
/// \return A vector of values defined above the region which correspond to the
/// new region arguments.
SmallVector<Value> createClosedRegion(
    RewriterBase &rewriter, Region &region,
    llvm::function_ref<bool(Operation *)> cloneOperationIntoRegion = {},
    llvm::function_ref<void(SmallVectorImpl<Value> &)> reorderCapturedValues =
        {});

} // namespace mlir

#endif // MLIR_TENSORRT_COMMON_UTILS_REGIONUTILS
