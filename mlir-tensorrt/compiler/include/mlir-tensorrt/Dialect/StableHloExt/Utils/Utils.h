//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
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
/// Generic utilities for the StableHlo dialect.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_STABLEHLOEXT_UTILS_UTILS
#define MLIR_TENSORRT_DIALECT_STABLEHLOEXT_UTILS_UTILS

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Operation.h"
namespace mlir {
class Value;
namespace stablehlo {

/// Returns true if the type of `result` can be updated and replaced (assuming
/// replacement type is compatible with respect to dynamic dim refinement)
/// without inserting a cast. This is true if:
/// 1. All users are StableHLO operations, since the StableHLO dialect was
///    designed with this in mind, or
/// 2. The operation passes a custom check defined by the `isCustomOp` lambda.
/// For operations that don't meet these criteria, we conservatively require
/// a cast to be inserted.
bool canUpdateTypeWithoutCast(
    Value result, const std::function<bool(Operation *)> &isCustomOp =
                      [](Operation *) { return false; });

} // namespace stablehlo
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_STABLEHLOEXT_UTILS_UTILS
