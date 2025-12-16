//===- Utils.h ------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Utilities for building MLIR IR that are missing from the upstream C API.
/// NOTE: don't include things here that are specific to a particular top-level
/// API
/// TODO: move these things to upstream C API.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_C_UTILS_UTILS
#define MLIR_TENSORRT_C_UTILS_UTILS

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Affine/Indexing Utilities
//===----------------------------------------------------------------------===//

/// This performs the equivalent of using
/// `affine::fullyComposeAffineMapAndOperands` and replacing the map and
/// operands of the `affineApplyOp` the map was changed.
MLIR_CAPI_EXPORTED void
mtrtSimplifyAffineApplyInPlace(MlirOperation affineApplyOp);

/// Returns true if the `affine.apply` operation's map retuns a single constant.
/// NOTE: we only need this and the below function since upstream MLIR Python
/// bindings need to be updated to get the AffineMap from an AffineMapAttribute,
/// otherwise this could be done in Python.
MLIR_CAPI_EXPORTED bool mtrtAffineApplyIsConstant(MlirOperation affineApplyOp);

/// Returns the constant value that the `affine.apply` operation's map returns.
/// This asserts that the map returns a single constant.
MLIR_CAPI_EXPORTED int64_t
mtrtAffineApplyGetConstant(MlirOperation affineApplyOp);

#ifdef __cplusplus
}
#endif

#endif // MLIR_TENSORRT_C_UTILS_UTILS
