//===- TensorRTDialect.h ------------------------------------------*- C -*-===//
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
/// Declaration of TensorRT dialect CAPI registration.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_C_TENSORRTDIALECT
#define MLIR_TENSORRT_DIALECT_C_TENSORRTDIALECT

#include "mlir-c/IR.h"

/// Include generated pass registration methods.
#include "mlir-tensorrt-dialect/TensorRT/Transforms/Passes.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TensorRT, tensorrt);

#ifdef __cplusplus
}
#endif

#endif // MLIR_TENSORRT_DIALECT_C_TENSORRTDIALECT
