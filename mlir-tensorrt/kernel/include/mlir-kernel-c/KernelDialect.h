//===- KernelDialect.h ----------------------------------------------------===//
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
/// Declarations for Kernel dialect using MLIR C API.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_C_DIALECTS_KERNEL_KERNELDIALECT
#define MLIR_TENSORRT_C_DIALECTS_KERNEL_KERNELDIALECT

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir-kernel/Kernel/Transforms/Passes.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Kernel, kernel);

#ifdef __cplusplus
}
#endif

#endif // MLIR_TENSORRT_C_DIALECTS_KERNEL_KERNELDIALECT
