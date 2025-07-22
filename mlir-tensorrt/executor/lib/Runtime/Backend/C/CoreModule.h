//===- CoreModule.h -----------------------------------------------*- C -*-===//
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
/// Core C runtime declarations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_RUNTIME_BACKEND_C_COREMODULE
#define MLIR_EXECUTOR_RUNTIME_BACKEND_C_COREMODULE

#include "mlir-tensorrt-common-c/Support/Status.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

MTRT_CAPI_EXPORTED void __memset_8(uintptr_t pointer, size_t offset,
                                   size_t numBytes, uint8_t fillInt);
MTRT_CAPI_EXPORTED void __memset_16(uintptr_t pointer, size_t offset,
                                    size_t numBytes, uint16_t fillInt);
MTRT_CAPI_EXPORTED void __memset_32(uintptr_t pointer, size_t offset,
                                    size_t numBytes, uint32_t fillInt);

#ifdef __cplusplus
}
#endif

#endif // MLIR_EXECUTOR_RUNTIME_BACKEND_C_COREMODULE
