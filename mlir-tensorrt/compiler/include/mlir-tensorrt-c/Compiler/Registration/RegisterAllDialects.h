//===- RegisterAllDialects.h --------------------------------------*- C -*-===//
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
/// Declarations for dialect and pass registration functions.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_C_COMPILER_REGISTRATION_REGISTERALLDIALECTS
#define MLIR_TENSORRT_C_COMPILER_REGISTRATION_REGISTERALLDIALECTS

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Add all the dialects used by MLIR-TensorRT to the registry.
MLIR_CAPI_EXPORTED void
mtrtCompilerRegisterDialects(MlirDialectRegistry registry);

/// Register all the compiler passes used by MLIR-TensorRT.
MLIR_CAPI_EXPORTED void mtrtCompilerRegisterPasses();

/// Register all the compiler task types (pass manager types).
MLIR_CAPI_EXPORTED void mtrtCompilerRegisterTasks();

#ifdef __cplusplus
}
#endif

#endif // MLIR_TENSORRT_C_COMPILER_REGISTRATION_REGISTERALLDIALECTS
