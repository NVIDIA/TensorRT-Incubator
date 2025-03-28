//===- Compiler.h -------------------------------------------------*- C -*-===//
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
///  MLIR-TensorRT Compiler CAPI declarations.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_C_COMPILER_COMPILER
#define MLIR_TENSORRT_C_COMPILER_COMPILER

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"
#include "mlir-executor-c/Common/Common.h"
#include "mlir-executor-c/Support/Status.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// MTRT_CompilerClient
//===----------------------------------------------------------------------===//

typedef struct MTRT_CompilerClient {
  void *ptr;
} MTRT_CompilerClient;

MLIR_CAPI_EXPORTED MTRT_Status
mtrtCompilerClientCreate(MlirContext context, MTRT_CompilerClient *client);

MLIR_CAPI_EXPORTED MTRT_Status
mtrtCompilerClientDestroy(MTRT_CompilerClient client);

static inline bool mtrtCompilerClientIsNull(MTRT_CompilerClient options) {
  return !options.ptr;
}

MLIR_CAPI_EXPORTED MTRT_Status mtrtCompilerClientGetCompilationTask(
    MTRT_CompilerClient client, MlirStringRef taskMnemonic,
    const MlirStringRef *argv, unsigned argc, MlirPassManager *result);

//===----------------------------------------------------------------------===//
// PassManagerReference APIs
//===----------------------------------------------------------------------===//

static inline bool mtrtPassManagerReferenceIsNull(MlirPassManager pm) {
  return !pm.ptr;
}

#ifdef __cplusplus
}
#endif

#endif // MLIR_TENSORRT_C_COMPILER_COMPILER
