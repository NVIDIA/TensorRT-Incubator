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
// MTRT_OptionsContext
//===----------------------------------------------------------------------===//

typedef struct MTRT_OptionsContext {
  void *ptr;
} MTRT_OptionsContext;

MLIR_CAPI_EXPORTED MTRT_Status mtrtOptionsContextCreateFromArgs(
    MTRT_CompilerClient client, MTRT_OptionsContext *options,
    MlirStringRef optionsType, const MlirStringRef *argv, unsigned argc);

MLIR_CAPI_EXPORTED void mtrtOptionsContextPrint(MTRT_OptionsContext options,
                                                MlirStringCallback append,
                                                void *userData);

MLIR_CAPI_EXPORTED MTRT_Status
mtrtOptionsContextDestroy(MTRT_OptionsContext options);

static inline bool mtrtOptionsConextIsNull(MTRT_OptionsContext options) {
  return !options.ptr;
}

//===----------------------------------------------------------------------===//
// MTRT_StableHLOToExecutableOptions
//===----------------------------------------------------------------------===//

/// Options for compiling StableHLO MLIR to an Executable.
typedef struct MTRT_StableHLOToExecutableOptions {
  void *ptr;
} MTRT_StableHLOToExecutableOptions;

MLIR_CAPI_EXPORTED MTRT_Status mtrtStableHloToExecutableOptionsCreate(
    MTRT_CompilerClient client, MTRT_StableHLOToExecutableOptions *options,
    int32_t tensorRTBuilderOptLevel, bool tensorRTStronglyTyped);

MLIR_CAPI_EXPORTED MTRT_Status mtrtStableHloToExecutableOptionsCreateFromArgs(
    MTRT_CompilerClient client, MTRT_StableHLOToExecutableOptions *options,
    const MlirStringRef *argv, unsigned argc);

/// Specifies whether to enable the global LLVM debug flag for the duration of
/// the compilation process. If the flag is enabled then the debug types
/// specified in the array of literals are used as the global LLVM debug types
/// (equivalent to `-debug-only=[list]`).
MLIR_CAPI_EXPORTED MTRT_Status mtrtStableHloToExecutableOptionsSetDebugOptions(
    MTRT_StableHLOToExecutableOptions options, bool enableDebugging,
    const char **debugTypes, size_t debugTypeSizes,
    const char *dumpIrTreeDir = nullptr, const char *dumpTensorRTDir = nullptr);

MLIR_CAPI_EXPORTED MTRT_Status mtrtStableHloToExecutableOptionsDestroy(
    MTRT_StableHLOToExecutableOptions options);

static inline bool mtrtStableHloToExecutableOptionsIsNull(
    MTRT_StableHLOToExecutableOptions options) {
  return !options.ptr;
}

//===----------------------------------------------------------------------===//
// PassManagerReference APIs
//===----------------------------------------------------------------------===//

static inline bool mtrtPassManagerReferenceIsNull(MlirPassManager pm) {
  return !pm.ptr;
}

//===----------------------------------------------------------------------===//
// Main StableHLO Compiler API Functions
//===----------------------------------------------------------------------===//

/// Compiler StableHLO to Executable.
MLIR_CAPI_EXPORTED MTRT_Status mtrtCompilerStableHLOToExecutable(
    MTRT_CompilerClient client, MlirOperation module,
    MTRT_StableHLOToExecutableOptions options, MTRT_Executable *result);

#ifdef __cplusplus
}
#endif

#endif // MLIR_TENSORRT_C_COMPILER_COMPILER
