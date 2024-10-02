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

//===----------------------------------------------------------------------===//
// MTRT_StableHLOToExecutableOptions
//===----------------------------------------------------------------------===//

/// Options for compiling StableHLO MLIR to an Executable.
typedef struct MTRT_StableHLOToExecutableOptions {
  void *ptr;
} MTRT_StableHLOToExecutableOptions;

/// A callback that allows the user to customize the metadata set for layers
/// corresponding to each MLIR operation. The callback should invoke the
/// provided append function in order to manipulate the result string.
typedef void (*MTRT_MetadataCallback)(MlirOperation op,
                                      MlirStringCallback append,
                                      void *appendCtx, void *userData);

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

/// Sets the layer metadata callback. The `userData` argument is passed along
/// to the callback when it is invoked.
MLIR_CAPI_EXPORTED MTRT_Status
mtrtStableHloToExecutableOptionsSetTensorRTTranslationMetadataCallback(
    MTRT_StableHLOToExecutableOptions options, MTRT_MetadataCallback callback,
    void *userData);

MLIR_CAPI_EXPORTED MTRT_Status mtrtStableHloToExecutableOptionsDestroy(
    MTRT_StableHLOToExecutableOptions options);

static inline bool mtrtStableHloToExecutableOptionsIsNull(
    MTRT_StableHLOToExecutableOptions options) {
  return !options.ptr;
}

//===----------------------------------------------------------------------===//
// Main StableHLO Compiler API Functions
//===----------------------------------------------------------------------===//

/// Compiler StableHLO to Executable.
MLIR_CAPI_EXPORTED MTRT_Status mtrtCompilerStableHLOToExecutable(
    MTRT_CompilerClient client, MlirOperation module,
    MTRT_StableHLOToExecutableOptions options, MTRT_Executable *result);

//===----------------------------------------------------------------------===//
// MTRT_StableHLOProgramSignatureRefinementOptions
//===----------------------------------------------------------------------===//

/// Options for compiling StableHLO MLIR to an Executable.
typedef struct MTRT_StableHLOProgramSignatureRefinementOptions {
  void *ptr;
} MTRT_StableHLOProgramSignatureRefinementOptions;

MLIR_CAPI_EXPORTED MTRT_Status
mtrtStableHloProgramSignatureRefinementOptionsCreate(
    MTRT_StringView funcName,
    MTRT_StableHLOProgramSignatureRefinementOptions *options);

MLIR_CAPI_EXPORTED MTRT_Status
mtrtStableHloProgramSignatureRefinementOptionsDestroy(
    MTRT_StableHLOProgramSignatureRefinementOptions options);

static inline bool mtrtStableHloProgramSignatureRefinementOptionsIsNull(
    MTRT_StableHLOProgramSignatureRefinementOptions options) {
  return !options.ptr;
}

//===----------------------------------------------------------------------===//
// Main StableHLO Program Signature Refinement API Functions
//===----------------------------------------------------------------------===//

/// Compiler StableHLO to Executable.
MLIR_CAPI_EXPORTED MTRT_Status mtrtGetStableHloProgramRefinedSignature(
    MTRT_CompilerClient client, MlirOperation module,
    MTRT_StableHLOProgramSignatureRefinementOptions options, MlirType *result);

#ifdef __cplusplus
}
#endif

#endif // MLIR_TENSORRT_C_COMPILER_COMPILER
