//===- Common.h ---------------------------------------------------*- C -*-===//
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
/// Common elements shared across the compiler and runtime APIS.
///
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTOR_C_COMMON_COMMON
#define MLIR_EXECUTOR_C_COMMON_COMMON

#include "mlir-executor-c/Support/Status.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct MTRT_ArrayRefI64 {
  int64_t size;
  int64_t *ptr;
};

inline static MTRT_ArrayRefI64 mtrtArrayRefI64GetEmpty() {
  return MTRT_ArrayRefI64{0, nullptr};
}

//===----------------------------------------------------------------------===//
// Data Enums
//===----------------------------------------------------------------------===//

/// MTRT_ScalarTypeCode is the C API equivalent to
/// runtime::impl::ScalarTypeCode.
/// TODO: Make some automation to update this enum.
typedef enum MTRT_ScalarTypeCode {
  MTRT_ScalarTypeCode_unknown = 0,
  MTRT_ScalarTypeCode_f8e4m3fn = 1,
  MTRT_ScalarTypeCode_f16 = 2,
  MTRT_ScalarTypeCode_f32 = 3,
  MTRT_ScalarTypeCode_f64 = 4,
  MTRT_ScalarTypeCode_i1 = 5,
  MTRT_ScalarTypeCode_i8 = 6,
  MTRT_ScalarTypeCode_ui8 = 7,
  MTRT_ScalarTypeCode_i16 = 8,
  MTRT_ScalarTypeCode_i32 = 9,
  MTRT_ScalarTypeCode_i64 = 10,
  MTRT_ScalarTypeCode_bf16 = 11,
  MTRT_ScalarTypeCode_i4 = 12,
  MTRT_ScalarTypeCode_complex32 = 13,
  MTRT_ScalarTypeCode_complex64 = 14,
  MTRT_ScalarTypeCode_MIN = MTRT_ScalarTypeCode_unknown,
  MTRT_ScalarTypeCode_MAX = MTRT_ScalarTypeCode_complex64
} MTRT_ScalarTypeCode;

typedef enum MTRT_PointerType {
  MTRT_PointerType_host = 0,
  MTRT_PointerType_pinned_host = 1,
  MTRT_PointerType_device = 2,
  MTRT_PointerType_unified = 3,
  MTRT_PointerType_unknown = 4,
  MTRT_PointerType_MIN = MTRT_PointerType_host,
  MTRT_PointerType_MAX = MTRT_PointerType_unknown,
} MTRT_PointerType;

/// Returns the number of bits that comprise the given scalar element type.
MTRT_CAPI_EXPORTED MTRT_Status
mtrtScalarTypeCodeBitsPerElement(MTRT_ScalarTypeCode code, int64_t *result);

/// Returns enum name string given the scalar element type.
MTRT_CAPI_EXPORTED const char *
mtrtScalarTypeCodeGetString(MTRT_ScalarTypeCode code);

//===----------------------------------------------------------------------===//
// MTRT_Executable
//===----------------------------------------------------------------------===//

typedef struct MTRT_Executable {
  void *ptr;
} MTRT_Executable;

MTRT_CAPI_EXPORTED bool mtrtExecutableIsNull(MTRT_Executable executable);

MTRT_CAPI_EXPORTED MTRT_Status
mtrtExecutableDestroy(MTRT_Executable executable);

/// Create a new instance of `MTRT_Executable`
MTRT_CAPI_EXPORTED MTRT_Status mtrtExecutableCreate(MTRT_StringView buffer,
                                                    MTRT_Executable *result);

/// Serialize `MTRT_Executable` to a string buffer
MTRT_CAPI_EXPORTED MTRT_Status mtrtExecutableGetStorageView(
    MTRT_Executable executable, MTRT_StringView *buffer,
    size_t *requiredAlignment);

//===----------------------------------------------------------------------===//
// MTRT_Type
//===----------------------------------------------------------------------===//

/// A MTRT_Type represents a union of the potential concrete types.
typedef struct MTRT_Type {
  void *ptr;
} MTRT_Type;

MTRT_CAPI_EXPORTED bool mtrtTypeIsNull(MTRT_Type type);

MTRT_CAPI_EXPORTED MTRT_Status mtrtTypeDestroy(MTRT_Type type);

//===----------------------------------------------------------------------===//
// MTRT_ScalarType
//===----------------------------------------------------------------------===//

typedef struct MTRT_ScalarType {
  void *ptr;
} MTRT_ScalarType;

MTRT_CAPI_EXPORTED bool mtrtScalarTypeIsNull(MTRT_ScalarType scalar);

MTRT_CAPI_EXPORTED MTRT_Status mtrtScalarTypeDestroy(MTRT_ScalarType scalar);

MTRT_CAPI_EXPORTED bool mtrtTypeIsaScalarType(MTRT_Type type);

/// Create a new instance of `MTRT_ScalarType`, which is upcasted and returned
/// as the generic `MTRT_Type`.
/// Note that the API only provides a method for destroying `MTRT_Type`, not
/// downcasted concrete types.
MTRT_CAPI_EXPORTED MTRT_Status mtrtScalarTypeCreate(MTRT_ScalarTypeCode code,
                                                    MTRT_Type *result);

/// Downcast to a `MTRT_ScalarType` from a generic `MTRT_Type`.
MTRT_CAPI_EXPORTED MTRT_ScalarType mtrtTypeGetScalarType(MTRT_Type type);

/// Retrieve the enum type code.
MTRT_CAPI_EXPORTED MTRT_ScalarTypeCode
mtrtScalarTypeGetCode(MTRT_ScalarType type);

//===----------------------------------------------------------------------===//
// MTRT_MemRefType
//===----------------------------------------------------------------------===//

typedef struct MTRT_MemRefType {
  void *ptr;
} MTRT_MemRefType;

/// Utility struct to store memref type data.
typedef struct MTRT_MemRefTypeInfo {
  int64_t rank;
  const int64_t *shape;
  int64_t *strides;
  MTRT_ScalarTypeCode elementType;
  MTRT_PointerType addressSpace;
} MTRT_MemRefTypeInfo;

MTRT_CAPI_EXPORTED bool mtrtMemRefTypeIsNull(MTRT_MemRefType memref);

MTRT_CAPI_EXPORTED MTRT_Status mtrtMemRefTypeDestroy(MTRT_MemRefType memref);

MTRT_CAPI_EXPORTED bool mtrtTypeIsaMemRefType(MTRT_Type type);

/// Create a new instance of `MTRT_MemRefType`, which is upcasted and returned
/// as the generic `MTRT_Type`.
/// Note that the API only provides a method for destroying `MTRT_Type`, not
/// downcasted concrete types.
MTRT_CAPI_EXPORTED MTRT_Status mtrtMemRefTypeCreate(
    int64_t rank, const int64_t *shape, MTRT_ScalarTypeCode elementType,
    MTRT_PointerType addressSpace, MTRT_Type *result);

/// Downcast to a `MTRT_MemRefType` from a generic `MTRT_Type`.
MTRT_CAPI_EXPORTED MTRT_MemRefType mtrtTypeGetMemRefType(MTRT_Type type);

/// Retrieve metadata for the provided memref.
MTRT_CAPI_EXPORTED MTRT_Status mtrtMemRefTypeGetInfo(MTRT_Type memref,
                                                     MTRT_MemRefTypeInfo *info);

// TODO: Add a methd to retrieve element type which return a MTRT_Type. We need
// this to support nested objects like memref<...xmemref<....> >.

//===----------------------------------------------------------------------===//
// MTRT_ExternalOpaqueRefType
//===----------------------------------------------------------------------===//

typedef struct MTRT_ExternalOpaqueRefType {
  void *ptr;
} MTRT_ExternalOpaqueRefType;

MTRT_CAPI_EXPORTED bool
mtrtExternalOpaqueRefTypeIsNull(MTRT_ExternalOpaqueRefType opaque);

MTRT_CAPI_EXPORTED MTRT_Status
mtrtExternalOpaqueRefTypeDestroy(MTRT_ExternalOpaqueRefType opaque);

//===----------------------------------------------------------------------===//
// MTRT_Bounds
//===----------------------------------------------------------------------===//

typedef struct MTRT_Bounds {
  void *ptr;
} MTRT_Bounds;

MTRT_CAPI_EXPORTED bool mtrtBoundsIsNull(MTRT_Bounds bounds);

MTRT_CAPI_EXPORTED MTRT_Status mtrtBoundsDestroy(MTRT_Bounds bounds);

MTRT_CAPI_EXPORTED MTRT_Status mtrtBoundsGetSize(MTRT_Bounds bounds,
                                                 MTRT_ArrayRefI64 *size);
MTRT_CAPI_EXPORTED MTRT_Status mtrtBoundsGetMin(MTRT_Bounds bounds,
                                                MTRT_ArrayRefI64 *out);
MTRT_CAPI_EXPORTED MTRT_Status mtrtBoundsGetMax(MTRT_Bounds bounds,
                                                MTRT_ArrayRefI64 *out);

//===----------------------------------------------------------------------===//
// MTRT_FunctionSignature
//===----------------------------------------------------------------------===//

typedef struct MTRT_FunctionSignature {
  void *ptr;
} MTRT_FunctionSignature;

MTRT_CAPI_EXPORTED bool
mtrtFunctionSignatureIsNull(MTRT_FunctionSignature signature);

MTRT_CAPI_EXPORTED MTRT_FunctionSignature
mtrtGetFunctionSignature(MTRT_Executable exec, const char *name);

MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetString(
    MTRT_FunctionSignature signature, MTRT_PrintCallbackInfo callback);

MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetNumArgs(
    MTRT_FunctionSignature signature, int64_t *numArgs);
MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetNumResults(
    MTRT_FunctionSignature signature, int64_t *numResults);
MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetNumInputArgs(
    MTRT_FunctionSignature signature, int64_t *numInputArgs);
MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetNumOutputArgs(
    MTRT_FunctionSignature signature, int64_t *numOutputArgs);
MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetArg(
    MTRT_FunctionSignature signature, int64_t index, MTRT_Type *type);
MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetResult(
    MTRT_FunctionSignature signature, int64_t index, MTRT_Type *type);
MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetArgBound(
    MTRT_FunctionSignature signature, int64_t index, MTRT_Bounds *bounds);
MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetResultBound(
    MTRT_FunctionSignature signature, int64_t index, MTRT_Bounds *bounds);
MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetNumArgBounds(
    MTRT_FunctionSignature signature, int64_t *numArgBounds);
MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetNumResBounds(
    MTRT_FunctionSignature signature, int64_t *numResBounds);
MTRT_CAPI_EXPORTED MTRT_Status mtrtFunctionSignatureGetShapeFuncName(
    MTRT_FunctionSignature signature, MTRT_StringView *name);

#ifdef __cplusplus
}
#endif

#endif // MLIR_EXECUTOR_C_COMMON_COMMON
