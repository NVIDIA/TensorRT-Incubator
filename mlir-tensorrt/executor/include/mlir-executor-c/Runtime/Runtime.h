//===- Runtime.h --------------------------------------------------*- C -*-===//
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
/// MLIR-TensorRT compiler and runtime C API intended to be used by
/// external consumers.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_C_RUNTIME_RUNTIME
#define MLIR_EXECUTOR_C_RUNTIME_RUNTIME

#include "dlpack/dlpack.h"
#include "mlir-c/Support.h"
#include "mlir-executor-c/Common/Common.h"
#include "mlir-executor-c/Support/Status.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
/// Conventions
/// - Runtime types are declared as
///  typedef struct MTRT_X {
///    void *ptr;
///  } MTRT_X;
/// where, `ptr` corresponds to the implementation.
/// - All function names should follow mtrt[ObjectName][ActionName], for example
/// mtrtStatusGetMessage or mtrtDevicesGet.
/// - Any function that can possibly fail should return a MTRT_Status .
/// - If the MTRT_Status returned from a function is `mtrtStatsGetOk()`, that
/// means the call succeeded.
/// - If the MTRT_Status returned from a function is not ok, then the user
/// should check the error code/status using the mtrtStatusGetMessage. The
/// caller must be sure to delete errors via mtrtStatusDestroy.
//===----------------------------------------------------------------------===//

typedef struct MTRT_RuntimeClient MTRT_RuntimeClient;

//===----------------------------------------------------------------------===//
// Global Initialization / Shutdown
//===----------------------------------------------------------------------===//

/// Perform global initialization of the runtime. This should only be called
/// once. Calling multiple times will result in an error.
MLIR_CAPI_EXPORTED void mtrtRuntimeInitialize();

/// Perform global de-initialization of the runtime.
MLIR_CAPI_EXPORTED void mtrtRuntimeShutdown();

//===----------------------------------------------------------------------===//
// MTRT_GlobalDebug
//===----------------------------------------------------------------------===//

/// Enables or disables global debug flag.
MLIR_CAPI_EXPORTED MTRT_Status mtrtEnableGlobalDebug(bool enable);

/// Retrieves the current state of the global debug flag.
MLIR_CAPI_EXPORTED MTRT_Status mtrtIsGlobalDebugEnabled(bool *enable);

/// Sets the current global debug type.
MLIR_CAPI_EXPORTED MTRT_Status mtrtSetGlobalDebugType(const char *type);

/// Sets multiple global debug types.
MLIR_CAPI_EXPORTED MTRT_Status mtrtSetGlobalDebugTypes(const char **types,
                                                       size_t n);

//===----------------------------------------------------------------------===//
// MTRT_Stream
//===----------------------------------------------------------------------===//

typedef struct MTRT_Stream {
  void *ptr;
} MTRT_Stream;

/// Creates `MTRT_Stream`.
MLIR_CAPI_EXPORTED MTRT_Status mtrtStreamCreate(MTRT_Stream *stream);

/// Checks nullity of `stream`.
static inline bool mtrtStreamIsNull(MTRT_Stream stream) { return !stream.ptr; }

/// Returns null stream.
static inline MTRT_Stream mtrtStreamGetNull() { return MTRT_Stream{NULL}; }

/// Synchronizes `MTRT_Stream`
MLIR_CAPI_EXPORTED MTRT_Status mtrtStreamSynchronize(MTRT_Stream stream);

/// Destroys for `MTRT_Stream` to be ready.
MLIR_CAPI_EXPORTED MTRT_Status mtrtStreamDestroy(MTRT_Stream stream);

//===----------------------------------------------------------------------===//
// MTRT_Device
//===----------------------------------------------------------------------===//

typedef struct MTRT_Device {
  void *ptr;
} MTRT_Device;

/// Check if MTRT_Device is null.
static inline bool mtrtDeviceIsNull(MTRT_Device device) { return !device.ptr; }

/// Return a null MTRT_Device. This should be used where MTRT_Device input
/// arguments are optional in functions below.
static inline MTRT_Device mtrtDeviceGetNull() { return MTRT_Device{NULL}; }

//===----------------------------------------------------------------------===//
// MTRT_MemRefValue
//===----------------------------------------------------------------------===//

typedef struct MTRT_MemRefValue {
  void *ptr;
} MTRT_MemRefValue;

typedef struct MTRT_DLPackManagedTensor {
  void *ptr;
} MTRT_DLPackManagedTensor;

typedef struct MTRT_DLPackDevice {
  void *ptr;
} MTRT_DLPackDevice;

/// Returns whether the memref is null.
static inline bool mtrtMemRefValueIsNull(MTRT_MemRefValue memref) {
  return !memref.ptr;
}

/// Creates a `MTRT_MemRefValue` for a tensor. Argument `rank` represents the
/// number of tensor dimensions, `shape` represents tensor shape, and
/// `strides` represents tensor strides in number of elements.
/// The RuntimeClient will take care of allocating memory in the appropraite
/// address space.
///
/// If `MTRT_PointerType` is `MTRT_MemRefValue_Device`, then:
///   - `device` must be provided, otherwise the result of `mtrtDeviceGetNull()`
///   should be passed.
///   - `stream` may optionally be provided in order to provide for asynchronous
///   allocation, otherwise
///      the result of `mtrtStreamGetNull()` should be passed.
MLIR_CAPI_EXPORTED MTRT_Status
mtrtMemRefCreate(MTRT_RuntimeClient client, MTRT_PointerType pointerKind,
                 int64_t bitsPerElement, int64_t rank, const int64_t *shape,
                 const int64_t *strides, MTRT_Device device, MTRT_Stream stream,
                 MTRT_ScalarTypeCode scalarType, MTRT_MemRefValue *result,
                 bool assertCanonicalStrides = false);

/// Creates an externally managed MemRef value. The caller provides all the
/// metadata for the MemRef including the shape, strides (in elements), pointer,
/// offset, and size of the element type in bits, and the device on which the
/// buffer resides (only if it is a device buffer). The RuntimeClient tracks
/// this MemRef as an externally managed resource and will not warn the user if
/// it is not freed when RuntimeClient is destroyed.
MLIR_CAPI_EXPORTED MTRT_Status mtrtMemRefCreateExternal(
    MTRT_RuntimeClient client, MTRT_PointerType pointerKind,
    int64_t bitsPerElement, uintptr_t ptr, int64_t offset, int64_t rank,
    const int64_t *shape, const int64_t *strides, MTRT_Device device,
    MTRT_ScalarTypeCode scalarType, MTRT_MemRefValue *result,
    bool assertCanonicalStrides = false);

/// Destroys `MTRT_MemRefValue` in a potentially asynchronous manner.
/// If `buffer` is a device buffer, device memory is freed in the stream
/// ordered asynchronous manner using `stream` if the stream is not null.
MLIR_CAPI_EXPORTED MTRT_Status
mtrtMemRefValueDestroyAsync(MTRT_MemRefValue buffer, MTRT_Stream stream);

/// Destroys the given memref. If the memref was allocated through the
/// `mtrtMemRefCreate` method, then the memory will be automatically freed
/// in a synchronous manner.
MLIR_CAPI_EXPORTED MTRT_Status mtrtMemRefValueDestroy(MTRT_MemRefValue buffer);

typedef struct MTRT_MemRefValueInfo {
  uintptr_t ptr;
  int64_t rank;
  int64_t offset;
  const int64_t *strides;
  const int64_t *shape;
  int64_t bitsPerElement;
  MTRT_ScalarTypeCode scalarType;
  MTRT_PointerType addressSpace;
} MTRT_MemRefValueInfo;

/// Retrieve metadata for the provided memref.
MLIR_CAPI_EXPORTED MTRT_Status
mtrtMemRefValueGetInfo(MTRT_MemRefValue memref, MTRT_MemRefValueInfo *info);

/// Create DL Managed tensor from MemRefValue.
MLIR_CAPI_EXPORTED MTRT_Status mtrtMemRefValueGetDLPackManagedTensor(
    MTRT_MemRefValue memrefValue, MTRT_DLPackManagedTensor *outTensor);

/// Retrieve DL Device from MemRefValue.
MLIR_CAPI_EXPORTED MTRT_Status mtrtMemRefValueGetDLPackDevice(
    MTRT_MemRefValue memrefValue, int32_t *device_type, int32_t *device_id);

MLIR_CAPI_EXPORTED MTRT_Status mtrtMemRefReferenceCount(
    MTRT_RuntimeClient client, uintptr_t ptr, int32_t *externalRefCount);

MLIR_CAPI_EXPORTED MTRT_Status mtrtMemRefIsReleasedInternally(
    MTRT_RuntimeClient client, uintptr_t ptr, bool *isReleasedInternally);

//===----------------------------------------------------------------------===//
// MTRT_RuntimeClient
//===----------------------------------------------------------------------===//

struct MTRT_RuntimeClient {
  void *ptr;
};

static inline bool mtrtRuntimeClientIsNull(MTRT_RuntimeClient client) {
  return !client.ptr;
}

/// Returns null client.
static inline MTRT_RuntimeClient mtrtRuntimeClientGetNull() {
  return MTRT_RuntimeClient{NULL};
}

/// Creates a `MTRT_RuntimeClient`. Client must be alive for the lifetime of the
/// program execution.
/// The `stream` passed to the client is used by all underlying CUDA methods
/// (which are stream ordered asynchronous, wherever
/// supported). `MTRT_RuntimeClient` keeps track of underlying
/// `MTRT_MemRefValue`s, to avoid un-intended memory leak in the device memory.
/// Generally, the user is responsible for destroying created
/// `MTRT_MemRefValue`s. If one or more `MTRT_MemRefValue` is not destroyed by
/// the user and it's a device buffer, `MTRT_RuntimeClient` uses passed `stream`
/// to free up the device memory in the stream ordered asynchronous manner, at
/// the end of program lifetime. However, if memory associated with the
/// `MTRT_MemRefValue` is the host memory, a leak warning is printed, without
/// freeing it. out: MTRT_RuntimeClient
/// **client
MLIR_CAPI_EXPORTED MTRT_Status
mtrtRuntimeClientCreate(MTRT_RuntimeClient *client);

/// Destroys the client. Any resources allocated through the client
/// (e.g. MTRT_MemRefValue) should be destroyed prior to destroying the client.
/// The client  will attempt to synchronously cleanup any dangling resources.
MLIR_CAPI_EXPORTED MTRT_Status
mtrtRuntimeClientDestroy(MTRT_RuntimeClient client);

/// Get the total number of devices available on the host.
MLIR_CAPI_EXPORTED MTRT_Status
mtrtRuntimeClientGetNumDevices(MTRT_RuntimeClient client, int32_t *numDevices);

/// Return the device at the specified index. Note that this is the internal
/// index, and it is not necessarily the same as the CUDA device ordinal.
MLIR_CAPI_EXPORTED MTRT_Status mtrtRuntimeClientGetDevice(
    MTRT_RuntimeClient client, int32_t index, MTRT_Device *device);

/// Retrieve the runtiem client that manages the specified memref.
MLIR_CAPI_EXPORTED MTRT_RuntimeClient
mtrtMemRefGetClient(MTRT_MemRefValue memref);

//===----------------------------------------------------------------------===//
// Data Transfer
//===----------------------------------------------------------------------===//

/// Copies `hostBuffer` to `device` using CUDA stream `stream`. Allocated device
/// memory pointer and a non-owing view of the `device` is attached to
/// MTRT_MemRefValue `deviceBuffer`.
/// out: MTRT_MemRefValue *deviceBuffer
MLIR_CAPI_EXPORTED MTRT_Status
mtrtCopyFromHostToDevice(MTRT_MemRefValue hostBuffer, MTRT_Device device,
                         MTRT_Stream stream, MTRT_MemRefValue *deviceBuffer);

/// Copies `hostBufferSource` to `hostBufferTarget`. General use case is
/// creating a new buffer from the view only host buffer, by copying the data.
/// out: MTRT_MemRefValue *hostBufferTarget
MLIR_CAPI_EXPORTED MTRT_Status mtrtCopyFromHostToHost(
    MTRT_MemRefValue hostBufferSource, MTRT_MemRefValue *hostBufferTarget);

/// Copies `deviceBuffer` to a newly allocated buffer `host` using CUDA stream
/// `stream`.
MLIR_CAPI_EXPORTED MTRT_Status mtrtCopyFromDeviceToNewHostMemRef(
    MTRT_MemRefValue deviceBuffer, MTRT_Stream stream,
    MTRT_MemRefValue *hostBuffer);

/// Copies `deviceBuffer` to an exising `hostBuffer` using the optional stream.
/// If a synchronous copy is desired, then pass the result of
/// `mtrtStreamGetNull()`.
MLIR_CAPI_EXPORTED MTRT_Status mtrtCopyFromDeviceToExistingHostMemRef(
    MTRT_MemRefValue deviceBuffer, MTRT_MemRefValue hostBuffer,
    MTRT_Stream stream);

//===----------------------------------------------------------------------===//
// MTRT_ScalarValue
//===----------------------------------------------------------------------===//

typedef struct MTRT_ScalarValue {
  void *ptr;
} MTRT_ScalarValue;

/// Returns whether the RuntimeValue is null.
static inline bool mtrtScalarValueIsNull(MTRT_ScalarValue value) {
  return !value.ptr;
}

//===----------------------------------------------------------------------===//
// MTRT_RuntimeValue
//===----------------------------------------------------------------------===//

typedef struct MTRT_RuntimeValue {
  void *ptr;
} MTRT_RuntimeValue;

/// Returns whether the RuntimeValue is null.
static inline bool mtrtRuntimeValueIsNull(MTRT_RuntimeValue value) {
  return !value.ptr;
}

// Returns whether the RuntimeValue is MemRef.
MLIR_CAPI_EXPORTED bool mtrtRuntimeValueIsMemRef(MTRT_RuntimeValue value);

// Returns whether the RuntimeValue is Scalar.
MLIR_CAPI_EXPORTED bool mtrtRuntimeValueIsScalar(MTRT_RuntimeValue value);

/// Cast a MTRT_MemRefValue to a generic MTRT_RuntimeValue.
MLIR_CAPI_EXPORTED MTRT_RuntimeValue
mtrtMemRefCastToRuntimeValue(MTRT_MemRefValue memref);

/// Cast a MTRT_RuntimeValue to a MTRT_MemRefValue. The result is null if the
/// cast is not valid.
MLIR_CAPI_EXPORTED MTRT_MemRefValue
mtrtRuntimeValueDynCastToMemRef(MTRT_RuntimeValue value);

/// Cast a MTRT_RuntimeValue to a MTRT_ScalarValue. The result is null if the
/// cast is not valid.
MLIR_CAPI_EXPORTED MTRT_ScalarValue
mtrtRuntimeValueDynCastToScalar(MTRT_RuntimeValue value);

/// Destroy a MTRT_RuntimeValue. This is equivalent to destorying the derived
/// concerete type.
MLIR_CAPI_EXPORTED MTRT_Status mtrtRuntimeValueDestroy(MTRT_RuntimeValue value);

/// Create a MTRT_ScalarValue representing an int64.
MLIR_CAPI_EXPORTED MTRT_Status
mtrtRuntimeValueScalarI64Create(int64_t data, MTRT_RuntimeValue *value);

/// Cast a MTRT_RuntimeValue to a generic MTRT_RuntimeValue.

MLIR_CAPI_EXPORTED MTRT_RuntimeValue
mtrtScalarValueCastToRuntimeValue(MTRT_ScalarValue v);

MLIR_CAPI_EXPORTED MTRT_Status
mtrtScalarValueGetType(MTRT_ScalarValue scalar, MTRT_ScalarTypeCode *code);

MLIR_CAPI_EXPORTED MTRT_Status mtrtScalarValueGet(MTRT_ScalarValue scalar,
                                                  int64_t *data);

//===----------------------------------------------------------------------===//
// MTRT_RuntimeSessionOptions
//===----------------------------------------------------------------------===//

typedef struct MTRT_RuntimeSessionOptions {
  void *ptr;
} MTRT_RuntimeSessionOptions;

/// Create an MTRT_RuntimeSessionOptions with the specified information.
/// Currently, `numDevices` is only the number of devices on the current host
/// (e.g. num local GPUs). `deviceId` is the CUDA ordinal for the CUDA device
/// thtat the session should be associated with. `ncclUuid` is a
/// caller-determined UUID returne by NCCL initialization methods on the rank-0
/// device and destributed to all ranks prior to creation of sessions.
MLIR_CAPI_EXPORTED MTRT_Status mtrtRuntimeSessionOptionsCreate(
    int32_t numDevices, int32_t deviceId, MTRT_StringView ncclUuid,
    MTRT_RuntimeSessionOptions *options);

/// Destroy `options` and free any resources.
MLIR_CAPI_EXPORTED MTRT_Status
mtrtRuntimeSessionOptionsDestroy(MTRT_RuntimeSessionOptions options);

/// Return if the session options is null.
static inline bool
mtrtRuntimeSessionOptionsIsNull(MTRT_RuntimeSessionOptions options) {
  return !options.ptr;
}

//===----------------------------------------------------------------------===//
// MTRT_RuntimeSession
//===----------------------------------------------------------------------===//

typedef struct MTRT_RuntimeSession {
  void *ptr;
} MTRT_RuntimeSession;

/// Create the RuntimeSession using the specified options and Executable. Note
/// that the session only has a read-only view in to the Executable for code and
/// constant data. Therefore the Executable must outlive the RuntimeSession.
MLIR_CAPI_EXPORTED MTRT_Status mtrtRuntimeSessionCreate(
    MTRT_RuntimeSessionOptions options, MTRT_Executable executable,
    MTRT_RuntimeSession *result);

/// Destory the session. This does not destroy the associated Executable, which
/// may be shared among many sessions.
MLIR_CAPI_EXPORTED MTRT_Status
mtrtRuntimeSessionDestroy(MTRT_RuntimeSession session);

/// Return if the session is null.
static inline bool mtrtRuntimeSessionIsNull(MTRT_RuntimeSession session) {
  return !session.ptr;
}

/// Using `session`, execute the public function with the specified name.
/// The `inArgs`, `outArgs`, and `results` are arrays for input arguments,
/// output arguments, and return values, respectively. Arguments and results
/// can be MemRefs, scalars, or other supported types. Both `outArgs` and
/// `results` can be used simultaneously, allowing for functions that both
/// modify arguments and return values.
/// A stream may optionally be specified, otherwise pass the result of
/// `mtrtStreamGetNull()`.
///
/// The `results` array must point to an array with at least the number of
/// elements returned by mtrtRuntimeSessionGetNumResults for the given function.
MLIR_CAPI_EXPORTED MTRT_Status mtrtRuntimeSessionExecuteFunction(
    MTRT_RuntimeSession session, MTRT_StringView name,
    const MTRT_RuntimeValue *inArgs, size_t numInArgs,
    const MTRT_RuntimeValue *outArgs, size_t numOutArgs,
    MTRT_RuntimeValue *results, MTRT_Stream stream, MTRT_RuntimeClient client);

/// Return number of results given a function name. Function name refers
/// to an exported function in the executable.
MLIR_CAPI_EXPORTED MTRT_Status mtrtRuntimeSessionGetNumResults(
    MTRT_RuntimeSession session, MTRT_StringView name, int64_t *numResults);

//===----------------------------------------------------------------------===//
// DLPack
//===----------------------------------------------------------------------===//

/// Converts a DLDeviceType to MTRT_PointerType. This function will throw a
/// runtime error if the device type is invalid.
MLIR_CAPI_EXPORTED MTRT_Status mtrtGetPointerTypeFromDLDeviceType(
    DLDeviceType device, MTRT_PointerType *result);

/// Converts a DLDataType to MTRT_ScalarTypeCode. This function will throw a
/// runtime error if the data type is invalid.
MLIR_CAPI_EXPORTED MTRT_Status mtrtGetScalarTypeCodeFromDLDataType(
    DLDataType dtype, MTRT_ScalarTypeCode *result);

#ifdef __cplusplus
}
#endif

#endif // MLIR_TENSORRT_C_RUNTIME_RUNTIME
