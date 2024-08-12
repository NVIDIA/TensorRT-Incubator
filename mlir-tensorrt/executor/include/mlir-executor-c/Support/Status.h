//===- Status.h -------------------------------------------------*- C++ -*-===//
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
/// Declarations for the MLIR-TensorRT Runtime Status C API.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_C_RUNTIME_STATUS_H
#define MLIR_TENSORRT_C_RUNTIME_STATUS_H

#include <stddef.h>

///===----------------------------------------------------------------------===//
// Define the visibility macro MTRT_CAPI_EXPORTED.
// This should be used to enable all symbols declared in these public header
// files to be exported when the CAPI libraries are bundled into a shared
// object library.
//
// We use MTRT_CAPI_EXPORTED when `mlir-c/IR/IR.h` is not included, otherwise
// one can also use MLIR_CAPI_EXPORTED.
//===----------------------------------------------------------------------===//

#if (defined(_WIN32) || defined(__CYGWIN__)) &&                                \
    !defined(MTRT_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC)
// Visibility annotations disabled.
#define MTRT_CAPI_EXPORTED
#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows visibility declarations.
#if MTRT_CAPI_BUILDING_LIBRARY
#define MTRT_CAPI_EXPORTED __declspec(dllexport)
#else
#define MTRT_CAPI_EXPORTED __declspec(dllimport)
#endif
#else
// Non-windows: use visibility attributes.
#define MTRT_CAPI_EXPORTED __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GEN_ENUM_DECLS
#include "mlir-executor/Support/StatusEnums.c.h.inc"

//===----------------------------------------------------------------------===//
// MTRT_Status
//===----------------------------------------------------------------------===//

typedef struct MTRT_Status {
  void *ptr;
} MTRT_Status;

/// Creates `MTRT_Status` from `MTRT_StatusCode` and a meesage.
MTRT_CAPI_EXPORTED MTRT_Status mtrtStatusCreate(MTRT_StatusCode code,
                                                const char *msg);

/// Returns true if MTRT_Status is contains a null pointer.
static inline bool mtrtStatusIsNull(MTRT_Status status) { return !status.ptr; }

/// Returns true if the contained status object is an "OK" status (no error),
/// equivalent to the result of `mtrtStatusGetOk()`.
static inline bool mtrtStatusIsOk(MTRT_Status status) {
  return mtrtStatusIsNull(status);
}

/// Return the OK status. Note that the OK status is equivalent to a NULL status
/// and thus cannot be passed to `mtrtStatusGetMessage` and does not need to be
/// destroyed.
static inline MTRT_Status mtrtStatusGetOk() { return MTRT_Status{nullptr}; }

/// Returns the string literal containing the message of the given error into
/// `dest`.
MTRT_CAPI_EXPORTED void mtrtStatusGetMessage(MTRT_Status error,
                                             const char **dest);

/// Destroys `MTRT_Status`.
MTRT_CAPI_EXPORTED void mtrtStatusDestroy(MTRT_Status error);

//===----------------------------------------------------------------------===//
// MTRT_StringView
//===----------------------------------------------------------------------===//

typedef struct MTRT_StringView {
  const char *data;
  size_t length;
} MTRT_StringView;

inline static MTRT_StringView mtrtStringViewCreate(const char *data,
                                                   size_t length) {
  return MTRT_StringView{data, length};
}

//===----------------------------------------------------------------------===//
// Other Utilities
//===----------------------------------------------------------------------===//

/// A callback for returning string references. The idea here is the same as
/// `MlirStringCallback`, but it is reproduced here to break the dependence on
/// MLIR-C headers for our runtime API.
///
/// This function is called back by formatting/printing functions that need to
/// return a reference to the portion of the string with the following
/// arguments:
/// - an MTRT_StringRef representing the current portion of the string
/// - a pointer to user data forwarded from the printing call.
typedef void (*MTRT_PrintCallback)(MTRT_StringView, void *);

/// Wraps the pair of the callback function pointer and the user data pointer.
typedef struct MTRT_PrintCallbackInfo {
  MTRT_PrintCallback callback;
  void *userData;
} MTRT_PrintCallbackInfo;

#ifdef __cplusplus
}
#endif

#endif // MLIR_TENSORRT_C_RUNTIME_STATUS_H
