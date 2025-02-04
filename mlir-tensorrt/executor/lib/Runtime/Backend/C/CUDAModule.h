//===- CUDAModule.h -----------------------------------------------*- C -*-===//
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
/// MTRT CUDA C runtime module declarations.
///
/// Note that this header is exposed for two use cases:
/// - Provide runtime support library implementation when C++ is generated.
/// - Reuse existing C backend implementations for alternative runtime backends
///   (e.g. Lua).
///
//===----------------------------------------------------------------------===//
#ifndef RUNTIME_BACKEND_C_CUDAMODULE
#define RUNTIME_BACKEND_C_CUDAMODULE

#include "cuda.h"
#include "stdint.h"

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
// Non-windows: use visibility attributes. Note that user can set this to empty
// if using the library to link privately for generated C code.
#define MTRT_CAPI_EXPORTED __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

MTRT_CAPI_EXPORTED CUstream mtrt_cuda_stream_create();

MTRT_CAPI_EXPORTED void mtrt_cuda_stream_destroy(CUstream stream);

MTRT_CAPI_EXPORTED void mtrt_cuda_stream_sync(CUstream stream);

MTRT_CAPI_EXPORTED void *mtrt_cuda_alloc_async(CUstream stream, int32_t device,
                                               int64_t size, int32_t alignment,
                                               int8_t isHostPinned,
                                               int8_t isManaged);

MTRT_CAPI_EXPORTED void mtrt_cuda_memcpy_async(CUstream stream, void *src,
                                               void *dest, int64_t size);

MTRT_CAPI_EXPORTED void mtrt_cuda_free(CUstream stream, void *ptr,
                                       int8_t isHostPinned, int8_t isManaged);

MTRT_CAPI_EXPORTED CUmodule mtrt_cuda_module_load_from_ptx(const char *ptxData,
                                                           size_t ptxLen);

MTRT_CAPI_EXPORTED CUmodule mtrt_cuda_module_load_from_ptx_file(
    const char *filename, size_t filenameLength);

MTRT_CAPI_EXPORTED void mtrt_cuda_module_unload(CUmodule module);

MTRT_CAPI_EXPORTED CUfunction mtrt_cuda_module_get_function(CUmodule cumodule,
                                                            const char *name,
                                                            int64_t length);

MTRT_CAPI_EXPORTED
void mtrt_cuda_launch_kernel(CUfunction cudaFuncPtr, int32_t gridX,
                             int32_t gridY, int32_t gridZ, int32_t blockX,
                             int32_t blockY, int32_t blockZ,
                             int32_t dynamicSharedMemory, CUstream stream,
                             void **args);

MTRT_CAPI_EXPORTED int32_t mtrt_cuda_get_current_device();

//===----------------------------------------------------------------------===//
// EmitC-Compatible Wrappers
//
// These wrappers are used by emitted C/C++ code.
// The C/C++ generation currently uses opaque pointers like LLVM, so
// we must provide type-erased wrappers around each of the above functions.
//===----------------------------------------------------------------------===//

inline static void *mtrt_cuda_stream_create_cwrapper() {
  return (CUstream)mtrt_cuda_stream_create();
}
inline static void mtrt_cuda_stream_destroy_cwrapper(void *stream) {
  return mtrt_cuda_stream_destroy((CUstream)stream);
}
inline static void mtrt_cuda_stream_sync_cwrapper(void *stream) {
  return mtrt_cuda_stream_sync((CUstream)stream);
}
inline static void *mtrt_cuda_alloc_async_cwrapper(void *stream, int32_t device,
                                                   int64_t size,
                                                   int32_t alignment,
                                                   int8_t isHostPinned,
                                                   int8_t isManaged) {
  return mtrt_cuda_alloc_async((CUstream)stream, device, size, alignment,
                               isHostPinned, isManaged);
}
inline static void mtrt_cuda_memcpy_async_cwrapper(void *stream, void *src,
                                                   void *dest, int64_t size) {
  return mtrt_cuda_memcpy_async((CUstream)stream, src, dest, size);
}
inline static void mtrt_cuda_free_cwrapper(void *stream, void *ptr,
                                           int8_t isHostPinned,
                                           int8_t isManaged) {
  return mtrt_cuda_free((CUstream)stream, ptr, isHostPinned, isManaged);
}
inline static void *mtrt_cuda_module_load_from_ptx_cwrapper(void *ptxData,
                                                            size_t ptxLen) {
  return mtrt_cuda_module_load_from_ptx((const char *)ptxData, ptxLen);
}
inline static void *
mtrt_cuda_module_load_from_ptx_file_cwrapper(void *filename,
                                             size_t filenameLength) {
  return mtrt_cuda_module_load_from_ptx_file((const char *)filename,
                                             filenameLength);
}
inline static void mtrt_cuda_module_unload_cwrapper(void *module) {
  return mtrt_cuda_module_unload((CUmodule)module);
}
inline static void *mtrt_cuda_module_get_function_cwrapper(void *cumodule,
                                                           void *name,
                                                           int64_t length) {
  return mtrt_cuda_module_get_function((CUmodule)cumodule, (const char *)name,
                                       length);
}
inline static void mtrt_cuda_launch_kernel_cwrapper(
    void *cudaFuncPtr, int32_t gridX, int32_t gridY, int32_t gridZ,
    int32_t blockX, int32_t blockY, int32_t blockZ, int32_t dynamicSharedMemory,
    void *stream, void *args) {
  return mtrt_cuda_launch_kernel((CUfunction)cudaFuncPtr, gridX, gridY, gridZ,
                                 blockX, blockY, blockZ, dynamicSharedMemory,
                                 (CUstream)stream, (void **)args);
}
inline static int32_t mtrt_cuda_get_current_device_cwrapper() {
  return mtrt_cuda_get_current_device();
}

#ifdef __cplusplus
}
#endif

#endif // RUNTIME_BACKEND_C_CUDAMODULE
