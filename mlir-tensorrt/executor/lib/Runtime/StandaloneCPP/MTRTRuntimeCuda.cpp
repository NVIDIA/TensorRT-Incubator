//===- MTRTRuntimeCuda.cpp ------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of CUDA runtime support library.
///
//===----------------------------------------------------------------------===//
#include "MTRTRuntimeCuda.h"
#include "MTRTRuntimeStatus.h"
#include "cuda_runtime_api.h"
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

using namespace mtrt;

namespace {
inline mtrt::Status statusFromCudaDriverError(CUresult err, const char *expr) {
  const char *name = nullptr;
  const char *msg = nullptr;
  (void)cuGetErrorName(err, &name);
  (void)cuGetErrorString(err, &msg);
  MTRT_RETURN_ERROR(mtrt::ErrorCode::CUDADriverError, "%s failed: %s (%s)",
                    expr ? expr : "<unknown>", name ? name : "",
                    msg ? msg : "");
}

inline mtrt::Status statusFromCudaRuntimeError(cudaError_t err,
                                               const char *expr) {
  const char *name = cudaGetErrorName(err);
  const char *msg = cudaGetErrorString(err);
  MTRT_RETURN_ERROR(mtrt::ErrorCode::CUDARuntimeError, "%s failed: %s (%s)",
                    expr ? expr : "<unknown>", name ? name : "",
                    msg ? msg : "");
}

#define MTRT_RETURN_IF_CUDADRV_ERROR(expr)                                     \
  do {                                                                         \
    CUresult _err = (expr);                                                    \
    if (_err != CUDA_SUCCESS)                                                  \
      return statusFromCudaDriverError(_err, #expr);                           \
  } while (false)

#define MTRT_RETURN_IF_CUDART_ERROR(expr)                                      \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess)                                                   \
      return statusFromCudaRuntimeError(_err, #expr);                          \
  } while (false)

constexpr int32_t kPlanMemorySpaceUnknown = 0;
constexpr int32_t kPlanMemorySpaceHost = 1;
constexpr int32_t kPlanMemorySpaceHostPinned = 2;
constexpr int32_t kPlanMemorySpaceDevice = 3;
constexpr int32_t kPlanMemorySpaceUnified = 4;
} // namespace

/// Ensure the CUDA driver API is initialized.
/// Some environments won't implicitly initialize the driver just because the
/// runtime API is used elsewhere.
static void initCudaDriverOnce() {
  static std::once_flag onceFlag;
  call_once(onceFlag, []() {
    // Ensure the CUDA runtime initializes a primary context. Without a current
    // context, driver APIs like `cuModuleLoadDataEx` may fail with
    // `CUDA_ERROR_INVALID_CONTEXT`.
    (void)cudaFree(0);
  });
}

namespace mtrt::detail {
Status readInputFile(const std::string &filename, std::vector<char> &buffer);
Status readInputFile(const std::string &filename, char *buffer,
                     size_t expectedSize);
Status getFileSize(const std::string &filename, size_t *outSize);
} // namespace mtrt::detail

//===----------------------------------------------------------------------===//
// CUDA Wrappers
//===----------------------------------------------------------------------===//

Status mtrt::cuda_module_create_from_ptx_file(const char *filename,
                                              CUmodule *outModule) {
  if (!outModule)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "outModule must not be null");
  *outModule = nullptr;
  if (!filename || filename[0] == '\0')
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "filename must not be empty");
  initCudaDriverOnce();
  CUmodule module;
  std::vector<char> buffer;
  Status st = detail::readInputFile(filename, buffer);
  if (st != mtrt::ok())
    return st;
  buffer.push_back('\0');
  MTRT_RETURN_IF_CUDADRV_ERROR(
      cuModuleLoadDataEx(&module, buffer.data(), 0, nullptr, nullptr));
  *outModule = module;
  return mtrt::ok();
}

Status mtrt::cuda_module_destroy(CUmodule module) {
  initCudaDriverOnce();
  MTRT_RETURN_IF_CUDADRV_ERROR(cuModuleUnload(module));
  return mtrt::ok();
}

Status mtrt::cuda_module_get_func(CUmodule module, const char *name,
                                  CUfunction *outFunc) {
  if (!outFunc)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "outFunc must not be null");
  *outFunc = nullptr;
  if (!name || name[0] == '\0')
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "name must not be empty");
  initCudaDriverOnce();
  CUfunction func;
  MTRT_RETURN_IF_CUDADRV_ERROR(cuModuleGetFunction(&func, module, name));
  *outFunc = func;
  return mtrt::ok();
}

Status mtrt::cuda_launch_kernel(CUfunction func, int32_t gridX, int32_t gridY,
                                int32_t gridZ, int32_t blockX, int32_t blockY,
                                int32_t blockZ,
                                int32_t dynamicSharedMemoryBytes,
                                CUstream stream, void **arguments) {
  initCudaDriverOnce();
  MTRT_RETURN_IF_CUDADRV_ERROR(
      cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY, blockZ,
                     dynamicSharedMemoryBytes, stream, arguments, nullptr));
  return mtrt::ok();
}

Status mtrt::cuda_stream_sync(CUstream stream) {
  MTRT_RETURN_IF_CUDART_ERROR(cudaStreamSynchronize(stream));
  return mtrt::ok();
}

/// Return the current CUDA device.
Status mtrt::cuda_get_current_device(int32_t *outDevice) {
  if (!outDevice)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "outDevice must not be null");
  int32_t device{0};
  MTRT_RETURN_IF_CUDART_ERROR(cudaGetDevice(&device));
  *outDevice = device;
  return mtrt::ok();
}

/// Perform a CUDA allocation.
Status mtrt::cuda_alloc(CUstream stream, int64_t sizeBytes, bool isHostPinned,
                        bool isManaged, void **outPtr) {
  if (!outPtr)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "outPtr must not be null");
  *outPtr = nullptr;
  if (sizeBytes < 0)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "sizeBytes must be >= 0");
  if (sizeBytes == 0)
    return mtrt::ok();

  void *result = nullptr;
  if (isHostPinned) {
    (void)stream;
    MTRT_RETURN_IF_CUDART_ERROR(cudaHostAlloc(
        &result, static_cast<size_t>(sizeBytes), cudaHostAllocDefault));
    *outPtr = result;
    return mtrt::ok();
  }
  if (isManaged) {
    MTRT_RETURN_IF_CUDART_ERROR(
        cudaMallocManaged(&result, static_cast<size_t>(sizeBytes)));
    *outPtr = result;
    return mtrt::ok();
  }
  MTRT_RETURN_IF_CUDART_ERROR(
      cudaMallocAsync(&result, static_cast<size_t>(sizeBytes), stream));
  *outPtr = result;
  return mtrt::ok();
}

Status mtrt::cuda_free(CUstream stream, void *ptr, int8_t isHostPinned,
                       int8_t isManaged) {
  if (!ptr)
    return mtrt::ok();
  if (isHostPinned) {
    (void)stream;
    MTRT_RETURN_IF_CUDART_ERROR(cudaFreeHost(ptr));
    return mtrt::ok();
  }
  if (isManaged) {
    (void)stream;
    MTRT_RETURN_IF_CUDART_ERROR(cudaFree(ptr));
    return mtrt::ok();
  }
  MTRT_RETURN_IF_CUDART_ERROR(cudaFreeAsync(ptr, stream));
  return mtrt::ok();
}

Status mtrt::cuda_copy(CUstream stream, void *src, void *dest,
                       int64_t sizeBytes) {
  if (sizeBytes < 0)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "sizeBytes must be >= 0");
  if (sizeBytes == 0)
    return mtrt::ok();
  MTRT_RETURN_IF_CUDART_ERROR(cudaMemcpyAsync(
      dest, src, static_cast<size_t>(sizeBytes), cudaMemcpyDefault, stream));
  return mtrt::ok();
}

//===----------------------------------------------------------------------===//
// Internal hooks used by MTRTRuntimeCore.cpp for non-host constant loading.
//===----------------------------------------------------------------------===//
namespace mtrt::detail {
Status cuda_alloc_and_copy_constant(int32_t space, const void *src,
                                    size_t bytes, void **outPtr) {
  if (!outPtr)
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "outPtr must not be null");
  *outPtr = nullptr;
  if (bytes == 0)
    return mtrt::ok();

  void *dst = nullptr;
  switch (space) {
  case kPlanMemorySpaceHostPinned: {
    MTRT_RETURN_IF_CUDART_ERROR(
        cudaHostAlloc(&dst, bytes, cudaHostAllocDefault));
    std::memcpy(dst, src, bytes);
    *outPtr = dst;
    return mtrt::ok();
  }
  case kPlanMemorySpaceUnified: {
    MTRT_RETURN_IF_CUDART_ERROR(cudaMallocManaged(&dst, bytes));
    std::memcpy(dst, src, bytes);
    *outPtr = dst;
    return mtrt::ok();
  }
  case kPlanMemorySpaceDevice: {
    MTRT_RETURN_IF_CUDART_ERROR(cudaMalloc(&dst, bytes));
    MTRT_RETURN_IF_CUDART_ERROR(
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    *outPtr = dst;
    return mtrt::ok();
  }
  default:
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "unsupported memory space for constant load: %d", space);
  }
}

Status cuda_free_constant(int32_t space, void *ptr) {
  if (!ptr)
    return mtrt::ok();
  switch (space) {
  case kPlanMemorySpaceHostPinned:
    MTRT_RETURN_IF_CUDART_ERROR(cudaFreeHost(ptr));
    return mtrt::ok();
  case kPlanMemorySpaceUnified:
  case kPlanMemorySpaceDevice:
    MTRT_RETURN_IF_CUDART_ERROR(cudaFree(ptr));
    return mtrt::ok();
  default:
    MTRT_RETURN_ERROR(mtrt::ErrorCode::InvalidArgument,
                      "unsupported memory space for constant free: %d", space);
  }
}
} // namespace mtrt::detail
