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
#include "cuda_runtime_api.h"
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#define HANDLE_CUDADRV_ERROR(x, ...)                                           \
  do {                                                                         \
    CUresult err = (x);                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *msg = "";                                                    \
      cuGetErrorString(err, &msg);                                             \
      std::fprintf(stderr, "%s:%d:%s(): CUDA Driver Error: %s\n",              \
                   "MTRTRuntimeCuda.cpp", __LINE__, __func__, msg);            \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  } while (false)

#define HANDLE_CUDART_ERROR(x, ...)                                            \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      const char *msg = "";                                                    \
      msg = cudaGetErrorString(err);                                           \
      std::fprintf(stderr, "%s:%d:%s(): CUDA Runtime Error: %s\n",             \
                   "MTRTRuntimeCuda.cpp", __LINE__, __func__, msg);            \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  } while (false)

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
int readInputFile(const std::string &filename, std::vector<char> &buffer);
int readInputFile(const std::string &filename, char *buffer,
                  size_t expectedSize);
size_t getFileSize(const std::string &filename);
} // namespace mtrt::detail

//===----------------------------------------------------------------------===//
// CUDA Wrappers
//===----------------------------------------------------------------------===//

CUmodule mtrt::cuda_module_create_from_ptx_file(const char *filename) {
  initCudaDriverOnce();
  CUmodule module;
  std::vector<char> buffer;
  if (detail::readInputFile(filename, buffer))
    return nullptr;
  buffer.push_back('\0');
  HANDLE_CUDADRV_ERROR(
      cuModuleLoadDataEx(&module, buffer.data(), 0, nullptr, nullptr), nullptr);
  return module;
}

void mtrt::cuda_module_destroy(CUmodule module) {
  initCudaDriverOnce();
  HANDLE_CUDADRV_ERROR(cuModuleUnload(module), );
}

CUfunction mtrt::cuda_module_get_func(CUmodule module, const char *name) {
  initCudaDriverOnce();
  CUfunction func;
  HANDLE_CUDADRV_ERROR(cuModuleGetFunction(&func, module, name), nullptr);
  return func;
}

void mtrt::cuda_launch_kernel(CUfunction func, int32_t gridX, int32_t gridY,
                              int32_t gridZ, int32_t blockX, int32_t blockY,
                              int32_t blockZ, int32_t dynamicSharedMemoryBytes,
                              CUstream stream, void **arguments) {
  initCudaDriverOnce();
  HANDLE_CUDADRV_ERROR(cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY,
                                      blockZ, dynamicSharedMemoryBytes, stream,
                                      arguments, nullptr), );
}

void mtrt::cuda_stream_sync(CUstream stream) {
  HANDLE_CUDART_ERROR(cudaStreamSynchronize(stream), );
}

/// Return the current CUDA device.
int32_t mtrt::cuda_get_current_device() {
  int32_t device{0};
  HANDLE_CUDART_ERROR(cudaGetDevice(&device), 0);
  return device;
}

/// Perform a CUDA allocation.
void *mtrt::cuda_alloc(CUstream stream, int64_t size, bool isHostPinned,
                       bool isManaged) {
  void *result{nullptr};
  if (isHostPinned || isManaged) {
    HANDLE_CUDART_ERROR(cudaMallocManaged(&result, size), nullptr);
    return result;
  }
  HANDLE_CUDART_ERROR(cudaMallocAsync(&result, size, stream), nullptr);
  return result;
}

void mtrt::cuda_free(CUstream stream, void *ptr, int8_t isHostPinned,
                     int8_t isManaged) {
  if (isHostPinned || isManaged) {
    HANDLE_CUDART_ERROR(cudaFree(ptr), );
    return;
  }
  HANDLE_CUDART_ERROR(cudaFreeAsync(ptr, stream), );
}

void mtrt::cuda_copy(CUstream stream, void *src, void *dest, int64_t size) {
  HANDLE_CUDART_ERROR(
      cudaMemcpyAsync(dest, src, size, cudaMemcpyDefault, stream), );
}

void *mtrt::constant_load_from_file(const char *filename, int32_t align,
                                    int32_t space) {
  size_t fileSize = detail::getFileSize(filename);
  if (fileSize == 0)
    return nullptr;
  void *buffer;
  HANDLE_CUDART_ERROR(cudaMallocManaged(&buffer, fileSize), nullptr);
  if (!detail::readInputFile(filename, reinterpret_cast<char *>(buffer),
                             fileSize))
    return nullptr;
  return buffer;
}

void mtrt::constant_destroy(void *data, int32_t space) {
  HANDLE_CUDART_ERROR(cudaFree(data), );
}
