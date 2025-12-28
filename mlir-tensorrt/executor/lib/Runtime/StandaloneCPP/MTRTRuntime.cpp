//===- MTRTRuntime.cpp ----------------------------------------------------===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// This file contains an example implementation of C++ functions required
/// to interact with generated C++ host code.
///
//===----------------------------------------------------------------------===//
#include "MTRTRuntime.h"
#include "cuda_runtime_api.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

using namespace mtrt;

#define MTRT_DBGF(fmt, ...)                                                    \
  do {                                                                         \
    if (isDebugEnabled())                                                      \
      std::fprintf(stderr, "%s:%d:%s(): " fmt "\n", "MTRTRuntime.cpp",         \
                   __LINE__, __func__, __VA_ARGS__);                           \
  } while (0)

#define HANDLE_CUDADRV_ERROR(x, ...)                                           \
  do {                                                                         \
    CUresult err = (x);                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *msg = "";                                                    \
      cuGetErrorString(err, &msg);                                             \
      std::fprintf(stderr, "%s:%d:%s(): CUDA Driver Error: %s\n",              \
                   "MTRTRuntime.cpp", __LINE__, __func__, msg);                \
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
                   "MTRTRuntime.cpp", __LINE__, __func__, msg);                \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  } while (false)

static const char *kDebugEnvironmentVariable = "MTRT_DEBUG";

/// Helper method that checks environment value for debugging.
[[maybe_unused]] static bool isDebugEnabled() {
  static bool isInitialized = false;
  static bool isEnabled = false;
  if (!isInitialized)
    isEnabled = getenv(kDebugEnvironmentVariable) != nullptr;
  return isEnabled;
}

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

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static size_t getFileSize(const std::string &filename) {
  // Open the binary file
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 0;
  }
  return file.tellg();
}

static int readInputFile(const std::string &filename,
                         std::vector<char> &buffer) {
  // Open the binary file
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }

  // Get the size of the file
  std::streamsize size = file.tellg();

  // Move back to the beginning of the file
  file.seekg(0, std::ios::beg);

  // Create a vector to hold the file contents
  buffer.resize(size);

  // Read the entire file into the vector
  if (file.read(buffer.data(), size))
    return 0;

  std::cerr << "Error reading file!" << std::endl;
  return 1;
}

static int readInputFile(const std::string &filename, char *buffer) {
  // Open the binary file
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }

  // Get the size of the file
  std::streamsize size = file.tellg();

  // Move back to the beginning of the file
  file.seekg(0, std::ios::beg);

  // Read the entire file into the vector
  if (file.read(buffer, size))
    return 0;

  std::cerr << "Error reading file!" << std::endl;
  return 1;
}

//===----------------------------------------------------------------------===//
// CUDA Wrappers
//===----------------------------------------------------------------------===//

CUmodule mtrt::cuda_module_create_from_ptx_file(const char *filename) {
  initCudaDriverOnce();
  CUmodule module;
  std::vector<char> buffer;
  if (readInputFile(filename, buffer))
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

//===----------------------------------------------------------------------===//
// Host Memory Management
//===----------------------------------------------------------------------===//

void *mtrt::host_alloc(int64_t size, int32_t alignment) {
  if (size % alignment != 0)
    size = ((size + alignment - 1) / alignment) * alignment;
  return std::aligned_alloc(size, alignment);
}

void mtrt::host_free(void *ptr) { ::free(ptr); }

void *mtrt::constant_load_from_file(const char *filename, int32_t align,
                                    int32_t space) {

  size_t fileSize = getFileSize(filename);
  if (fileSize == 0)
    return nullptr;
  void *buffer;
  HANDLE_CUDART_ERROR(cudaMallocManaged(&buffer, fileSize), nullptr);
  if (!readInputFile(filename, reinterpret_cast<char *>(buffer)))
    return nullptr;
  return buffer;
}

void mtrt::constant_destroy(void *data, int32_t space) {
  HANDLE_CUDART_ERROR(cudaFree(data), );
}