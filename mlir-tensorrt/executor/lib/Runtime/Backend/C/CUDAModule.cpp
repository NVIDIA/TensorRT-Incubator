//===- MTRTRuntime.cpp ----------------------------------------------------===//
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of C runtime support library for `mlir-runner` style
/// tests.
///
//===----------------------------------------------------------------------===//
#include "CUDAModule.h"
#include "cuda.h"
#include "mlir-executor/Runtime/Backend/Common/NvPtxCompilerUtils.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <fstream>
#include <string>
#include <vector>

#define HANDLE_CUDADRV_ERROR(x, ...)                                           \
  do {                                                                         \
    CUresult err = (x);                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *msg = "";                                                    \
      cuGetErrorString(err, &msg);                                             \
      llvm::report_fatal_error(                                                \
          llvm::formatv("{0}#{1}: {2}\n", __FILE__, __LINE__, msg)             \
              .str()                                                           \
              .c_str());                                                       \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  } while (false)

#define HANDLE_CUDART_ERROR(x, ...)                                            \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      const char *msg = "";                                                    \
      msg = cudaGetErrorString(err);                                           \
      llvm::report_fatal_error(                                                \
          llvm::formatv("{0}#{1}: {2}\n", __FILE__, __LINE__, msg)             \
              .str()                                                           \
              .c_str());                                                       \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  } while (false)

using namespace mlirtrt;
using namespace mlirtrt::runtime;

CUstream mtrt_cuda_stream_create() {
  CUstream stream;
  HANDLE_CUDART_ERROR(cudaStreamCreate(&stream), nullptr);
  return stream;
}

void mtrt_cuda_stream_destroy(CUstream stream) {
  HANDLE_CUDART_ERROR(cudaStreamDestroy(stream), );
}

void mtrt_cuda_stream_sync(CUstream stream) {
  HANDLE_CUDART_ERROR(cudaStreamSynchronize(stream), );
}

void *mtrt_cuda_alloc_async(CUstream stream, int32_t device, int64_t size,
                            int32_t alignment, int8_t isHostPinned,
                            int8_t isManaged) {
  void *result{nullptr};
  if (isHostPinned || isManaged) {
    HANDLE_CUDART_ERROR(cudaMallocManaged(&result, size), nullptr);
    return result;
  }
  HANDLE_CUDART_ERROR(cudaMallocAsync(&result, size, stream), nullptr);
  return result;
}

void mtrt_cuda_memcpy_async(CUstream stream, void *src, void *dest,
                            int64_t size) {

  HANDLE_CUDART_ERROR(
      cudaMemcpyAsync(dest, src, size, cudaMemcpyDefault, stream), );
}

void mtrt_cuda_free(CUstream stream, void *ptr, int8_t isHostPinned,
                    int8_t isManaged) {
  if (isHostPinned || isManaged) {
    HANDLE_CUDART_ERROR(cudaFree(ptr), );
    return;
  }
  HANDLE_CUDART_ERROR(cudaFreeAsync(ptr, stream), );
}

CUfunction mtrt_cumodule_load_func(CUmodule module, const char *funcName) {
  CUfunction func;
  cuModuleGetFunction(&func, module, funcName);
  return func;
}

static StatusOr<std::string> getDeviceArch(int32_t deviceNumber) {
  CUdevice deviceID;
  RETURN_ERROR_WITH_MSG_IF_CUDADRV_ERROR(cuDeviceGet(&deviceID, deviceNumber),
                                         "could not get CUDA driver device");
  int smMinor = 0;
  int smMajor = 0;
  RETURN_ERROR_WITH_MSG_IF_CUDADRV_ERROR(
      cuDeviceGetAttribute(
          &smMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, deviceID),
      "could not query compute capability of device");
  RETURN_ERROR_WITH_MSG_IF_CUDADRV_ERROR(
      cuDeviceGetAttribute(
          &smMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, deviceID),
      "could not query compute capability of device");
  std::string arch = "sm_" + std::to_string(smMajor) + std::to_string(smMinor);
  return arch;
}

CUmodule mtrt_cuda_module_load_from_ptx(const char *ptxData, size_t ptxLen) {
  int32_t device{0};
  HANDLE_CUDART_ERROR(cudaGetDevice(&device), nullptr);
  StatusOr<std::string> arch = getDeviceArch(device);
  if (!arch.isOk()) {
    llvm::report_fatal_error(arch.getString().c_str());
    return nullptr;
  }

  StatusOr<std::unique_ptr<mlirtrt::runtime::CuBinWrapper>> cubinWrapper =
      mlirtrt::runtime::compilePtxToCuBin(
          reinterpret_cast<const char *>(ptxData), ptxLen, *arch);
  if (!cubinWrapper.isOk()) {
    llvm::errs() << cubinWrapper.getString() << "\n";
    return nullptr;
  }

  CUmodule module{nullptr};
  CUresult result = cuModuleLoadDataEx(
      &module, reinterpret_cast<const void *>((*cubinWrapper)->data.data()), 0,
      0, 0);
  HANDLE_CUDADRV_ERROR(result, nullptr);
  return module;
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

CUmodule mtrt_cuda_module_load_from_ptx_file(const char *filename,
                                             size_t filenameLength) {
  int32_t device{0};
  HANDLE_CUDART_ERROR(cudaGetDevice(&device), nullptr);
  StatusOr<std::string> arch = getDeviceArch(device);
  if (!arch.isOk()) {
    llvm::report_fatal_error(arch.getString().c_str());
    return nullptr;
  }

  std::fstream fs(filename, std::fstream::in | std::fstream::binary);

  std::vector<char> buffer;
  if (readInputFile(std::string(filename, filenameLength), buffer))
    return nullptr;

  StatusOr<std::unique_ptr<mlirtrt::runtime::CuBinWrapper>> cubinWrapper =
      mlirtrt::runtime::compilePtxToCuBin(buffer.data(), buffer.size(), *arch);
  if (!cubinWrapper.isOk()) {
    llvm::errs() << cubinWrapper.getString() << "\n";
    return nullptr;
  }

  CUmodule module{nullptr};
  CUresult result = cuModuleLoadDataEx(
      &module, reinterpret_cast<const void *>((*cubinWrapper)->data.data()), 0,
      0, 0);
  HANDLE_CUDADRV_ERROR(result, nullptr);
  return module;
}

void mtrt_cuda_module_unload(CUmodule module) {
  HANDLE_CUDADRV_ERROR(cuModuleUnload(module), );
}

CUfunction mtrt_cuda_module_get_function(CUmodule cumodule, const char *name,
                                         int64_t length) {
  CUfunction result;
  std::string name_(name, length);
  HANDLE_CUDADRV_ERROR(cuModuleGetFunction(&result, cumodule, name_.c_str()),
                       nullptr);
  return result;
}

void mtrt_cuda_launch_kernel(CUfunction cudaFuncPtr, int32_t gridX,
                             int32_t gridY, int32_t gridZ, int32_t blockX,
                             int32_t blockY, int32_t blockZ,
                             int32_t dynamicSharedMemory, CUstream stream,
                             void **args) {
  llvm::dbgs() << "launching kernel:\n";
  CUresult result =
      cuLaunchKernel(reinterpret_cast<CUfunction>(cudaFuncPtr), gridX, gridY,
                     gridZ, blockX, blockY, blockZ, dynamicSharedMemory,
                     reinterpret_cast<CUstream>(stream), args,
                     /*extra=*/0);
  HANDLE_CUDADRV_ERROR(result, );
}

int32_t mtrt_cuda_get_current_device() {
  int32_t device{0};
  HANDLE_CUDART_ERROR(cudaGetDevice(&device), 0);
  return device;
}
