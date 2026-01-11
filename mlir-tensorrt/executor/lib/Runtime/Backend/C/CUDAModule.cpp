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
#include "FileUtilities.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
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

using namespace mtrt;
using namespace mtrt;

int32_t mtrt_cuda_get_active_device() {
  int32_t device{0};
  HANDLE_CUDART_ERROR(cudaGetDevice(&device), 0);
  return device;
}

int32_t mtrt_cuda_set_active_device(int32_t device) {
  HANDLE_CUDART_ERROR(cudaSetDevice(device), 0);
  return device;
}

int32_t mtrt_cuda_get_device(int32_t device) { return device; }

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

cudaEvent_t mtrt_cuda_event_create() {
  cudaEvent_t event{nullptr};
  HANDLE_CUDART_ERROR(cudaEventCreateWithFlags(&event, cudaEventDefault),
                      nullptr);
  return event;
}

void mtrt_cuda_event_release(cudaEvent_t event) {
  HANDLE_CUDART_ERROR(cudaEventDestroy(event), );
}

void mtrt_cuda_stream_record_event(CUstream stream, cudaEvent_t event) {
  HANDLE_CUDART_ERROR(cudaEventRecord(event, stream), );
}

void mtrt_cuda_stream_wait_event(CUstream stream, cudaEvent_t event) {
  HANDLE_CUDART_ERROR(cudaStreamWaitEvent(stream, event, /*flags=*/0), );
}

void mtrt_cuda_event_sync(cudaEvent_t event) {
  HANDLE_CUDART_ERROR(cudaEventSynchronize(event), );
}

float mtrt_cuda_event_elapsed_msec(cudaEvent_t start, cudaEvent_t end) {
  float ms = 0.0f;
  HANDLE_CUDART_ERROR(cudaEventElapsedTime(&ms, start, end), 0.0f);
  return ms;
}

void *mtrt_cuda_alloc_async(CUstream stream, int64_t size, int32_t alignment,
                            int8_t isHostPinned, int8_t isManaged) {
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
  mtrt::cantFail(arch);

  StatusOr<std::unique_ptr<mtrt::CuBinWrapper>> cubinWrapper =
      mtrt::compilePtxToCuBin(reinterpret_cast<const char *>(ptxData), ptxLen,
                              *arch);
  if (!cubinWrapper.isOk()) {
    mtrt::cantFail(cubinWrapper);
    return nullptr;
  }

  CUmodule module{nullptr};
  CUresult result = cuModuleLoadDataEx(
      &module, reinterpret_cast<const void *>((*cubinWrapper)->data.data()), 0,
      0, 0);
  HANDLE_CUDADRV_ERROR(result, nullptr);
  return module;
}

CUmodule mtrt_cuda_module_load_from_ptx_file(const char *filename,
                                             size_t filenameLength) {
  int32_t device{0};
  HANDLE_CUDART_ERROR(cudaGetDevice(&device), nullptr);
  StatusOr<std::string> arch = getDeviceArch(device);
  mtrt::cantFail(arch);

  std::fstream fs(filename, std::fstream::in | std::fstream::binary);

  std::vector<char> buffer;
  if (mtrtReadInputFile(std::string(filename, filenameLength), buffer))
    return nullptr;

  StatusOr<std::unique_ptr<mtrt::CuBinWrapper>> cubinWrapper =
      mtrt::compilePtxToCuBin(buffer.data(), buffer.size(), *arch);
  mtrt::cantFail(cubinWrapper);

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

int32_t mtrt_cuda_get_program_device(int32_t logicalId) {
  assert(logicalId == 0 && "only single-device mode is supported");

  int32_t deviceId = 0;
  HANDLE_CUDART_ERROR(cudaGetDevice(&deviceId), 0);
  return deviceId;
}
