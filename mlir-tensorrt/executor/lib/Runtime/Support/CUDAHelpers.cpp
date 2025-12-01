//===- CUDAHelpers.cpp ----------------------------------------------------===//
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
#include "mlir-executor/Runtime/Support/CUDAHelpers.h"

#ifdef MLIR_TRT_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace mtrt;

namespace {
/// CUDADeviceGuard is an abstract RAII handle that scopes a temporary
/// activation of a device, restoring the old active device on destruction.
class CUDADeviceGuard {
public:
  CUDADeviceGuard(CUDADeviceGuard &&) = delete;
  CUDADeviceGuard(const CUDADeviceGuard &) = delete;
  CUDADeviceGuard &operator=(CUDADeviceGuard &&) = delete;
  CUDADeviceGuard &operator=(const CUDADeviceGuard &) = delete;

  static StatusOr<std::unique_ptr<CUDADeviceGuard>>
  create(int32_t deviceNumber) {
    MTRT_ASSIGN_OR_RETURN(int32_t originalDeviceNumber, getCurrentCUDADevice());

    RETURN_STATUS_IF_ERROR(setCurrentCUDADevice(deviceNumber));
    MTRT_DBG("CUDAGPUDeviceGuard: original={0} new={1}", originalDeviceNumber,
             deviceNumber);
    return std::unique_ptr<CUDADeviceGuard>(
        new CUDADeviceGuard(originalDeviceNumber));
  }

  ~CUDADeviceGuard() {
    if (originalDeviceNumber < 0)
      return;
    MTRT_DBG("CUDAGPUDeviceGuard: restoring original device {0}",
             originalDeviceNumber);
    mtrt::cantFail(setCurrentCUDADevice(originalDeviceNumber));
  }

private:
  CUDADeviceGuard(int32_t originalDeviceNumber)
      : originalDeviceNumber(originalDeviceNumber) {}

  int32_t originalDeviceNumber;
};
} // namespace

namespace mtrt {

StatusOr<int32_t> getCurrentCUDADevice() {
#ifdef MLIR_TRT_ENABLE_CUDA
  int deviceNumber = -1;
  RETURN_ERROR_IF_CUDART_ERROR(cudaGetDevice(&deviceNumber));
  CUDA_DBGV("getCurrentCUDADevice: {0}", deviceNumber);
  return static_cast<int32_t>(deviceNumber);
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status setCurrentCUDADevice(int32_t deviceNumber) {
#ifdef MLIR_TRT_ENABLE_CUDA
  int32_t currDeviceNumber = -1;
  RETURN_ERROR_IF_CUDART_ERROR(cudaGetDevice(&currDeviceNumber));
  CUDA_DBGV("setCurrentCUDADevice: {0} -> {1}", currDeviceNumber, deviceNumber);
  if (currDeviceNumber == deviceNumber)
    return getOkStatus();
  RETURN_ERROR_IF_CUDART_ERROR(cudaSetDevice(deviceNumber));
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<int32_t> getCUDADeviceCount() {
#ifdef MLIR_TRT_ENABLE_CUDA
  int count = 0;
  RETURN_ERROR_IF_CUDART_ERROR(cudaGetDeviceCount(&count));
  CUDA_DBGV("getCUDADeviceCount: {0}", count);
  return static_cast<int32_t>(count);
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status warmupCUDA() {
#ifdef MLIR_TRT_ENABLE_CUDA
  // Trigger CUDA runtime initialization.
  cudaError_t e = cudaFree(nullptr);
  RETURN_ERROR_IF_CUDART_ERROR(e);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<uintptr_t> createCUDAStream() {
#ifdef MLIR_TRT_ENABLE_CUDA
  cudaStream_t stream;
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#ifndef NDEBUG
  StatusOr<int32_t> device = getCurrentCUDADevice();
  if (!device.isOk())
    return device.getStatus();
  CUDA_DBGV("createCUDAStream: {0:x} for device {1}",
            reinterpret_cast<uintptr_t>(stream), *device);
#endif
  return reinterpret_cast<uintptr_t>(stream);
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status destroyCUDAStream(uintptr_t stream) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
  CUDA_DBGV("destroyCUDAStream: {0:x}", stream);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status synchronizeCUDAStream(uintptr_t stream) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
  CUDA_DBGV("synchronizeCUDAStream: {0:x}", stream);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status copyCUDAAsync(void *dst, const void *src, uint64_t numBytes,
                     int32_t copyKind, uintptr_t stream) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaMemcpyAsync(dst, src, static_cast<size_t>(numBytes),
                      static_cast<cudaMemcpyKind>(copyKind),
                      reinterpret_cast<cudaStream_t>(stream)));
  CUDA_DBGV(
      "copyCUDAAsync: dst={0:x} src={1:x} bytes={2} kind={3} stream={4:x}",
      reinterpret_cast<uintptr_t>(dst), reinterpret_cast<uintptr_t>(src),
      numBytes, copyKind, stream);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status copyCUDADeviceToDeviceAsync(void *dstDevice, const void *srcDevice,
                                   uint64_t numBytes, uintptr_t stream) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(cudaMemcpyAsync(
      dstDevice, srcDevice, static_cast<size_t>(numBytes),
      cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));
  CUDA_DBGV(
      "copyCUDADeviceToDeviceAsync: dst={0:x} src={1:x} bytes={2} stream={3:x}",
      reinterpret_cast<uintptr_t>(dstDevice),
      reinterpret_cast<uintptr_t>(srcDevice), numBytes, stream);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status copyCUDAHostToDeviceAsync(void *dstDevice, const void *srcHost,
                                 uint64_t numBytes, uintptr_t stream) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(cudaMemcpyAsync(
      dstDevice, srcHost, static_cast<size_t>(numBytes), cudaMemcpyHostToDevice,
      reinterpret_cast<cudaStream_t>(stream)));
  CUDA_DBGV(
      "copyCUDAHostToDeviceAsync: dst={0:x} src={1:x} bytes={2} stream={3:x}",
      reinterpret_cast<uintptr_t>(dstDevice),
      reinterpret_cast<uintptr_t>(srcHost), numBytes, stream);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status copyCUDADeviceToHostAsync(void *dstHost, const void *srcDevice,
                                 uint64_t numBytes, uintptr_t stream) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(cudaMemcpyAsync(
      dstHost, srcDevice, static_cast<size_t>(numBytes), cudaMemcpyDeviceToHost,
      reinterpret_cast<cudaStream_t>(stream)));
  CUDA_DBGV(
      "copyCUDADeviceToHostAsync: dst={0:x} src={1:x} bytes={2} stream={3:x}",
      reinterpret_cast<uintptr_t>(dstHost),
      reinterpret_cast<uintptr_t>(srcDevice), numBytes, stream);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status copyCUDAPeerAsync(void *dstDevice, int32_t dstDeviceId,
                         const void *srcDevice, int32_t srcDeviceId,
                         uint64_t numBytes, uintptr_t stream) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(cudaMemcpyPeerAsync(
      dstDevice, dstDeviceId, srcDevice, srcDeviceId,
      static_cast<size_t>(numBytes), reinterpret_cast<cudaStream_t>(stream)));
  CUDA_DBGV("copyCUDAPeerAsync: dst={0:x} dstDev={1} src={2:x} srcDev={3} "
            "bytes={4} stream={5:x}",
            reinterpret_cast<uintptr_t>(dstDevice), dstDeviceId,
            reinterpret_cast<uintptr_t>(srcDevice), srcDeviceId, numBytes,
            stream);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<uintptr_t> createCUDAEvent() {
#ifdef MLIR_TRT_ENABLE_CUDA
#ifndef NDEBUG
  MTRT_ASSIGN_OR_RETURN(int32_t device, getCurrentCUDADevice());
  CUDA_DBGV("createCUDAEvent: current device = {0}", device);
#endif
  cudaEvent_t event;
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaEventCreateWithFlags(&event, cudaEventDefault));
#ifndef NDEBUG
  CUDA_DBGV("createCUDAEvent: {0:x} ", reinterpret_cast<uintptr_t>(event));
#endif
  return reinterpret_cast<uintptr_t>(event);
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<uintptr_t> createCUDAEventForDevice(int32_t deviceNumber) {
  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<CUDADeviceGuard> guard,
                        CUDADeviceGuard::create(deviceNumber));
  return createCUDAEvent();
}

Status destroyCUDAEvent(uintptr_t event) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event)));
  CUDA_DBGV("destroyCUDAEvent: {0:x}", event);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status recordCUDAEvent(uintptr_t event, uintptr_t stream) {
#ifdef MLIR_TRT_ENABLE_CUDA
  CUDA_DBGV("recordCUDAEvent: event={0:x} stream={1:x}", event, stream);
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                      reinterpret_cast<cudaStream_t>(stream)));
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status waitCUDAEventOnStream(uintptr_t stream, uintptr_t event) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream),
                          reinterpret_cast<cudaEvent_t>(event), 0));
  CUDA_DBGV("waitCUDAEventOnStream: stream={0:x} event={1:x}", stream, event);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status synchronizeCUDAEvent(uintptr_t event) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)));
  CUDA_DBGV("synchronizeCUDAEvent: {0:x}", event);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<bool> queryCUDAEvent(uintptr_t event) {
#ifdef MLIR_TRT_ENABLE_CUDA
  cudaError_t eventQueryStatus =
      cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
  if (eventQueryStatus == cudaSuccess) {
    CUDA_DBGV("queryCUDAEvent: {0:x} -> ready", event);
    return true;
  }
  if (eventQueryStatus == cudaErrorNotReady) {
    CUDA_DBGV("queryCUDAEvent: {0:x} -> not ready", event);
    return false;
  }
  RETURN_ERROR_IF_CUDART_ERROR(eventQueryStatus);
  return false; // Unreachable, but satisfies control flow.
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<float> getCUDAEventElapsedTimeMs(uintptr_t startEvent,
                                          uintptr_t endEvent) {
#ifdef MLIR_TRT_ENABLE_CUDA
  float ms = 0.0f;
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaEventElapsedTime(&ms, reinterpret_cast<cudaEvent_t>(startEvent),
                           reinterpret_cast<cudaEvent_t>(endEvent)));
  CUDA_DBGV("getCUDAEventElapsedTimeMs: start={0:x} end={1:x} -> {2}",
            startEvent, endEvent, ms);
  return ms;
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<std::string> getCUDADeviceName(int32_t deviceNumber) {
#ifdef MLIR_TRT_ENABLE_CUDA
  cudaDeviceProp prop{};
  RETURN_ERROR_IF_CUDART_ERROR(cudaGetDeviceProperties(&prop, deviceNumber));
  return std::string(prop.name);
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<uintptr_t> mallocCUDA(uint64_t numBytes) {
#ifdef MLIR_TRT_ENABLE_CUDA
  void *alloc{nullptr};
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaMalloc(&alloc, static_cast<size_t>(numBytes)));
  CUDA_DBGV("mallocCUDA: size={0} -> {1:x}", numBytes,
            reinterpret_cast<uintptr_t>(alloc));
  return reinterpret_cast<uintptr_t>(alloc);
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<uintptr_t> mallocCUDAAsync(uint64_t numBytes, uintptr_t stream) {
#ifdef MLIR_TRT_ENABLE_CUDA
  void *alloc{nullptr};
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaMallocAsync(&alloc, static_cast<size_t>(numBytes),
                      reinterpret_cast<cudaStream_t>(stream)));
  CUDA_DBGV("mallocCUDAAsync: size={0} stream={1:x} -> {2:x}", numBytes, stream,
            reinterpret_cast<uintptr_t>(alloc));
  return reinterpret_cast<uintptr_t>(alloc);
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<uintptr_t> mallocCUDAManaged(uint64_t numBytes) {
#ifdef MLIR_TRT_ENABLE_CUDA
  void *alloc{nullptr};
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaMallocManaged(&alloc, static_cast<size_t>(numBytes)));
  CUDA_DBGV("mallocCUDAManaged: size={0} -> {1:x}", numBytes,
            reinterpret_cast<uintptr_t>(alloc));
  return reinterpret_cast<uintptr_t>(alloc);
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<uintptr_t> mallocCUDAPinnedHost(uint64_t numBytes) {
#ifdef MLIR_TRT_ENABLE_CUDA
  void *alloc{nullptr};
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaMallocHost(&alloc, static_cast<size_t>(numBytes)));
  CUDA_DBGV("mallocCUDAPinnedHost: size={0} -> {1:x}", numBytes,
            reinterpret_cast<uintptr_t>(alloc));
  return reinterpret_cast<uintptr_t>(alloc);
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status freeCUDA(uintptr_t ptr) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(cudaFree(reinterpret_cast<void *>(ptr)));
  CUDA_DBGV("freeCUDA: ptr={0:x}", ptr);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status freeCUDAAsync(uintptr_t ptr, uintptr_t stream) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(cudaFreeAsync(
      reinterpret_cast<void *>(ptr), reinterpret_cast<cudaStream_t>(stream)));
  CUDA_DBGV("freeCUDAAsync: ptr={0:x} stream={1:x}", ptr, stream);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status freeCUDAPinnedHost(uintptr_t ptr) {
#ifdef MLIR_TRT_ENABLE_CUDA
  RETURN_ERROR_IF_CUDART_ERROR(cudaFreeHost(reinterpret_cast<void *>(ptr)));
  CUDA_DBGV("freeCUDAPinnedHost: ptr={0:x}", ptr);
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

Status launchCUDAHostFunc(uintptr_t stream, void (*callback)(void *),
                          void *userData) {
#ifdef MLIR_TRT_ENABLE_CUDA
  CUDA_DBGV("launchCUDAHostFunc: stream={0:x} callback={1:x} userData={2:x}",
            stream, reinterpret_cast<uintptr_t>(callback),
            reinterpret_cast<uintptr_t>(userData));
  RETURN_ERROR_IF_CUDART_ERROR(cudaLaunchHostFunc(
      reinterpret_cast<cudaStream_t>(stream), callback, userData));
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

} // namespace mtrt
