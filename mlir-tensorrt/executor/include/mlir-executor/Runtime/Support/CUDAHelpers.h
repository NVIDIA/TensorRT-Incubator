//===- CUDAHelpers.h ----------------------------------------------------===//
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
#ifndef MLIR_EXECUTOR_RUNTIME_SUPPORT_CUDAHELPERS
#define MLIR_EXECUTOR_RUNTIME_SUPPORT_CUDAHELPERS

#include "mlir-executor/Runtime/Support/Support.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include <cstdint>

#define CUDA_DBGV(fmt, ...) MTRT_DBG("[cuda] " fmt, __VA_ARGS__)

// CUDA runtime headers are only required in the implementation.

namespace mtrt {

/// Wrapper around `cudaGetDevice`.
StatusOr<int32_t> getCurrentCUDADevice();

/// Wrapper around `cudaSetDevice`.
Status setCurrentCUDADevice(int32_t deviceNumber);

/// Wrapper around `cudaGetDeviceCount`.
StatusOr<int32_t> getCUDADeviceCount();

/// Convenience helper to initialize the CUDA runtime (calls cudaFree(0)).
Status warmupCUDA();

/// Wrapper around `cudaStreamCreateWithFlags` to create a non-blocking stream.
StatusOr<uintptr_t> createCUDAStream();

/// Wrapper around `cudaStreamDestroy`.
Status destroyCUDAStream(uintptr_t stream);

/// Wrapper around `cudaStreamSynchronize`.
Status synchronizeCUDAStream(uintptr_t stream);

/// CUDA memcpy async wrappers
/// Generic wrapper around `cudaMemcpyAsync` accepting a numeric copy kind.
Status copyCUDAAsync(void *dst, const void *src, uint64_t numBytes,
                     int32_t copyKind, uintptr_t stream);

/// Wrapper around `cudaMemcpyAsync` for device-to-device copies.
Status copyCUDADeviceToDeviceAsync(void *dstDevice, const void *srcDevice,
                                   uint64_t numBytes, uintptr_t stream);

/// Wrapper around `cudaMemcpyAsync` for host-to-device copies.
Status copyCUDAHostToDeviceAsync(void *dstDevice, const void *srcHost,
                                 uint64_t numBytes, uintptr_t stream);

/// Wrapper around `cudaMemcpyAsync` for device-to-host copies.
Status copyCUDADeviceToHostAsync(void *dstHost, const void *srcDevice,
                                 uint64_t numBytes, uintptr_t stream);

/// Wrapper around `cudaMemcpyPeerAsync` for device-peer copies.
Status copyCUDAPeerAsync(void *dstDevice, int32_t dstDeviceId,
                         const void *srcDevice, int32_t srcDeviceId,
                         uint64_t numBytes, uintptr_t stream);

/// CUDA Event API wrappers
/// Wrapper around `cudaEventCreateWithFlags`. Creates an event with default
/// flags (timing enabled) so it can be used for elapsed time measurements.
StatusOr<uintptr_t> createCUDAEvent();

/// Wrapper around `cudaEventDestroy`.
Status destroyCUDAEvent(uintptr_t event);

/// Wrapper around `cudaEventRecord`.
Status recordCUDAEvent(uintptr_t event, uintptr_t stream);

/// Wrapper around `cudaStreamWaitEvent` with flags=0.
Status waitCUDAEventOnStream(uintptr_t stream, uintptr_t event);

/// Wrapper around `cudaEventSynchronize`.
Status synchronizeCUDAEvent(uintptr_t event);

/// Wrapper around `cudaEventQuery`. Returns true if ready, false if not ready.
StatusOr<bool> queryCUDAEvent(uintptr_t event);

/// Wrapper around `cudaEventElapsedTime`. Returns milliseconds between events.
StatusOr<float> getCUDAEventElapsedTimeMs(uintptr_t startEvent,
                                          uintptr_t endEvent);

/// CUDA device properties helpers
/// Wrapper to retrieve device name via cudaGetDeviceProperties.
StatusOr<std::string> getCUDADeviceName(int32_t deviceNumber);

/// CUDA Memory API wrappers
/// Wrapper around `cudaMalloc`. Returns device pointer.
StatusOr<uintptr_t> mallocCUDA(uint64_t numBytes);
/// Wrapper around `cudaMallocAsync`. Returns device pointer.
StatusOr<uintptr_t> mallocCUDAAsync(uint64_t numBytes, uintptr_t stream);
/// Wrapper around `cudaMallocManaged`. Returns managed pointer.
StatusOr<uintptr_t> mallocCUDAManaged(uint64_t numBytes);
/// Wrapper around `cudaMallocHost`. Returns host pinned pointer.
StatusOr<uintptr_t> mallocCUDAPinnedHost(uint64_t numBytes);
/// Wrapper around `cudaFree` for device or managed memory.
Status freeCUDA(uintptr_t ptr);
/// Wrapper around `cudaFreeAsync` for device or managed memory.
Status freeCUDAAsync(uintptr_t ptr, uintptr_t stream);
/// Wrapper around `cudaFreeHost` for pinned host memory.
Status freeCUDAPinnedHost(uintptr_t ptr);

/// Wrapper around `cudaLaunchHostFunc`.
Status launchCUDAHostFunc(uintptr_t stream, void (*callback)(void *),
                          void *userData);

} // namespace mtrt

#endif // MLIR_EXECUTOR_RUNTIME_SUPPORT_CUDAHELPERS
