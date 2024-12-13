//===- CUDAModule.cpp -------------------------------------------*- C++ -*-===//
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
/// Executor CUDA module runtime implementation.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/Backend/Lua/Modules/CUDA/CudaModule.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Common/CUDACommon.h"
#include "mlir-executor/Runtime/Backend/Common/CommonRuntime.h"
#include "mlir-executor/Runtime/Backend/Common/NvPtxCompilerUtils.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaErrorHandling.h"
#include "mlir-executor/Runtime/Backend/Lua/Modules/Utils/MemRefUtils.h"
#include "mlir-executor/Runtime/Backend/Utils/NvtxUtils.h"
#include "mlir-executor/Support/Allocators.h"
#include "llvm/Support/Alignment.h"
#include <memory>
#include <string>

using namespace mlirtrt;
using namespace mlirtrt::runtime;

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

static StatusOr<int32_t> getDevice(int32_t deviceNumber) {
  CUdevice deviceID;
  RETURN_ERROR_WITH_MSG_IF_CUDADRV_ERROR(cuDeviceGet(&deviceID, deviceNumber),
                                         "could not get CUDA driver device");
  int smCount = 0;
  RETURN_ERROR_WITH_MSG_IF_CUDADRV_ERROR(
      cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                           deviceID),
      "could not query compute capability of device");
  StatusOr<std::string> arch = getDeviceArch(deviceNumber);
  if (!arch.isOk())
    return arch.getStatus();
  MTRT_DBGF("created device %d with arch= %s, num_sm=%d", deviceNumber,
            arch->c_str(), smCount);
  return deviceNumber;
}

static void registerCudaOps(sol::state_view &lua, AllocTracker *allocTracker,
                            PinnedMemoryAllocator *pinnedMemoryAllocator,
                            ResourceTracker *resourceTracker) {
  //===----------------------------------------------------------------------===//
  // CUDA - Device Management Ops
  //===----------------------------------------------------------------------===//
  lua["__cuda_num_devices"] = [](sol::this_state state) -> int32_t {
    ADD_CUDA_MODULE_RANGE("cuda_device_count");
    int count = 0;
    SET_LUA_ERROR_IF_CUDART_ERROR(cudaGetDeviceCount(&count), state);
    return count;
  };

  lua["__cuda_get_device"] = [](sol::this_state state,
                                int32_t deviceNumber) -> int32_t {
    ADD_CUDA_MODULE_RANGE("cuda_device_get");
    StatusOr<int32_t> device = getDevice(deviceNumber);
    SET_LUA_ERROR_AND_RETURN_IF_ERROR(device, state, 0);
    return *device;
  };

  //===----------------------------------------------------------------------===//
  // CUDA - Event Management Ops
  //===----------------------------------------------------------------------===//
  lua["__cuda_event_create"] =
      [resourceTracker](sol::this_state state) -> CudaEventPtr {
    ADD_CUDA_MODULE_RANGE("cuda_event_create");
    StatusOr<CudaEventPtr> event = CudaEventPtr::create(*resourceTracker);
    SET_LUA_ERROR_IF_ERROR(event, state);
    return *event;
  };

  lua["__cuda_event_elapsed_msec"] =
      [](sol::this_state state, CudaEventPtr start, CudaEventPtr end) -> float {
    ADD_CUDA_MODULE_RANGE("cuda_event_elapsed_msec");
    float ms = 0.0f;
    SET_LUA_ERROR_AND_RETURN_IF_CUDART_ERROR(
        cudaEventElapsedTime(&ms, start, end), state, ms);
    return ms;
  };
}

//===----------------------------------------------------------------------===//
// CUDA - Memory Management Ops
//===----------------------------------------------------------------------===//

static void
registerCudaMemoryManagementOps(sol::state_view &lua,
                                AllocTracker *allocTracker,
                                PinnedMemoryAllocator *pinnedMemoryAllocator) {

  //===----------------------------------------------------------------------===//
  // MemSet Ops
  //===----------------------------------------------------------------------===//
  lua["__cuda_memset_32"] = [](sol::this_state state, uintptr_t pointer,
                               size_t offset, size_t numBytes,
                               uint32_t fillInt) {
    MTRT_DBGF("cudaMemset32 @ 0x%lx, %lu bytes fill value = %u", pointer,
              numBytes, fillInt);
    SET_LUA_ERROR_IF_CUDA_ERROR(cuMemsetD32(static_cast<CUdeviceptr>(pointer),
                                            fillInt,
                                            numBytes / sizeof(fillInt)),
                                state);
  };

  lua["__cuda_memset_16"] = [](sol::this_state state, uintptr_t pointer,
                               size_t offset, size_t numBytes,
                               uint16_t fillInt) {
    MTRT_DBGF("cudaMemset16 @ 0x%lx, %lu bytes fill value = %u", pointer,
              numBytes, fillInt);
    SET_LUA_ERROR_IF_CUDA_ERROR(cuMemsetD16(static_cast<CUdeviceptr>(pointer),
                                            fillInt,
                                            numBytes / sizeof(fillInt)),
                                state);
  };

  lua["__cuda_memset_8"] = [](sol::this_state state, uintptr_t pointer,
                              size_t offset, size_t numBytes, uint8_t fillInt) {
    MTRT_DBGF("cudaMemset8 @ 0x%lx, %lu bytes fill value = %u", pointer,
              numBytes, fillInt);
    SET_LUA_ERROR_IF_CUDA_ERROR(
        cuMemsetD8(static_cast<CUdeviceptr>(pointer + offset), fillInt,
                   numBytes / sizeof(fillInt)),
        state);
  };

  //===----------------------------------------------------------------------===//
  // strided memcpy operations
  //===----------------------------------------------------------------------===//

  auto getCudaMemcpyFunc = [](cudaMemcpyKind kind, const char *dbgKind) {
    return [kind, dbgKind](sol::this_state state, CudaStreamPtr stream,
                           int64_t rank, int64_t elemSize, uintptr_t srcPointer,
                           size_t srcOffset, uintptr_t srcShapeAndStrides,
                           uintptr_t dstPointer, size_t dstOffset,
                           uintptr_t dstShapeAndStrides) {
      ADD_CUDA_MODULE_RANGE(dbgKind);
      MTRT_DBGF("%s: %lx to %lx rank = %ld", dbgKind, srcPointer + srcOffset,
                dstPointer + dstOffset, rank);

      const auto *srcShapeAndStridesPtr =
          reinterpret_cast<const int64_t *>(srcShapeAndStrides);
      const auto *dstShapeAndStridesPtr =
          reinterpret_cast<const int64_t *>(dstShapeAndStrides);

      std::vector<int64_t> srcShape(srcShapeAndStridesPtr,
                                    srcShapeAndStridesPtr + rank);
      std::vector<int64_t> srcStrides(srcShapeAndStridesPtr + rank,
                                      srcShapeAndStridesPtr + 2 * rank);
      std::vector<int64_t> dstShape(dstShapeAndStridesPtr,
                                    dstShapeAndStridesPtr + rank);
      std::vector<int64_t> dstStrides(dstShapeAndStridesPtr + rank,
                                      dstShapeAndStridesPtr + 2 * rank);

      executeStridedCopy(elemSize, srcPointer, srcOffset, srcShape, srcStrides,
                         dstPointer, dstOffset, dstShape, dstStrides,
                         [&](void *dst, void *src, size_t size) {
                           cudaMemcpyAsync(dst, src, size, kind, stream);
                         });
    };
  };

  lua["__cuda_memcpy_strided_async_device2device"] =
      getCudaMemcpyFunc(cudaMemcpyDeviceToDevice, "strided_device_memcpy_d2d");
  lua["__cuda_memcpy_strided_async_device2host"] =
      getCudaMemcpyFunc(cudaMemcpyDeviceToHost, "strided_device_memcpy_d2h");
  lua["__cuda_memcpy_strided_async_host2device"] =
      getCudaMemcpyFunc(cudaMemcpyHostToDevice, "strided_device_memcpy_h2d");
  lua["__cuda_memcpy_strided_async_host_pinned2device"] =
      lua["__cuda_memcpy_strided_async_host2device"];
  lua["__cuda_memcpy_strided_async_device2host_pinned"] =
      lua["__cuda_memcpy_strided_async_device2host"];

  //===----------------------------------------------------------------------===//
  // New Versions
  //===----------------------------------------------------------------------===//

  lua["__cuda_get_device"] = [](sol::this_state state,
                                int32_t deviceNumber) -> int32_t {
    ADD_CUDA_MODULE_RANGE("cuda_device_get");
    int32_t device{0};
    SET_LUA_ERROR_AND_RETURN_IF_CUDART_ERROR(cudaGetDevice(&device), state, 0);
    return device;
  };

  lua["__cuda_stream_create"] = [](sol::this_state state) -> CudaStreamPtr {
    ADD_CUDA_MODULE_RANGE("cuda_stream_create");
    cudaStream_t stream{nullptr};
    SET_LUA_ERROR_IF_CUDART_ERROR(cudaStreamCreate(&stream), state);
    return reinterpret_cast<uintptr_t>(stream);
  };

  lua["__cuda_stream_sync"] = [](sol::this_state state, CudaStreamPtr stream) {
    MTRT_DBG("__cuda_stream_sync @ {0}", reinterpret_cast<void *>(stream.ptr));
    ADD_CUDA_MODULE_RANGE("cuda_stream_sync");
    SET_LUA_ERROR_IF_CUDART_ERROR(cudaStreamSynchronize(stream), state);
  };

  lua["__cuda_stream_destroy"] = [](sol::this_state state,
                                    CudaStreamPtr stream) {
    ADD_CUDA_MODULE_RANGE("cuda_stream_destroy");
    SET_LUA_ERROR_IF_CUDART_ERROR(cudaStreamDestroy(stream), state);
  };

  lua["__cuda_get_function"] = [](sol::this_state state, uintptr_t cuModulePtr,
                                  std::string functionName) -> uintptr_t {
    ADD_CUDA_MODULE_RANGE("cuda_get_function");
    CUmodule module = reinterpret_cast<CUmodule>(cuModulePtr);
    CUfunction func;
    CUresult result = cuModuleGetFunction(&func, module, functionName.c_str());
    SET_LUA_ERROR_IF_CUDA_ERROR(result, state);
    return reinterpret_cast<uintptr_t>(func);
  };

  lua["__cuda_load_module"] =
      [allocTracker](sol::this_state state, uint32_t device, uintptr_t ptxData,
                     uint64_t ptxDataSize) -> uintptr_t {
    ADD_CUDA_MODULE_RANGE("cuda_get_function");
    // JIT compile the PTX.

    StatusOr<std::string> arch = getDeviceArch(device);
    SET_LUA_ERROR_AND_RETURN_IF_ERROR(arch, state, 0);

    const PointerInfo &info = allocTracker->get(ptxData);
    assert(info.isHostVisible());
    MTRT_DBGF("given size = %lu, actual size = %lu", ptxDataSize, info.size);
    assert(info.size == ptxDataSize);

    std::unique_ptr<runtime::CuBinWrapper> cubinWrapper =
        runtime::compilePtxToCuBin(reinterpret_cast<const char *>(ptxData),
                                   info.size, *arch);
    if (cubinWrapper == nullptr) {
      auto err = getInternalErrorStatus("failed to load PTX to cubin");
      SET_LUA_ERROR_AND_RETURN_IF_ERROR(err, state, 0);
    }

    CUmodule module{nullptr};
    CUresult result = cuModuleLoadDataEx(
        &module, reinterpret_cast<const void *>(cubinWrapper->data.data()), 0,
        0, 0);
    SET_LUA_ERROR_AND_RETURN_IF_CUDA_ERROR(result, state, 0);

    return reinterpret_cast<uintptr_t>(module);
  };

  lua["__cuda_launch"] = [](sol::this_state state, uintptr_t cudaFuncPtr,
                            int32_t gridX, int32_t gridY, int32_t gridZ,
                            int32_t blockX, int32_t blockY, int32_t blockZ,
                            int32_t dynamicSharedMemory,
                            CudaStreamPtr streamPtr, uintptr_t callArgsHostPtr,
                            uint32_t /*numCallArgs*/) {
    assert(cudaFuncPtr);
    assert(callArgsHostPtr);
    assert(streamPtr);
    CUresult result =
        cuLaunchKernel(reinterpret_cast<CUfunction>(cudaFuncPtr), gridX, gridY,
                       gridZ, blockX, blockY, blockZ, dynamicSharedMemory,
                       reinterpret_cast<CUstream>(cudaStream_t(streamPtr)),
                       reinterpret_cast<void **>(callArgsHostPtr),
                       /*extra=*/0);
    if (result != CUDA_SUCCESS) {
      MTRT_DBGF("%s", "error launching cuda kernel");
      SET_LUA_ERROR_IF_CUDA_ERROR(result, state);
    }
  };

  lua["__cuda_alloc_device"] = [allocTracker](sol::this_state state,
                                              CudaStreamPtr stream,
                                              int32_t device, size_t numBytes,
                                              int32_t alignment) -> uintptr_t {
    ADD_CUDA_MODULE_RANGE("__cuda_alloc");
    SET_LUA_ERROR_AND_RETURN_IF_CUDART_ERROR(cudaSetDevice(device), state, 0);
    StatusOr<PointerInfo> info = allocate(*allocTracker, PointerType::device,
                                          numBytes, alignment, stream);
    SET_LUA_ERROR_AND_RETURN_IF_ERROR(info, state, 0);
    return info->ptr;
  };

  lua["__cuda_alloc_host_pinned"] =
      [allocTracker, pinnedMemoryAllocator](sol::this_state state, size_t bytes,
                                            int32_t alignment) -> uintptr_t {
    ADD_CORE_MODULE_RANGE("core_alloc_host_pinned");
    sol::state_view lua(state);
    AllocTracker &tracker = *allocTracker;
    MTRT_DBGF("executor_alloc_host_pinned: %lu bytes", bytes);
    StatusOr<PinnedMemoryBlock> allocated =
        pinnedMemoryAllocator->allocate(bytes);
    SET_LUA_ERROR_AND_RETURN_IF_ERROR(allocated, state, 0);
    PointerInfo info(allocated->ptr, static_cast<int64_t>(allocated->size),
                     PointerType::pinned_host, PointerOwner::internal);
    if (!llvm::isAligned(llvm::Align(alignment), allocated->ptr)) {
      SET_LUA_ERROR_AND_RETURN_IF_ERROR(
          getInternalErrorStatus("allocated host pinned pointer {0} does not "
                                 "have desired alignment {1}",
                                 reinterpret_cast<void *>(allocated->ptr),
                                 alignment),
          state, 0);
    }
    tracker.track(info);
    return info.ptr;
  };

  lua["__memcpy_host2host_pinned"] = [allocTracker](
                                         sol::this_state state, uintptr_t src,
                                         size_t srcOffset, uintptr_t dst,
                                         size_t destOffset, size_t numBytes) {
    ADD_CORE_MODULE_RANGE("__memcpy_host2host_pinned");
    void *srcPtr = reinterpret_cast<void *>(src + srcOffset);
    void *dstPtr = reinterpret_cast<void *>(dst + destOffset);

    AllocTracker &tracker = *allocTracker;
    if (tracker.contains(src)) {
      const PointerInfo &srcInfo = tracker.get(src);
      assert(srcInfo.isHostVisible() && "expected host visible src pointer");
    }
    if (tracker.contains(dst)) {
      const PointerInfo &dstInfo = tracker.get(dst);
      assert(dstInfo.isHostVisible() && "expected host visible dst pointer");
    }
    MTRT_DBGF("executor_memcpy host-host %lu bytes src %lx + %lu dst %lx + %lu",
              numBytes, src, srcOffset, dst, destOffset);
    std::memcpy(dstPtr, srcPtr, numBytes);
  };

  lua["__cuda_free_host_pinned"] = [allocTracker, pinnedMemoryAllocator](
                                       sol::this_state state,
                                       CudaStreamPtr stream, uintptr_t ptr) {
    ADD_CORE_MODULE_RANGE("core_dealloc_host_pinned");
    auto &tracker = *allocTracker;
    PointerInfo info = tracker.get(ptr);
    MTRT_DBGF("executor_dealloc_host_pinned %lu bytes @ %lx", info.size,
              info.ptr);
    SET_LUA_ERROR_IF_ERROR(pinnedMemoryAllocator->freeAsync(info.ptr, stream),
                           state);
    tracker.untrack(ptr);
  };

  lua["__cuda_free_device"] = [allocTracker](sol::this_state state,
                                             CudaStreamPtr stream,
                                             uintptr_t ptr) {
    ADD_CUDA_MODULE_RANGE("cuda_memory_free_async");
    AllocTracker &tracker = *allocTracker;
    PointerInfo info = tracker.get(ptr);
    assert(info.isDeviceVisible() && "expected device-visible pointer");
    SET_LUA_ERROR_IF_CUDART_ERROR(
        cudaFreeAsync(reinterpret_cast<void *>(ptr), stream), state);
    tracker.untrack(ptr);
  };

  lua["__cuda_memcpy_host2device"] =
      [allocTracker](sol::this_state state, CudaStreamPtr stream, uintptr_t src,
                     size_t srcOffset, uintptr_t dest, size_t destOffset,
                     size_t numBytes) {
        ADD_CUDA_MODULE_RANGE("cuda_memcpy_async_h2d");
        MTRT_DBGF("cuda_memcpy_h2d %lu bytes from 0x%lx + %lu to 0x%lx + %lu",
                  numBytes, src, srcOffset, dest, destOffset);
        void *srcPtr = reinterpret_cast<void *>(src + srcOffset);
        void *dstPtr = reinterpret_cast<void *>(dest + destOffset);
#ifndef NDEBUG
        {
          AllocTracker &tracker = *allocTracker;
          assert(tracker.get(src).isHostVisible() &&
                 tracker.get(dest).isDeviceVisible() &&
                 "expected src to be a host ptr and dest to be a device ptr");
          assert(tracker.get(src).size >= srcOffset + numBytes &&
                 tracker.get(dest).size >= destOffset + numBytes &&
                 "src and/or dst buffers are insufficiently sized");
        }
#endif
        SET_LUA_ERROR_IF_CUDART_ERROR(cudaMemcpyAsync(dstPtr, srcPtr, numBytes,
                                                      cudaMemcpyHostToDevice,
                                                      stream),
                                      state);
      };

  lua["__cuda_memcpy_device2host"] =
      [allocTracker](sol::this_state state, CudaStreamPtr stream, uintptr_t src,
                     size_t srcOffset, uintptr_t dest, size_t destOffset,
                     size_t numBytes) {
        ADD_CUDA_MODULE_RANGE("cuda_memcpy_async_d2h");
        MTRT_DBGF("cuda_memcpy_d2h %lu bytes from 0x%lx + %lu to 0x%lx + %lu",
                  numBytes, src, srcOffset, dest, destOffset);
        void *srcPtr = reinterpret_cast<void *>(src + srcOffset);
        void *dstPtr = reinterpret_cast<void *>(dest + destOffset);
#ifndef NDEBUG
        {
          AllocTracker &tracker = *allocTracker;
          assert(tracker.get(src).isDeviceVisible() &&
                 tracker.get(dest).isHostVisible() &&
                 "expected src to be a host ptr and dest to be a device ptr");
          assert(tracker.get(src).size >= srcOffset + numBytes &&
                 tracker.get(dest).size >= destOffset + numBytes &&
                 "src and/or dst buffers are insufficiently sized");
        }
#endif
        SET_LUA_ERROR_IF_CUDART_ERROR(cudaMemcpyAsync(dstPtr, srcPtr, numBytes,
                                                      cudaMemcpyDeviceToHost,
                                                      stream),
                                      state);
      };

  lua["__cuda_memcpy_host_pinned2device"] =
      [allocTracker](sol::this_state state, CudaStreamPtr stream, uintptr_t src,
                     size_t srcOffset, uintptr_t dest, size_t destOffset,
                     size_t numBytes) {
        ADD_CUDA_MODULE_RANGE("cuda_memcpy_host_pinned2device");
        MTRT_DBGF("__cuda_memcpy_host_pinned2device: %lu bytes from 0x%lx + "
                  "%lu to 0x%lx + %lu",
                  numBytes, src, srcOffset, dest, destOffset);
        void *srcPtr = reinterpret_cast<void *>(src + srcOffset);
        void *dstPtr = reinterpret_cast<void *>(dest + destOffset);
#ifndef NDEBUG
        {
          AllocTracker &tracker = *allocTracker;
          assert(tracker.get(src).isHostVisible() &&
                 tracker.get(dest).isDeviceVisible() &&
                 "expected src to be a host ptr and dest to be a device ptr");
          assert(tracker.get(src).size >= srcOffset + numBytes &&
                 tracker.get(dest).size >= destOffset + numBytes &&
                 "src and/or dst buffers are insufficiently sized");
        }
#endif
        SET_LUA_ERROR_IF_CUDART_ERROR(cudaMemcpyAsync(dstPtr, srcPtr, numBytes,
                                                      cudaMemcpyHostToDevice,
                                                      stream),
                                      state);
      };

  lua["__cuda_memcpy_device2host_pinned"] =
      [allocTracker](sol::this_state state, CudaStreamPtr stream, uintptr_t src,
                     size_t srcOffset, uintptr_t dest, size_t destOffset,
                     size_t numBytes) {
        ADD_CUDA_MODULE_RANGE("cuda_memcpy_async_d2h");
        void *srcPtr = reinterpret_cast<void *>(src + srcOffset);
        void *dstPtr = reinterpret_cast<void *>(dest + destOffset);
#ifndef NDEBUG
        {
          AllocTracker &tracker = *allocTracker;
          assert(tracker.get(src).isDeviceVisible() &&
                 tracker.get(dest).isHostVisible() &&
                 "expected src to be a device ptr and dest to be a host ptr");
        }
#endif
        MTRT_DBGF("__cuda_memcpy_device2host_pinned: %lu bytes from 0x%lx + "
                  "%lu to 0x%lx + %lu",
                  numBytes, src, srcOffset, dest, destOffset);
        SET_LUA_ERROR_IF_CUDART_ERROR(cudaMemcpyAsync(dstPtr, srcPtr, numBytes,
                                                      cudaMemcpyDeviceToHost,
                                                      stream),
                                      state);
      };
  lua["__cuda_memcpy_device2device"] = [allocTracker](
                                           sol::this_state state,
                                           CudaStreamPtr stream, uintptr_t src,
                                           size_t srcOffset, uintptr_t dest,
                                           size_t destOffset, size_t numBytes) {
    ADD_CUDA_MODULE_RANGE("cuda_memcpy_async_d2d");
    void *srcPtr = reinterpret_cast<void *>(src + srcOffset);
    void *dstPtr = reinterpret_cast<void *>(dest + destOffset);
#ifndef NDEBUG
    {
      AllocTracker &tracker = *allocTracker;
      assert(tracker.get(src).isDeviceVisible() &&
             tracker.get(dest).isDeviceVisible() &&
             "expected src to be a device ptr and dest to be a device ptr");
    }
#endif
    MTRT_DBGF(
        "executor_memcpy device-device %lu bytes from %lx + %lu to %lx to %lu",
        numBytes, src, srcOffset, dest, destOffset);
    SET_LUA_ERROR_IF_CUDART_ERROR(cudaMemcpyAsync(dstPtr, srcPtr, numBytes,
                                                  cudaMemcpyDeviceToDevice,
                                                  stream),
                                  state);
    return;
  };
}

void mlirtrt::runtime::registerExecutorCUDAModuleLuaRuntimeMethods(
    lua_State *state, AllocTracker *allocTracker,
    PinnedMemoryAllocator *pinnedMemoryAllocator,
    ResourceTracker *resourceTracker) {
  sol::state_view lua(state);
  registerCudaOps(lua, allocTracker, pinnedMemoryAllocator, resourceTracker);
  registerCudaMemoryManagementOps(lua, allocTracker, pinnedMemoryAllocator);
}
