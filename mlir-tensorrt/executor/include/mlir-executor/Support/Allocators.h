//===- Allocators.h ----------------------------------------------*- C++-*-===//
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
/// Declarations for common runtime resource allocators.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_SUPPORT_ALLOCATORS_H
#define MLIR_TENSORRT_SUPPORT_ALLOCATORS_H

#include "cuda_runtime_api.h"
#include "mlir-executor/Support/Status.h"
#include <memory>

namespace mlirtrt {

struct EventPool;

//===----------------------------------------------------------------------===//
// GpuAllocator and CustomTensorRTAllocator
//===----------------------------------------------------------------------===//

class GpuAllocator {
public:
  GpuAllocator() = default;
  virtual ~GpuAllocator() = default;
  virtual void *allocate(uint64_t const size, uint64_t const alignment,
                         uint32_t flags, cudaStream_t* stream) {
    return nullptr;
  }
  virtual bool deallocate(void *const memory,
                          cudaStream_t* stream) {
    return false;
  }
};

class CustomTensorRTAllocator : public GpuAllocator {
public:
  CustomTensorRTAllocator() = default;
  ~CustomTensorRTAllocator() = default;
  void *allocate(uint64_t const size, uint64_t const alignment, uint32_t flags,
                 cudaStream_t* stream) override;
  bool deallocate(void *const memory,
                  cudaStream_t* stream) override;
};

//===----------------------------------------------------------------------===//
// OutputAllocator and CustomTensorRTOuputAllocator
//===----------------------------------------------------------------------===//

//!
//! Class to allocate memory for outputs with data-dependent shapes. The sizes
//! of those are unknown so pre-allocation is not possible.
//!
class OutputAllocator {
public:
  virtual ~OutputAllocator() = default;
  virtual void setGpuAllocator(GpuAllocator* gpuAllocator) = 0;
  virtual void setTensorName(const char *tensorName) = 0;
  virtual void setCurrentMemory(void *currentMemory) = 0;
  virtual void setOutputSize(const int64_t outputSize) = 0;
  virtual void *reallocateOutputAsync(char const *tensorName,
                                      void *currentMemory, uint64_t size,
                                      uint64_t alignment,
                                      cudaStream_t * /*stream*/) = 0;
  virtual void notifyShape(char const *tensorName, const int64_t *dims,
                           int64_t nbDims) = 0;
};

class CustomTensorRTOuputAllocator : public OutputAllocator {
public:
  CustomTensorRTOuputAllocator() = default;
  ~CustomTensorRTOuputAllocator() {
    if (mOutputPtr != nullptr) {
      cudaFree(mOutputPtr);
    }
  }

  void setGpuAllocator(GpuAllocator* gpuAllocator) override {
    mGpuAllocator = gpuAllocator;
  }

  //! Methods are called just after construction. TODO: can they be called
  //! during construction?
  void setTensorName(const char *tensorName) override {
    mTensorName = tensorName;
  }

  void setCurrentMemory(void *currentMemory) override {
    mCurrentMemory = currentMemory;
  }
  
  void setOutputSize(int64_t outputSize) override { mOutputSize = outputSize; }

  void *reallocateOutputAsync(char const *tensorName, void *currentMemory,
                              uint64_t size, uint64_t alignment,
                              cudaStream_t * /*stream*/) override;

  void notifyShape(char const *tensorName, const int64_t *dims,
                           int64_t nbDims) override;

  //! nullptr if memory could not be allocated
  void *mOutputPtr{nullptr};

  //! Size of allocation pointed to by output.
  uint64_t mOutputSize{0};

  bool mReallocateOutputCalled{false};

  bool mNotifyShapeCalled{false};

  //! Dimensions of tensor.
  std::vector<int64_t> mOutputDims;

private:
  GpuAllocator* mGpuAllocator;
  const char *mTensorName;
  void *mCurrentMemory;
};

class OutputAllocatorTracker {
public:
  OutputAllocatorTracker() = default;
  ~OutputAllocatorTracker() = default;

  OutputAllocatorTracker(const OutputAllocatorTracker &) = delete;
  OutputAllocatorTracker &operator=(const OutputAllocatorTracker &) = delete;
  OutputAllocatorTracker(OutputAllocatorTracker &&) = default;
  OutputAllocatorTracker &operator=(OutputAllocatorTracker &&) = default;

  // Add a new OutputAllocator
  void addAllocator(void *ptr, OutputAllocator *allocator) {
    mOutputAllocatorRegistry.emplace_back(std::make_pair(ptr, allocator));
  }

  // Get a reference to an OutputAllocator
  OutputAllocator *getAllocator(void *ptr) {
    auto it = std::find_if(
        mOutputAllocatorRegistry.begin(), mOutputAllocatorRegistry.end(),
        [ptr](const auto &pair) { return pair.first == ptr; });

    if (it != mOutputAllocatorRegistry.end()) {
      return it->second;
    }
    return nullptr;
  }

private:
  std::vector<std::pair<void *, OutputAllocator *>> mOutputAllocatorRegistry;
};

//===----------------------------------------------------------------------===//
// PoolTrackedCudaEvent
//===----------------------------------------------------------------------===//
struct PoolTrackedCudaEvent {
public:
  PoolTrackedCudaEvent() = delete;
  PoolTrackedCudaEvent(PoolTrackedCudaEvent &&) = default;
  PoolTrackedCudaEvent(const PoolTrackedCudaEvent &) = delete;
  PoolTrackedCudaEvent &operator=(const PoolTrackedCudaEvent &) = delete;
  PoolTrackedCudaEvent(cudaEvent_t event, EventPool *pool)
      : event(event), owningPool(pool) {}
  ~PoolTrackedCudaEvent();

  void releaseBackToPool();

  cudaEvent_t getEvent() const { return event; }
  EventPool *getPool() const { return owningPool; }

  static StatusOr<std::unique_ptr<PoolTrackedCudaEvent>>
  get(EventPool *eventPool);

private:
  cudaEvent_t event;
  EventPool *owningPool;
};

//===----------------------------------------------------------------------===//
// EventPool
//===----------------------------------------------------------------------===//

/// An event pool that is associated with a particular device. All events in the
/// pool are considered free for use. When an event is being used, it is removed
/// from the pool.
struct EventPool {
  std::vector<PoolTrackedCudaEvent> pool;

  bool empty() const { return pool.empty(); }

  /// Push a new event onto the pool and return it.
  void push(PoolTrackedCudaEvent event);

  /// Return a cuda event from pool. Asserts pool is non-empty.
  std::unique_ptr<PoolTrackedCudaEvent> pop();

  // Retrieve an event from the pool or create a new one.
  StatusOr<std::unique_ptr<PoolTrackedCudaEvent>> getCudaEvent();
};

//===----------------------------------------------------------------------===//
// PinnedMemoryAllocator
//===----------------------------------------------------------------------===//

/// Represents an allocated contiguous block of page-locked host memory.
struct PinnedMemoryBlock {
  uintptr_t ptr{0};
  size_t size{0};
};

/// An allocator that manages the creation of host-pinned memory via
/// `cudaMallocHost` and `cudaFreeHost`. It provides a synchronous `allocate`
/// function and an asynchronous `freeAsync` function. Note that CUDA RT does
/// not provide asynchronous host-pinned memory deallocation in the
/// stream-ordered API as of CUDA 12.1. We create this capability by using CUDA
/// events to know when it is save to release an allocation back into a managed
/// pool.
class PinnedMemoryAllocator {
public:
  PinnedMemoryAllocator();
  ~PinnedMemoryAllocator();

  StatusOr<PinnedMemoryBlock> allocate(size_t size);

  /// Free the block associated with the given pointer on the given stream. An
  /// event is pushed onto the stream and the memory won't be released into the
  /// free pool until after the stream has lapsed.
  Status freeAsync(uintptr_t ptr, cudaStream_t stream);

private:
  EventPool eventPool;

  /// Tracks all blocks allocated by the allocator.
  struct BlockTracker;
  std::unique_ptr<BlockTracker> blockTracker;

  /// Tracks the free blocks available to the allocator.
  struct BlockSet;
  std::unique_ptr<BlockSet> freeBlocks;

  /// Tracks the events CUDA events that must elapse before a block can be
  /// released back into the freeBlocksSet.
  struct BlockEventQueue;
  std::unique_ptr<BlockEventQueue> pendingBlockEvents;
};

} // namespace mlirtrt

#endif // MLIR_TENSORRT_SUPPORT_ALLOCATORS_H
