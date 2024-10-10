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

#include "mlir-executor/Support/Status.h"
#include <memory>

namespace mlirtrt {

using CudaStream = uintptr_t;
using CudaEvent = uintptr_t;

struct EventPool;

//===----------------------------------------------------------------------===//
// PoolTrackedCudaEvent
//===----------------------------------------------------------------------===//
struct PoolTrackedCudaEvent {
public:
  PoolTrackedCudaEvent() = delete;
  PoolTrackedCudaEvent(PoolTrackedCudaEvent &&) = default;
  PoolTrackedCudaEvent(const PoolTrackedCudaEvent &) = delete;
  PoolTrackedCudaEvent &operator=(const PoolTrackedCudaEvent &) = delete;
  PoolTrackedCudaEvent(CudaEvent event, EventPool *pool)
      : event(event), owningPool(pool) {}
  ~PoolTrackedCudaEvent();

  void releaseBackToPool();

  CudaEvent getEvent() const { return event; }
  EventPool *getPool() const { return owningPool; }

  static StatusOr<std::unique_ptr<PoolTrackedCudaEvent>>
  get(EventPool *eventPool);

private:
  CudaEvent event;
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
  Status freeAsync(uintptr_t ptr, CudaStream stream);

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

//===----------------------------------------------------------------------===//
// OutputAllocator and CustomTensorRTOuputAllocator
//===----------------------------------------------------------------------===//

/// Class to allocate memory for outputs with data-dependent shapes. The sizes
/// of those are unknown so pre-allocation is not possible.
class OutputAllocator {
public:
  virtual ~OutputAllocator() = default;

  /// Set the name of the tensor.
  virtual void setTensorName(const char *tensorName) = 0;

  /// Set the current memory pointer.
  virtual void setCurrentMemory(void *currentMemory) = 0;

  /// Set the size of the output.
  virtual void setOutputSize(const int64_t outputSize) = 0;

  /// Reallocate output memory asynchronously.
  virtual void *reallocateOutputAsync(char const *tensorName,
                                      void *currentMemory, uint64_t size,
                                      uint64_t alignment,
                                      CudaStream /*stream*/) = 0;

  /// Notify the shape of the tensor.
  virtual void notifyShape(char const *tensorName, const int64_t *dims,
                           int64_t nbDims) = 0;
};

/// Custom TensorRT output allocator implementation.
class CustomTensorRTOuputAllocator : public OutputAllocator {
public:
  CustomTensorRTOuputAllocator() = default;
  ~CustomTensorRTOuputAllocator();

  /// Set the name of the tensor.
  /// \note Methods are called just after construction. TODO: can they be called
  /// during construction?
  void setTensorName(const char *tensorName) override {
    mTensorName = tensorName;
  }

  /// Set the current memory pointer.
  void setCurrentMemory(void *currentMemory) override {
    mCurrentMemory = currentMemory;
  }

  /// Set the size of the output.
  void setOutputSize(int64_t outputSize) override { mOutputSize = outputSize; }

  /// Reallocate output memory asynchronously.
  void *reallocateOutputAsync(char const *tensorName, void *currentMemory,
                              uint64_t size, uint64_t alignment,
                              CudaStream /*stream*/) override;

  /// Notify the shape of the tensor.
  void notifyShape(char const *tensorName, const int64_t *dims,
                   int64_t nbDims) override;

  void *mOutputPtr{nullptr};           ///< nullptr if memory could not be allocated
  uint64_t mOutputSize{0};             ///< Size of allocation pointed to by output.
  bool mReallocateOutputCalled{false}; ///< Flag indicating if reallocateOutput was called
  bool mNotifyShapeCalled{false};      ///< Flag indicating if notifyShape was called
  std::vector<int64_t> mOutputDims;    ///< Dimensions of tensor.

private:
  const char *mTensorName;              ///< Name of the tensor
  void *mCurrentMemory;                 ///< Current memory pointer
};

/// Class to track and manage multiple OutputAllocators.
class OutputAllocatorTracker {
public:
  OutputAllocatorTracker() = default;
  ~OutputAllocatorTracker() = default;

  // Disable copy operations
  OutputAllocatorTracker(const OutputAllocatorTracker &) = delete;
  OutputAllocatorTracker &operator=(const OutputAllocatorTracker &) = delete;

  // Enable move operations
  OutputAllocatorTracker(OutputAllocatorTracker &&) = default;
  OutputAllocatorTracker &operator=(OutputAllocatorTracker &&) = default;

  /// Track a new OutputAllocator.
  /// \param allocator Unique pointer to the OutputAllocator to be tracked.
  void track(std::unique_ptr<OutputAllocator> allocator) {
    mOutputAllocatorRegistry.emplace_back(std::move(allocator));
  }

  /// Get an OutputAllocator by index.
  /// \param index The index of the OutputAllocator to retrieve.
  /// \return Pointer to the OutputAllocator at the specified index.
  OutputAllocator *get(int64_t index) {
    return mOutputAllocatorRegistry[index].get();
  }

private:
  /// Registry of OutputAllocators.
  std::vector<std::unique_ptr<OutputAllocator>> mOutputAllocatorRegistry;
};

/// Manages output tensor descriptors for TensorRT execution.
class OutputDescriptor {
public:
    /// Constructs an OutputDescriptor from a raw pointer.
    ///
    /// \param ptr Raw pointer to the descriptor data.
    OutputDescriptor(uintptr_t ptr);

    /// Returns the number of results in the descriptor.
    int64_t getNumberOfResults() const;

    /// Gets the rank of a specific tensor result.
    ///
    /// \param resultIndex Index of the result.
    unsigned getRank(int resultIndex) const;

    /// Sets the data pointer for a specific tensor result.
    ///
    /// \param resultIndex Index of the result to update.
    /// \param ptr New data pointer value.
    void setTensorDataPtr(int resultIndex, uintptr_t ptr);

    /// Sets the shape for a specific tensor result.
    ///
    /// \param resultIndex Index of the result to update.
    /// \param shape Vector containing the shape dimensions.
    void setShape(int resultIndex, const std::vector<int64_t>& shape);

    /// Sets the stride for a specific tensor result.
    ///
    /// \param resultIndex Index of the result to update.
    /// \param stride Vector containing the stride values.
    void setStride(int resultIndex, const std::vector<int64_t>& stride);

private:
    /// Pointer to the raw descriptor data.
    int64_t* mData;

    /// Total size of the descriptor data.
    size_t mSize;

    /// Calculates the index for a specific result in the descriptor.
    size_t getIndexForResult(int resultIndex) const;

    /// Calculates the total size of the descriptor.
    static size_t calculateTotalSize(uintptr_t ptr);

    /// Calculates the offset for a specific result in the descriptor.
    static size_t calculateOffsetForResult(const int64_t* desc, int64_t resultIndex);

    /// Fixed fields corresponding to rank, data ptr.
    static constexpr int OUTPUT_DESC_FIXED_FIELDS = 2;
};

} // namespace mlirtrt

#endif // MLIR_TENSORRT_SUPPORT_ALLOCATORS_H
