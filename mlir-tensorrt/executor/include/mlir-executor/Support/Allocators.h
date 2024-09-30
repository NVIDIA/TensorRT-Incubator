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

  /// Untracks
  void untrack(uintptr_t ptr);

  StatusOr<PinnedMemoryBlock> allocate(size_t size);

  /// Free the block associated with the given pointer on the given stream. An
  /// event is pushed onto the stream and the memory won't be released into the
  /// free pool until after the stream has lapsed.
  Status freeAsync(uintptr_t ptr, CudaStream stream);

private:
  EventPool eventPool;

  /// Tracks all the pointers which need not to freed up.
  static std::vector<uintptr_t> untrackedPtrs;

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
