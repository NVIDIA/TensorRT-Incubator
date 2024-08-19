//===- Allocators.cpp -----------------------------------------------------===//
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
/// Implementation of runtime resource allocators.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Support/Allocators.h"
#include "mlir-executor/Support/Status.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "cuda_runtime_api.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <deque>
#include <set>
#include <sstream>
#include <string>

using namespace mlirtrt;

#define ALLOC_DBGF(fmt, ...)                                                   \
  DEBUG_WITH_TYPE("allocators", fprintf(stderr, "%s:%d " fmt "\n", __FILE__,   \
                                        __LINE__, __VA_ARGS__))

//===----------------------------------------------------------------------===//
// CustomTensorRTAllocator
//===----------------------------------------------------------------------===//

void *CustomTensorRTAllocator::allocate(uint64_t const size) {
  uint8_t *memory;
  cudaMalloc(reinterpret_cast<void **>(&memory), size);
  return memory;
}

bool CustomTensorRTAllocator::deallocate(void *const memory) {
  cudaFree(memory);
  return true;
}

//===----------------------------------------------------------------------===//
// PoolTrackedCudaEvent
//===----------------------------------------------------------------------===//

StatusOr<std::unique_ptr<PoolTrackedCudaEvent>>
PoolTrackedCudaEvent::get(EventPool *pool) {
  assert(pool != nullptr && "expected valid device event pool");
  cudaEvent_t event;
  RETURN_ERROR_IF_CUDART_ERROR(cudaEventCreate(&event));
  ALLOC_DBGF("creating pool-tracked cuda event %lu on pool %lu",
             reinterpret_cast<uintptr_t>(event),
             reinterpret_cast<uintptr_t>(pool));
  return std::make_unique<PoolTrackedCudaEvent>(event, pool);
}

PoolTrackedCudaEvent::~PoolTrackedCudaEvent() { releaseBackToPool(); }

void PoolTrackedCudaEvent::releaseBackToPool() {
  if (event && owningPool) {
    ALLOC_DBGF("releasing pool-tracked cuda event %lu back into pool %lu",
               reinterpret_cast<uintptr_t>(event),
               reinterpret_cast<uintptr_t>(owningPool));
    this->owningPool->push(PoolTrackedCudaEvent(event, nullptr));
    this->owningPool = nullptr;
    this->event = nullptr;
  }
}

//===----------------------------------------------------------------------===//
// EventPool
//===----------------------------------------------------------------------===//

void EventPool::push(PoolTrackedCudaEvent event) {
  pool.emplace_back(std::move(event));
}

std::unique_ptr<PoolTrackedCudaEvent> EventPool::pop() {
  assert(!empty() && "expected non-empty pool");
  std::unique_ptr<PoolTrackedCudaEvent> event =
      std::make_unique<PoolTrackedCudaEvent>(pool.back().getEvent(), this);
  pool.pop_back();
  return event;
}

StatusOr<std::unique_ptr<PoolTrackedCudaEvent>> EventPool::getCudaEvent() {
  if (empty())
    return PoolTrackedCudaEvent::get(this);
  return pop();
}

//===----------------------------------------------------------------------===//
// PinnedMemoryAllocator
//===----------------------------------------------------------------------===//

namespace {
/// Encapsulates a block of memory allocated by the pooling allocator.
struct Block {
  /// The byte size of the block.
  size_t size{0};
  /// The pointer to the start of the block.
  uintptr_t ptr{0};
  /// The number of pending events that must lapse before this block can be
  /// returned to the free pool.
  unsigned pendingEvents{0};
};

/// A functor that compares blocks. For use with `std::set`.
struct BlockComparison {
  bool operator()(const Block *lhs, const Block *rhs) const {
    if (lhs->size != rhs->size)
      return lhs->size < rhs->size;
    return lhs->ptr < rhs->ptr;
  }

  using is_transparent = void;
  bool operator()(const Block *lhs, Block rhs) const {
    if (lhs->size != rhs.size)
      return lhs->size < rhs.size;
    return lhs->ptr < rhs.ptr;
  }
  bool operator()(Block lhs, const Block *rhs) const {
    if (lhs.size != rhs->size)
      return lhs.size < rhs->size;
    return lhs.ptr < rhs->ptr;
  }
};
} // namespace

struct PinnedMemoryAllocator::BlockSet {
  template <typename... Args>
  auto insert(Args &&...args) {
    return set.insert(std::forward<Args>(args)...);
  }
  std::set<Block *, BlockComparison> set;
};

struct PinnedMemoryAllocator::BlockEventQueue {
  /// A BlockEvent is a pair (cudaEvent, Block*) that represents a dependency of
  /// Block* on cudaDevent before the block can be released into the free pool.
  using BlockEvent = std::pair<std::unique_ptr<PoolTrackedCudaEvent>, Block *>;

  /// The queue of BlockEvents currently in play. New BlockEvents go into the
  /// front.
  std::deque<BlockEvent> eventQueue;

  /// If the event queue is non-empty, return the oldest event.
  std::optional<BlockEvent> getNextOldestBlockEvent() {
    if (eventQueue.empty())
      return {};
    BlockEvent event = std::move(eventQueue.back());
    eventQueue.pop_back();
    return event;
  }

  /// Checks the event queue, and if a block can be freed, move it into
  /// `freeBlocks`.
  Status checkForFreeBlocks(
      std::unique_ptr<PinnedMemoryAllocator::BlockSet> &freeBlocks);
};

Status PinnedMemoryAllocator::BlockEventQueue::checkForFreeBlocks(
    std::unique_ptr<PinnedMemoryAllocator::BlockSet> &freeBlocks) {
  while (std::optional<BlockEvent> blockEvent = getNextOldestBlockEvent()) {
    auto &[eventPtr, block] = *blockEvent;
    cudaError_t status = cudaEventQuery(eventPtr->getEvent());
    if (status == cudaErrorNotReady) {
      (void)cudaGetLastError();
      eventQueue.emplace_back(std::move(*blockEvent));
      return getOkStatus();
    }
    RETURN_ERROR_IF_CUDART_ERROR(status);

    // Decrement use and push block if available.
    if (--block->pendingEvents == 0)
      freeBlocks->insert(block);
  }
  return getOkStatus();
}

struct PinnedMemoryAllocator::BlockTracker {
  std::set<Block *, BlockComparison> blocks;
  llvm::DenseMap<uintptr_t, Block *> pointerToBlock;

  ~BlockTracker() {
    ALLOC_DBGF(
        "[PinnedMemoryAllocator] Releasing block tracker that has %lu blocks",
        blocks.size());
    for (Block *block : blocks) {
      ALLOC_DBGF("[PinnedMemoryAllocator] releasing block %lu of size %lu",
                 block->ptr, block->size);
      (void)cudaFreeHost(reinterpret_cast<void *>(block->ptr));
    }
  }
};

PinnedMemoryAllocator::PinnedMemoryAllocator()
    : blockTracker(std::make_unique<BlockTracker>()),
      freeBlocks(std::make_unique<BlockSet>()),
      pendingBlockEvents(std::make_unique<BlockEventQueue>()) {}

PinnedMemoryAllocator::~PinnedMemoryAllocator() {}

StatusOr<PinnedMemoryBlock> PinnedMemoryAllocator::allocate(size_t size) {
  if (size == 0)
    return PinnedMemoryBlock{0, 0};

  Status processEventsResult =
      pendingBlockEvents->checkForFreeBlocks(freeBlocks);
  if (!processEventsResult.isOk())
    return processEventsResult.getStatus();

  // Return the smallest size block that meets the specified size.
  auto lowerBound = freeBlocks->set.lower_bound(Block{size, 0, 0});
  if (lowerBound != freeBlocks->set.end()) {
    Block *result = *lowerBound;
    freeBlocks->set.erase(result);
    ALLOC_DBGF("re-using block %lu of size %lu", result->ptr, result->size);
    return PinnedMemoryBlock{result->ptr, result->size};
  }

  // Allocate the block and insert it into the list of tracked blocks. Round up
  // to the nearest power of two since we want to create a nice distribution of
  // block-sizes for reuse.
  size_t allocatedSize = llvm::PowerOf2Ceil(size);
  void *allocatedPtr{nullptr};
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaHostAlloc(&allocatedPtr, allocatedSize, cudaHostAllocDefault));
  Block *result =
      new Block{allocatedSize, reinterpret_cast<uintptr_t>(allocatedPtr), 0};
  ALLOC_DBGF("allocated new block %lu of size %lu (rounded up from %lu)",
             result->ptr, result->size, size);
  blockTracker->blocks.insert(result);
  blockTracker->pointerToBlock.insert({result->ptr, result});
  return PinnedMemoryBlock{result->ptr, result->size};
}

// Free the given block.
Status PinnedMemoryAllocator::freeAsync(uintptr_t ptr, cudaStream_t stream) {
  assert(ptr && "expected valid ptr");
  Block *block = blockTracker->pointerToBlock.lookup(ptr);
  assert(block && "expected valid block");
  // If this block is associated with a stream, then we create an event to track
  // the dependency. It won't be released until the dependency is met.
  StatusOr<std::unique_ptr<PoolTrackedCudaEvent>> event =
      eventPool.getCudaEvent();
  RETURN_ERROR_IF_CUDART_ERROR(cudaEventRecord((*event)->getEvent(), stream));
  block->pendingEvents++;
  // Add the event and block to processing list.
  ALLOC_DBGF("enqueing asynchronously free of block %lu on stream %lu using "
             "stream %lu",
             block->ptr, reinterpret_cast<uintptr_t>(stream),
             reinterpret_cast<uintptr_t>((*event)->getEvent()));
  pendingBlockEvents->eventQueue.emplace_front(std::move(*event), block);
  return getOkStatus();
}