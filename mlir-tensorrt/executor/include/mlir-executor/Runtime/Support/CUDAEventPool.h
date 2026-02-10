//===- CUDAEventPool.h --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025-2026 NVIDIA CORPORATION & AFFILIATES.
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
/// \file
/// This file contains the declaration of CudaEventPool, a thread-safe CUDA
/// event pool for efficient event reuse.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_RUNTIME_SUPPORT_CUDAEVENTPOOL
#define MLIR_EXECUTOR_RUNTIME_SUPPORT_CUDAEVENTPOOL

#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/ADT/DenseMap.h"
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace mtrt {

// A lightweight wrapper around a cudaEvent_t plus its pool classification.
struct EventHandle {
public:
  EventHandle(uintptr_t event, int32_t device, unsigned int flags)
      : device(device), flags(flags), event(event) {}

  int32_t getDevice() const { return device; }
  unsigned int getFlags() const { return flags; }
  uintptr_t getEvent() const { return event; }

private:
  int32_t device = -1;
  unsigned int flags = 0;
  uintptr_t event = 0;
};

// A lightweight wrapper around a cudaEvent_t plus its pool classification.
struct OwningEventHandle {

public:
  OwningEventHandle() = default;
  OwningEventHandle(int32_t dev, unsigned int fl, uintptr_t ev)
      : device(dev), flags(fl), event(ev) {}
  OwningEventHandle(EventHandle handle)
      : device(handle.getDevice()), flags(handle.getFlags()),
        event(handle.getEvent()) {}

  // Non-copyable (owning handle)
  OwningEventHandle(const OwningEventHandle &) = delete;
  OwningEventHandle &operator=(const OwningEventHandle &) = delete;

  // Movable
  OwningEventHandle(OwningEventHandle &&other) noexcept;
  OwningEventHandle &operator=(OwningEventHandle &&other) noexcept;

  ~OwningEventHandle(); // destroys uintptr_t if still owned

  int32_t getDevice() const { return device; }
  unsigned int getFlags() const { return flags; }
  uintptr_t getEvent() const { return event; }

  friend class CudaEventPool;

  EventHandle release() {
    EventHandle ret(this->event, this->device, this->flags);
    this->device = -1;
    this->flags = 0;
    this->event = 0;
    return ret;
  }

private:
  int32_t device = -1;
  unsigned int flags = 0;
  uintptr_t event = 0;
};

// Thread-safe CUDA event pool keyed by (device, flags).
class CudaEventPool {
public:
  struct Options {
    Options();

    // Max number of completed events to reclaim from pending per Acquire()
    // call. 0 = unlimited reclaim (may be more work under large pending
    // queues).
    std::size_t max_reclaim_per_acquire{64};

    // Max number of free events to keep per bucket. Extra will be destroyed on
    // trim. 0 = unlimited retention.
    std::size_t max_free_per_bucket{0};

    // Batch size for event creation when empty.
    std::size_t batch_size{32};
  };

  struct Key {
    int device;
    unsigned int flags;

    bool operator==(const Key &o) const {
      return device == o.device && flags == o.flags;
    }
  };

  explicit CudaEventPool(Options opts = Options());
  ~CudaEventPool();

  CudaEventPool(const CudaEventPool &) = delete;
  CudaEventPool &operator=(const CudaEventPool &) = delete;

  // Acquire an event for (device, flags). Returned handle is non-owning and
  // must be Release()'d back to the pool when done.
  //
  // flags: same bitmask used for cudaEventCreateWithFlags (e.g.
  // cudaEventDisableTiming).
  StatusOr<EventHandle> Acquire(int32_t device, unsigned int flags);

  // Release an event back to the pool. The event is placed into the "pending"
  // queue until cudaEventQuery reports completion, at which point it becomes
  // reusable.
  //
  // Safe even if the event is still in-flight.
  Status Release(EventHandle handle);

  // Query completion.
  static StatusOr<bool> IsComplete(const EventHandle &handle);

  // Attempt to reclaim completed pending events across all buckets.
  // Returns number of events moved from pending -> free (or destroyed due to
  // max_free policy).
  StatusOr<std::size_t> ReclaimAll();

  // Trim free lists across all buckets to obey max_free_per_bucket.
  // Returns number of events destroyed.
  StatusOr<std::size_t> TrimFree();

private:
  struct Bucket {
    std::mutex mu;
    std::vector<std::unique_ptr<OwningEventHandle>> free;
    std::deque<std::unique_ptr<OwningEventHandle>> pending;
  };

  Bucket &GetOrCreateBucketLocked_(const Key &key);

  // Reclaim up to limit events from pending -> free within this bucket.
  // If limit == 0, reclaim all possible (until first incomplete, FIFO
  // behavior).
  StatusOr<std::size_t> ReclaimDoneLocked_(Bucket &b, std::size_t limit);

  // Create a single event (owning handle).
  static StatusOr<std::unique_ptr<OwningEventHandle>>
  CreateEvent_(int device, unsigned int flags);

  Options opts_{Options()};
  std::mutex map_mu_;
  llvm::DenseMap<Key, std::unique_ptr<Bucket>> buckets_;
};

} // namespace mtrt

// DenseMapInfo specialization for CudaEventPool::Key
namespace llvm {
template <>
struct DenseMapInfo<mtrt::CudaEventPool::Key> {
  static inline mtrt::CudaEventPool::Key getEmptyKey() {
    return {DenseMapInfo<int>::getEmptyKey(),
            DenseMapInfo<unsigned int>::getEmptyKey()};
  }
  static inline mtrt::CudaEventPool::Key getTombstoneKey() {
    return {DenseMapInfo<int>::getTombstoneKey(),
            DenseMapInfo<unsigned int>::getTombstoneKey()};
  }
  static unsigned getHashValue(const mtrt::CudaEventPool::Key &k) {
    return DenseMapInfo<int>::getHashValue(k.device) ^
           (DenseMapInfo<unsigned int>::getHashValue(k.flags) << 1);
  }
  static bool isEqual(const mtrt::CudaEventPool::Key &lhs,
                      const mtrt::CudaEventPool::Key &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // MLIR_EXECUTOR_RUNTIME_SUPPORT_CUDAEVENTPOOL
