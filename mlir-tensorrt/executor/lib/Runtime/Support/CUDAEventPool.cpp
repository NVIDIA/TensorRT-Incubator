//===- CUDAEventPool.cpp ------------------------------------------------===//
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
#include "mlir-executor/Runtime/Support/CUDAEventPool.h"
#include "mlir-executor/Runtime/Support/CUDAHelpers.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include <cassert>

using namespace mtrt;

//===----------------------------------------------------------------------===//
// OwningEventHandle
//===----------------------------------------------------------------------===//

OwningEventHandle::OwningEventHandle(OwningEventHandle &&other) noexcept {
  device = other.device;
  flags = other.flags;
  event = other.event;

  other.device = -1;
  other.flags = 0;
  other.event = 0;
}

OwningEventHandle &
OwningEventHandle::operator=(OwningEventHandle &&other) noexcept {
  if (this == &other)
    return *this;
  // Destroy our current event (if any)
  if (event) {
    // Best-effort destroy; don't throw in noexcept move assignment.
    mtrt::logUnhandledErrors(mtrt::destroyCUDAEvent(event), llvm::errs());
  }

  device = other.device;
  flags = other.flags;
  event = other.event;

  other.device = -1;
  other.flags = 0;
  other.event = 0;
  return *this;
}

OwningEventHandle::~OwningEventHandle() {
  if (event) {
    StatusOr<std::unique_ptr<CUDADeviceGuard>> guard =
        CUDADeviceGuard::create(device);
    mtrt::cantFail(guard);
    mtrt::cantFail(destroyCUDAEvent(event));
    event = 0;
  }
}

//===----------------------------------------------------------------------===//
// CudaEventPool::Options
//===----------------------------------------------------------------------===//

CudaEventPool::Options::Options() = default;

//===----------------------------------------------------------------------===//
// CudaEventPool
//===----------------------------------------------------------------------===//

CudaEventPool::CudaEventPool(Options opts) : opts_(opts) {
  if (opts_.batch_size == 0)
    opts_.batch_size = 1;
}

CudaEventPool::~CudaEventPool() {
  // Destroy all events by letting the owning unique_ptr<EventHandle> destruct.
  // We must ensure device is set correctly for destruction.
  std::lock_guard<std::mutex> lk(map_mu_);
  buckets_.clear();
}

StatusOr<std::unique_ptr<OwningEventHandle>>
CudaEventPool::CreateEvent_(int32_t device, unsigned int flags) {
  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<CUDADeviceGuard> guard,
                        CUDADeviceGuard::create(device));
  MTRT_ASSIGN_OR_RETURN(uintptr_t ev, createCUDAEventForDevice(device));
  return std::make_unique<OwningEventHandle>(device, flags, ev);
}

CudaEventPool::Bucket &CudaEventPool::GetOrCreateBucketLocked_(const Key &key) {
  auto it = buckets_.find(key);
  if (it != buckets_.end())
    return *it->second;

  auto b = std::make_unique<Bucket>();
  Bucket &ref = *b;
  buckets_.insert({key, std::move(b)});
  return ref;
}

StatusOr<std::size_t> CudaEventPool::ReclaimDoneLocked_(Bucket &b,
                                                        std::size_t limit) {
  std::size_t reclaimed = 0;

  while (!b.pending.empty()) {
    if (limit != 0 && reclaimed >= limit)
      break;

    auto &front = b.pending.front();
    // FIFO reclaim optimization: stop at first incomplete
    MTRT_ASSIGN_OR_RETURN(bool isComplete, queryCUDAEvent(front->event));
    if (isComplete) {
      // Move from pending -> free (or destroy if exceeding max_free policy)
      std::unique_ptr<OwningEventHandle> done = std::move(front);
      b.pending.pop_front();

      if (opts_.max_free_per_bucket != 0 &&
          b.free.size() >= opts_.max_free_per_bucket) {
        // Drop (destroy) extra
        done.reset();
      } else {
        b.free.push_back(std::move(done));
      }
      ++reclaimed;
    } else {
      break;
    }
  }

  return reclaimed;
}

StatusOr<EventHandle> CudaEventPool::Acquire(int32_t device,
                                             unsigned int flags) {
  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<CUDADeviceGuard> guard,
                        CUDADeviceGuard::create(device));
  const Key key{device, flags};

  Bucket *bucket_ptr = nullptr;
  {
    std::lock_guard<std::mutex> lk(map_mu_);
    bucket_ptr = &GetOrCreateBucketLocked_(key);
  }

  Bucket &b = *bucket_ptr;
  std::lock_guard<std::mutex> lk(b.mu);

  // Reclaim completed pending events
  MTRT_ASSIGN_OR_RETURN(std::size_t reclaimed,
                        ReclaimDoneLocked_(b, opts_.max_reclaim_per_acquire));
  MTRT_DBG("reclaimed {0} events", reclaimed);

  if (!b.free.empty()) {
    auto h = std::move(b.free.back());
    b.free.pop_back();
    return h->release();
  }

  // No free events: allocate one (or a batch) and keep extras in free.
  if (opts_.batch_size > 1) {
    std::vector<std::unique_ptr<OwningEventHandle>> batch;
    batch.reserve(opts_.batch_size);
    for (std::size_t i = 0; i < opts_.batch_size; ++i) {
      MTRT_ASSIGN_OR_RETURN(std::unique_ptr<OwningEventHandle> event,
                            CreateEvent_(device, flags));
      batch.emplace_back(std::move(event));
    }
    // Return one, keep the rest
    auto ret = std::move(batch.back());
    batch.pop_back();

    for (auto &e : batch) {
      if (opts_.max_free_per_bucket != 0 &&
          b.free.size() >= opts_.max_free_per_bucket) {
        e.reset();
      } else {
        b.free.push_back(std::move(e));
      }
    }
    return ret->release();
  }

  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<OwningEventHandle> event,
                        CreateEvent_(device, flags));
  return event->release();
}

Status CudaEventPool::Release(EventHandle handle) {
  const Key key{handle.getDevice(), handle.getFlags()};
  Bucket *bucket_ptr = nullptr;
  {
    std::lock_guard<std::mutex> lk(map_mu_);
    bucket_ptr = &GetOrCreateBucketLocked_(key);
  }

  Bucket &b = *bucket_ptr;
  std::lock_guard<std::mutex> lk(b.mu);

  // Always goes to pending; reclaim will move it to free when complete.
  b.pending.push_back(std::make_unique<OwningEventHandle>(handle));
  return getOkStatus();
}

StatusOr<bool> CudaEventPool::IsComplete(const EventHandle &handle) {
  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<CUDADeviceGuard> guard,
                        CUDADeviceGuard::create(handle.getDevice()));
  MTRT_ASSIGN_OR_RETURN(bool isComplete, queryCUDAEvent(handle.getEvent()));
  return isComplete;
}

StatusOr<std::size_t> CudaEventPool::ReclaimAll() {
  std::vector<Bucket *> bucket_list;

  {
    std::lock_guard<std::mutex> lk(map_mu_);
    bucket_list.reserve(buckets_.size());
    for (auto &kv : buckets_)
      bucket_list.push_back(kv.second.get());
  }

  std::size_t total = 0;
  for (Bucket *b : bucket_list) {
    std::lock_guard<std::mutex> lk(b->mu);
    MTRT_ASSIGN_OR_RETURN(std::size_t reclaimed,
                          ReclaimDoneLocked_(*b, 0 /*unlimited*/));
    total += reclaimed;
  }
  return total;
}

StatusOr<std::size_t> CudaEventPool::TrimFree() {
  if (opts_.max_free_per_bucket == 0)
    return getInvalidArgStatus("max_free_per_bucket is 0");

  std::vector<Bucket *> bucket_list;
  {
    std::lock_guard<std::mutex> lk(map_mu_);
    bucket_list.reserve(buckets_.size());
    for (auto &kv : buckets_)
      bucket_list.push_back(kv.second.get());
  }

  std::size_t total = 0;
  for (Bucket *b : bucket_list) {
    std::lock_guard<std::mutex> lk(b->mu);
    while (b->free.size() > opts_.max_free_per_bucket) {
      b->free.back().reset(); // destroy event
      b->free.pop_back();
      ++total;
    }
  }
  return total;
}
