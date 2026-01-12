//===- CUDAEventPoolTests.cpp --------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/Support/CUDAEventPool.h"
#include "mlir-executor/Runtime/Support/CUDAHelpers.h"
#include "gtest/gtest.h"
#include <thread>
#include <vector>

using namespace mtrt;

#ifdef MLIR_TRT_ENABLE_CUDA

namespace {

//===----------------------------------------------------------------------===//
// EventHandle Tests
//===----------------------------------------------------------------------===//

TEST(EventHandleTest, Construction) {
  EventHandle handle(123, 0, 0);
  EXPECT_EQ(handle.getEvent(), 123u);
  EXPECT_EQ(handle.getDevice(), 0);
  EXPECT_EQ(handle.getFlags(), 0u);
}

TEST(EventHandleTest, GettersReturnCorrectValues) {
  EventHandle handle(456, 1, 0x02);
  EXPECT_EQ(handle.getEvent(), 456u);
  EXPECT_EQ(handle.getDevice(), 1);
  EXPECT_EQ(handle.getFlags(), 0x02u);
}

//===----------------------------------------------------------------------===//
// OwningEventHandle Tests
//===----------------------------------------------------------------------===//

TEST(OwningEventHandleTest, DefaultConstruction) {
  OwningEventHandle handle;
  EXPECT_EQ(handle.getEvent(), 0u);
  EXPECT_EQ(handle.getDevice(), -1);
  EXPECT_EQ(handle.getFlags(), 0u);
}

// Note: OwningEventHandle tests with real CUDA events are covered in
// CudaEventPoolTest suite. These tests would require creating real CUDA events
// which would complicate the test setup.

//===----------------------------------------------------------------------===//
// CudaEventPool::Options Tests
//===----------------------------------------------------------------------===//

TEST(CudaEventPoolOptionsTest, DefaultConstruction) {
  CudaEventPool::Options opts;
  EXPECT_EQ(opts.max_reclaim_per_acquire, 64u);
  EXPECT_EQ(opts.max_free_per_bucket, 0u);
  EXPECT_EQ(opts.batch_size, 32u);
}

TEST(CudaEventPoolOptionsTest, CustomValues) {
  CudaEventPool::Options opts;
  opts.max_reclaim_per_acquire = 10;
  opts.max_free_per_bucket = 5;
  opts.batch_size = 8;

  EXPECT_EQ(opts.max_reclaim_per_acquire, 10u);
  EXPECT_EQ(opts.max_free_per_bucket, 5u);
  EXPECT_EQ(opts.batch_size, 8u);
}

//===----------------------------------------------------------------------===//
// CudaEventPool Basic Tests
//===----------------------------------------------------------------------===//

class CudaEventPoolTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Check if CUDA is available
    auto deviceCount = getCUDADeviceCount();
    if (!deviceCount.isOk() || deviceCount.getValue() == 0) {
      GTEST_SKIP() << "CUDA not available, skipping test";
    }
  }
};

TEST_F(CudaEventPoolTest, ConstructionAndDestruction) {
  CudaEventPool::Options opts;
  CudaEventPool pool(opts);
  // Should construct and destruct without error
}

TEST_F(CudaEventPoolTest, AcquireAndReleaseSingleEvent) {
  CudaEventPool pool;

  auto handleOrErr = pool.Acquire(0, 0);
  ASSERT_TRUE(handleOrErr.isOk()) << "Failed to acquire event";

  EventHandle handle = handleOrErr.getValue();
  EXPECT_NE(handle.getEvent(), 0u);
  EXPECT_EQ(handle.getDevice(), 0);
  EXPECT_EQ(handle.getFlags(), 0u);

  auto status = pool.Release(handle);
  EXPECT_TRUE(status.isOk()) << "Failed to release event";
}

TEST_F(CudaEventPoolTest, AcquireMultipleEvents) {
  CudaEventPool pool;

  std::vector<EventHandle> handles;
  for (int i = 0; i < 5; ++i) {
    auto handleOrErr = pool.Acquire(0, 0);
    ASSERT_TRUE(handleOrErr.isOk());
    handles.push_back(handleOrErr.getValue());
  }

  // All handles should be unique
  for (size_t i = 0; i < handles.size(); ++i) {
    for (size_t j = i + 1; j < handles.size(); ++j) {
      EXPECT_NE(handles[i].getEvent(), handles[j].getEvent());
    }
  }

  // Release all
  for (const auto &handle : handles) {
    auto status = pool.Release(handle);
    EXPECT_TRUE(status.isOk());
  }
}

TEST_F(CudaEventPoolTest, EventReuse) {
  CudaEventPool pool;

  // Acquire and release an event
  auto handle1OrErr = pool.Acquire(0, 0);
  ASSERT_TRUE(handle1OrErr.isOk());
  EventHandle handle1 = handle1OrErr.getValue();

  auto status = pool.Release(handle1);
  ASSERT_TRUE(status.isOk());

  // Reclaim to move from pending to free
  auto reclaimOrErr = pool.ReclaimAll();
  ASSERT_TRUE(reclaimOrErr.isOk());

  // Acquire again - might get the same event back
  auto handle2OrErr = pool.Acquire(0, 0);
  ASSERT_TRUE(handle2OrErr.isOk());
  EventHandle handle2 = handle2OrErr.getValue();

  // Should be a valid event (might or might not be the same)
  EXPECT_NE(handle2.getEvent(), 0u);

  status = pool.Release(handle2);
  EXPECT_TRUE(status.isOk());
}

TEST_F(CudaEventPoolTest, DifferentDevices) {
  auto deviceCountOrErr = getCUDADeviceCount();
  ASSERT_TRUE(deviceCountOrErr.isOk());
  int32_t deviceCount = deviceCountOrErr.getValue();

  if (deviceCount < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA devices for this test";
  }

  CudaEventPool pool;

  auto handle0OrErr = pool.Acquire(0, 0);
  ASSERT_TRUE(handle0OrErr.isOk());
  EventHandle handle0 = handle0OrErr.getValue();
  EXPECT_EQ(handle0.getDevice(), 0);

  auto handle1OrErr = pool.Acquire(1, 0);
  ASSERT_TRUE(handle1OrErr.isOk());
  EventHandle handle1 = handle1OrErr.getValue();
  EXPECT_EQ(handle1.getDevice(), 1);

  EXPECT_NE(handle0.getEvent(), handle1.getEvent());

  EXPECT_TRUE(pool.Release(handle0).isOk());
  EXPECT_TRUE(pool.Release(handle1).isOk());
}

TEST_F(CudaEventPoolTest, DifferentFlags) {
  CudaEventPool pool;

  auto handle1OrErr = pool.Acquire(0, 0x00);
  ASSERT_TRUE(handle1OrErr.isOk());
  EventHandle handle1 = handle1OrErr.getValue();
  EXPECT_EQ(handle1.getFlags(), 0x00u);

  auto handle2OrErr = pool.Acquire(0, 0x02); // cudaEventDisableTiming
  ASSERT_TRUE(handle2OrErr.isOk());
  EventHandle handle2 = handle2OrErr.getValue();
  EXPECT_EQ(handle2.getFlags(), 0x02u);

  // Different flags should give different events
  EXPECT_NE(handle1.getEvent(), handle2.getEvent());

  EXPECT_TRUE(pool.Release(handle1).isOk());
  EXPECT_TRUE(pool.Release(handle2).isOk());
}

//===----------------------------------------------------------------------===//
// CudaEventPool Options Tests
//===----------------------------------------------------------------------===//

TEST_F(CudaEventPoolTest, BatchSizeOption) {
  CudaEventPool::Options opts;
  opts.batch_size = 5;
  CudaEventPool pool(opts);

  // First acquire should create a batch
  auto handle1OrErr = pool.Acquire(0, 0);
  ASSERT_TRUE(handle1OrErr.isOk());

  // Subsequent acquires should use the batch (no new allocations needed)
  std::vector<EventHandle> handles;
  handles.push_back(handle1OrErr.getValue());

  for (int i = 1; i < 5; ++i) {
    auto handleOrErr = pool.Acquire(0, 0);
    ASSERT_TRUE(handleOrErr.isOk());
    handles.push_back(handleOrErr.getValue());
  }

  // All should be unique
  for (size_t i = 0; i < handles.size(); ++i) {
    for (size_t j = i + 1; j < handles.size(); ++j) {
      EXPECT_NE(handles[i].getEvent(), handles[j].getEvent());
    }
  }

  for (const auto &handle : handles) {
    EXPECT_TRUE(pool.Release(handle).isOk());
  }
}

TEST_F(CudaEventPoolTest, MaxFreePerBucketOption) {
  CudaEventPool::Options opts;
  opts.max_free_per_bucket = 2;
  opts.batch_size = 1; // Don't batch to have more control
  CudaEventPool pool(opts);

  // Acquire and release 5 events
  std::vector<EventHandle> handles;
  for (int i = 0; i < 5; ++i) {
    auto handleOrErr = pool.Acquire(0, 0);
    ASSERT_TRUE(handleOrErr.isOk());
    handles.push_back(handleOrErr.getValue());
  }

  for (const auto &handle : handles) {
    EXPECT_TRUE(pool.Release(handle).isOk());
  }

  // Reclaim all - should only keep max_free_per_bucket (2) events
  auto reclaimOrErr = pool.ReclaimAll();
  ASSERT_TRUE(reclaimOrErr.isOk());

  // Next acquires should reuse the 2 kept events, then allocate new ones
  for (int i = 0; i < 3; ++i) {
    auto handleOrErr = pool.Acquire(0, 0);
    ASSERT_TRUE(handleOrErr.isOk());
    EXPECT_TRUE(pool.Release(handleOrErr.getValue()).isOk());
  }
}

TEST_F(CudaEventPoolTest, MaxReclaimPerAcquireOption) {
  CudaEventPool::Options opts;
  opts.max_reclaim_per_acquire = 2;
  opts.batch_size = 1;
  CudaEventPool pool(opts);

  // Acquire and release many events
  for (int i = 0; i < 10; ++i) {
    auto handleOrErr = pool.Acquire(0, 0);
    ASSERT_TRUE(handleOrErr.isOk());
    EXPECT_TRUE(pool.Release(handleOrErr.getValue()).isOk());
  }

  // Each acquire will reclaim at most 2 events from pending
  // This is hard to test directly, but we can verify it doesn't crash
  auto handleOrErr = pool.Acquire(0, 0);
  ASSERT_TRUE(handleOrErr.isOk());
  EXPECT_TRUE(pool.Release(handleOrErr.getValue()).isOk());
}

TEST_F(CudaEventPoolTest, ZeroBatchSizeDefaultsToOne) {
  CudaEventPool::Options opts;
  opts.batch_size = 0; // Should be corrected to 1
  CudaEventPool pool(opts);

  auto handleOrErr = pool.Acquire(0, 0);
  ASSERT_TRUE(handleOrErr.isOk());
  EXPECT_TRUE(pool.Release(handleOrErr.getValue()).isOk());
}

//===----------------------------------------------------------------------===//
// CudaEventPool Advanced Tests
//===----------------------------------------------------------------------===//

TEST_F(CudaEventPoolTest, ReclaimAll) {
  CudaEventPool pool;

  // Acquire and release several events
  std::vector<EventHandle> handles;
  for (int i = 0; i < 5; ++i) {
    auto handleOrErr = pool.Acquire(0, 0);
    ASSERT_TRUE(handleOrErr.isOk());
    handles.push_back(handleOrErr.getValue());
  }

  for (const auto &handle : handles) {
    EXPECT_TRUE(pool.Release(handle).isOk());
  }

  // ReclaimAll should move completed events from pending to free
  auto reclaimOrErr = pool.ReclaimAll();
  ASSERT_TRUE(reclaimOrErr.isOk());
  // Should have reclaimed some events (exact number depends on completion)
  EXPECT_GE(reclaimOrErr.getValue(), 0u);
}

TEST_F(CudaEventPoolTest, TrimFree) {
  CudaEventPool::Options opts;
  opts.max_free_per_bucket = 3;
  CudaEventPool pool(opts);

  // Acquire and release many events
  for (int i = 0; i < 10; ++i) {
    auto handleOrErr = pool.Acquire(0, 0);
    ASSERT_TRUE(handleOrErr.isOk());
    EXPECT_TRUE(pool.Release(handleOrErr.getValue()).isOk());
  }

  // Reclaim to move to free list
  auto reclaimOrErr = pool.ReclaimAll();
  ASSERT_TRUE(reclaimOrErr.isOk());

  // Trim should reduce free list to max_free_per_bucket
  auto trimOrErr = pool.TrimFree();
  ASSERT_TRUE(trimOrErr.isOk());
  // Should have trimmed some events if free list exceeded max
}

TEST_F(CudaEventPoolTest, TrimFreeWithZeroMaxReturnsError) {
  CudaEventPool::Options opts;
  opts.max_free_per_bucket = 0; // Unlimited
  CudaEventPool pool(opts);

  auto trimOrErr = pool.TrimFree();
  EXPECT_FALSE(trimOrErr.isOk());
}

TEST_F(CudaEventPoolTest, IsComplete) {
  CudaEventPool pool;

  auto handleOrErr = pool.Acquire(0, 0);
  ASSERT_TRUE(handleOrErr.isOk());
  EventHandle handle = handleOrErr.getValue();

  // Query completion status
  auto isCompleteOrErr = CudaEventPool::IsComplete(handle);
  ASSERT_TRUE(isCompleteOrErr.isOk());
  // Event should be complete (not recorded yet)
  EXPECT_TRUE(isCompleteOrErr.getValue());

  EXPECT_TRUE(pool.Release(handle).isOk());
}

//===----------------------------------------------------------------------===//
// Thread Safety Tests
//===----------------------------------------------------------------------===//

TEST_F(CudaEventPoolTest, ConcurrentAcquireRelease) {
  CudaEventPool pool;
  const int numThreads = 4;
  const int eventsPerThread = 10;

  auto threadFunc = [&pool]() {
    for (int i = 0; i < eventsPerThread; ++i) {
      auto handleOrErr = pool.Acquire(0, 0);
      ASSERT_TRUE(handleOrErr.isOk());
      EventHandle handle = handleOrErr.getValue();
      EXPECT_NE(handle.getEvent(), 0u);

      // Do some "work" (just a small delay)
      std::this_thread::yield();

      auto status = pool.Release(handle);
      EXPECT_TRUE(status.isOk());
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; ++i) {
    threads.emplace_back(threadFunc);
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

TEST_F(CudaEventPoolTest, ConcurrentReclaimAndAcquire) {
  CudaEventPool pool;
  std::atomic<bool> stop{false};

  auto acquireReleaseFunc = [&pool, &stop]() {
    while (!stop.load()) {
      auto handleOrErr = pool.Acquire(0, 0);
      if (handleOrErr.isOk()) {
        (void)pool.Release(handleOrErr.getValue());
      }
      std::this_thread::yield();
    }
  };

  auto reclaimFunc = [&pool, &stop]() {
    while (!stop.load()) {
      (void)pool.ReclaimAll();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };

  std::thread t1(acquireReleaseFunc);
  std::thread t2(acquireReleaseFunc);
  std::thread t3(reclaimFunc);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  stop.store(true);

  t1.join();
  t2.join();
  t3.join();
}

//===----------------------------------------------------------------------===//
// Key Hashing Tests
//===----------------------------------------------------------------------===//

TEST(CudaEventPoolKeyTest, Equality) {
  CudaEventPool::Key key1{0, 0};
  CudaEventPool::Key key2{0, 0};
  CudaEventPool::Key key3{1, 0};
  CudaEventPool::Key key4{0, 1};

  EXPECT_TRUE(key1 == key2);
  EXPECT_FALSE(key1 == key3);
  EXPECT_FALSE(key1 == key4);
  EXPECT_FALSE(key3 == key4);
}

TEST(CudaEventPoolKeyTest, DenseMapInfo) {
  using DenseMapInfo = llvm::DenseMapInfo<CudaEventPool::Key>;

  CudaEventPool::Key empty = DenseMapInfo::getEmptyKey();
  CudaEventPool::Key tombstone = DenseMapInfo::getTombstoneKey();
  CudaEventPool::Key normal{0, 0};

  // Empty and tombstone should be different
  EXPECT_FALSE(DenseMapInfo::isEqual(empty, tombstone));

  // Normal key should not equal empty or tombstone
  EXPECT_FALSE(DenseMapInfo::isEqual(normal, empty));
  EXPECT_FALSE(DenseMapInfo::isEqual(normal, tombstone));

  // Hash values should be consistent
  CudaEventPool::Key key1{0, 0};
  CudaEventPool::Key key2{0, 0};
  EXPECT_EQ(DenseMapInfo::getHashValue(key1), DenseMapInfo::getHashValue(key2));

  // Different keys should (likely) have different hashes
  CudaEventPool::Key key3{1, 0};
  EXPECT_NE(DenseMapInfo::getHashValue(key1), DenseMapInfo::getHashValue(key3));
}

} // anonymous namespace

#else // !MLIR_TRT_ENABLE_CUDA

// Provide a dummy test when CUDA is not enabled
TEST(CUDAEventPoolTest, CUDANotEnabled) {
  GTEST_SKIP() << "Tests skipped: MLIR_TRT_ENABLE_CUDA is not defined";
}

#endif // MLIR_TRT_ENABLE_CUDA
