//===- MemRefOffsetTests.cpp ----------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/API/API.h"
#include "gtest/gtest.h"
#include <numeric>

using namespace mtrt;

namespace {

TEST(RuntimeMemRefOffset, ExternalMemRefDataPointerAndPointerInfo) {
  StatusOr<Ref<RuntimeClient>> client = RuntimeClient::create();
  ASSERT_TRUE(client.isOk()) << client.getStatus();

  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0.0f);
  std::vector<int64_t> shape{2, 2};
  std::vector<int64_t> strides{3, 1};
  constexpr int64_t offset = 1;

  BufferType type = BufferType::createWithElementStrides(
      ScalarType(ScalarTypeCode::f32), shape, strides, PointerType::host,
      offset);
  StatusOr<std::unique_ptr<MemRefValue>> memref =
      (*client)->createExternalMemRef(type,
                                      reinterpret_cast<uintptr_t>(data.data()));
  ASSERT_TRUE(memref.isOk()) << memref.getStatus();

  EXPECT_EQ((*memref)->getLayout().getOffset(), offset);
  EXPECT_EQ((*memref)->getMemory(), reinterpret_cast<uintptr_t>(data.data()));
  EXPECT_EQ((*memref)->getDataPtr(),
            reinterpret_cast<uintptr_t>(data.data() + offset));

  PointerInfo info = (*memref)->getPointerInfo(PointerOwner::external);
  EXPECT_EQ(info.ptr, reinterpret_cast<uintptr_t>(data.data()));
  EXPECT_EQ(info.size,
            static_cast<uint64_t>((*memref)->getTotalFootprintInBytes() +
                                  offset * sizeof(float)));
}

TEST(RuntimeMemRefOffset, CopyHostToHostResetsOffsetAndData) {
  StatusOr<Ref<RuntimeClient>> client = RuntimeClient::create();
  ASSERT_TRUE(client.isOk()) << client.getStatus();

  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0.0f);
  std::vector<int64_t> shape{2, 2};
  std::vector<int64_t> strides{3, 1};
  constexpr int64_t offset = 1;

  BufferType type = BufferType::createWithElementStrides(
      ScalarType(ScalarTypeCode::f32), shape, strides, PointerType::host,
      offset);
  StatusOr<std::unique_ptr<MemRefValue>> memref =
      (*client)->createExternalMemRef(type,
                                      reinterpret_cast<uintptr_t>(data.data()));
  ASSERT_TRUE(memref.isOk()) << memref.getStatus();

  StatusOr<std::unique_ptr<MemRefValue>> copy =
      (*client)->copyHostToHost(**memref);
  ASSERT_TRUE(copy.isOk()) << copy.getStatus();

  EXPECT_EQ((*copy)->getLayout().getOffset(), 0);
  EXPECT_EQ((*copy)->getShape(), llvm::ArrayRef(shape));
  EXPECT_EQ((*copy)->getStrides(), llvm::ArrayRef(strides));

  auto *copyData = reinterpret_cast<float *>((*copy)->getDataPtr());
  EXPECT_EQ(copyData[0], 1.0f);
  EXPECT_EQ(copyData[1], 2.0f);
  EXPECT_EQ(copyData[(*copy)->getStrides()[0]], 4.0f);
  EXPECT_EQ(copyData[(*copy)->getStrides()[0] + (*copy)->getStrides()[1]],
            5.0f);
}

TEST(RuntimeMemRefOffset, AllocateMemRefRequiresZeroOffset) {
  StatusOr<Ref<RuntimeClient>> client = RuntimeClient::create();
  ASSERT_TRUE(client.isOk()) << client.getStatus();

  std::vector<int64_t> shape{2, 2};
  std::vector<int64_t> strides{2, 1};
  BufferType type = BufferType::createWithElementStrides(
      ScalarType(ScalarTypeCode::f32), shape, strides, PointerType::host,
      /*offset=*/1);
  StatusOr<std::unique_ptr<MemRefValue>> memref =
      (*client)->allocateMemRef(type);
  EXPECT_TRUE(memref.isError());
}

} // namespace
