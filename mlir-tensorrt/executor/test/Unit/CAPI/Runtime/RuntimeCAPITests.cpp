//===- RuntimeCAPITests.cpp  ----------------------------------------------===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Unit tests for the runtime C API implementation.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor-c/Common/Common.h"
#include "mlir-executor-c/Runtime/Runtime.h"
#include "llvm/ADT/ArrayRef.h"
#include "gtest/gtest.h"
#include <numeric>
#include <utility>

namespace {

class RuntimeCAPIClientTest : public ::testing::Test {
protected:
  void SetUp() override {
    MTRT_Status status = mtrtRuntimeClientCreate(&client);
    ASSERT_TRUE(mtrtStatusIsOk(status));
    ASSERT_FALSE(mtrtRuntimeClientIsNull(client));
  }

  void TearDown() override {
    if (!mtrtRuntimeClientIsNull(client))
      ASSERT_TRUE(mtrtStatusIsOk(mtrtRuntimeClientDestroy(client)));
  }

  /// Create a host memref from externally owned data.
  MTRT_MemRefValue createHostMemRef(const std::vector<float> &data,
                                    llvm::ArrayRef<int64_t> shape,
                                    llvm::ArrayRef<int64_t> strides,
                                    int64_t offset = 0) {
    MTRT_MemRefValue buffer{nullptr};
    MTRT_Status status = mtrtMemRefCreateExternal(
        client, MTRT_PointerType::MTRT_PointerType_host,
        MTRT_ScalarTypeCode_f32, reinterpret_cast<uintptr_t>(data.data()),
        offset, shape.size(), shape.data(), strides.data(), mtrtDeviceGetNull(),
        &buffer);
    if (!mtrtStatusIsOk(status)) {
      ADD_FAILURE() << "mtrtMemRefCreateExternal failed";
      return buffer;
    }
    if (mtrtMemRefValueIsNull(buffer)) {
      ADD_FAILURE() << "created memref is null";
      return buffer;
    }
    return buffer;
  }

  /// Retrieve memref metadata for assertions.
  MTRT_MemRefValueInfo getMemRefInfo(MTRT_MemRefValue memref) {
    MTRT_MemRefValueInfo info{};
    MTRT_Status status = mtrtMemRefValueGetInfo(memref, &info);
    if (!mtrtStatusIsOk(status))
      ADD_FAILURE() << "mtrtMemRefValueGetInfo failed";
    return info;
  }

  /// Destroy a memref and assert success.
  void destroyMemRef(MTRT_MemRefValue memref) {
    ASSERT_TRUE(mtrtStatusIsOk(mtrtMemRefValueDestroy(memref)));
  }

  MTRT_RuntimeClient client = mtrtRuntimeClientGetNull();
};

} // namespace

TEST(RuntimeCAPI, ScalarTypeCodeSize) {
  std::vector<std::pair<MTRT_ScalarTypeCode, int64_t>> testData = {
      std::make_pair(MTRT_ScalarTypeCode_f8e4m3fn, 8),
      std::make_pair(MTRT_ScalarTypeCode_f16, 16),
      std::make_pair(MTRT_ScalarTypeCode_f32, 32),
      std::make_pair(MTRT_ScalarTypeCode_f64, 64),
      std::make_pair(MTRT_ScalarTypeCode_i1, 8),
      std::make_pair(MTRT_ScalarTypeCode_i8, 8),
      std::make_pair(MTRT_ScalarTypeCode_ui8, 8),
      std::make_pair(MTRT_ScalarTypeCode_i16, 16),
      std::make_pair(MTRT_ScalarTypeCode_i32, 32),
      std::make_pair(MTRT_ScalarTypeCode_i64, 64),
      std::make_pair(MTRT_ScalarTypeCode_bf16, 16)};

  for (auto [enumVal, expectedBitsPerElement] : testData) {
    int64_t bitsPerElement{0};
    MTRT_Status s = mtrtScalarTypeCodeBitsPerElement(enumVal, &bitsPerElement);
    ASSERT_TRUE(mtrtStatusIsOk(s));
    ASSERT_EQ(bitsPerElement, expectedBitsPerElement);
  }
}

TEST(RuntimeCAPI, TestStatusCreate) {

  std::string statusMessage{"This is status test!"};
  MTRT_Status status = mtrtStatusCreate(
      MTRT_StatusCode::MTRT_StatusCode_InternalError, statusMessage.c_str());
  ASSERT_FALSE(mtrtStatusIsOk(status));
  mtrtStatusDestroy(status);
}

TEST(RuntimeCAPI, TestClientCreate) {
  MTRT_RuntimeClient client;
  mtrtRuntimeClientCreate(&client);
  ASSERT_TRUE(!mtrtRuntimeClientIsNull(client));
  mtrtRuntimeClientDestroy(client);
}

TEST(RuntimeCAPI, TestClientGetDevices) {
  MTRT_RuntimeClient client;
  MTRT_Status status = mtrtRuntimeClientCreate(&client);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_TRUE(!mtrtRuntimeClientIsNull(client));

  int32_t numDevices;
  status = mtrtRuntimeClientGetNumDevices(client, &numDevices);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  // Can't run this test without an available CUDA device.
  if (numDevices < 1)
    return;

  MTRT_Device device;
  status = mtrtRuntimeClientGetDevice(client, 0, &device);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  // get the stream.
  MTRT_Stream stream;
  status = mtrtDeviceGetStream(device, &stream);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_TRUE(!mtrtStreamIsNull(stream));
  ASSERT_TRUE(mtrtStatusIsOk(mtrtStreamDestroy(stream)));

  status = mtrtRuntimeClientDestroy(client);
  ASSERT_TRUE(mtrtStatusIsOk(status));
}

TEST_F(RuntimeCAPIClientTest, TestHostBufferCreateExternalAndTracking) {
  int32_t numDevices;
  MTRT_Status status = mtrtRuntimeClientGetNumDevices(client, &numDevices);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  std::vector<int64_t> shape{2, 2};
  std::vector<int64_t> strides{2, 1};
  std::vector<float> data(4);
  std::iota(data.begin(), data.end(), 0.0);
  MTRT_PointerType addressSpace{MTRT_PointerType::MTRT_PointerType_host};
  MTRT_MemRefValue buffer = createHostMemRef(data, shape, strides);
  MTRT_MemRefValueInfo info = getMemRefInfo(buffer);

  ASSERT_EQ(llvm::ArrayRef(info.shape, info.rank), llvm::ArrayRef(shape));
  ASSERT_EQ(llvm::ArrayRef(info.strides, info.rank), llvm::ArrayRef(strides));
  ASSERT_EQ(info.addressSpace, addressSpace);
  ASSERT_EQ(info.rank, 2);
  ASSERT_EQ(info.bitsPerElement, 32);

  destroyMemRef(buffer);
}

TEST_F(RuntimeCAPIClientTest, TestHostToHostCopy) {
  std::vector<float> data{1, 2, 3, 4};
  std::vector<int64_t> shape{2, 2};
  std::vector<int64_t> strides{2, 1};
  MTRT_MemRefValue hostBuffer = createHostMemRef(data, shape, strides);

  MTRT_MemRefValue newHostBuffer;
  MTRT_Status status = mtrtCopyFromHostToHost(hostBuffer, &newHostBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_FALSE(mtrtMemRefValueIsNull(newHostBuffer));

  destroyMemRef(hostBuffer);
  destroyMemRef(newHostBuffer);
}

TEST_F(RuntimeCAPIClientTest, TestMemRefCreateExternalOffsetAndCopy) {
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0.0f);
  std::vector<int64_t> shape{2, 2};
  std::vector<int64_t> strides{3, 1};
  constexpr int64_t offset = 1;

  MTRT_MemRefValue hostBuffer = createHostMemRef(data, shape, strides, offset);
  MTRT_MemRefValueInfo info = getMemRefInfo(hostBuffer);
  ASSERT_EQ(info.ptr, reinterpret_cast<uintptr_t>(data.data()));
  ASSERT_EQ(info.offset, offset);
  ASSERT_EQ(llvm::ArrayRef(info.shape, info.rank), llvm::ArrayRef(shape));
  ASSERT_EQ(llvm::ArrayRef(info.strides, info.rank), llvm::ArrayRef(strides));

  auto *dataPtr =
      reinterpret_cast<float *>(info.ptr) + static_cast<size_t>(info.offset);
  ASSERT_EQ(dataPtr[0], 1.0f);
  ASSERT_EQ(dataPtr[1], 2.0f);
  ASSERT_EQ(dataPtr[info.strides[0]], 4.0f);
  ASSERT_EQ(dataPtr[info.strides[0] + info.strides[1]], 5.0f);

  MTRT_MemRefValue copyBuffer;
  MTRT_Status status = mtrtCopyFromHostToHost(hostBuffer, &copyBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_FALSE(mtrtMemRefValueIsNull(copyBuffer));

  MTRT_MemRefValueInfo copyInfo = getMemRefInfo(copyBuffer);
  ASSERT_EQ(copyInfo.offset, 0);
  ASSERT_EQ(llvm::ArrayRef(copyInfo.shape, copyInfo.rank),
            llvm::ArrayRef(shape));
  ASSERT_EQ(llvm::ArrayRef(copyInfo.strides, copyInfo.rank),
            llvm::ArrayRef(strides));

  auto *copyDataPtr = reinterpret_cast<float *>(copyInfo.ptr);
  ASSERT_EQ(copyDataPtr[0], 1.0f);
  ASSERT_EQ(copyDataPtr[1], 2.0f);
  ASSERT_EQ(copyDataPtr[copyInfo.strides[0]], 4.0f);
  ASSERT_EQ(copyDataPtr[copyInfo.strides[0] + copyInfo.strides[1]], 5.0f);

  destroyMemRef(copyBuffer);
  destroyMemRef(hostBuffer);
}

TEST_F(RuntimeCAPIClientTest, TestMemRefCreateViewRef) {
  std::vector<float> data(20);
  std::vector<int64_t> shape{4, 5};
  std::vector<int64_t> strides{5, 1};
  constexpr int64_t baseOffset = 3;
  MTRT_MemRefValue baseBuffer =
      createHostMemRef(data, shape, strides, baseOffset);

  uint32_t refCountBefore = mtrtMemRefReferenceCount(baseBuffer);
  ASSERT_GT(refCountBefore, 0u);

  std::vector<int64_t> offsets{1, 2};
  std::vector<int64_t> sizes{2, 2};
  std::vector<int64_t> sliceStrides{1, 2};
  MTRT_MemRefValue viewBuffer{nullptr};
  MTRT_Status status = mtrtMemRefCreateViewRef(
      baseBuffer, offsets.data(), sizes.data(), sliceStrides.data(),
      /*squeezeUnitDims=*/false, &viewBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_FALSE(mtrtMemRefValueIsNull(viewBuffer));

  uint32_t refCountAfter = mtrtMemRefReferenceCount(baseBuffer);
  ASSERT_EQ(refCountAfter, refCountBefore + 1u);
  ASSERT_EQ(mtrtMemRefReferenceCount(viewBuffer), refCountAfter);

  MTRT_MemRefValueInfo baseInfo = getMemRefInfo(baseBuffer);
  MTRT_MemRefValueInfo viewInfo = getMemRefInfo(viewBuffer);

  ASSERT_EQ(baseInfo.ptr, reinterpret_cast<uintptr_t>(data.data()));
  ASSERT_EQ(viewInfo.ptr, baseInfo.ptr);

  int64_t expectedOffset =
      baseOffset + offsets[0] * strides[0] + offsets[1] * strides[1];
  std::vector<int64_t> expectedStrides{sliceStrides[0] * strides[0],
                                       sliceStrides[1] * strides[1]};

  ASSERT_EQ(viewInfo.offset, expectedOffset);
  ASSERT_EQ(viewInfo.rank, static_cast<int64_t>(sizes.size()));
  ASSERT_EQ(llvm::ArrayRef(viewInfo.shape, viewInfo.rank),
            llvm::ArrayRef(sizes));
  ASSERT_EQ(llvm::ArrayRef(viewInfo.strides, viewInfo.rank),
            llvm::ArrayRef(expectedStrides));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(data.data() + expectedOffset),
            viewInfo.ptr + expectedOffset * sizeof(float));

  destroyMemRef(viewBuffer);
  ASSERT_EQ(mtrtMemRefReferenceCount(baseBuffer), refCountBefore);

  destroyMemRef(baseBuffer);
}

TEST_F(RuntimeCAPIClientTest, TestMemRefCreateViewRefSqueezeUnitDims) {
  constexpr int64_t baseOffset = 4;
  const int64_t numElements = baseOffset + 2 * 32 * 3;
  std::vector<float> data(numElements);
  std::iota(data.begin(), data.end(), 0.0f);
  std::vector<int64_t> shape{2, 32, 3};
  std::vector<int64_t> strides{96, 3, 1};
  MTRT_MemRefValue baseBuffer =
      createHostMemRef(data, shape, strides, baseOffset);

  std::vector<int64_t> offsets{1, 0, 2};
  std::vector<int64_t> sizes{1, 32, 1};
  std::vector<int64_t> sliceStrides{1, 1, 1};
  MTRT_MemRefValue viewBuffer{nullptr};
  MTRT_Status status = mtrtMemRefCreateViewRef(
      baseBuffer, offsets.data(), sizes.data(), sliceStrides.data(),
      /*squeezeUnitDims=*/true, &viewBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_FALSE(mtrtMemRefValueIsNull(viewBuffer));

  MTRT_MemRefValueInfo viewInfo = getMemRefInfo(viewBuffer);
  int64_t expectedOffset =
      baseOffset + 1 * strides[0] + 0 * strides[1] + 2 * strides[2];
  ASSERT_EQ(viewInfo.offset, expectedOffset);
  ASSERT_EQ(viewInfo.rank, 1);
  ASSERT_EQ(viewInfo.shape[0], 32);
  ASSERT_EQ(viewInfo.strides[0], strides[1]);

  destroyMemRef(viewBuffer);

  std::vector<int64_t> allUnitSizes{1, 1, 1};
  std::vector<int64_t> zeroOffsets{0, 0, 0};
  MTRT_MemRefValue scalarView{nullptr};
  status = mtrtMemRefCreateViewRef(baseBuffer, zeroOffsets.data(),
                                   allUnitSizes.data(), sliceStrides.data(),
                                   /*squeezeUnitDims=*/true, &scalarView);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_FALSE(mtrtMemRefValueIsNull(scalarView));

  MTRT_MemRefValueInfo scalarInfo = getMemRefInfo(scalarView);
  ASSERT_EQ(scalarInfo.rank, 0);
  ASSERT_EQ(scalarInfo.offset, baseOffset);

  destroyMemRef(scalarView);
  destroyMemRef(baseBuffer);
}

#ifdef MLIR_TRT_ENABLE_CUDA
TEST_F(RuntimeCAPIClientTest, TestHostToDeviceAndBackCopy) {
  int32_t numDevices{0};
  MTRT_Status status = mtrtRuntimeClientGetNumDevices(client, &numDevices);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  if (numDevices < 1) {
    mtrtRuntimeClientDestroy(client);
    return;
  }
  MTRT_Device device;
  status = mtrtRuntimeClientGetDevice(client, 0, &device);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_TRUE(!mtrtDeviceIsNull(device));

  MTRT_Stream stream;
  status = mtrtDeviceGetStream(device, &stream);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  std::vector<float> data{1, 2, 3, 4};
  std::vector<int64_t> shape{2, 2};
  std::vector<int64_t> strides{2, 1};
  MTRT_MemRefValue hostBuffer = createHostMemRef(data, shape, strides);

  MTRT_MemRefValue deviceBuffer{nullptr};
  status = mtrtCopyFromHostToDevice(hostBuffer, device, mtrtStreamGetNull(),
                                    &deviceBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_FALSE(mtrtMemRefValueIsNull(deviceBuffer));

  MTRT_MemRefValue finalBuffer{nullptr};
  status = mtrtCopyFromDeviceToNewHostMemRef(deviceBuffer, mtrtStreamGetNull(),
                                             &finalBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_FALSE(mtrtMemRefValueIsNull(finalBuffer));

  status = mtrtMemRefValueDestroy(finalBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  status = mtrtMemRefValueDestroy(deviceBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  destroyMemRef(hostBuffer);

  status = mtrtStreamDestroy(stream);
  ASSERT_TRUE(mtrtStatusIsOk(status));
}
#endif

TEST(RuntimeCAPI, TestScalarType) {
  MTRT_Type type;
  MTRT_Status status = mtrtScalarTypeCreate(MTRT_ScalarTypeCode_f32, &type);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  ASSERT_TRUE(mtrtTypeIsaScalarType(type));

  MTRT_ScalarType scalarType = mtrtTypeGetScalarType(type);
  ASSERT_FALSE(mtrtScalarTypeIsNull(scalarType));

  ASSERT_EQ(mtrtScalarTypeGetCode(scalarType), MTRT_ScalarTypeCode_f32);
}

TEST(RuntimeCAPI, TestMemRefType) {
  MTRT_Type type;
  std::vector<int64_t> shape = {1, 2, 3};
  MTRT_Status status =
      mtrtMemRefTypeCreate(shape.size(), shape.data(), MTRT_ScalarTypeCode_f32,
                           MTRT_PointerType_host, &type);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  ASSERT_TRUE(mtrtTypeIsaMemRefType(type));

  MTRT_MemRefTypeInfo info;
  mtrtMemRefTypeGetInfo(type, &info);

  ASSERT_EQ(static_cast<unsigned>(info.rank), shape.size());
  ASSERT_EQ(info.shape[0], 1);
  ASSERT_EQ(info.shape[1], 2);
  ASSERT_EQ(info.shape[2], 3);
  ASSERT_EQ(info.elementType, MTRT_ScalarTypeCode_f32);
  ASSERT_EQ(info.addressSpace, MTRT_PointerType_host);

  MTRT_MemRefType memrefType = mtrtTypeGetMemRefType(type);
  ASSERT_FALSE(mtrtMemRefTypeIsNull(memrefType));
}
