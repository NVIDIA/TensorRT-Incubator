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

static constexpr int64_t kBitsPerByte = 8;

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

TEST(RuntimeCAPI, TestStreamCreate) {
  MTRT_Stream stream;
  MTRT_Status streamStatus = mtrtStreamCreate(&stream);
  ASSERT_TRUE(mtrtStatusIsOk(streamStatus));
  ASSERT_TRUE(!mtrtStreamIsNull(stream));
  mtrtStreamDestroy(stream);
}

TEST(RuntimeCAPI, TestClientCreate) {
  MTRT_Stream stream;
  MTRT_Status streamStatus = mtrtStreamCreate(&stream);
  ASSERT_TRUE(mtrtStatusIsOk(streamStatus));
  ASSERT_TRUE(!mtrtStreamIsNull(stream));

  MTRT_RuntimeClient client;
  mtrtRuntimeClientCreate(&client);
  ASSERT_TRUE(!mtrtRuntimeClientIsNull(client));

  mtrtRuntimeClientDestroy(client);
  mtrtStreamDestroy(stream);
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

  status = mtrtRuntimeClientDestroy(client);
  ASSERT_TRUE(mtrtStatusIsOk(status));
}

TEST(RuntimeCAPI, TestHostBufferCreateExternalAndTracking) {
  MTRT_RuntimeClient client;
  MTRT_Status status = mtrtRuntimeClientCreate(&client);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_TRUE(!mtrtRuntimeClientIsNull(client));

  int32_t numDevices;
  status = mtrtRuntimeClientGetNumDevices(client, &numDevices);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  std::vector<int64_t> shape{2, 2};
  std::vector<int64_t> strides{2, 1};
  std::vector<float> data(4);
  std::iota(data.begin(), data.end(), 0.0);
  MTRT_MemRefValue buffer{nullptr};
  MTRT_PointerType addressSpace{MTRT_PointerType::MTRT_PointerType_host};
  status = mtrtMemRefCreateExternal(
      client, addressSpace, sizeof(float) * kBitsPerByte,
      reinterpret_cast<uintptr_t>(data.data()), /*offsetInElements=*/0,
      shape.size(), shape.data(), strides.data(), mtrtDeviceGetNull(),
      MTRT_ScalarTypeCode_unknown, &buffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_FALSE(mtrtMemRefValueIsNull(buffer));

  MTRT_MemRefValueInfo info;
  status = mtrtMemRefValueGetInfo(buffer, &info);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  ASSERT_EQ(llvm::ArrayRef(info.shape, info.rank), llvm::ArrayRef(shape));
  ASSERT_EQ(llvm::ArrayRef(info.strides, info.rank), llvm::ArrayRef(strides));
  ASSERT_EQ(info.addressSpace, addressSpace);
  ASSERT_EQ(info.rank, 2);
  ASSERT_EQ(info.bitsPerElement, 32);

  status = mtrtMemRefValueDestroy(buffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  status = mtrtRuntimeClientDestroy(client);
  ASSERT_TRUE(mtrtStatusIsOk(status));
}

TEST(RuntimeCAPI, TestHostToHostCopy) {
  MTRT_RuntimeClient client;
  MTRT_Status status = mtrtRuntimeClientCreate(&client);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_TRUE(!mtrtRuntimeClientIsNull(client));

  std::vector<float> data{1, 2, 3, 4};
  std::vector<int64_t> shape{2, 2};
  std::vector<int64_t> strides{2, 1};
  MTRT_MemRefValue hostBuffer;
  status = mtrtMemRefCreateExternal(
      client, MTRT_PointerType::MTRT_PointerType_host,
      sizeof(float) * kBitsPerByte, reinterpret_cast<uintptr_t>(data.data()),
      /*offset=*/0, shape.size(), shape.data(), strides.data(),
      mtrtDeviceGetNull(), MTRT_ScalarTypeCode_unknown, &hostBuffer);

  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_FALSE(mtrtMemRefValueIsNull(hostBuffer));

  MTRT_MemRefValue newHostBuffer;
  status = mtrtCopyFromHostToHost(hostBuffer, &newHostBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_FALSE(mtrtMemRefValueIsNull(newHostBuffer));

  status = mtrtMemRefValueDestroy(hostBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  status = mtrtMemRefValueDestroy(newHostBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  status = mtrtRuntimeClientDestroy(client);
  ASSERT_TRUE(mtrtStatusIsOk(status));
}

TEST(RuntimeCAPI, TestHostToDeviceAndBackCopy) {

  MTRT_RuntimeClient client;
  MTRT_Status status = mtrtRuntimeClientCreate(&client);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_TRUE(!mtrtRuntimeClientIsNull(client));

  MTRT_Stream stream;
  status = mtrtStreamCreate(&stream);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  int numDevices{0};
  status = mtrtRuntimeClientGetNumDevices(client, &numDevices);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  if (numDevices < 1) {
    mtrtStreamDestroy(stream);
    mtrtRuntimeClientDestroy(client);
    return;
  }

  MTRT_Device device;
  status = mtrtRuntimeClientGetDevice(client, 0, &device);
  ASSERT_TRUE(mtrtStatusIsOk(status));
  ASSERT_TRUE(!mtrtDeviceIsNull(device));

  std::vector<float> data{1, 2, 3, 4};
  std::vector<int64_t> shape{2, 2};
  std::vector<int64_t> strides{2, 1};
  MTRT_MemRefValue hostBuffer;
  status = mtrtMemRefCreateExternal(
      client, MTRT_PointerType::MTRT_PointerType_host,
      sizeof(float) * kBitsPerByte, reinterpret_cast<uintptr_t>(data.data()),
      /*offset=*/0, shape.size(), shape.data(), strides.data(),
      mtrtDeviceGetNull(), MTRT_ScalarTypeCode_unknown, &hostBuffer);

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

  status = mtrtMemRefValueDestroy(hostBuffer);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  status = mtrtStreamDestroy(stream);
  ASSERT_TRUE(mtrtStatusIsOk(status));

  status = mtrtRuntimeClientDestroy(client);
  ASSERT_TRUE(mtrtStatusIsOk(status));
}

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
