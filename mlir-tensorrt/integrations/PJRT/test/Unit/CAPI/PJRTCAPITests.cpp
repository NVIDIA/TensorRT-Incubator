//===- PJRTCAPITests.cpp --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Tests for the PJRT plugin that require multiple GPUs.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt-pjrt/CAPI/API.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "llvm/ADT/ScopeExit.h"
#include "gtest/gtest.h"
#include <chrono>
#include <thread>

static constexpr std::string_view kMlirFormat = "mlir";

static constexpr std::string_view kStablehloAddOneProgram =
    R"(
module @test_module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.add %arg0, %cst : tensor<f32>
    return %1 : tensor<f32>
  }
})";

static void destroyError(const PJRT_Api *api, PJRT_Error *err) {
  PJRT_Error_Destroy_Args args;
  args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  args.extension_start = 0;
  args.error = err;
  api->PJRT_Error_Destroy(&args);
}

static std::string takeError(const PJRT_Api *api, PJRT_Error *err) {
  PJRT_Error_Message_Args args;
  args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  args.error = err;
  args.message_size = 0;
  args.message = nullptr;
  api->PJRT_Error_Message(&args);
  std::string result(args.message, args.message_size);
  destroyError(api, err);
  return result;
}

static mtrt::StatusOr<const PJRT_Api *> getApiImpl() {
  const PJRT_Api *api = GetPjrtApi();

  PJRT_Plugin_Initialize_Args args;
  args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  PJRT_Error *e = api->PJRT_Plugin_Initialize(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus("failed to initialize plugin: {0}",
                                        takeError(api, e));
  return api;
}

static mtrt::StatusOr<PJRT_Client *> getPJRTClient(const PJRT_Api *api) {

  PJRT_Client_Create_Args create_arg;
  create_arg.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create_arg.extension_start = nullptr;
  create_arg.client = nullptr;
  create_arg.create_options = nullptr;
  create_arg.num_options = 0;

  PJRT_Error *e = api->PJRT_Client_Create(&create_arg);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus("failed to create client: {0}",
                                        takeError(api, e));

  return mtrt::StatusOr<PJRT_Client *>(create_arg.client);
}

static mtrt::Status destroyClient(const PJRT_Api *api, PJRT_Client *client) {
  PJRT_Client_Destroy_Args args;
  args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = client;

  PJRT_Error *e = api->PJRT_Client_Destroy(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus("failed to destroy client: {0}",
                                        takeError(api, e));
  return mtrt::getOkStatus();
}

static mtrt::Status destroyBuffer(const PJRT_Api *api, PJRT_Buffer *buffer) {
  PJRT_Buffer_Destroy_Args args;
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;

  PJRT_Error *e = api->PJRT_Buffer_Destroy(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus("failed to destroy buffer: {0}",
                                        takeError(api, e));
  return mtrt::getOkStatus();
}

static mtrt::StatusOr<std::vector<PJRT_Device *>>
getAddressableDevices(const PJRT_Api *api, PJRT_Client *client) {

  PJRT_Client_AddressableDevices_Args args;
  args.struct_size = PJRT_Client_AddressableMemories_Args_STRUCT_SIZE;
  args.extension_start = 0;
  args.addressable_devices = nullptr;
  args.num_addressable_devices = 0;
  args.client = client;

  PJRT_Error *e = api->PJRT_Client_AddressableDevices(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus(
        "failed to get addressable devices: {0}", takeError(api, e));

  return std::vector<PJRT_Device *>(args.addressable_devices,
                                    args.addressable_devices +
                                        args.num_addressable_devices);
}

static mtrt::StatusOr<PJRT_Memory *> getDeviceMemory(const PJRT_Api *api,
                                                     PJRT_Device *device) {
  PJRT_Device_DefaultMemory_Args args;
  args.struct_size = PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device = device;
  PJRT_Error *e = api->PJRT_Device_DefaultMemory(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus(
        "failed to get device default memory: {0}", takeError(api, e));
  return args.memory;
}

static mtrt::StatusOr<PJRT_Buffer *> copyBufferToMemory(const PJRT_Api *api,
                                                        PJRT_Buffer *buffer,
                                                        PJRT_Memory *memory) {
  PJRT_Buffer_CopyToMemory_Args args;
  args.struct_size = PJRT_Buffer_CopyToMemory_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.dst_memory = memory;
  args.buffer = buffer;
  PJRT_Error *e = api->PJRT_Buffer_CopyToMemory(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus("failed to copy buffer to memory: {0}",
                                        takeError(api, e));
  return args.dst_buffer;
}

static mtrt::Status eventOnReady(const PJRT_Api *api, PJRT_Event *event,
                                 std::atomic<bool> &ready) {
  PJRT_Event_OnReady_Args args;
  args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.event = event;
  args.user_arg = new std::function<void(PJRT_Error *)>(
      [api, &ready](PJRT_Error *error) -> void {
        ready = true;
        // delete error.
        if (error != nullptr) {
          std::cerr << "event has error status: " << takeError(api, error)
                    << std::endl;
        }
      });
  args.callback = [](PJRT_Error *error, void *callback_ptr) {
    auto callback =
        static_cast<std::function<void(PJRT_Error *)> *>(callback_ptr);
    assert(callback != nullptr);
    (*callback)(error);
    delete callback;
  };

  PJRT_Error *e = api->PJRT_Event_OnReady(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus("failed to run PJRT_Event_OnReady: {0}",
                                        takeError(api, e));
  return mtrt::getOkStatus();
}

static mtrt::StatusOr<PJRT_Buffer *> bufferFromHostBuffer(
    const PJRT_Api *api, PJRT_Client *client, const void *data,
    const std::vector<int64_t> &shape, PJRT_Buffer_Type elementType,
    const std::vector<int64_t> &byteStrides, PJRT_Memory *destMemory) {

  PJRT_Client_BufferFromHostBuffer_Args args;
  args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = client;
  args.data = data;
  args.type = elementType;

  PJRT_Buffer_MemoryLayout_Strides strides;
  strides.struct_size = PJRT_Buffer_MemoryLayout_Strides_STRUCT_SIZE;
  strides.extension_start = 0;
  strides.byte_strides = byteStrides.data();
  strides.num_byte_strides = byteStrides.size();

  args.dims = shape.data();
  args.num_dims = shape.size();
  args.byte_strides = byteStrides.data();
  args.num_byte_strides = byteStrides.size();

  args.host_buffer_semantics = PJRT_HostBufferSemantics::
      PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;

  args.device = nullptr;
  args.memory = destMemory;
  args.device_layout = nullptr;

  PJRT_Error *e = api->PJRT_Client_BufferFromHostBuffer(&args);

  if (e != nullptr)
    return mtrt::getInternalErrorStatus("failed to create buffer: {0}",
                                        takeError(api, e));

  std::atomic<bool> bufferReady{false};
  mtrt::Status s = eventOnReady(api, args.done_with_host_buffer, bufferReady);
  if (!s.isOk())
    return s.getStatus();

  // wait for buffer to become ready.
  while (!bufferReady) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }

  return args.buffer;
}

static mtrt::StatusOr<PJRT_Buffer *> bufferFromHostBuffer(
    const PJRT_Api *api, PJRT_Client *client, const std::vector<float> &data,
    const std::vector<int64_t> &shape, const std::vector<int64_t> &byteStrides,
    PJRT_Memory *destMemory) {
  return bufferFromHostBuffer(api, client, data.data(), shape,
                              PJRT_Buffer_Type::PJRT_Buffer_Type_F32,
                              byteStrides, destMemory);
}

static mtrt::StatusOr<PJRT_Buffer_Type>
getBufferElementType(const PJRT_Api *api, PJRT_Buffer *buffer) {
  PJRT_Buffer_ElementType_Args args;
  args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;

  PJRT_Error *e = api->PJRT_Buffer_ElementType(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus(
        "failed to get buffer element type: {0}", takeError(api, e));
  return args.type;
}

static mtrt::StatusOr<size_t>
getBufferOnDeviceSizeInBytes(const PJRT_Api *api, PJRT_Buffer *buffer) {
  PJRT_Buffer_OnDeviceSizeInBytes_Args args;
  args.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;

  PJRT_Error *e = api->PJRT_Buffer_OnDeviceSizeInBytes(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus(
        "failed to get buffer on device size in bytes: {0}", takeError(api, e));
  return args.on_device_size_in_bytes;
}

static mtrt::Status destroyLoadedExecutable(const PJRT_Api *api,
                                            PJRT_LoadedExecutable *executable) {

  PJRT_LoadedExecutable_Destroy_Args args;
  args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = executable;
  PJRT_Error *error = api->PJRT_LoadedExecutable_Destroy(&args);
  if (error != nullptr)
    return mtrt::getInternalErrorStatus("failed to destroy executable: {0}",
                                        takeError(api, error));
  return mtrt::getOkStatus();
}

namespace {
class PJRTCApiTest : public ::testing::Test {
public:
  PJRTCApiTest()
      : api(getApiImpl()),
        client(api.isError() ? nullptr : getPJRTClient(*api)) {}

  void SetUp() override {
    ASSERT_TRUE(api.isOk()) << api.getStatus();
    ASSERT_TRUE(client.isOk()) << client.getStatus();
  }

  ~PJRTCApiTest() override {
    if (client.isOk())
      mtrt::cantFail(destroyClient(*api, *client));
  }

  PJRTCApiTest(const PJRTCApiTest &) = delete;
  void operator=(const PJRTCApiTest &) = delete;

  const PJRT_Api *getApi() const { return *api; }
  PJRT_Client *getClient() const { return *client; }

protected:
  mtrt::StatusOr<const PJRT_Api *> api;
  mtrt::StatusOr<PJRT_Client *> client;
};
} // namespace

// Tests PJRT_Buffer_BufferFromHostBuffer, PJRT_Buffer_ElementType,
// PJRT_Buffer_Destroy, PJRT_Buffer_OnDeviceSizeInBytes.
TEST_F(PJRTCApiTest, PJRT_Buffer_BufferFromHostBuffer) {
  mtrt::StatusOr<std::vector<PJRT_Device *>> devices =
      getAddressableDevices(getApi(), getClient());
  ASSERT_TRUE(devices.isOk()) << devices.getStatus();

  if (devices->size() < 1) {
    GTEST_SKIP() << "requires >=1 GPUs";
  }

  mtrt::StatusOr<PJRT_Memory *> memory =
      getDeviceMemory(*api, devices->front());
  ASSERT_TRUE(memory.isOk()) << memory.getStatus();

  // F32 buffer with some data.
  {
    std::vector<float> data{1.0, 2.0, 3.0, 4.0};

    mtrt::StatusOr<PJRT_Buffer *> buffer =
        bufferFromHostBuffer(*api, *client, data, {4}, {4}, *memory);
    ASSERT_TRUE(buffer.isOk()) << buffer.getStatus();

    mtrt::StatusOr<PJRT_Buffer_Type> elType =
        getBufferElementType(*api, *buffer);
    ASSERT_TRUE(elType.isOk()) << elType.getStatus();
    ASSERT_EQ(*elType, PJRT_Buffer_Type::PJRT_Buffer_Type_F32);

    mtrt::StatusOr<size_t> onDeviceSize =
        getBufferOnDeviceSizeInBytes(*api, *buffer);
    ASSERT_TRUE(onDeviceSize.isOk()) << onDeviceSize.getStatus();
    ASSERT_EQ(*onDeviceSize, 16U);

    mtrt::Status s = destroyBuffer(*api, *buffer);
    ASSERT_TRUE(s.isOk()) << s.getStatus();
  }

  // Scalar f32 buffer with with valid pointer.
  {
    std::vector<float> data{1.0, 2.0, 3.0, 4.0};
    mtrt::StatusOr<PJRT_Buffer *> buffer =
        bufferFromHostBuffer(*api, *client, data, {0}, {0}, *memory);
    ASSERT_TRUE(buffer.isOk()) << buffer.getStatus();

    mtrt::StatusOr<PJRT_Buffer_Type> elType =
        getBufferElementType(*api, *buffer);
    ASSERT_TRUE(elType.isOk()) << elType.getStatus();
    ASSERT_EQ(*elType, PJRT_Buffer_Type::PJRT_Buffer_Type_F32);

    mtrt::StatusOr<size_t> onDeviceSize =
        getBufferOnDeviceSizeInBytes(*api, *buffer);
    ASSERT_TRUE(onDeviceSize.isOk()) << onDeviceSize.getStatus();
    ASSERT_EQ(*onDeviceSize, 0U);

    mtrt::Status s = destroyBuffer(*api, *buffer);
    ASSERT_TRUE(s.isOk()) << s.getStatus();
  }

  // Empty f32 buffer with nullptr for data.
  {
    mtrt::StatusOr<PJRT_Buffer *> buffer = bufferFromHostBuffer(
        *api, *client, nullptr, {0}, PJRT_Buffer_Type_F32, {0}, *memory);
    ASSERT_TRUE(buffer.isOk()) << buffer.getStatus();

    mtrt::StatusOr<PJRT_Buffer_Type> elType =
        getBufferElementType(*api, *buffer);
    ASSERT_TRUE(elType.isOk()) << elType.getStatus();
    ASSERT_EQ(*elType, PJRT_Buffer_Type::PJRT_Buffer_Type_F32);

    mtrt::StatusOr<size_t> onDeviceSize =
        getBufferOnDeviceSizeInBytes(*api, *buffer);
    ASSERT_TRUE(onDeviceSize.isOk()) << onDeviceSize.getStatus();
    ASSERT_EQ(*onDeviceSize, 0U);

    mtrt::Status s = destroyBuffer(*api, *buffer);
    ASSERT_TRUE(s.isOk()) << s.getStatus();
  }

  // Valid i4 buffer.
  {
    std::vector<int8_t> data{1, 2, 3, 4};
    mtrt::StatusOr<PJRT_Buffer *> buffer = bufferFromHostBuffer(
        *api, *client, data.data(), {4}, PJRT_Buffer_Type_S4, {1}, *memory);
    ASSERT_TRUE(buffer.isOk()) << buffer.getStatus();

    mtrt::StatusOr<PJRT_Buffer_Type> elType =
        getBufferElementType(*api, *buffer);
    ASSERT_TRUE(elType.isOk()) << elType.getStatus();
    ASSERT_EQ(*elType, PJRT_Buffer_Type::PJRT_Buffer_Type_S4);

    mtrt::StatusOr<size_t> onDeviceSize =
        getBufferOnDeviceSizeInBytes(*api, *buffer);
    ASSERT_TRUE(onDeviceSize.isOk()) << onDeviceSize.getStatus();
    ASSERT_EQ(*onDeviceSize, 4U);

    mtrt::Status s = destroyBuffer(*api, *buffer);
    ASSERT_TRUE(s.isOk()) << s.getStatus();
  }
}

TEST(PJRTPluginCAPI, PJRT_Buffer_CopyToMemory) {
  mtrt::StatusOr<const PJRT_Api *> api = getApiImpl();
  ASSERT_TRUE(api.isOk()) << api.getStatus();

  mtrt::StatusOr<PJRT_Client *> client = getPJRTClient(*api);
  ASSERT_TRUE(client.isOk()) << client.getStatus();
  auto clientCleanup = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyClient(*api, *client);
    ASSERT_TRUE(s.isOk()) << s.getStatus();
  });

  mtrt::StatusOr<std::vector<PJRT_Device *>> devices =
      getAddressableDevices(*api, *client);
  ASSERT_TRUE(devices.isOk()) << devices.getStatus();

  if (devices->size() < 2) {
    GTEST_SKIP() << "requires >=2 GPUs";
  }

  mtrt::StatusOr<PJRT_Memory *> memorySrc =
      getDeviceMemory(*api, (*devices)[0]);
  mtrt::StatusOr<PJRT_Memory *> memoryDst =
      getDeviceMemory(*api, (*devices)[1]);

  ASSERT_TRUE(memorySrc.isOk()) << memorySrc.getStatus();
  ASSERT_TRUE(memoryDst.isOk()) << memoryDst.getStatus();

  std::vector<float> data{1.0, 2.0, 3.0, 4.0};

  mtrt::StatusOr<PJRT_Buffer *> buffer =
      bufferFromHostBuffer(*api, *client, data, {4}, {4}, *memorySrc);
  ASSERT_TRUE(buffer.isOk()) << buffer.getStatus();

  auto cleanupBuffer = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyBuffer(*api, *buffer);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });

  mtrt::StatusOr<PJRT_Buffer *> secondBuffer =
      copyBufferToMemory(*api, *buffer, *memoryDst);
  ASSERT_TRUE(secondBuffer.isOk()) << secondBuffer.getStatus();

  auto cleanupBuffer2 = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyBuffer(*api, *secondBuffer);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });
}

TEST(PJRTPluginCAPI, Buffer_IsDeleted) {
  mtrt::StatusOr<const PJRT_Api *> api = getApiImpl();
  ASSERT_TRUE(api.isOk()) << api.getStatus();

  mtrt::StatusOr<PJRT_Client *> client = getPJRTClient(*api);
  ASSERT_TRUE(client.isOk()) << client.getStatus();
  auto clientCleanup = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyClient(*api, *client);
    ASSERT_TRUE(s.isOk()) << s.getStatus();
  });

  mtrt::StatusOr<std::vector<PJRT_Device *>> devices =
      getAddressableDevices(*api, *client);
  ASSERT_TRUE(devices.isOk()) << devices.getStatus();

  if (devices->size() < 1) {
    GTEST_SKIP() << "requires >=1 GPUs";
  }

  std::vector<float> data{1.0, 2.0, 3.0, 4.0};

  mtrt::StatusOr<PJRT_Memory *> memory =
      getDeviceMemory(*api, devices->front());
  ASSERT_TRUE(memory.isOk()) << memory.getStatus();

  mtrt::StatusOr<PJRT_Buffer *> buffer =
      bufferFromHostBuffer(*api, *client, data, {4}, {4}, *memory);
  ASSERT_TRUE(buffer.isOk()) << buffer.getStatus();

  auto cleanupBuffer = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyBuffer(*api, *buffer);
    ASSERT_TRUE(s.isOk()) << s.getStatus();
  });

  PJRT_Buffer_IsDeleted_Args is_deleted_args;
  is_deleted_args.struct_size = PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
  is_deleted_args.extension_start = nullptr;
  is_deleted_args.buffer = *buffer;
  PJRT_Error *is_deleted_error =
      (*api)->PJRT_Buffer_IsDeleted(&is_deleted_args);
  ASSERT_EQ(is_deleted_error, nullptr);
  ASSERT_FALSE(is_deleted_args.is_deleted);

  PJRT_Buffer_Delete_Args delete_args;
  delete_args.struct_size = PJRT_Buffer_Delete_Args_STRUCT_SIZE;
  delete_args.extension_start = nullptr;
  delete_args.buffer = *buffer;
  PJRT_Error *delete_error = (*api)->PJRT_Buffer_Delete(&delete_args);
  ASSERT_EQ(delete_error, nullptr);

  is_deleted_error = (*api)->PJRT_Buffer_IsDeleted(&is_deleted_args);
  ASSERT_EQ(is_deleted_error, nullptr);
  ASSERT_TRUE(is_deleted_args.is_deleted);
}

TEST_F(PJRTCApiTest, Client_Compile) {
  PJRT_Client_Compile_Args args;
  args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = getClient();

  // Since TensorRT may be required, we only test compilation if a GPU is
  // present.
  mtrt::StatusOr<std::vector<PJRT_Device *>> devices =
      getAddressableDevices(*api, *client);
  ASSERT_TRUE(devices.isOk()) << devices.getStatus();
  if (devices->size() < 1)
    GTEST_SKIP() << "requires >=1 GPUs";

  std::string options_str = "";
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  std::string format(kMlirFormat);
  std::string program_code{kStablehloAddOneProgram};
  PJRT_Program program;
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.extension_start = nullptr;
  program.code = program_code.data();
  program.code_size = program_code.length();
  program.format = format.c_str();
  program.format_size = format.size();
  args.program = &program;

  PJRT_Error *error = getApi()->PJRT_Client_Compile(&args);
  ASSERT_EQ(error, nullptr)
      << "failed to compile program: " << takeError(getApi(), error);

  // Check `is_deleted` is false.
  PJRT_LoadedExecutable_IsDeleted_Args isDeleted;
  isDeleted.struct_size = PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE;
  isDeleted.extension_start = nullptr;
  isDeleted.executable = args.executable;
  PJRT_Error *isDeletedStatus =
      (*api)->PJRT_LoadedExecutable_IsDeleted(&isDeleted);
  ASSERT_EQ(isDeletedStatus, nullptr) << takeError(getApi(), isDeletedStatus);
  ASSERT_FALSE(isDeleted.is_deleted);

  mtrt::Status s = destroyLoadedExecutable(getApi(), args.executable);
  ASSERT_TRUE(s.isOk()) << s.getStatus();
}

TEST_F(PJRTCApiTest, Client_Compile_concurrent) {
  // Since TensorRT may be required, we only test compilation if a GPU is
  // present.
  mtrt::StatusOr<std::vector<PJRT_Device *>> devices =
      getAddressableDevices(*api, *client);
  ASSERT_TRUE(devices.isOk()) << devices.getStatus();
  if (devices->size() < 1)
    GTEST_SKIP() << "requires >=1 GPUs";

  std::atomic<uint32_t> numErrors = 0;
  auto doCompile = [&]() {
    std::string options_str = "";
    PJRT_Client_Compile_Args args;
    args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.client = getClient();
    args.compile_options = options_str.c_str();
    args.compile_options_size = options_str.size();

    std::string format(kMlirFormat);
    std::string program_code{kStablehloAddOneProgram};
    PJRT_Program program;
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.extension_start = nullptr;
    program.code = program_code.data();
    program.code_size = program_code.length();
    program.format = format.c_str();
    program.format_size = format.size();
    args.program = &program;

    PJRT_Error *error = getApi()->PJRT_Client_Compile(&args);
    if (error != nullptr) {
      std::cerr << "failed to compile program: " << takeError(getApi(), error)
                << std::endl;
      numErrors++;
      return;
    }

    mtrt::Status s = destroyLoadedExecutable(getApi(), args.executable);
    if (!s.isOk()) {
      std::cerr << "failed to destroy executable: " << s.getMessage()
                << std::endl;
      numErrors++;
    }
  };

  std::vector<std::thread> threads;
  for (uint32_t i = 0; i < 4; i++)
    threads.emplace_back(doCompile);
  for (auto &thread : threads)
    thread.join();

  ASSERT_EQ(numErrors, 0U) << "expected no errors";
}
