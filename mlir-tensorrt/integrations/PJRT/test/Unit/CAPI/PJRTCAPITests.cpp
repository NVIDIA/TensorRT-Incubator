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

// Helper to reduce verbosity when setting up PJRT C API structs.
// Most PJRT entrypoints require `.struct_size` and `.extension_start` to be
// set. `T##_STRUCT_SIZE` is defined by the PJRT C API headers for each struct
// `T`.
#define MTRT_PJRT_INIT_STRUCT(T, var)                                          \
  T var{};                                                                     \
  var.struct_size = T##_STRUCT_SIZE;                                           \
  var.extension_start = nullptr

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
  MTRT_PJRT_INIT_STRUCT(PJRT_Error_Destroy_Args, args);
  args.error = err;
  api->PJRT_Error_Destroy(&args);
}

static std::string takeError(const PJRT_Api *api, PJRT_Error *err) {
  MTRT_PJRT_INIT_STRUCT(PJRT_Error_Message_Args, args);
  args.error = err;
  args.message_size = 0;
  args.message = nullptr;
  api->PJRT_Error_Message(&args);
  std::string result(args.message, args.message_size);
  destroyError(api, err);
  return result;
}

static mtrt::StatusOr<PJRT_LoadedExecutable *>
compileTestProgram(const PJRT_Api *api, PJRT_Client *client,
                   std::string_view programText,
                   std::string_view programFormat = kMlirFormat,
                   std::string_view compileOptions = "") {
  MTRT_PJRT_INIT_STRUCT(PJRT_Client_Compile_Args, args);
  args.client = client;

  std::string optionsStr(compileOptions);
  args.compile_options = optionsStr.c_str();
  args.compile_options_size = optionsStr.size();

  std::string formatStr(programFormat);
  std::string programCode(programText);

  MTRT_PJRT_INIT_STRUCT(PJRT_Program, program);
  program.code = programCode.data();
  program.code_size = programCode.length();
  program.format = formatStr.c_str();
  program.format_size = formatStr.size();
  args.program = &program;

  PJRT_Error *error = api->PJRT_Client_Compile(&args);
  if (error != nullptr)
    return mtrt::getInternalErrorStatus("failed to compile program: {0}",
                                        takeError(api, error));

  return args.executable;
}

static mtrt::StatusOr<const PJRT_Api *> getApiImpl() {
  const PJRT_Api *api = GetPjrtApi();

  MTRT_PJRT_INIT_STRUCT(PJRT_Plugin_Initialize_Args, args);
  PJRT_Error *e = api->PJRT_Plugin_Initialize(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus("failed to initialize plugin: {0}",
                                        takeError(api, e));
  return api;
}

static mtrt::StatusOr<PJRT_Client *> getPJRTClient(const PJRT_Api *api) {

  MTRT_PJRT_INIT_STRUCT(PJRT_Client_Create_Args, create_arg);
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
  MTRT_PJRT_INIT_STRUCT(PJRT_Client_Destroy_Args, args);
  args.client = client;

  PJRT_Error *e = api->PJRT_Client_Destroy(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus("failed to destroy client: {0}",
                                        takeError(api, e));
  return mtrt::getOkStatus();
}

static mtrt::Status destroyBuffer(const PJRT_Api *api, PJRT_Buffer *buffer) {
  MTRT_PJRT_INIT_STRUCT(PJRT_Buffer_Destroy_Args, args);
  args.buffer = buffer;

  PJRT_Error *e = api->PJRT_Buffer_Destroy(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus("failed to destroy buffer: {0}",
                                        takeError(api, e));
  return mtrt::getOkStatus();
}

static mtrt::StatusOr<std::vector<PJRT_Device *>>
getAddressableDevices(const PJRT_Api *api, PJRT_Client *client) {

  MTRT_PJRT_INIT_STRUCT(PJRT_Client_AddressableDevices_Args, args);
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
  MTRT_PJRT_INIT_STRUCT(PJRT_Device_DefaultMemory_Args, args);
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
  MTRT_PJRT_INIT_STRUCT(PJRT_Buffer_CopyToMemory_Args, args);
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
  MTRT_PJRT_INIT_STRUCT(PJRT_Event_OnReady_Args, args);
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

  MTRT_PJRT_INIT_STRUCT(PJRT_Client_BufferFromHostBuffer_Args, args);
  args.client = client;
  args.data = data;
  args.type = elementType;

  MTRT_PJRT_INIT_STRUCT(PJRT_Buffer_MemoryLayout_Strides, strides);
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
  MTRT_PJRT_INIT_STRUCT(PJRT_Buffer_ElementType_Args, args);
  args.buffer = buffer;

  PJRT_Error *e = api->PJRT_Buffer_ElementType(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus(
        "failed to get buffer element type: {0}", takeError(api, e));
  return args.type;
}

static mtrt::StatusOr<size_t>
getBufferOnDeviceSizeInBytes(const PJRT_Api *api, PJRT_Buffer *buffer) {
  MTRT_PJRT_INIT_STRUCT(PJRT_Buffer_OnDeviceSizeInBytes_Args, args);
  args.buffer = buffer;

  PJRT_Error *e = api->PJRT_Buffer_OnDeviceSizeInBytes(&args);
  if (e != nullptr)
    return mtrt::getInternalErrorStatus(
        "failed to get buffer on device size in bytes: {0}", takeError(api, e));
  return args.on_device_size_in_bytes;
}

static mtrt::Status destroyLoadedExecutable(const PJRT_Api *api,
                                            PJRT_LoadedExecutable *executable) {

  MTRT_PJRT_INIT_STRUCT(PJRT_LoadedExecutable_Destroy_Args, args);
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

  MTRT_PJRT_INIT_STRUCT(PJRT_Buffer_IsDeleted_Args, is_deleted_args);
  is_deleted_args.buffer = *buffer;
  PJRT_Error *is_deleted_error =
      (*api)->PJRT_Buffer_IsDeleted(&is_deleted_args);
  ASSERT_EQ(is_deleted_error, nullptr);
  ASSERT_FALSE(is_deleted_args.is_deleted);

  MTRT_PJRT_INIT_STRUCT(PJRT_Buffer_Delete_Args, delete_args);
  delete_args.buffer = *buffer;
  PJRT_Error *delete_error = (*api)->PJRT_Buffer_Delete(&delete_args);
  ASSERT_EQ(delete_error, nullptr);

  is_deleted_error = (*api)->PJRT_Buffer_IsDeleted(&is_deleted_args);
  ASSERT_EQ(is_deleted_error, nullptr);
  ASSERT_TRUE(is_deleted_args.is_deleted);
}

TEST_F(PJRTCApiTest, Client_Compile) {
  // Since TensorRT may be required, we only test compilation if a GPU is
  // present.
  mtrt::StatusOr<std::vector<PJRT_Device *>> devices =
      getAddressableDevices(*api, *client);
  ASSERT_TRUE(devices.isOk()) << devices.getStatus();
  if (devices->size() < 1)
    GTEST_SKIP() << "requires >=1 GPUs";

  mtrt::StatusOr<PJRT_LoadedExecutable *> executable =
      compileTestProgram(getApi(), getClient(), kStablehloAddOneProgram);
  ASSERT_TRUE(executable.isOk()) << executable.getStatus();

  // Check `is_deleted` is false.
  MTRT_PJRT_INIT_STRUCT(PJRT_LoadedExecutable_IsDeleted_Args, isDeleted);
  isDeleted.executable = *executable;
  PJRT_Error *isDeletedStatus =
      (*api)->PJRT_LoadedExecutable_IsDeleted(&isDeleted);
  ASSERT_EQ(isDeletedStatus, nullptr) << takeError(getApi(), isDeletedStatus);
  ASSERT_FALSE(isDeleted.is_deleted);

  mtrt::Status s = destroyLoadedExecutable(getApi(), *executable);
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
    mtrt::StatusOr<PJRT_LoadedExecutable *> executable =
        compileTestProgram(getApi(), getClient(), kStablehloAddOneProgram);
    if (!executable.isOk()) {
      std::cerr << executable.getStatus() << std::endl;
      numErrors++;
      return;
    }

    mtrt::Status s = destroyLoadedExecutable(getApi(), *executable);
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

static mtrt::Status destroyExecutable(const PJRT_Api *api,
                                      PJRT_Executable *executable) {
  MTRT_PJRT_INIT_STRUCT(PJRT_Executable_Destroy_Args, args);
  args.executable = executable;
  PJRT_Error *error = api->PJRT_Executable_Destroy(&args);
  if (error != nullptr)
    return mtrt::getInternalErrorStatus("failed to destroy executable: {0}",
                                        takeError(api, error));
  return mtrt::getOkStatus();
}

static mtrt::StatusOr<std::string>
getExecutableFingerprint(const PJRT_Api *api, PJRT_Executable *e) {
  MTRT_PJRT_INIT_STRUCT(PJRT_Executable_Fingerprint_Args, args);
  args.executable = e;
  PJRT_Error *error = api->PJRT_Executable_Fingerprint(&args);
  if (error != nullptr)
    return mtrt::getInternalErrorStatus("failed to get fingerprint: {0}",
                                        takeError(api, error));
  return std::string(args.executable_fingerprint,
                     args.executable_fingerprint_size);
}

static mtrt::StatusOr<std::vector<PJRT_Buffer_Type>>
getExecutableOutputElementTypes(const PJRT_Api *api, PJRT_Executable *e) {
  MTRT_PJRT_INIT_STRUCT(PJRT_Executable_OutputElementTypes_Args, args);
  args.executable = e;
  PJRT_Error *error = api->PJRT_Executable_OutputElementTypes(&args);
  if (error != nullptr)
    return mtrt::getInternalErrorStatus(
        "failed to get output element types: {0}", takeError(api, error));
  return std::vector<PJRT_Buffer_Type>(
      args.output_types, args.output_types + args.num_output_types);
}

static mtrt::StatusOr<std::vector<char>>
serializeExecutable(const PJRT_Api *api, PJRT_Executable *e) {
  MTRT_PJRT_INIT_STRUCT(PJRT_Executable_Serialize_Args, args);
  args.executable = e;
  PJRT_Error *error = api->PJRT_Executable_Serialize(&args);
  if (error != nullptr)
    return mtrt::getInternalErrorStatus("failed to serialize executable: {0}",
                                        takeError(api, error));

  auto cleanup = llvm::make_scope_exit([&]() {
    if (args.serialized_executable_deleter && args.serialized_executable)
      args.serialized_executable_deleter(args.serialized_executable);
  });

  if (args.serialized_bytes_size != 0 && args.serialized_bytes == nullptr)
    return mtrt::getInternalErrorStatus(
        "serialize returned non-zero size but null bytes");

  std::vector<char> out;
  out.assign(args.serialized_bytes,
             args.serialized_bytes + args.serialized_bytes_size);
  return out;
}

static mtrt::StatusOr<PJRT_LoadedExecutable *>
deserializeAndLoadExecutable(const PJRT_Api *api, PJRT_Client *client,
                             const std::vector<char> &bytes) {
  MTRT_PJRT_INIT_STRUCT(PJRT_Executable_DeserializeAndLoad_Args, args);
  args.client = client;
  args.serialized_executable = bytes.data();
  args.serialized_executable_size = bytes.size();
  args.loaded_executable = nullptr;

  PJRT_Error *error = api->PJRT_Executable_DeserializeAndLoad(&args);
  if (error != nullptr)
    return mtrt::getInternalErrorStatus("failed to deserialize and load: {0}",
                                        takeError(api, error));
  return args.loaded_executable;
}

// Test for PJRT_Executable_OutputElementTypes
TEST_F(PJRTCApiTest, Executable_OutputElementTypes) {
  mtrt::StatusOr<std::vector<PJRT_Device *>> devices =
      getAddressableDevices(*api, *client);
  ASSERT_TRUE(devices.isOk()) << devices.getStatus();
  if (devices->size() < 1)
    GTEST_SKIP() << "requires >=1 GPUs";

  // Compile the program
  mtrt::StatusOr<PJRT_LoadedExecutable *> loaded =
      compileTestProgram(getApi(), getClient(), kStablehloAddOneProgram);
  ASSERT_TRUE(loaded.isOk()) << loaded.getStatus();

  auto cleanupLoaded = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyLoadedExecutable(getApi(), *loaded);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });

  // Get the PJRT_Executable from the loaded executable
  MTRT_PJRT_INIT_STRUCT(PJRT_LoadedExecutable_GetExecutable_Args, get_exe_args);
  get_exe_args.loaded_executable = *loaded;
  get_exe_args.executable = nullptr;

  PJRT_Error *error =
      getApi()->PJRT_LoadedExecutable_GetExecutable(&get_exe_args);
  ASSERT_EQ(error, nullptr)
      << "failed to get executable: " << takeError(getApi(), error);

  auto cleanupExe = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyExecutable(getApi(), get_exe_args.executable);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });

  // Test PJRT_Executable_OutputElementTypes
  MTRT_PJRT_INIT_STRUCT(PJRT_Executable_OutputElementTypes_Args,
                        output_types_args);
  output_types_args.executable = get_exe_args.executable;

  error = getApi()->PJRT_Executable_OutputElementTypes(&output_types_args);
  ASSERT_EQ(error, nullptr)
      << "failed to get output element types: " << takeError(getApi(), error);

  // The program has 1 output of type f32
  ASSERT_EQ(output_types_args.num_output_types, 1U);
  ASSERT_NE(output_types_args.output_types, nullptr);
  ASSERT_EQ(output_types_args.output_types[0], PJRT_Buffer_Type_F32);
}

// Test for PJRT_Executable_OutputDimensions
TEST_F(PJRTCApiTest, Executable_OutputDimensions) {
  mtrt::StatusOr<std::vector<PJRT_Device *>> devices =
      getAddressableDevices(*api, *client);
  ASSERT_TRUE(devices.isOk()) << devices.getStatus();
  if (devices->size() < 1)
    GTEST_SKIP() << "requires >=1 GPUs";

  // Compile the program
  mtrt::StatusOr<PJRT_LoadedExecutable *> loaded =
      compileTestProgram(getApi(), getClient(), kStablehloAddOneProgram);
  ASSERT_TRUE(loaded.isOk()) << loaded.getStatus();

  auto cleanupLoaded = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyLoadedExecutable(getApi(), *loaded);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });

  // Get the PJRT_Executable from the loaded executable
  MTRT_PJRT_INIT_STRUCT(PJRT_LoadedExecutable_GetExecutable_Args, get_exe_args);
  get_exe_args.loaded_executable = *loaded;
  get_exe_args.executable = nullptr;

  PJRT_Error *error =
      getApi()->PJRT_LoadedExecutable_GetExecutable(&get_exe_args);
  ASSERT_EQ(error, nullptr)
      << "failed to get executable: " << takeError(getApi(), error);

  auto cleanupExe = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyExecutable(getApi(), get_exe_args.executable);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });

  // Test PJRT_Executable_OutputDimensions
  MTRT_PJRT_INIT_STRUCT(PJRT_Executable_OutputDimensions_Args,
                        output_dims_args);
  output_dims_args.executable = get_exe_args.executable;

  error = getApi()->PJRT_Executable_OutputDimensions(&output_dims_args);
  ASSERT_EQ(error, nullptr)
      << "failed to get output dimensions: " << takeError(getApi(), error);

  // The program has 1 output of shape [] (scalar), so no dims
  ASSERT_EQ(output_dims_args.num_outputs, 1U);
  ASSERT_NE(output_dims_args.dim_sizes, nullptr);
  ASSERT_EQ(output_dims_args.dim_sizes[0], 0U); // scalar has 0 dimensions
}

// Test for PJRT_Executable_Fingerprint
TEST_F(PJRTCApiTest, Executable_Fingerprint) {
  mtrt::StatusOr<std::vector<PJRT_Device *>> devices =
      getAddressableDevices(*api, *client);
  ASSERT_TRUE(devices.isOk()) << devices.getStatus();
  if (devices->size() < 1)
    GTEST_SKIP() << "requires >=1 GPUs";

  // Compile the program
  mtrt::StatusOr<PJRT_LoadedExecutable *> loaded =
      compileTestProgram(getApi(), getClient(), kStablehloAddOneProgram);
  ASSERT_TRUE(loaded.isOk()) << loaded.getStatus();

  auto cleanupLoaded = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyLoadedExecutable(getApi(), *loaded);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });

  // Get the PJRT_Executable from the loaded executable
  MTRT_PJRT_INIT_STRUCT(PJRT_LoadedExecutable_GetExecutable_Args, get_exe_args);
  get_exe_args.loaded_executable = *loaded;
  get_exe_args.executable = nullptr;

  PJRT_Error *error =
      getApi()->PJRT_LoadedExecutable_GetExecutable(&get_exe_args);
  ASSERT_EQ(error, nullptr)
      << "failed to get executable: " << takeError(getApi(), error);

  auto cleanupExe = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyExecutable(getApi(), get_exe_args.executable);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });

  // Test PJRT_Executable_Fingerprint
  MTRT_PJRT_INIT_STRUCT(PJRT_Executable_Fingerprint_Args, fingerprint_args);
  fingerprint_args.executable = get_exe_args.executable;

  error = getApi()->PJRT_Executable_Fingerprint(&fingerprint_args);
  ASSERT_EQ(error, nullptr)
      << "failed to get fingerprint: " << takeError(getApi(), error);

  // The fingerprint should be a non-empty hex string
  ASSERT_NE(fingerprint_args.executable_fingerprint, nullptr);
  ASSERT_GT(fingerprint_args.executable_fingerprint_size, 0U);

  std::string fingerprint(fingerprint_args.executable_fingerprint,
                          fingerprint_args.executable_fingerprint_size);
  // Check it's a valid hex string (at least has some characters)
  ASSERT_FALSE(fingerprint.empty());

  // Calling fingerprint again should return the same value (consistency check)
  MTRT_PJRT_INIT_STRUCT(PJRT_Executable_Fingerprint_Args, fingerprint_args2);
  fingerprint_args2.executable = get_exe_args.executable;

  error = getApi()->PJRT_Executable_Fingerprint(&fingerprint_args2);
  ASSERT_EQ(error, nullptr)
      << "failed to get fingerprint (2nd call): " << takeError(getApi(), error);

  std::string fingerprint2(fingerprint_args2.executable_fingerprint,
                           fingerprint_args2.executable_fingerprint_size);
  ASSERT_EQ(fingerprint, fingerprint2) << "fingerprint should be consistent";
}

TEST_F(PJRTCApiTest, Executable_Serialize_DeserializeAndLoad_RoundTrip) {
  mtrt::StatusOr<std::vector<PJRT_Device *>> devices =
      getAddressableDevices(*api, *client);
  ASSERT_TRUE(devices.isOk()) << devices.getStatus();
  if (devices->size() < 1)
    GTEST_SKIP() << "requires >=1 GPUs";

  // Compile the program.
  mtrt::StatusOr<PJRT_LoadedExecutable *> loaded =
      compileTestProgram(getApi(), getClient(), kStablehloAddOneProgram);
  ASSERT_TRUE(loaded.isOk()) << loaded.getStatus();

  auto cleanupLoaded = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyLoadedExecutable(getApi(), *loaded);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });

  // Get the PJRT_Executable from the loaded executable.
  MTRT_PJRT_INIT_STRUCT(PJRT_LoadedExecutable_GetExecutable_Args, get_exe_args);
  get_exe_args.loaded_executable = *loaded;
  get_exe_args.executable = nullptr;
  PJRT_Error *error =
      getApi()->PJRT_LoadedExecutable_GetExecutable(&get_exe_args);
  ASSERT_EQ(error, nullptr) << takeError(getApi(), error);

  auto cleanupExe = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyExecutable(getApi(), get_exe_args.executable);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });

  // Serialize.
  mtrt::StatusOr<std::vector<char>> bytes =
      serializeExecutable(getApi(), get_exe_args.executable);
  ASSERT_TRUE(bytes.isOk()) << bytes.getStatus();
  ASSERT_GT(bytes->size(), 0U);

  // Deserialize and load.
  mtrt::StatusOr<PJRT_LoadedExecutable *> loaded2 =
      deserializeAndLoadExecutable(getApi(), getClient(), *bytes);
  ASSERT_TRUE(loaded2.isOk()) << loaded2.getStatus();

  auto cleanupLoaded2 = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyLoadedExecutable(getApi(), *loaded2);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });

  // Get executable from loaded2.
  MTRT_PJRT_INIT_STRUCT(PJRT_LoadedExecutable_GetExecutable_Args,
                        get_exe_args2);
  get_exe_args2.loaded_executable = *loaded2;
  get_exe_args2.executable = nullptr;
  error = getApi()->PJRT_LoadedExecutable_GetExecutable(&get_exe_args2);
  ASSERT_EQ(error, nullptr) << takeError(getApi(), error);

  auto cleanupExe2 = llvm::make_scope_exit([&]() {
    mtrt::Status s = destroyExecutable(getApi(), get_exe_args2.executable);
    EXPECT_TRUE(s.isOk()) << s.getStatus();
  });

  // Validate stable invariants.
  mtrt::StatusOr<std::string> fp1 =
      getExecutableFingerprint(getApi(), get_exe_args.executable);
  mtrt::StatusOr<std::string> fp2 =
      getExecutableFingerprint(getApi(), get_exe_args2.executable);
  ASSERT_TRUE(fp1.isOk()) << fp1.getStatus();
  ASSERT_TRUE(fp2.isOk()) << fp2.getStatus();
  ASSERT_EQ(*fp1, *fp2);

  mtrt::StatusOr<std::vector<PJRT_Buffer_Type>> tys1 =
      getExecutableOutputElementTypes(getApi(), get_exe_args.executable);
  mtrt::StatusOr<std::vector<PJRT_Buffer_Type>> tys2 =
      getExecutableOutputElementTypes(getApi(), get_exe_args2.executable);
  ASSERT_TRUE(tys1.isOk()) << tys1.getStatus();
  ASSERT_TRUE(tys2.isOk()) << tys2.getStatus();
  ASSERT_EQ(*tys1, *tys2);
}
