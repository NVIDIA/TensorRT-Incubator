//===- Runtime.cpp --------------------------------------------------------===//
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
/// MLIR-TensorRT runtime C API implementation.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-c/Runtime/Runtime.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaRuntime.h"
#include "mlir-executor/Support/Status.h"
#include "mlir-tensorrt-c/Common/Common.h"
#include "mlir-tensorrt-c/Support/Status.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <memory>

struct MTRT_StreamImpl;

#define DEFINE_C_API_PTR_METHODS(name, cpptype)                                \
  static inline name wrap(cpptype *cpp) { return name{cpp}; }                  \
  static inline cpptype *unwrap(name c) {                                      \
    return static_cast<cpptype *>(c.ptr);                                      \
  }

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
DEFINE_C_API_PTR_METHODS(MTRT_RuntimeSession,
                         ::mlirtrt::runtime::RuntimeSession)
DEFINE_C_API_PTR_METHODS(MTRT_RuntimeSessionOptions,
                         ::mlirtrt::runtime::RuntimeSessionOptions)
DEFINE_C_API_PTR_METHODS(MTRT_Executable, ::mlirtrt::runtime::Executable)
DEFINE_C_API_PTR_METHODS(MTRT_Stream, MTRT_StreamImpl)
DEFINE_C_API_PTR_METHODS(MTRT_RuntimeValue, ::mlirtrt::runtime::RuntimeValue)
DEFINE_C_API_PTR_METHODS(MTRT_ScalarValue, ::mlirtrt::runtime::ScalarValue)
DEFINE_C_API_PTR_METHODS(MTRT_RuntimeClient, ::mlirtrt::runtime::RuntimeClient)
DEFINE_C_API_PTR_METHODS(MTRT_MemRefValue, ::mlirtrt::runtime::MemRefValue)
DEFINE_C_API_PTR_METHODS(MTRT_Device, ::mlirtrt::runtime::Device)
#ifdef MLIR_TRT_ENABLE_PYTHON
DEFINE_C_API_PTR_METHODS(MTRT_DLPackManagedTensor, DLManagedTensor)
DEFINE_C_API_PTR_METHODS(MTRT_DLPackDevice, DLDevice)
#endif // MLIR_TRT_ENABLE_PYTHON
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace mlirtrt;
using namespace mlirtrt::runtime;

/// Return the MTRT_StatusCode. These are auto-generated from the same schema as
/// the `mlirtrt::StatusCode`.
static MTRT_StatusCode getMTRTStatusCodeFromRuntimeStatusCode(StatusCode code) {
  return static_cast<MTRT_StatusCode>(code);
}

template <typename Enum1, typename Enum2>
constexpr bool checkEnumPair(Enum1 e1, Enum2 e2) {
  return static_cast<int64_t>(e1) == static_cast<int64_t>(e2);
}

template <auto e1, auto e2>
struct EnumPairChecker {
  static constexpr bool value = checkEnumPair(e1, e2);
};

/// Statically assert that enum values of two enum types match. Enum values are
/// expected to be of MTRT_<EnumType>_<EnumValue> and <EnumType>_<EnumValue>.
#define ASSERT_ENUM_MATCH(Category, x)                                         \
  static_assert(EnumPairChecker<MTRT_##Category::MTRT_##Category##_##x,        \
                                Category::x>::value,                           \
                "expected equal enum values for equivalent MTRT_" #Category    \
                "/" #Category " values");

/// Convert C API ScalarTypeCode to the C++ API Enum.
/// TODO: we have a basic static assert to check that the number of types did
/// not change, but eventually we need a more comprehensive automation to keep
/// them in sync.
static ScalarTypeCode unwrap(MTRT_ScalarTypeCode type) {
  ASSERT_ENUM_MATCH(ScalarTypeCode, f8e4m3fn);
  ASSERT_ENUM_MATCH(ScalarTypeCode, f16);
  ASSERT_ENUM_MATCH(ScalarTypeCode, f32);
  ASSERT_ENUM_MATCH(ScalarTypeCode, f64);
  ASSERT_ENUM_MATCH(ScalarTypeCode, i1);
  ASSERT_ENUM_MATCH(ScalarTypeCode, i8);
  ASSERT_ENUM_MATCH(ScalarTypeCode, ui8);
  ASSERT_ENUM_MATCH(ScalarTypeCode, i16);
  ASSERT_ENUM_MATCH(ScalarTypeCode, i32);
  ASSERT_ENUM_MATCH(ScalarTypeCode, i64);
  ASSERT_ENUM_MATCH(ScalarTypeCode, bf16);
  ASSERT_ENUM_MATCH(ScalarTypeCode, i4);
  ASSERT_ENUM_MATCH(ScalarTypeCode, unknown);
  ASSERT_ENUM_MATCH(ScalarTypeCode, MIN);
  ASSERT_ENUM_MATCH(ScalarTypeCode, MAX);
  return static_cast<ScalarTypeCode>(type);
}

static PointerType unwrap(MTRT_PointerType pointerType) {
  // Since this is a compile time check, it is OK to only have it here
  // and not repeat the same in corresponding wrap() method.
  ASSERT_ENUM_MATCH(PointerType, host);
  ASSERT_ENUM_MATCH(PointerType, pinned_host);
  ASSERT_ENUM_MATCH(PointerType, device);
  ASSERT_ENUM_MATCH(PointerType, unified);
  ASSERT_ENUM_MATCH(PointerType, unknown);
  ASSERT_ENUM_MATCH(PointerType, MIN);
  ASSERT_ENUM_MATCH(PointerType, MAX);
  return static_cast<PointerType>(pointerType);
}

static MTRT_PointerType wrap(PointerType pointerType) {
  return static_cast<MTRT_PointerType>(pointerType);
}

static MTRT_Status wrap(const Status &status) {
  if (status.isOk())
    return mtrtStatusGetOk();
  return mtrtStatusCreate(
      getMTRTStatusCodeFromRuntimeStatusCode(status.getCode()),
      status.getString().c_str());
}

//===----------------------------------------------------------------------===//
// MTRT_Stream
//===----------------------------------------------------------------------===//

struct MTRT_StreamImpl {
public:
  MTRT_StreamImpl() = delete;
  MTRT_StreamImpl(const MTRT_StreamImpl &) = delete;
  MTRT_StreamImpl &operator=(const MTRT_StreamImpl &) = delete;
  MTRT_StreamImpl(MTRT_StreamImpl &&other) {
    stream = other.stream;
    other.stream = nullptr;
  };
  MTRT_StreamImpl &operator=(MTRT_StreamImpl &&other) {
    if (this != &other) {
      stream = other.stream;
      other.stream = nullptr;
    }
    return *this;
  };
  static StatusOr<std::unique_ptr<MTRT_StreamImpl>> create();
  ~MTRT_StreamImpl();
  cudaStream_t getRawStream() { return stream; }
  Status sync();

private:
  MTRT_StreamImpl(cudaStream_t s) : stream(s) {}
  cudaStream_t stream{nullptr};
};

StatusOr<std::unique_ptr<MTRT_StreamImpl>> MTRT_StreamImpl::create() {
  cudaStream_t s;
  RETURN_ERROR_IF_CUDART_ERROR(cudaStreamCreate(&s));
  return std::unique_ptr<MTRT_StreamImpl>(new MTRT_StreamImpl(std::move(s)));
}

MTRT_StreamImpl::~MTRT_StreamImpl() {
  if (stream)
    cudaStreamDestroy(stream);
}

Status MTRT_StreamImpl::sync() {
  RETURN_ERROR_IF_CUDART_ERROR(cudaStreamSynchronize(stream));
  return getOkStatus();
}

MTRT_Status mtrtStreamCreate(MTRT_Stream *stream) {
  StatusOr<std::unique_ptr<MTRT_StreamImpl>> streamImpl =
      MTRT_StreamImpl::create();
  if (!streamImpl.isOk())
    return wrap(streamImpl.getStatus());
  *stream = MTRT_Stream{streamImpl->release()};
  return mtrtStatusGetOk();
}

MTRT_Status mtrtStreamSynchronize(MTRT_Stream stream) {
  Status syncStatus = unwrap(stream)->sync();
  if (!syncStatus.isOk())
    return wrap(syncStatus.getStatus());
  return mtrtStatusGetOk();
}

MTRT_Status mtrtStreamDestroy(MTRT_Stream stream) {
  delete unwrap(stream);
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_MemRefValue
//===----------------------------------------------------------------------===//

MTRT_Status
mtrtMemRefCreate(MTRT_RuntimeClient client, MTRT_PointerType pointerKind,
                 int64_t bitsPerElement, int64_t rank, const int64_t *shape,
                 const int64_t *strides, MTRT_Device device, MTRT_Stream stream,
                 MTRT_ScalarTypeCode scalarType, MTRT_MemRefValue *result) {
  StatusOr<std::unique_ptr<MemRefValue>> bufferImpl =
      unwrap(client)->allocateMemRef(
          unwrap(pointerKind), bitsPerElement,
          llvm::ArrayRef(shape, shape + rank),
          llvm::ArrayRef(strides, strides + rank),
          mtrtDeviceIsNull(device) ? std::nullopt
                                   : std::optional(unwrap(device)),
          mtrtStreamIsNull(stream)
              ? std::nullopt
              : std::optional(unwrap(stream)->getRawStream()),
          scalarType != MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_unknown
              ? std::optional(ScalarType(unwrap(scalarType)))
              : std::nullopt);

  if (bufferImpl.isError())
    return wrap(bufferImpl.getStatus());

  *result = wrap(bufferImpl->release());
  return mtrtStatusGetOk();
}

MTRT_Status mtrtMemRefCreateExternal(
    MTRT_RuntimeClient client, MTRT_PointerType pointerKind,
    int64_t bitsPerElement, uintptr_t ptr, int64_t offset, int64_t rank,
    const int64_t *shape, const int64_t *strides, MTRT_Device device,
    MTRT_ScalarTypeCode scalarType, MTRT_MemRefValue *result) {
  StatusOr<std::unique_ptr<MemRefValue>> bufferImpl =
      unwrap(client)->createExternalMemRef(
          unwrap(pointerKind), bitsPerElement, ptr, offset,
          llvm::ArrayRef(shape, shape + rank),
          llvm::ArrayRef(strides, strides + rank),
          mtrtDeviceIsNull(device) ? std::nullopt
                                   : std::optional(unwrap(device)),
          scalarType == MTRT_ScalarTypeCode_unknown
              ? std::nullopt
              : std::optional(ScalarType(unwrap(scalarType))));

  if (bufferImpl.isError())
    return wrap(bufferImpl.getStatus());

  *result = wrap(bufferImpl->release());

  return mtrtStatusGetOk();
}

MTRT_Status mtrtMemRefValueDestroyAsync(MTRT_MemRefValue buffer,
                                        MTRT_Stream stream) {
  MemRefValue *memref = unwrap(buffer);
  Status s = memref->getClient()->deallocate(
      std::unique_ptr<MemRefValue>(memref),
      mtrtStreamIsNull(stream) ? std::nullopt
                               : std::optional(unwrap(stream)->getRawStream()));
  if (!s.isOk())
    return wrap(s);
  return mtrtStatusGetOk();
}

MTRT_Status mtrtMemRefValueDestroy(MTRT_MemRefValue buffer) {
  MemRefValue *memref = unwrap(buffer);
  Status s =
      memref->getClient()->deallocate(std::unique_ptr<MemRefValue>(memref));
  if (!s.isOk())
    return wrap(s);

  return mtrtStatusGetOk();
}

MTRT_Status mtrtMemRefValueGetInfo(MTRT_MemRefValue memref,
                                   MTRT_MemRefValueInfo *info) {
  MemRefValue *cppMemRef = unwrap(memref);

  MTRT_MemRefValueInfo result;
  result.ptr = cppMemRef->getMemory();
  result.rank = cppMemRef->getRank();
  result.bitsPerElement = cppMemRef->getElementBitWidth();
  result.offset = cppMemRef->getOffset();
  result.shape = cppMemRef->getShape().data();
  result.strides = cppMemRef->getStrides().data();
  result.addressSpace = wrap(cppMemRef->getBufferKind());

  const std::optional<ScalarType> &scalarType = cppMemRef->getScalarType();
  if (!scalarType)
    result.scalarType = MTRT_ScalarTypeCode_unknown;
  else
    result.scalarType = static_cast<MTRT_ScalarTypeCode>(scalarType->getCode());

  *info = std::move(result);

  return mtrtStatusGetOk();
}

MTRT_RuntimeClient mtrtMemRefGetClient(MTRT_MemRefValue memref) {
  return wrap(unwrap(memref)->getClient());
}

#ifdef MLIR_TRT_ENABLE_PYTHON

static StatusOr<DLDeviceType> toDLPackDeviceType(PointerType address) {
  switch (address) {
  case PointerType::device:
    return DLDeviceType::kDLCUDA;
  case PointerType::host:
    return DLDeviceType::kDLCPU;
  case PointerType::pinned_host:
    return DLDeviceType::kDLCUDAHost;
  case PointerType::unified:
    return DLDeviceType::kDLCUDAManaged;
  default:
    return getStatusWithMsg(
        StatusCode::InvalidArgument, "Address space [",
        stringifyPointerType(address),
        "] conversion to DLPackDeviceType is not supported.");
  }
  return DLDeviceType::kDLCPU;
}

static StatusOr<DLDataTypeCode> toDLPackDataTypeCode(ScalarTypeCode type) {
  switch (type) {
  case ScalarTypeCode::i1:
    return DLDataTypeCode::kDLBool;
  case ScalarTypeCode::i4:
  case ScalarTypeCode::i8:
  case ScalarTypeCode::i16:
  case ScalarTypeCode::i32:
  case ScalarTypeCode::i64:
    return DLDataTypeCode::kDLInt;
  case ScalarTypeCode::ui8:
    return DLDataTypeCode::kDLUInt;
  case ScalarTypeCode::f8e4m3fn:
  case ScalarTypeCode::f16:
  case ScalarTypeCode::f32:
  case ScalarTypeCode::f64:
    return DLDataTypeCode::kDLFloat;
  case ScalarTypeCode::bf16:
    return DLDataTypeCode::kDLBfloat;
  default:
    return getStatusWithMsg(
        StatusCode::InvalidArgument, "Scalar type code [",
        EnumNameScalarTypeCode(type),
        "] conversion to DLPackDataTypeCode is not supported.");
  }
  return DLDataTypeCode::kDLFloat;
}

void dlpackManagedTensorDeleter(DLManagedTensor *tensor) {
  if (tensor) {
    delete[] tensor->dl_tensor.shape;
    delete[] tensor->dl_tensor.strides;
    static_cast<RuntimeClient *>(tensor->manager_ctx)
        ->getAllocTracker()
        .decrementExternalCount(
            reinterpret_cast<uintptr_t>(tensor->dl_tensor.data));
    delete tensor;
  }
}

MLIR_CAPI_EXPORTED MTRT_Status mtrtMemRefValueGetDLPackManagedTensor(
    MTRT_MemRefValue memrefValue, MTRT_DLPackManagedTensor *outTensor) {
  MemRefValue memref = *unwrap(memrefValue);

  std::unique_ptr<DLManagedTensor> managedTensor;
  managedTensor.reset(new DLManagedTensor());

  managedTensor->dl_tensor.data = memref.getVoidPtr();
  int device = memref.getDevice().has_value()
                   ? memref.getDevice().value()->getDeviceNumber()
                   : 0;

  StatusOr<DLDeviceType> deviceType =
      toDLPackDeviceType(memref.getAddressSpace());
  if (!deviceType.isOk())
    return wrap(deviceType.getStatus());

  managedTensor->dl_tensor.device = {*deviceType, device};
  managedTensor->dl_tensor.ndim = memref.getRank();

  StatusOr<DLDataTypeCode> dtypeCode =
      toDLPackDataTypeCode(memref.getScalarType()->getCode());
  if (!dtypeCode.isOk())
    return wrap(dtypeCode.getStatus());

  managedTensor->dl_tensor.dtype = {
      static_cast<uint8_t>(*dtypeCode),
      static_cast<uint8_t>(memref.getElementBitWidth()),
      1}; // Assume data is non-vectorized.
  managedTensor->dl_tensor.shape = new int64_t[managedTensor->dl_tensor.ndim];
  managedTensor->dl_tensor.strides = new int64_t[managedTensor->dl_tensor.ndim];
  for (int i = 0; i < managedTensor->dl_tensor.ndim; ++i) {
    managedTensor->dl_tensor.shape[i] = memref.getShape()[i];
    managedTensor->dl_tensor.strides[i] = memref.getStrides()[i];
  }
  managedTensor->dl_tensor.byte_offset = memref.getOffset();
  managedTensor->manager_ctx = memref.getClient();
  managedTensor->deleter = dlpackManagedTensorDeleter;

  // Increment reference count to ensure memory is not released prematurely.
  memref.getClient()->getAllocTracker().incrementExternalCount(
      memref.getMemory());

  *outTensor = wrap(managedTensor.release());
  return mtrtStatusGetOk();
}

MLIR_CAPI_EXPORTED MTRT_Status mtrtMemRefValueGetDLPackDevice(
    MTRT_MemRefValue memrefValue, int32_t *device_type, int32_t *device_id) {
  MemRefValue memref = *unwrap(memrefValue);
  int device = memref.getDevice().has_value()
                   ? memref.getDevice().value()->getDeviceNumber()
                   : 0;

  StatusOr<DLDeviceType> deviceType =
      toDLPackDeviceType(memref.getAddressSpace());
  if (!deviceType.isOk())
    return wrap(deviceType.getStatus());
  *device_type = *deviceType;
  *device_id = device;
  return mtrtStatusGetOk();
}

#endif // MLIR_TRT_ENABLE_PYTHON

//===----------------------------------------------------------------------===//
// Data Transfer
//===----------------------------------------------------------------------===//

// Copy from host to device happens through a pinned host buffer acting as a
// staging buffer.
MTRT_Status mtrtCopyFromHostToDevice(MTRT_MemRefValue hostBuffer,
                                     MTRT_Device device, MTRT_Stream stream,
                                     MTRT_MemRefValue *deviceBuffer) {
  StatusOr<std::unique_ptr<MemRefValue>> deviceBufferImpl =
      unwrap(hostBuffer)
          ->getClient()
          ->copyToDevice(*unwrap(hostBuffer), *unwrap(device),
                         mtrtStreamIsNull(stream)
                             ? std::nullopt
                             : std::optional(unwrap(stream)->getRawStream()));
  if (!deviceBufferImpl.isOk())
    return wrap(deviceBufferImpl.getStatus());

  *deviceBuffer = wrap(deviceBufferImpl->release());
  return mtrtStatusGetOk();
}

MTRT_Status mtrtCopyFromHostToHost(MTRT_MemRefValue hostBufferSource,
                                   MTRT_MemRefValue *hostBufferTarget) {
  StatusOr<std::unique_ptr<MemRefValue>> hostBufferImpl =
      unwrap(hostBufferSource)
          ->getClient()
          ->copyHostToHost(*unwrap(hostBufferSource));
  if (!hostBufferImpl.isOk())
    return wrap(hostBufferImpl.getStatus());
  *hostBufferTarget = wrap(hostBufferImpl->release());
  return mtrtStatusGetOk();
}

MTRT_Status mtrtCopyFromDeviceToNewHostMemRef(MTRT_MemRefValue deviceBuffer,
                                              MTRT_Stream stream,
                                              MTRT_MemRefValue *result) {
  StatusOr<std::unique_ptr<MemRefValue>> hostMemRef =
      unwrap(deviceBuffer)
          ->getClient()
          ->copyToHost(*unwrap(deviceBuffer),
                       mtrtStreamIsNull(stream)
                           ? std::nullopt
                           : std::optional(unwrap(stream)->getRawStream()));
  if (!hostMemRef.isOk())
    return wrap(hostMemRef.getStatus());

  *result = wrap(hostMemRef->release());

  return mtrtStatusGetOk();
}

MTRT_Status
mtrtCopyFromDeviceToExistingHostMemRef(MTRT_MemRefValue deviceBuffer,
                                       MTRT_MemRefValue hostBuffer,
                                       MTRT_Stream stream) {
  Status s =
      unwrap(deviceBuffer)
          ->getClient()
          ->copyToHost(*unwrap(deviceBuffer), *unwrap(hostBuffer),
                       mtrtStreamIsNull(stream)
                           ? std::nullopt
                           : std::optional(unwrap(stream)->getRawStream()));
  if (!s.isOk())
    return wrap(s);

  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_RuntimeValue
//===----------------------------------------------------------------------===//

MTRT_Status mtrtRuntimeValueDestroy(MTRT_RuntimeValue value) {
  delete unwrap(value);
  return mtrtStatusGetOk();
}

MTRT_Status mtrtRuntimeValueScalarI64Create(int64_t data,
                                            MTRT_RuntimeValue *value) {
  *value =
      MTRT_RuntimeValue{new ScalarValue(data, ScalarType(ScalarTypeCode::i64))};
  return mtrtStatusGetOk();
}

MTRT_RuntimeValue mtrtMemRefCastToRuntimeValue(MTRT_MemRefValue memref) {
  RuntimeValue *cppMemref = unwrap(memref);
  return wrap(cppMemref);
}

MTRT_RuntimeValue mtrtScalarValueCastToRuntimeValue(MTRT_ScalarValue v) {
  RuntimeValue *x = unwrap(v);
  return wrap(x);
}

MTRT_MemRefValue mtrtRuntimeValueDynCastToMemRef(MTRT_RuntimeValue v) {
  RuntimeValue *x = unwrap(v);
  assert(x->getKind() == RuntimeValue::Kind::MemRef);
  return wrap(static_cast<MemRefValue *>(x));
}

MTRT_ScalarValue mtrtRuntimeValueDynCastToScalar(MTRT_RuntimeValue v) {
  RuntimeValue *x = unwrap(v);
  assert(x->getKind() == RuntimeValue::Kind::Scalar);
  return wrap(static_cast<ScalarValue *>(x));
}

//===----------------------------------------------------------------------===//
// MTRT_RuntimeSessionOptions
//===----------------------------------------------------------------------===//

MTRT_Status
mtrtRuntimeSessionOptionsCreate(int32_t numDevices, int32_t deviceId,
                                MTRT_StringView ncclUuid,
                                MTRT_RuntimeSessionOptions *options) {
  RuntimeSessionOptions result(numDevices, deviceId,
                               llvm::StringRef(ncclUuid.data, ncclUuid.length));
  *options = MTRT_RuntimeSessionOptions{
      /*ptr=*/new RuntimeSessionOptions(std::move(result))};
  return mtrtStatusGetOk();
}

MTRT_Status
mtrtRuntimeSessionOptionsDestroy(MTRT_RuntimeSessionOptions options) {
  delete unwrap(options);
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_RuntimeSession
//===----------------------------------------------------------------------===//

MTRT_Status mtrtRuntimeSessionCreate(MTRT_RuntimeSessionOptions options,
                                     MTRT_Executable executable,
                                     MTRT_RuntimeSession *result) {
  RuntimeSessionOptions *cppOptions = unwrap(options);
  Executable *cppExecutable = unwrap(executable);

  StatusOr<std::unique_ptr<RuntimeSession>> session =
      createRuntimeSessionWithLuaBackend(cppExecutable->getView(), *cppOptions);
  if (session.isError())
    return wrap(session.getStatus());

  *result = wrap(session->release());
  return mtrtStatusGetOk();
}

MTRT_Status mtrtRuntimeSessionDestroy(MTRT_RuntimeSession session) {
  delete unwrap(session);
  return mtrtStatusGetOk();
}

MTRT_Status mtrtRuntimeSessionExecuteFunction(
    MTRT_RuntimeSession session, MTRT_StringView name,
    const MTRT_RuntimeValue *inArgs, size_t numInArgs,
    const MTRT_RuntimeValue *outArgs, size_t numOutArgs, MTRT_Stream stream) {
  RuntimeSession *cppSession = unwrap(session);

  llvm::SmallVector<RuntimeValue *> inArgValues =
      llvm::map_to_vector(llvm::ArrayRef(inArgs, numInArgs),
                          [](MTRT_RuntimeValue arg) { return unwrap(arg); });
  llvm::SmallVector<RuntimeValue *> outArgValues =
      llvm::map_to_vector(llvm::ArrayRef(outArgs, numOutArgs),
                          [](MTRT_RuntimeValue arg) { return unwrap(arg); });

  StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>> result =
      executeFunctionWithLuaBackend(
          *cppSession, std::string_view(name.data, name.length), inArgValues,
          outArgValues,
          !mtrtStreamIsNull(stream)
              ? std::optional(unwrap(stream)->getRawStream())
              : std::nullopt);
  if (!result.isOk())
    return wrap(result.getStatus());

  return mtrtStatusGetOk();
}
//===----------------------------------------------------------------------===//
// MTRT_RuntimeClient
//===----------------------------------------------------------------------===//

MTRT_Status mtrtRuntimeClientCreate(MTRT_RuntimeClient *client) {
  StatusOr<std::unique_ptr<RuntimeClient>> cppClient = RuntimeClient::create();
  if (!cppClient.isOk())
    return wrap(cppClient.getStatus());

  *client = MTRT_RuntimeClient{cppClient->release()};
  return mtrtStatusGetOk();
}

MTRT_Status mtrtRuntimeClientDestroy(MTRT_RuntimeClient client) {
  delete unwrap(client);
  return mtrtStatusGetOk();
}

MTRT_Status mtrtRuntimeClientGetNumDevices(MTRT_RuntimeClient client,
                                           int32_t *numDevices) {
  RuntimeClient *cppClient = unwrap(client);
  *numDevices = cppClient->getDevices().size();
  return mtrtStatusGetOk();
}

MTRT_Status mtrtRuntimeClientGetDevice(MTRT_RuntimeClient client, int32_t index,
                                       MTRT_Device *device) {
  RuntimeClient *cppClient = unwrap(client);
  if (index >= static_cast<int64_t>(cppClient->getDevices().size()))
    return wrap(getInvalidArgStatus(
        "the provided index is greater than the number of devices"));
  *device = wrap(cppClient->getDevices()[index].get());
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_ScalarValue
//===----------------------------------------------------------------------===//

MTRT_Status mtrtScalarValueGetType(MTRT_ScalarValue scalar,
                                   MTRT_ScalarTypeCode *code) {
  ScalarValue *cppScalar = unwrap(scalar);
  *code = static_cast<MTRT_ScalarTypeCode>(cppScalar->getType().getCode());
  return mtrtStatusGetOk();
}
