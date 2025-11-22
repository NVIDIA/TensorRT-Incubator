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
#include "mlir-executor-c/Runtime/Runtime.h"
#include "mlir-executor-c/Common/Common.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#include "mlir-executor/Runtime/Backend/Common/DataTypes.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensions.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaRuntime.h"
#include "mlir-executor/Runtime/Support/CUDAHelpers.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "mlir-tensorrt-common-c/Support/Status.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <memory>
#include <mutex>

#ifdef MLIR_TRT_ENABLE_CUDA
#include "cuda_runtime_api.h"
#endif

namespace {
struct RuntimeClientRef {
  mtrt::Ref<mtrt::RuntimeClient> ref;
};

struct StreamRef {
  mtrt::Ref<mtrt::Stream> ref;
};
} // namespace

#define DEFINE_C_API_PTR_METHODS(name, cpptype)                                \
  static inline name wrap(cpptype *cpp) { return name{cpp}; }                  \
  static inline cpptype *unwrap(name c) {                                      \
    return static_cast<cpptype *>(c.ptr);                                      \
  }

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
DEFINE_C_API_PTR_METHODS(MTRT_RuntimeSession, ::mtrt::RuntimeSession)
DEFINE_C_API_PTR_METHODS(MTRT_RuntimeSessionOptions,
                         ::mtrt::RuntimeSessionOptions)
DEFINE_C_API_PTR_METHODS(MTRT_Executable, ::mtrt::Executable)
DEFINE_C_API_PTR_METHODS(MTRT_Stream, ::StreamRef)
DEFINE_C_API_PTR_METHODS(MTRT_RuntimeValue, ::mtrt::RuntimeValue)
DEFINE_C_API_PTR_METHODS(MTRT_ScalarValue, ::mtrt::ScalarValue)
DEFINE_C_API_PTR_METHODS(MTRT_RuntimeClient, RuntimeClientRef)
DEFINE_C_API_PTR_METHODS(MTRT_MemRefValue, ::mtrt::MemRefValue)
DEFINE_C_API_PTR_METHODS(MTRT_Device, ::mtrt::Device)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace mtrt;
using namespace mtrt;

/// Return the MTRT_StatusCode. These are auto-generated from the same schema as
/// the `mtrt::StatusCode`.
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
  ASSERT_ENUM_MATCH(ScalarTypeCode, complex32);
  ASSERT_ENUM_MATCH(ScalarTypeCode, complex64);
  ASSERT_ENUM_MATCH(ScalarTypeCode, f4e2m1fn);
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
      status.getMessage().c_str());
}

//===----------------------------------------------------------------------===//
// Global Initialization / Shutdown
//===----------------------------------------------------------------------===//

namespace {
struct ExecutorRuntimeGlobalInit {
  static std::mutex m;

  ExecutorRuntimeGlobalInit() { mtrt::registerLuaRuntimeExtensions(); }
};
} // namespace

std::mutex ExecutorRuntimeGlobalInit::m;

static ExecutorRuntimeGlobalInit *globalInit = nullptr;

/// Perform global initialization of the runtime. This should only be called
/// once. Calling multiple times will result in an error.
void mtrtRuntimeInitialize() {
  std::scoped_lock<std::mutex> lock(ExecutorRuntimeGlobalInit::m);
  if (globalInit)
    llvm::report_fatal_error("mtrtRuntimeInitialize invoked multiple times");

  globalInit = new ExecutorRuntimeGlobalInit();
}

/// Perform global de-initialization of the runtime.
void mtrtRuntimeShutdown() {
  std::scoped_lock<std::mutex> lock(ExecutorRuntimeGlobalInit::m);
  if (!globalInit)
    llvm::report_fatal_error(
        "mtrtRuntimeShutdown called, but runtime was not initialized "
        "(mtrtRuntimeInitialize) or is already shutdown");
  delete globalInit;
}

//===----------------------------------------------------------------------===//
// MTRT_Stream
//===----------------------------------------------------------------------===//

MTRT_Status mtrtEnableGlobalDebug(bool enable) {
  llvm::DebugFlag = enable;
  return mtrtStatusGetOk();
}

MTRT_Status mtrtIsGlobalDebugEnabled(bool *enable) {
  *enable = llvm::DebugFlag;
  return mtrtStatusGetOk();
}

MTRT_Status mtrtSetGlobalDebugType(const char *type) {
  // Depending on the NDEBUG flag, this name can be either a function or a macro
  // that expands to something that isn't a funciton call, so we cannot
  // explicitly prefix it with `llvm::` or declare `using` it.
  using namespace llvm;
  setCurrentDebugType(type);
  return mtrtStatusGetOk();
}

MTRT_Status mtrtSetGlobalDebugTypes(const char **types, size_t n) {
  using namespace llvm;
  setCurrentDebugTypes(types, n);
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_Stream
//===----------------------------------------------------------------------===//

void mtrtStreamPrint(MTRT_Stream stream, MlirStringCallback append,
                     void *userData) {
  mlir::detail::CallbackOstream printStream(append, userData);
  std::stringstream ss;
  ss << std::hex
     << reinterpret_cast<uintptr_t>(unwrap(stream)->ref->getCUDAHandle());
  printStream << "CUDA Stream @ 0x" << ss.str();
}

MTRT_Status mtrtStreamGetPointer(MTRT_Stream stream, uintptr_t *ptr) {
  *ptr = reinterpret_cast<uintptr_t>(unwrap(stream)->ref->getCUDAHandle());
  return mtrtStatusGetOk();
}

MTRT_Status mtrtDeviceGetStream(MTRT_Device device, MTRT_Stream *stream) {
  Ref<Stream> streamRef = unwrap(device)->getStream();
  *stream = wrap(new StreamRef{std::move(streamRef)});
  return mtrtStatusGetOk();
}

MTRT_Status mtrtStreamDestroy(MTRT_Stream stream) {
  delete unwrap(stream);
  return mtrtStatusGetOk();
}

MTRT_Status mtrtStreamSynchronize(MTRT_Stream stream) {
  Status syncStatus = unwrap(stream)->ref->sync();
  if (!syncStatus.isOk())
    return wrap(syncStatus.getStatus());
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_MemRefValue
//===----------------------------------------------------------------------===//

MTRT_Status mtrtMemRefCreate(MTRT_RuntimeClient client,
                             MTRT_PointerType pointerKind,
                             MTRT_ScalarTypeCode scalarType, int64_t rank,
                             const int64_t *shape, const int64_t *strides,
                             MTRT_Device device, MTRT_Stream stream,
                             MTRT_MemRefValue *result,
                             bool assertCanonicalStrides) {
  Ref<Stream> streamRef = {nullptr};
  if (!mtrtStreamIsNull(stream))
    streamRef = unwrap(stream)->ref;
  BufferType bufferType = BufferType::createWithElementStrides(
      unwrap(scalarType), llvm::ArrayRef(shape, shape + rank),
      llvm::ArrayRef(strides, strides + rank), unwrap(pointerKind),
      /*offset=*/0);
  StatusOr<std::unique_ptr<MemRefValue>> bufferImpl =
      unwrap(client)->ref->allocateMemRef(bufferType, unwrap(device), streamRef,
                                          assertCanonicalStrides);

  if (bufferImpl.isError())
    return wrap(bufferImpl.getStatus());

  [[maybe_unused]] uintptr_t ptr = (*bufferImpl)->getMemory();
  *result = wrap(bufferImpl->release());
  MTRT_DBGF("mtrtMemRefCreate[%p]: ptr = %p",
            reinterpret_cast<void *>(result->ptr),
            reinterpret_cast<void *>(ptr));

  return mtrtStatusGetOk();
}

static std::function<void()>
unwrapDestroyCallback(MTRT_MemRefDestroyCallback callback) {
  if (mtrtMemRefDestroyCallbackIsNull(callback))
    return nullptr;
  return [callback = std::move(callback)]() {
    callback.callback(callback.userData);
  };
}

MTRT_Status mtrtMemRefCreateExternal(
    MTRT_RuntimeClient client, MTRT_PointerType pointerKind,
    MTRT_ScalarTypeCode scalarType, uintptr_t ptr, int64_t offset, int64_t rank,
    const int64_t *shape, const int64_t *strides, MTRT_Device device,
    MTRT_MemRefValue *result, bool assertCanonicalStrides,
    MTRT_MemRefDestroyCallback destroyCallback) {
  BufferType bufferType = BufferType::createWithElementStrides(
      unwrap(scalarType), llvm::ArrayRef(shape, shape + rank),
      llvm::ArrayRef(strides, strides + rank), unwrap(pointerKind),
      /*offset=*/0);
  StatusOr<std::unique_ptr<MemRefValue>> bufferImpl =
      unwrap(client)->ref->createExternalMemRef(
          bufferType, ptr, unwrap(device), assertCanonicalStrides,
          unwrapDestroyCallback(std::move(destroyCallback)));

  if (bufferImpl.isError())
    return wrap(bufferImpl.getStatus());

  *result = wrap(bufferImpl->release());

  MTRT_DBGF("mtrtMemRefCreateExternal[%p]: ptr = %p",
            reinterpret_cast<void *>(result->ptr),
            reinterpret_cast<void *>(ptr));

  return mtrtStatusGetOk();
}

MTRT_Status mtrtMemRefValueDestroyAsync(MTRT_MemRefValue buffer,
                                        MTRT_Stream stream) {
  MemRefValue *memref = unwrap(buffer);
  MTRT_DBGF("mtrtMemRefValueDestroyAsync[%p]: ptr = %p",
            reinterpret_cast<void *>(memref),
            reinterpret_cast<void *>(memref->getMemory()));
  delete memref;
  return mtrtStatusGetOk();
}

MTRT_Status mtrtMemRefValueDestroy(MTRT_MemRefValue buffer) {
  MemRefValue *memref = unwrap(buffer);
  MTRT_DBGF("mtrtMemRefValueDestroy[%p]: ptr = %p",
            reinterpret_cast<void *>(memref),
            reinterpret_cast<void *>(memref->getMemory()));
  delete memref;
  return mtrtStatusGetOk();
}

MTRT_Status mtrtMemRefValueGetInfo(MTRT_MemRefValue memref,
                                   MTRT_MemRefValueInfo *info) {
  MemRefValue *cppMemRef = unwrap(memref);

  MTRT_MemRefValueInfo result;
  result.ptr = cppMemRef->getMemory();
  result.rank = cppMemRef->getRank();
  result.bitsPerElement = cppMemRef->getElementBitWidth();
  result.offset = cppMemRef->getLayout().getOffset();
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
  return wrap(new RuntimeClientRef{unwrap(memref)->getClient()});
}

uint32_t mtrtMemRefReferenceCount(MTRT_MemRefValue memref) {
  unsigned refCount = unwrap(memref)->getStorageRefCount();
  MTRT_DBGF("mtrtMemRefReferenceCount[%p]: ref_count = %d",
            reinterpret_cast<void *>(unwrap(memref)), refCount);
  return refCount;
}

MTRT_MemRefValue mtrtMemRefCreateRef(MTRT_MemRefValue memref) {
  return wrap(unwrap(memref)->createRef().release());
}

MTRT_Status mtrtMemRefValueGetStream(MTRT_MemRefValue memref,
                                     MTRT_Stream *stream) {
  Device *device = unwrap(memref)->getDevice();
  if (!device) {
    *stream = MTRT_Stream{nullptr};
    return mtrtStatusGetOk();
  }
  *stream = wrap(new StreamRef{device->getStream()});
  return mtrtStatusGetOk();
}

/// Wait for the current value to be "ready". If the value is a device
/// memrefvalue, then it will incur a CUDA stream synchronization.
MTRT_Status mtrtMemRefValueWaitForReady(MTRT_MemRefValue value) {
  Device *device = unwrap(value)->getDevice();
  if (!device)
    return mtrtStatusGetOk();
  Status s = device->getStream()->sync();
  if (!s.isOk())
    return wrap(s);
  return mtrtStatusGetOk();
}

/// Add a wait on the `externalStream` for the `stream` to complete all
/// outstanding operations as of now.
MTRT_Status mtrtExternalStreamWaitOnMTRTStream(uintptr_t externalWaitingStream,
                                               MTRT_Stream streamToWaitOn) {

  mtrt::StatusOr<std::unique_ptr<mtrt::Event>> event =
      mtrt::Event::create(unwrap(streamToWaitOn)->ref);
  if (!event.isOk())
    return wrap(event.getStatus());

  /// Get the raw cuda event handle that will be ready when the streamToWaitOn
  /// is ready.
  uintptr_t cudaEventHandle = (*event)->getCudaHandle();
  Status waitStatus =
      mtrt::waitCUDAEventOnStream(externalWaitingStream, cudaEventHandle);
  if (!waitStatus.isOk())
    return wrap(waitStatus.getStatus());

  // Release the event when the wait is complete
  mtrt::Event::releaseWhenReady(std::move(*event));

  return mtrtStatusGetOk();
}

MTRT_Device mtrtMemRefValueGetDevice(MTRT_MemRefValue memref) {
  Device *device = unwrap(memref)->getDevice();
  if (!device)
    return mtrtDeviceGetNull();
  return wrap(device);
}

MTRT_PointerType mtrtMemRefValueGetAddressSpace(MTRT_MemRefValue memref) {
  return wrap(unwrap(memref)->getAddressSpace());
}

//===----------------------------------------------------------------------===//
// Data Transfer
//===----------------------------------------------------------------------===//

// Copy from host to device happens through a pinned host buffer acting as a
// staging buffer.
MTRT_Status mtrtCopyFromHostToDevice(MTRT_MemRefValue hostBuffer,
                                     MTRT_Device device, MTRT_Stream stream,
                                     MTRT_MemRefValue *deviceBuffer) {
  Ref<Stream> streamRef = {nullptr};
  if (!mtrtStreamIsNull(stream))
    streamRef = unwrap(stream)->ref;
  StatusOr<std::unique_ptr<MemRefValue>> deviceBufferImpl =
      unwrap(hostBuffer)
          ->getClient()
          ->copyToDevice(*unwrap(hostBuffer), *unwrap(device),
                         streamRef, /*doneWithHostBuffer=*/
                         nullptr);
  if (!deviceBufferImpl.isOk())
    return wrap(deviceBufferImpl.getStatus());

  [[maybe_unused]] uintptr_t deviceBufferPtr = (*deviceBufferImpl)->getMemory();
  *deviceBuffer = wrap(deviceBufferImpl->release());

  MTRT_DBGF("mtrtCopyFromHostToDevice: from MTRT_MemRefValue %p [ptr = %p] to "
            "MTRT_MemRefValue %p [ptr = %p]",
            reinterpret_cast<void *>(hostBuffer.ptr),
            reinterpret_cast<void *>(unwrap(hostBuffer)->getMemory()),
            reinterpret_cast<void *>(deviceBuffer->ptr),
            reinterpret_cast<void *>(deviceBufferPtr));

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
  Ref<Stream> streamRef = {nullptr};
  if (!mtrtStreamIsNull(stream))
    streamRef = unwrap(stream)->ref;
  StatusOr<std::unique_ptr<MemRefValue>> hostMemRef =
      unwrap(deviceBuffer)
          ->getClient()
          ->copyToHost(*unwrap(deviceBuffer), streamRef);
  if (!hostMemRef.isOk())
    return wrap(hostMemRef.getStatus());

  *result = wrap(hostMemRef->release());

  return mtrtStatusGetOk();
}

MTRT_Status
mtrtCopyFromDeviceToExistingHostMemRef(MTRT_MemRefValue deviceBuffer,
                                       MTRT_MemRefValue hostBuffer,
                                       MTRT_Stream stream) {
  Ref<Stream> streamRef = {nullptr};
  if (!mtrtStreamIsNull(stream))
    streamRef = unwrap(stream)->ref;
  Status s =
      unwrap(deviceBuffer)
          ->getClient()
          ->copyToHost(*unwrap(deviceBuffer), *unwrap(hostBuffer), streamRef);
  if (!s.isOk())
    return wrap(s);

  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_ScalarValue
//===----------------------------------------------------------------------===//

MTRT_Status mtrtScalarValueDestroy(MTRT_ScalarValue value) {
  if (value.ptr)
    delete unwrap(value);
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MTRT_RuntimeValue
//===----------------------------------------------------------------------===//

MTRT_Status mtrtRuntimeValueDestroy(MTRT_RuntimeValue value) {
  if (mtrtRuntimeValueIsMemRef(value))
    return mtrtMemRefValueDestroy(mtrtRuntimeValueDynCastToMemRef(value));
  if (mtrtRuntimeValueIsScalar(value))
    return mtrtScalarValueDestroy(mtrtRuntimeValueDynCastToScalar(value));
  return mtrtStatusGetOk();
}

MTRT_Status mtrtRuntimeValueScalarCreate(int64_t data, MTRT_ScalarTypeCode code,
                                         MTRT_RuntimeValue *value) {
  *value = MTRT_RuntimeValue{new ScalarValue(data, ScalarType(unwrap(code)))};
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

bool mtrtRuntimeValueIsMemRef(MTRT_RuntimeValue value) {
  RuntimeValue *x = unwrap(value);
  return x->getKind() == RuntimeValue::Kind::MemRef;
}

bool mtrtRuntimeValueIsScalar(MTRT_RuntimeValue value) {
  RuntimeValue *x = unwrap(value);
  return x->getKind() == RuntimeValue::Kind::Scalar;
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

void mtrtRuntimeSessionOptionsEnableFeature(MTRT_RuntimeSessionOptions options,
                                            MTRT_StringView feature) {
  RuntimeSessionOptions *cppOptions = unwrap(options);
  cppOptions->enableFeatures(std::string(feature.data, feature.length));
}

//===----------------------------------------------------------------------===//
// MTRT_RuntimeSession
//===----------------------------------------------------------------------===//

MTRT_Status mtrtRuntimeSessionCreate(MTRT_RuntimeClient client,
                                     MTRT_RuntimeSessionOptions options,
                                     MTRT_Executable executable,
                                     MTRT_RuntimeSession *result) {
  Ref<RuntimeClient> cppClient = unwrap(client)->ref;
  RuntimeSessionOptions *cppOptions = unwrap(options);
  Executable *cppExecutable = unwrap(executable);
  StatusOr<std::unique_ptr<LuaRuntimeSession>> session =
      LuaRuntimeSession::create(cppClient, *cppOptions,
                                cppExecutable->getView(), {});
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
    const MTRT_RuntimeValue *outArgs, size_t numOutArgs,
    MTRT_RuntimeValue *results, MTRT_Stream stream) {
  LuaRuntimeSession *cppSession =
      static_cast<LuaRuntimeSession *>(unwrap(session));

  llvm::SmallVector<RuntimeValue *> inArgValues =
      llvm::map_to_vector(llvm::ArrayRef(inArgs, numInArgs),
                          [](MTRT_RuntimeValue arg) { return unwrap(arg); });
  llvm::SmallVector<RuntimeValue *> outArgValues =
      llvm::map_to_vector(llvm::ArrayRef(outArgs, numOutArgs),
                          [](MTRT_RuntimeValue arg) { return unwrap(arg); });
  Ref<Stream> streamRef = {nullptr};
  if (!mtrtStreamIsNull(stream)) {
    streamRef = unwrap(stream)->ref;
    Status s = cppSession->setStream(streamRef);
    if (!s.isOk())
      return wrap(s);
  }
  StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>> resultValues =
      cppSession->executeFunction(std::string_view(name.data, name.length),
                                  inArgValues, outArgValues);
  if (!resultValues.isOk())
    return wrap(resultValues.getStatus());

  for (size_t i = 0; i < resultValues->size(); ++i)
    results[i] = wrap((*resultValues)[i].release());

  return mtrtStatusGetOk();
}

MTRT_Status mtrtRuntimeSessionGetNumResults(MTRT_RuntimeSession session,
                                            MTRT_StringView name,
                                            int64_t *numResults) {
  LuaRuntimeSession *cppSession =
      static_cast<LuaRuntimeSession *>(unwrap(session));
  StatusOr<FunctionView> func = cppSession->getExecutable().getFunction(
      std::string_view(name.data, name.length));
  if (func.isError()) {
    return wrap(func.getStatus());
  }
  *numResults = (*func).getSignature().getNumResults();
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_RuntimeClient
//===----------------------------------------------------------------------===//

MTRT_Status mtrtRuntimeClientCreate(MTRT_RuntimeClient *client) {
  StatusOr<Ref<RuntimeClient>> cppClient = RuntimeClient::create();
  if (!cppClient.isOk())
    return wrap(cppClient.getStatus());

  *client = MTRT_RuntimeClient{new RuntimeClientRef{std::move(*cppClient)}};
  return mtrtStatusGetOk();
}

MTRT_Status mtrtRuntimeClientDestroy(MTRT_RuntimeClient client) {
  delete unwrap(client);
  return mtrtStatusGetOk();
}

MTRT_Status mtrtRuntimeClientGetNumDevices(MTRT_RuntimeClient client,
                                           int32_t *numDevices) {
  RuntimeClient *cppClient = unwrap(client)->ref.get();
  *numDevices = cppClient->getDevices().size();
  return mtrtStatusGetOk();
}

MTRT_Status mtrtRuntimeClientGetDevice(MTRT_RuntimeClient client, int32_t index,
                                       MTRT_Device *device) {
  RuntimeClient *cppClient = unwrap(client)->ref.get();
  if (index >= static_cast<int64_t>(cppClient->getDevices().size()))
    return wrap(getInvalidArgStatus(
        "the provided index is greater than the number of devices"));
  *device = wrap(cppClient->getDevices()[index].get());
  return mtrtStatusGetOk();
}

//===----------------------------------------------------------------------===//
// MTRT_Device
//===----------------------------------------------------------------------===//

MTRT_Status mtrtDeviceGetIndex(MTRT_Device device, int32_t *index) {
  Device *cppDevice = unwrap(device);
  *index = cppDevice->getDeviceNumber();
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

MTRT_Status mtrtScalarValueGet(MTRT_ScalarValue scalar, int64_t *data) {
  ScalarValue *cppScalar = unwrap(scalar);
  ScalarTypeCode code = cppScalar->getType().getCode();
  switch (code) {
  case ScalarTypeCode::f8e4m3fn:
    *data = static_cast<int64_t>(cppScalar->get<F8E4M3FN>());
    break;
  case ScalarTypeCode::f16:
    *data = static_cast<int64_t>(cppScalar->get<Float16>());
    break;
  case ScalarTypeCode::bf16:
    *data = static_cast<int64_t>(cppScalar->get<BFloat16>());
    break;
  case ScalarTypeCode::f32:
    *data = static_cast<int64_t>(cppScalar->get<float>());
    break;
  case ScalarTypeCode::f64:
    *data = static_cast<int64_t>(cppScalar->get<double>());
    break;
  case ScalarTypeCode::i1:
    *data = static_cast<int64_t>(cppScalar->get<int8_t>());
    break;
  case ScalarTypeCode::i4:
    *data = static_cast<int64_t>(cppScalar->get<int8_t>());
    break;
  case ScalarTypeCode::i8:
    *data = static_cast<int64_t>(cppScalar->get<int8_t>());
    break;
  case ScalarTypeCode::ui8:
    *data = static_cast<int64_t>(cppScalar->get<uint8_t>());
    break;
  case ScalarTypeCode::i16:
    *data = static_cast<int64_t>(cppScalar->get<int16_t>());
    break;
  case ScalarTypeCode::i32:
    *data = static_cast<int64_t>(cppScalar->get<int32_t>());
    break;
  case ScalarTypeCode::i64:
    *data = cppScalar->get<int64_t>();
    break;
  default:
    return wrap(getInvalidArgStatus(
        "function input argument with scalar type {0} is unsupported",
        mtrt::flat::EnumNameScalarTypeCode(code)));
  }
  return mtrtStatusGetOk();
}
