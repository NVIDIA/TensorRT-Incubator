//===- Executable.cpp ------ ----------------------------------------------===//
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
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/API/Executable.h"
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#include "mlir-executor/Runtime/Support/CUDAHelpers.h"
#include "mlir-executor/Runtime/Support/StridedCopy.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "mlir-executor/Support/Allocators.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdlib>
#include <numeric>

#ifdef MLIR_TRT_ENABLE_NCCL
#define OMPI_SKIP_MPICXX
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "mpi.h"
#include "nccl.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif //  MLIR_TRT_ENABLE_NCCL

using namespace mtrt;

//===----------------------------------------------------------------------===//
// Scalar Type
//===----------------------------------------------------------------------===//

ScalarTypeCode mtrt::parseElementType(std::string_view str) {
  const char *const *names = mtrt::flat::EnumNamesScalarTypeCode();
  const ScalarTypeCode *values = mtrt::flat::EnumValuesScalarTypeCode();
  // Flatbuffers' 'enum::MAX' is inclusive (equals largest value).
  constexpr unsigned maxEnum = static_cast<unsigned>(mtrt::ScalarTypeCode::MAX);
  for (unsigned i = 0; i <= maxEnum; i++) {
    if (str == names[i])
      return values[i];
  }
  return ScalarTypeCode::unknown;
}

int64_t mtrt::getBitsPerElement(ScalarTypeCode elType) {
  switch (elType) {
  case ScalarTypeCode::i64:
  case ScalarTypeCode::f64:
    return 64;
  case ScalarTypeCode::f32:
  case ScalarTypeCode::i32:
    return 32;
  case ScalarTypeCode::f16:
  case ScalarTypeCode::bf16:
  case ScalarTypeCode::i16:
    return 16;
  case ScalarTypeCode::i8:
  case ScalarTypeCode::ui8:
  case ScalarTypeCode::f8e4m3fn:
    return 8;
  case ScalarTypeCode::i4:
    return 4;
  case ScalarTypeCode::f4e2m1fn:
    return 4;
  // We treat i1 types as having byte-level storage currently.
  case ScalarTypeCode::i1:
    return 8;
  case ScalarTypeCode::complex32:
    return 64;
  case ScalarTypeCode::complex64:
    return 128;
  default:
    llvm_unreachable("unhandled element type bit width conversion");
  }
}

ScalarType::ScalarType(ScalarTypeCode code) : code(code) {
  assert(code != ScalarTypeCode::unknown && "expected known element type code");
}

StatusOr<ScalarType> mtrt::ScalarType::fromString(std::string_view str) {
  auto code = parseElementType(str);
  assert(code != ScalarTypeCode::unknown && "expected known element type code");
  if (code != ScalarTypeCode::unknown)
    return ScalarType(code);
  return getStatusWithMsg(StatusCode::InvalidArgument, "unknown element type (",
                          str, ")");
}

int64_t mtrt::ScalarType::getBitWidth() const {
  int64_t result = getBitsPerElement(code);
  assert(result != 0 && "expected positive bitwidth");
  return result;
}

StatusOr<ScalarTypeCode>
mtrt::ScalarType::getIntegerTypeWithBitWidth(int64_t bitWidth) {
  switch (bitWidth) {
  case 4:
    return ScalarTypeCode::i4;
  case 8:
    return ScalarTypeCode::i8;
  case 16:
    return ScalarTypeCode::i16;
  case 32:
    return ScalarTypeCode::i32;
  case 64:
    return ScalarTypeCode::i64;
  }
  return getInvalidArgStatus("unknown integer type with bit width ({0})",
                             bitWidth);
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

PointerType mtrt::parsePointerType(std::string_view str) {
  if (str == "host")
    return PointerType::host;
  if (str == "pinned_host")
    return PointerType::pinned_host;
  if (str == "device")
    return PointerType::device;
  if (str == "unified")
    return PointerType::unified;
  return PointerType::unknown;
}

llvm::raw_ostream &mtrt::operator<<(llvm::raw_ostream &os,
                                    PointerType ptrType) {
  return os << mtrt::flat::EnumNamePointerType(ptrType);
}

std::string_view mtrt::stringifyPointerType(PointerType ptrType) {
  return mtrt::flat::EnumNamePointerType(ptrType);
}

static bool isDeviceVisible(PointerType type) {
  return type == PointerType::device || type == PointerType::unified;
}

static bool isHostVisible(PointerType type) {
  return !isDeviceVisible(type) || type == PointerType::unified;
}

//===----------------------------------------------------------------------===//
// ExecutableView
//===----------------------------------------------------------------------===//

StatusOr<FunctionView>
ExecutableView::getFunction(std::string_view name) const {
  const flatbuffers::Vector<flatbuffers::Offset<mtrt::flat::Function>>
      &functions = *view->functions();
  auto it = std::find_if(functions.begin(), functions.end(),
                         [&](const mtrt::flat::Function *x) {
                           return x->name()->string_view() == name;
                         });
  if (it == view->functions()->end())
    return getStatusWithMsg(StatusCode::InvalidArgument, "Function with name (",
                            name, ") is not present in the executable");
  return FunctionView(*it);
}

llvm::SmallVector<DataSegmentInfo> ExecutableView::getDataSegments() const {
  llvm::SmallVector<DataSegmentInfo> views;
  views.reserve(view->data_segments()->size());
  for (unsigned i = 0; i < view->data_segments()->size(); i++)
    views.push_back(view->data_segments()->Get(i));
  return views;
}

llvm::SmallVector<FunctionView> ExecutableView::getFunctions() const {
  llvm::SmallVector<FunctionView> views;
  views.reserve(view->functions()->size());
  for (unsigned i = 0; i < view->functions()->size(); i++)
    views.push_back(view->functions()->Get(i));
  return views;
}

//===----------------------------------------------------------------------===//
// ExecutableStorage (Implementations)
//===----------------------------------------------------------------------===//

namespace {
class ExecutableStorageMemBuffer : public ExecutableStorage {
public:
  ExecutableStorageMemBuffer(std::unique_ptr<llvm::MemoryBuffer> storage)
      : storage(std::move(storage)) {}

  std::unique_ptr<ExecutableStorage> getCopy() const final {
    return std::make_unique<ExecutableStorageMemBuffer>(
        llvm::MemoryBuffer::getMemBufferCopy(
            storage->getBuffer(), storage->getBufferIdentifier() + "_copy"));
  }
  const void *data() const final { return storage->getBuffer().data(); }
  size_t size() const final { return storage->getBufferSize(); }

private:
  std::unique_ptr<llvm::MemoryBuffer> storage;
};
} // namespace

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

Executable::Executable(Executable &&other)
    : ExecutableView(nullptr), storage(std::move(other.storage)) {
  other.storage.reset();
  this->view = mtrt::flat::GetExecutable(this->storage->data());
}

StatusOr<std::unique_ptr<Executable>>
Executable::loadFromFile(std::string_view path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (!buffer)
    return getStatusWithMsg(
        StatusCode::InternalError,
        "error loading executable from file: ", buffer.getError().message());

  auto result = std::unique_ptr<Executable>(new Executable(
      std::make_unique<ExecutableStorageMemBuffer>(std::move(*buffer))));

  Status verifyResult = result->verify();
  if (!verifyResult.isOk())
    return verifyResult;
  return result;
}

StatusOr<std::unique_ptr<Executable>>
Executable::loadFromBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer) {
  auto result = std::make_unique<Executable>(
      std::make_unique<ExecutableStorageMemBuffer>(std::move(buffer)));
  Status verifyResult = result->verify();
  if (!verifyResult.isOk())
    return verifyResult;
  return result;
}

StatusOr<std::unique_ptr<Executable>>
Executable::loadFromUnalignedRef(llvm::ArrayRef<char> data) {
  const llvm::Align alignment(16);
  std::unique_ptr<llvm::WritableMemoryBuffer> alignedBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(data.size(), "",
                                                        alignment);
  // `getNewUninitMemBuffer` will return null on failure.
  if (!alignedBuffer)
    return getInvalidArgStatus("failed to create uninitizlied memory buffer of "
                               "size {0} with alignment {1}",
                               data.size(), alignment.value());

  llvm::copy(data, alignedBuffer->getBuffer().begin());
  auto result = std::make_unique<Executable>(
      std::make_unique<ExecutableStorageMemBuffer>(std::move(alignedBuffer)));

  Status verifyResult = result->verify();
  if (!verifyResult.isOk())
    return verifyResult;
  return result;
}

mtrt::Executable::Executable(std::unique_ptr<ExecutableStorage> storage_)
    : ExecutableView(nullptr), storage(std::move(storage_)) {
  assert(this->storage && "expected valid storage pointer");
  this->view = mtrt::flat::GetExecutable(this->storage->data());
}

Executable::~Executable() {}

std::unique_ptr<Executable> Executable::getCopy() const {
  std::unique_ptr<llvm::WritableMemoryBuffer> alignedBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(storage->size(), "",
                                                        llvm::Align(16));
  std::copy_n(reinterpret_cast<const char *>(storage->data()), storage->size(),
              alignedBuffer->getBuffer().begin());
  return std::make_unique<Executable>(
      std::make_unique<ExecutableStorageMemBuffer>(std::move(alignedBuffer)));
}

Status Executable::verify() const {
  flatbuffers::Verifier::Options options{};
  options.max_size = FLATBUFFERS_MAX_64_BUFFER_SIZE;
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t *>(getStorage()->data()),
      getStorage()->size(), options);
  if (!mtrt::flat::VerifyExecutableBuffer(verifier))
    return getStatusWithMsg(
        StatusCode::InvalidArgument,
        "failed to verify that the provided buffer contains "
        "a valid MLIR-TRT Executable");
  return getOkStatus();
}

//===----------------------------------------------------------------------===//
// RuntimeSessionOptions
//===----------------------------------------------------------------------===//
RuntimeSessionOptions::RuntimeSessionOptions(int32_t numDevices,
                                             int32_t deviceId,
                                             llvm::StringRef ncclUuid)
    : numDevices(numDevices), deviceId(deviceId), ncclUuid(ncclUuid) {
  this->enableFeatures({"core"});
}

void RuntimeSessionOptions::enableFeatures(
    llvm::ArrayRef<std::string> toEnable) {
  for (const auto &feature : toEnable) {
    if (feature == "cuda") {
      // If the "cuda" feature is enabled, then we need to enable a module
      // that provides the device ID and number of devices. This is either SPMD
      // (default module for single-device, no NCCL/MPI) or NCCL.
      if (!features.contains("single-device") && !features.contains("nccl"))
        this->enableFeatures({"single-device"});
    }
    // If we enable the NCCL feature, disable the SPMD feature if present.
    if (feature == "nccl") {
      // If the "nccl" feature is enabled, then we override the default
      // "single-device" feature.
      if (auto it = features.find("single-device"); it != features.end())
        features.erase(it);

      // If NCCL is enabled, then enable CUDA as well.
      this->features.insert("cuda");
    }
    this->features.insert(feature);
  }
}

bool RuntimeSessionOptions::isFeatureEnabled(llvm::StringRef feature) const {
  return features.contains(feature);
}

StatusOr<RuntimeSessionOptions>
RuntimeSessionOptions::createUsingSingleHostMpi() {
#ifdef MLIR_TRT_ENABLE_NCCL
  auto getErrStatus = [](llvm::StringRef msg, int32_t errCode) {
    llvm::SmallString<MPI_MAX_ERROR_STRING> str;
    str.resize(MPI_MAX_ERROR_STRING);
    int errClass = 0;
    int errStrLen = 0;
    MPI_Error_class(errCode, &errClass);
    MPI_Error_string(errCode, str.data(), &errStrLen);
    str.resize(errStrLen);
    return getInternalErrorStatus("{0}: [class={1}, msg={2}]", msg, errClass,
                                  str);
  };

  int32_t rank;
  int32_t errCode = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (errCode != MPI_SUCCESS)
    return getErrStatus("MPI_Comm_rank failed", errCode);

  int size;
  errCode = MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (errCode != MPI_SUCCESS)
    return getErrStatus("MPI_Comm_rank failed", errCode);

  std::string uniqueIdStr(NCCL_UNIQUE_ID_BYTES, 0);
  ncclResult_t ncclError = ncclSuccess;
  Status errorStatus = Status::getOk();
  if (rank == 0) {
    ncclUniqueId id;
    ncclError = ncclGetUniqueId(&id);
    if (ncclError == ncclSuccess)
      std::copy_n(id.internal, NCCL_UNIQUE_ID_BYTES, uniqueIdStr.data());
  }

  // Broadcast the NCCL status so we know whether to abort with error on all
  // ranks.
  errCode =
      MPI_Bcast(&ncclError, sizeof(ncclResult_t), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (errCode != MPI_SUCCESS)
    return getErrStatus("MPI_Bcast NCCL unique id status failed", errCode);
  RETURN_ERROR_IF_NCCL_ERROR(ncclError, nullptr);

  // Broadcast the unique ID string.
  errCode = MPI_Bcast(uniqueIdStr.data(), uniqueIdStr.size(), MPI_BYTE, 0,
                      MPI_COMM_WORLD);
  if (errCode != MPI_SUCCESS)
    return getErrStatus("MPI_Comm_rank failed", errCode);

  return RuntimeSessionOptions(size, rank, uniqueIdStr);
#else  // MLIR_TRT_ENABLE_NCCL
  return getInternalErrorStatus(
      "MLIR-TensorRT was not configured and built with MPI and NCCL support");
#endif // MLIR_TRT_ENABLE_NCCL
}

//===----------------------------------------------------------------------===//
// RuntimeSession
//===----------------------------------------------------------------------===//

RuntimeSession::RuntimeSession(RuntimeSessionOptions options,
                               ExecutableView exe, Ref<RuntimeClient> client)
    : client(std::move(client)), options(std::move(options)), executable(exe),
      pinnedMemoryAllocator(std::make_unique<PinnedMemoryAllocator>()),
      allocTracker(std::make_unique<AllocTracker>()),
      resourceTracker(std::make_unique<ResourceTracker>()) {}

//===----------------------------------------------------------------------===//
// AllocTracker
//===----------------------------------------------------------------------===//

AllocTracker::~AllocTracker() {
  MTRT_DBGF("Destroying alloc tracker %p", static_cast<void *>(this));
  MTRT_DBGF("checking %u allocations", map.size());
  llvm::SmallVector<PointerInfo> ptrsToFree;
  ptrsToFree.reserve(map.size());
  for (const auto &[ptrVal, metadata] : map) {
    if (metadata->info.isInternallyManaged()) {
      MTRT_DBGF("still live: 0x%lx type %d size %lu", ptrVal,
                static_cast<int>(metadata->info.type), metadata->info.size);
      ptrsToFree.push_back(metadata->info);
    }
  }

  size_t totalSize = 0;
  for (const PointerInfo &ptr : ptrsToFree) {
    Status s = safeDeallocate(*this, ptr.ptr);
    totalSize += ptr.size;
    if (!s.isOk())
      MTRT_DBGF("error while deallocating dangling memory: %s",
                s.getString().c_str());
  }

  if (totalSize > 0)
    MTRT_DBGF("freed %zu bytes of unfreed memory", totalSize);
}

void AllocTracker::track(PointerInfo info) {
  MTRT_DBGF(
      "AllocTracker %p is now tracking 0x%lx size=%lu space=%s ownership=%s",
      static_cast<void *>(this), info.ptr, info.size,
      mtrt::flat::EnumNamePointerType(info.type),
      mtrt::flat::EnumNamePointerOwner(info.owner));
  auto value = std::make_unique<Metadata>();
  value->info = info;
  if (!contains(info.ptr)) {
    map.insert(std::make_pair(info.ptr, std::move(value)));
    return;
  }
  untrack(info.ptr);
  map.insert(std::make_pair(info.ptr, std::move(value)));
}

void AllocTracker::untrack(uintptr_t ptr) {
  MTRT_DBGF("AllocTracker %p is now untracking 0x%lx)",
            static_cast<void *>(this), ptr);
  MTRT_CHECK(
      !llvm::is_contained(map, ptr),
      llvm::formatv("Pointer {0:X} not found in AllocTracker::untrack\n", ptr));
  map.erase(map.find(ptr));
}

bool AllocTracker::contains(uintptr_t ptr) const { return map.contains(ptr); }

const PointerInfo &AllocTracker::get(uintptr_t ptr) const {
  auto it = map.find(ptr);
  assert(it != map.end() && "expected valid pointer info");
  return it->second->info;
}

PointerInfo AllocTracker::lookupOrDefault(uintptr_t ptr) const {
  if (!contains(ptr))
    return PointerInfo{ptr, PointerInfo::kUnknownSize, PointerType::unknown,
                       PointerOwner::unknown};
  return map.at(ptr)->info;
}

StatusOr<PointerInfo> mtrt::allocate(AllocTracker &tracker, PointerType type,
                                     uint64_t size,
                                     std::optional<uint32_t> alignment,
                                     std::optional<CudaStream> stream) {
  if (type == PointerType::host) {
    assert(alignment && !stream &&
           "expected alignment, no stream for host allocation");
    // Alignment has to be at a multiple of `size`. For small size
    // allocations, make sure to adjust size upward. The frontend may request
    // e.g. 4 bytes aligned to 16 byte boundary because it chose some minimum
    // alignment dumbly.
    alignment = std::max<uint32_t>(*alignment, alignof(std::max_align_t));
    size = llvm::alignTo(size, *alignment);
    uintptr_t mem =
        reinterpret_cast<uintptr_t>(std::aligned_alloc(*alignment, size));
    if (mem == 0)
      return mtrt::getInternalErrorStatus("failed to allocate memory on host");
    MTRT_DBGF("Allocated %lu host bytes at 0x%lx", size, mem);
    PointerInfo info{mem, size, type, PointerOwner::internal};
    tracker.track(info);
    return info;
  }

  if (type == PointerType::device) {
    size = std::max<size_t>(size, 16);
    if (!stream) {
      MTRT_ASSIGN_OR_RETURN(uintptr_t devPtr, mallocCUDA(size));
      PointerInfo info{devPtr, size, type, PointerOwner::internal};
      tracker.track(info);
      return info;
    }
    MTRT_ASSIGN_OR_RETURN(
        uintptr_t devPtr,
        mallocCUDAAsync(size, reinterpret_cast<uintptr_t>(*stream)));
    PointerInfo info{devPtr, size, type, PointerOwner::internal};
    tracker.track(info);
    return info;
  }
  if (type == PointerType::unified) {
    if (stream)
      return getInvalidArgStatus(
          "stream is not allowed when using unified memory allocator");
    MTRT_ASSIGN_OR_RETURN(uintptr_t managedPtr, mallocCUDAManaged(size));
    PointerInfo info{managedPtr, size, type, PointerOwner::internal};
    tracker.track(info);
    return info;
  }
  return getStatusWithMsg(
      mtrt::StatusCode::InvalidArgument,
      "unimplemented allocation type: ", stringifyPointerType(type));
}

mtrt::Status mtrt::safeDeallocate(AllocTracker &tracker, uintptr_t ptr,
                                  std::optional<CudaStream> stream) {
  if (!tracker.contains(ptr)) {
    MTRT_DBGF("ignoring ptr 0x%lx because it was either already freed, "
              "externally managed, or has an external reference",
              ptr);
    return mtrt::Status::getOk();
  }

  PointerInfo obj = tracker.get(ptr);
  if (obj.owner == PointerOwner::external) {
    MTRT_DBGF("Untracking externally managed 0x%lx", ptr);
    tracker.untrack(obj.ptr);
    return mtrt::Status::getOk();
  }

  if (obj.type == PointerType::host) {
    MTRT_DBGF("Freeing host memory %lx", ptr);
    std::free(reinterpret_cast<void *>(obj.ptr));
    tracker.untrack(ptr);
    return Status::getOk();
  }

  if (obj.type == PointerType::device || obj.type == PointerType::unified) {
    if (stream && *stream != 0) {
      MTRT_RETURN_IF_ERROR(
          freeCUDAAsync(obj.ptr, reinterpret_cast<uintptr_t>(*stream)));
    } else {
      MTRT_RETURN_IF_ERROR(freeCUDA(obj.ptr));
    }
    tracker.untrack(ptr);
    return Status::getOk();
  }

  if (obj.type == PointerType::pinned_host) {
    MTRT_DBG(
        "Deferring freeing pinned host memory {0} to pinned memory allocator",
        reinterpret_cast<void *>(obj.ptr));
    return Status::getOk();
  }

  return mtrt::getInternalErrorStatus("unhandled allocation type");
}

//===----------------------------------------------------------------------===//
// ResourceTracker
//===----------------------------------------------------------------------===//

ResourceTracker::~ResourceTracker() {
  for (auto [ptr, deleter] : llvm::reverse(tracker)) {
    assert(ptr && "expected valid pointer");
    assert(deleter && "expected valid deleter pointer");
    MTRT_DBGF("cleaning up resource at 0x%lx", ptr);
    deleter(ptr);
  }
}

void ResourceTracker::track(uintptr_t ptr, Deleter deleter) {
  assert(ptr && deleter && "expected valid ptr and deleter");
  MTRT_DBGF("tracking resource at 0x%lx", ptr);
  tracker.insert(std::make_pair(ptr, deleter));
}

void ResourceTracker::untrack(uintptr_t ptr) { tracker.erase(ptr); }

//===----------------------------------------------------------------------===//
// Device
//===----------------------------------------------------------------------===//

namespace {

/// Wrapper around `cudaSetDevice` and `cudaGetDevice` to ensure that at
/// destruction time, the CUDA device is set to the device active immediately
/// prior to construction.
struct CUDAGPUDeviceGuard final : public DeviceGuard {
  static StatusOr<std::unique_ptr<DeviceGuard>> create(int32_t deviceNumber) {
    StatusOr<int32_t> currentDevice = getCurrentCUDADevice();
    if (!currentDevice.isOk())
      return currentDevice.getStatus();
    int32_t originalDeviceNumber = *currentDevice;
    Status setDeviceStatus = setCurrentCUDADevice(deviceNumber);
    RETURN_STATUS_IF_ERROR(setDeviceStatus.getStatus());
    MTRT_DBG("CUDAGPUDeviceGuard: original={0} new={1}", originalDeviceNumber,
             deviceNumber);
    return std::unique_ptr<DeviceGuard>(
        new CUDAGPUDeviceGuard(originalDeviceNumber));
  }

  ~CUDAGPUDeviceGuard() {
    if (originalDeviceNumber < 0)
      return;
    MTRT_DBG("CUDAGPUDeviceGuard: restoring original device {0}",
             originalDeviceNumber);
    Status setDeviceStatus = setCurrentCUDADevice(originalDeviceNumber);
    if (!setDeviceStatus.isOk())
      llvm::report_fatal_error(setDeviceStatus.getStatus().getString().c_str());
  }

private:
  CUDAGPUDeviceGuard(int32_t originalDeviceNumber)
      : originalDeviceNumber(originalDeviceNumber) {}

  int32_t originalDeviceNumber;
};

class CUDAStream : public Stream {
public:
  ~CUDAStream() {
    Status s = destroyCUDAStream(stream);
    if (!s.isOk())
      mtrt::logUnhandledErrors(s, llvm::errs());
  }
  CudaStream getCUDAHandle() const final { return stream; }
  Device *getDevice() const final { return device; }
  Status sync() final { return synchronizeCUDAStream(stream); }
  static StatusOr<Ref<CUDAStream>> create(Device *device) {
    MTRT_ASSIGN_OR_RETURN(
        std::unique_ptr<DeviceGuard> deviceGuard,
        CUDAGPUDeviceGuard::create(device->getDeviceNumber()));
    MTRT_ASSIGN_OR_RETURN(uintptr_t stream, createCUDAStream());
    return Ref<CUDAStream>(new CUDAStream(stream, device));
  }

private:
  CUDAStream(CudaStream stream, Device *device)
      : stream(stream), device(device) {}

  /// Opaque handle to the underlying CUDA stream object.
  CudaStream stream;

  /// Pointer to the associated Device. Device's are uniquely owned by
  /// the RuntimeClient, so their lifetime should always outlive other
  /// objects, including Streams, which can only be created from Device
  /// objects.
  Device *device;
};

/// CUDA device implementation of the Device abstract class.
class CUDADevice final : public Device {
public:
  ~CUDADevice() = default;
  llvm::StringRef getDeviceKind() const final { return "cuda"; }
  HardwareId getDeviceNumber() const final { return deviceNumber; }
  StatusOr<std::unique_ptr<DeviceGuard>> createDeviceGuard() const final {
    MTRT_ASSIGN_OR_RETURN(std::unique_ptr<DeviceGuard> deviceGuard,
                          CUDAGPUDeviceGuard::create(deviceNumber));
    return deviceGuard;
  }
  llvm::StringRef getDeviceName() const final { return deviceName; }

  Ref<Stream> getStream() const final {
    assert(stream && "expected valid stream");
    return stream;
  }

  llvm::ThreadPoolInterface &getThreadPool() final {
    assert(stream && "expected valid stream");
    return threadPool;
  }

  static StatusOr<std::unique_ptr<Device>> create(HardwareId deviceNumber) {
    auto result =
        std::unique_ptr<CUDADevice>(new CUDADevice(deviceNumber, nullptr));
    MTRT_ASSIGN_OR_RETURN(Ref<CUDAStream> stream,
                          CUDAStream::create(result.get()));
    result->stream = std::move(stream);
    // Convert the pointer type to Device.
    return StatusOr<std::unique_ptr<Device>>(std::move(result));
  }

protected:
  CUDADevice(HardwareId deviceNumber, Ref<CUDAStream> stream)
      : deviceNumber(deviceNumber), stream(std::move(stream)), threadPool() {
    deviceName = llvm::formatv("cuda:{0}", deviceNumber).str();
  }

  const HardwareId deviceNumber;
  std::string deviceName;

  /// Reference the the stream associated with this device, and streams take a
  /// back reference-by-pointer to the owning device. This is the initial
  /// stream reference, but as the program runs references will also be taken by
  /// potentially long-lived objects such as MemRefStorage.
  /// RuntimeClient must always outlive all other objects, so the lifetime
  /// of the Device should outlive any child stream associated with it.
  Ref<CUDAStream> stream;

  llvm::StdThreadPool threadPool;
};
} // namespace

StatusOr<std::unique_ptr<Device>>
mtrt::createCUDADevice(HardwareId deviceNumber) {
  return CUDADevice::create(deviceNumber);
}

//===----------------------------------------------------------------------===//
// Event
//===----------------------------------------------------------------------===//

namespace {
// Completion token transported to CUDA callback and IO worker.
// T is the result type.
struct CompletionToken
    : public llvm::ThreadSafeRefCountedBase<CompletionToken> {
  CompletionToken(Device *device, llvm::ThreadPoolInterface *ioThreadPool,
                  Event *event)
      : device(device), ioThreadPool(ioThreadPool), event(std::move(event)) {
    assert(this->device && "expected valid device");
    assert(this->ioThreadPool && "expected valid io thread pool");
    assert(this->event && "expected valid event");
    assert(this->event->getCudaHandle() && "expected valid cuda event handle");
  }
  Device *device;
  llvm::ThreadPoolInterface *ioThreadPool;
  Event *event;
};
} // namespace

// Host callback invoked by CUDA runtime when stream work preceding it
// completes. userData is a heap-allocated pointer to
// std::shared_ptr<CompletionToken<T>>. Implementation posts a small job to IO
// thread pool and returns.
void cuda_event_host_callback(void *userData) {
  // // userData is pointer-to-heap-allocated Ref<CompletionToken>.
  Ref<CompletionToken> *tokenPtr =
      static_cast<Ref<CompletionToken> *>(userData);
  // Increment reference count by making a copy.
  Ref<CompletionToken> token = *tokenPtr;
  // Then delete the heap allocated reference.
  delete tokenPtr;

  // Use the IO thread pool to run the callback.
  llvm::ThreadPoolInterface *ioThreadPool = token->ioThreadPool;
  ioThreadPool->async([token = std::move(token)]() {
    Status s = Status::getOk();

    // Create device guard
    StatusOr<std::unique_ptr<DeviceGuard>> deviceGuard =
        token->device->createDeviceGuard();
    mtrt::cantFail(deviceGuard.getStatus());
    mtrt::StatusOr<bool> eventStatus =
        mtrt::queryCUDAEvent(token->event->getCudaHandle());
    if (!eventStatus.isOk())
      s = std::move(eventStatus.getStatus());
    else if (!*eventStatus)
      s = mtrt::getInternalErrorStatus(
          "event is not ready, but host callback was invoked");

    // May invoke callbacks.
    token->event->setReady(std::move(s));
  });
}

Event::Event(bool ready, Status status, CudaEvent cudaEventHandle)
    : cudaEventHandle(cudaEventHandle), ready(ready),
      status(std::move(status)) {}

std::unique_ptr<Event> Event::createReadyEvent() {
  return std::unique_ptr<Event>(new Event(true, Status::getOk(), CudaEvent(0)));
}

StatusOr<std::unique_ptr<Event>> Event::create(Ref<Stream> stream) {
  assert(stream && "expected valid stream");
  MTRT_ASSIGN_OR_RETURN(CudaEvent eventHandle, mtrt::createCUDAEvent());
  MTRT_RETURN_IF_ERROR(
      mtrt::recordCUDAEvent(eventHandle, stream->getCUDAHandle()));

  auto eventRef =
      std::unique_ptr<Event>(new Event(false, Status::getOk(), eventHandle));
  auto heapPtr = new Ref<CompletionToken>(new CompletionToken(
      stream->getDevice(), &stream->getDevice()->getThreadPool(),
      eventRef.get()));
  // Schedule host callback to run after all previously enqueued work on
  // stream finishes. The callback runs on a CUDA runtime thread and is
  // forbidden from invoking CUDA APis. Instead, it just sets up the completion
  // handler to run on the device's thread pool. The completion handler checks
  // for CUDA errors, sets the event state to ready, and invokes the callbacks,
  // which may include deleting the event.
  MTRT_RETURN_IF_ERROR(mtrt::launchCUDAHostFunc(stream->getCUDAHandle(),
                                                &cuda_event_host_callback,
                                                static_cast<void *>(heapPtr)));
  return eventRef;
}

/// Complete the lifecycle of an Event by appending an on-ready callback that
/// deletes the event.
void Event::releaseWhenReady(std::unique_ptr<Event> event) {
  // Move the event to the callback so that the event is deleted after the
  // callback is invoked. Deletion will occur on whatever thread is executing
  // the callbacks unless the Event is already ready, in which case deletion
  // occurs immediately.
  event->addReadyCallback(
      [](Status, void *userData) mutable {
        auto event =
            std::unique_ptr<Event>(reinterpret_cast<Event *>(userData));
        event.reset();
      },
      event.release());
}

Event::~Event() {
  MTRT_DBG("destroying mtrt::Event {0} ready={1} status={2}", this, ready,
           status.isOk() ? "ok" : status.getString());
  if (cudaEventHandle) {
    mtrt::logUnhandledErrors(mtrt::destroyCUDAEvent(cudaEventHandle),
                             llvm::errs());
    this->cudaEventHandle = 0;
    return;
  }
}

bool Event::checkIsReady() {
  std::lock_guard<std::mutex> g(lock);
  return ready;
}

void Event::setReady(Status s) {
  std::vector<std::pair<OnReadyCallback, void *>> callbacksCopy;

  {
    // Copy callbacks to avoid race condition.
    std::lock_guard<std::mutex> g(lock);
    if (ready)
      return;
    this->ready = true;
    this->status = s;
    std::swap(this->callbacks, callbacksCopy);
  }

  for (const auto &[callback, userData] : callbacksCopy) {
    MTRT_DBG("mtrt::Event {0} invoking callback with data {1}",
             reinterpret_cast<void *>(this), userData);
    callback(s, userData);
  }
}

Status Event::waitForReady(std::chrono::microseconds timeout) {
  if (this->checkIsReady())
    return getStatus();
  CudaEvent e = getCudaHandle();
  if (!e)
    return getInternalErrorStatus("Event {0} has invalid CUDA event handle",
                                  this);
  std::chrono::duration<double, std::micro> elapsed(0.0);
  while (elapsed < timeout) {
    MTRT_ASSIGN_OR_RETURN(bool evReady, mtrt::queryCUDAEvent(e));
    // The first boolean check is for the CUDA event, the second is to ensure
    // the event callbacks have been invoked.
    if (!evReady || !checkIsReady()) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
      elapsed += std::chrono::microseconds(10);
      continue;
    }
    return getStatus();
  }
  return getInternalErrorStatus("Event {0} timed out", this);
}

void Event::setStatus(Status status) {
  std::lock_guard<std::mutex> g(lock);
  this->status = std::move(status);
}

Status Event::getStatus() {
  std::lock_guard<std::mutex> g(lock);
  return status;
}

void Event::addReadyCallback(OnReadyCallback callback, void *userData) {
  Status s = Status::getOk();
  {
    std::lock_guard<std::mutex> g(lock);
    if (!ready) {
      MTRT_DBG("mtrt::Event {0} is not ready, enqueueing callback",
               reinterpret_cast<void *>(this));
      callbacks.push_back(std::make_pair(callback, userData));
      return;
    }
    s = this->status;
  }
  MTRT_DBG(
      "mtrt::Event {0} is already in 'ready' state, call callback immediately",
      reinterpret_cast<void *>(this));
  callback(s, userData);
}

//===----------------------------------------------------------------------===//
// ScalarValue
//===----------------------------------------------------------------------===//

ScalarValue::ScalarValue(ScalarValue &&other) noexcept
    : RuntimeValue(Kind::Scalar), type(other.type) {
  if (other.isComplex()) {
    data.complex = other.data.complex;
    other.data.complex = nullptr;
  } else {
    data.real = other.data.real;
  }
}

ScalarValue &ScalarValue::operator=(ScalarValue &&other) noexcept {
  if (this != &other) {
    cleanup();
    type = other.type;
    if (other.isComplex()) {
      data.complex = other.data.complex;
      other.data.complex = nullptr;
    } else {
      data.real = other.data.real;
    }
  }
  return *this;
}

ScalarValue::~ScalarValue() { cleanup(); }

void ScalarValue::cleanup() {
  if (isComplex()) {
    if (type.getCode() == ScalarTypeCode::complex32) {
      delete static_cast<std::complex<float> *>(data.complex);
    } else {
      delete static_cast<std::complex<double> *>(data.complex);
    }
  }
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

BufferType BufferType::createWithByteStrides(
    mtrt::ScalarType elementType, const std::vector<int64_t> &shape,
    const std::vector<int64_t> &byteStrides, mtrt::PointerType addressSpace,
    int64_t offset) {
  BufferType result;
  result.elementType = elementType.getCode();
  result.shape = shape;
  result.addressSpace = addressSpace;
  result.layout.offset = offset;

  int64_t bytesPerElement = llvm::divideCeil(elementType.getBitWidth(), 8);
  for (int64_t s : byteStrides) {
    assert(s % bytesPerElement == 0 &&
           "expected stride to be a multiple of the element size");
    result.layout.strides.push_back(s / bytesPerElement);
  }

  return result;
}

BufferType BufferType::createWithElementStrides(
    mtrt::ScalarType elementType, const std::vector<int64_t> &shape,
    const std::vector<int64_t> &elementStrides, mtrt::PointerType addressSpace,
    int64_t offset) {
  BufferType result;
  result.elementType = elementType.getCode();
  result.shape = shape;
  result.layout.strides = elementStrides;
  result.layout.offset = offset;
  result.addressSpace = addressSpace;
  return result;
}

BufferType
BufferType::createWithCanonicalLayout(mtrt::ScalarType elementType,
                                      const std::vector<int64_t> &shape,
                                      mtrt::PointerType addressSpace) {
  return BufferType(elementType, shape,
                    BufferStridedLayout::createCanonicalRowMajor(shape),
                    addressSpace);
}

bool BufferType::hasStaticShape() const {
  return llvm::all_of(shape, [](int64_t v) { return v != mtrt::kDynamicSize; });
}

int64_t BufferType::getNumElements() const {
  assert(hasStaticShape() && "expected known shape");
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

BufferType BufferType::getFromSerializedType(const mtrt::MemRefTypeView &type) {
  /// Serialized types in function signature currently have implicit zero
  /// offset.
  return BufferType::createWithElementStrides(
      type.getElementType(), type.getShape(), type.getStrides(),
      type.getAddressSpace(), /*offset=*/0);
}

bool BufferType::isCanonicalRowMajor() const {
  if (llvm::any_of(shape, [](int64_t v) { return v == 0; }))
    return true;
  llvm::ArrayRef<int64_t> strides = getLayout().getStrides();
  assert(strides.size() == shape.size() &&
         "expected equal rank shape and strides");
  if (!strides.empty()) {
    int64_t stride = 1;
    for (int64_t i = layout.strides.size() - 1; i >= 0; --i) {
      if (strides[i] != stride) {
        if (shape[i] > 1 || (shape[i] == 1 && strides[i] != 1))
          return false;
      }
      stride *= shape[i];
    }
  }
  return true;
}

bool BufferType::isCanonicalColMajor() const {
  if (llvm::any_of(getShape(), [](int64_t v) { return v == 0; }))
    return true;
  if (getRank() > 0) {
    int64_t stride = 1;
    llvm::ArrayRef<int64_t> strides = getLayout().getStrides();
    assert(shape.size() == strides.size() &&
           "expected equal rank shape and strides");
    for (unsigned i = 0; i < strides.size(); i++) {
      if (strides[i] != stride) {
        if (shape[i] > 1 || (shape[i] == 1 && strides[i] != 1))
          return false;
      }
      stride *= getShape()[i];
    }
  }
  return true;
}

std::vector<int64_t> BufferType::getByteStrides() const {
  std::vector<int64_t> byteStrides(layout.strides.size());
  unsigned bytesPerElement =
      llvm::divideCeil(mtrt::ScalarType(elementType).getBitWidth(), 8);
  for (unsigned i = 0, e = layout.strides.size(); i < e; i++)
    byteStrides[i] = layout.strides[i] * bytesPerElement;
  return byteStrides;
}

bool BufferType::isCanonicalPacked() const {
  return isCanonicalRowMajor() || isCanonicalColMajor();
}

static StatusOr<int64_t> getFootprintInBytes(llvm::ArrayRef<int64_t> shape,
                                             llvm::ArrayRef<int64_t> strides,
                                             int64_t bitsPerElement) {
  assert(shape.size() == strides.size() &&
         "expected equal rank shape and strides");
  if (llvm::find(shape, 0) != shape.end())
    return 0;
  if (!llvm::all_of(strides, [](int64_t x) { return x >= 0; }))
    return getInvalidArgStatus("only positive strides are supported");

  int64_t elementByteSize = llvm::divideCeil(bitsPerElement, 8);
  int64_t sizeBytes = elementByteSize;
  if (shape.empty())
    return sizeBytes;

  // Add the offset for the element positioned furthest away.
  for (auto [dimExtent, stride] : llvm::zip_equal(shape, strides))
    sizeBytes += (dimExtent - 1) * stride * elementByteSize;

  return sizeBytes;
}

size_t BufferType::getFootprintSizeInBytes() const {
  auto shape = getShape();
  assert(hasStaticShape() && "expected known shape");
  StatusOr<int64_t> sizeBytes = getFootprintInBytes(
      shape, layout.strides, mtrt::ScalarType(elementType).getBitWidth());
  assert(sizeBytes.isOk() && "expected valid size bytes");
  return *sizeBytes;
}

std::string BufferStridedLayout::toString() const {
  std::string str = "layout<";
  llvm::raw_string_ostream ss(str);
  ss << "offset=" << offset << ", strides=[";
  llvm::interleaveComma(strides, ss);
  ss << "]>";
  return str;
}

/// Compute canonical row major strides. Here we use the PyTorch convention that
/// unit dimensions have unit stride.
static std::vector<int64_t>
getCanonicalRowMajorStrides(llvm::ArrayRef<int64_t> shape) {
  if (shape.empty())
    return {};
  if (llvm::any_of(shape, [](int64_t v) { return v == 0; }))
    return std::vector<int64_t>(shape.size(), 0);
  // Compute reverse suffix product for strides.
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = shape[i] == 1 ? 1 : stride;
    stride *= shape[i];
  }
  return strides;
}

BufferStridedLayout
BufferStridedLayout::createCanonicalRowMajor(llvm::ArrayRef<int64_t> shape) {
  return BufferStridedLayout(getCanonicalRowMajorStrides(shape), 0);
}

std::ostream &mtrt::operator<<(std::ostream &os, const BufferType &t) {
  os << "buffer<";
  for (auto x : t.getShape())
    os << x << "x";
  os << mtrt::flat::EnumNameScalarTypeCode(t.getElementType().getCode());
  os << ", " << t.getLayout().toString() << ">";
  return os;
}

std::string BufferType::toString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

//===----------------------------------------------------------------------===//
// MemRefValue
//===----------------------------------------------------------------------===//

namespace {
class HostOwnedMemRefStorage : public MemRefStorage {
public:
  HostOwnedMemRefStorage(uintptr_t ptr, Ref<RuntimeClient> client)
      : MemRefStorage(ptr, nullptr, std::move(client)) {}

  ~HostOwnedMemRefStorage();

  PointerType getMemorySpace() const final { return PointerType::host; }
};

class DeviceOwnedMemRefStorage : public MemRefStorage {
public:
  DeviceOwnedMemRefStorage(uintptr_t ptr, Device *device,
                           Ref<RuntimeClient> client, PointerType type,
                           Ref<Stream> stream)
      : MemRefStorage(ptr, device, std::move(client)), type(type),
        stream(std::move(stream)) {}

  ~DeviceOwnedMemRefStorage();

  PointerType getMemorySpace() const final { return type; }
  Ref<Stream> getStream() const final { return stream; }

  PointerType type;
  Ref<Stream> stream;
};

class ViewMemRefStorage : public MemRefStorage {
public:
  ViewMemRefStorage(uintptr_t ptr, Device *device, PointerType type,
                    std::function<void()> destroyCallback,
                    Ref<RuntimeClient> client)
      : MemRefStorage(ptr, device, std::move(client)), type(type),
        destroyCallback(std::move(destroyCallback)) {}
  ~ViewMemRefStorage();

  PointerType getMemorySpace() const final { return type; }

  PointerType type;
  std::function<void()> destroyCallback;
};

} // namespace

static StatusOr<Ref<MemRefStorage>>
getOwnedMemRefStorage(uintptr_t ptr, PointerType kind, Device *device,
                      Ref<RuntimeClient> client, Ref<Stream> stream) {
  switch (kind) {
  case PointerType::host:
    return Ref<MemRefStorage>(
        new HostOwnedMemRefStorage(ptr, std::move(client)));
  case PointerType::device:
  case PointerType::unified:
  case PointerType::pinned_host:
    return Ref<MemRefStorage>(new DeviceOwnedMemRefStorage(
        ptr, device, std::move(client), kind, std::move(stream)));
  default:
    return getInvalidArgStatus("unsupported MemRefStorage address space");
  }
}

static StatusOr<Ref<MemRefStorage>>
getViewMemRefStorage(uintptr_t ptr, PointerType kind, Device *device,
                     std::function<void()> destroyCallback,
                     Ref<RuntimeClient> client) {
  return Ref<MemRefStorage>(new ViewMemRefStorage(
      ptr, device, kind, std::move(destroyCallback), std::move(client)));
}

HostOwnedMemRefStorage::~HostOwnedMemRefStorage() {
  MTRT_DBGF("HostOwnedMemRefStorage::~HostOwnedMemRefStorage() ptr = %p",
            reinterpret_cast<void *>(ptr));
  mtrt::logUnhandledErrors(client->getAllocator().deallocate(*this),
                           llvm::errs());
}

DeviceOwnedMemRefStorage::~DeviceOwnedMemRefStorage() {
  MTRT_DBGF("DeviceOwnedMemRefStorage::~DeviceOwnedMemRefStorage() ptr = %p",
            reinterpret_cast<void *>(ptr));
  mtrt::logUnhandledErrors(client->getAllocator().deallocate(*this),
                           llvm::errs());
}

ViewMemRefStorage::~ViewMemRefStorage() {
  MTRT_DBGF("ViewMemRefStorage::~ViewMemRefStorage() ptr = %p",
            reinterpret_cast<void *>(ptr));
  if (destroyCallback)
    destroyCallback();
}

static llvm::SmallVector<int64_t>
getCanonicalStride(const llvm::ArrayRef<int64_t> &shape) {
  if (shape.empty())
    return {};

  llvm::SmallVector<int64_t> canonicalStride(shape.size(), 1);
  int64_t cumulativeProduct = 1;

  for (int64_t dimIndex = shape.size() - 1; dimIndex >= 0; --dimIndex) {
    bool isFirstZeroDim = (shape[dimIndex] == 0 &&
                           dimIndex != static_cast<int64_t>(shape.size()) - 1);
    // For dimensions with size 0 or 1, the stride can be arbitrary.
    // We set it to 1 here, but other values would also be valid.
    if (isFirstZeroDim || shape[dimIndex] == 1)
      canonicalStride[dimIndex] = 1;
    else
      canonicalStride[dimIndex] = cumulativeProduct;
    // For zero-sized dimensions (except the last one), we don't update the
    // cumulative product This allows for consistent handling of zero-sized
    // dimensions across different frameworks
    cumulativeProduct *= isFirstZeroDim ? 1 : shape[dimIndex];
  }

  return canonicalStride;
}

static bool areStridesEquivalent(llvm::ArrayRef<int64_t> shape,
                                 llvm::ArrayRef<int64_t> stride,
                                 llvm::ArrayRef<int64_t> expectedStride) {
  if (shape.size() != stride.size() || shape.size() != expectedStride.size())
    return false;

  for (size_t i = 0; i < shape.size(); ++i)
    // Allow arbitrary strides for dimensions with size 0 or 1
    // This accounts for discrepancies in how different frameworks handle
    // these cases
    if (stride[i] != expectedStride[i] && shape[i] != 0 && shape[i] != 1)
      return false;

  return true;
}

StatusOr<std::unique_ptr<MemRefValue>>
MemRefValue::create(mtrt::PointerType addressSpace, ScalarTypeCode elementType,
                    Ref<MemRefStorage> storage, int64_t offset,
                    llvm::ArrayRef<int64_t> shape,
                    llvm::ArrayRef<int64_t> strides, Device *device,
                    bool assertCanonicalStrides) {
  if (!storage->getClient())
    return getInvalidArgStatus("a valid RuntimeClient must be provided to "
                               "create a tracked MemRef object");
  if (!::getFootprintInBytes(shape, strides,
                             ScalarType(elementType).getBitWidth())
           .isOk())
    return getInvalidArgStatus(
        "only memrefs with non-negative strides are allowed");

  auto isEmptyTensor = [](llvm::ArrayRef<int64_t> shape) -> bool {
    return std::any_of(shape.begin(), shape.end(),
                       [](int64_t s) { return s == 0; });
  };

  if (!storage->getPtr() && !isEmptyTensor(shape))
    return getInvalidArgStatus("MemRef objects must be created with a "
                               "valid pointer for a non-empty "
                               "tensor");

  if (isDeviceVisible(addressSpace) && !device)
    return getInvalidArgStatus("a specific device must be provided for MemRefs "
                               "that are device-visible");

  // Check if given strides match canonical stride
  if (assertCanonicalStrides) {
    llvm::SmallVector<int64_t> canonicalStride = getCanonicalStride(shape);
    if (!strides.empty() &&
        !areStridesEquivalent(shape, strides, canonicalStride)) {
      std::string errorMsg =
          llvm::formatv("Given strides [{0}] do not match canonical strides "
                        "[{1}] for shape [{2}]",
                        strides, canonicalStride, shape);
      return getInvalidArgStatus(errorMsg.c_str());
    }
  }

  return std::unique_ptr<MemRefValue>(
      new MemRefValue(addressSpace, elementType, std::move(storage), offset,
                      shape, strides, device));
}

MemRefValue::MemRefValue(mtrt::PointerType addressSpace,
                         ScalarTypeCode elementType, Ref<MemRefStorage> storage,
                         int64_t offset, llvm::ArrayRef<int64_t> shape,
                         llvm::ArrayRef<int64_t> strides, Device *device)
    : RuntimeValue(Kind::MemRef), storage(std::move(storage)), device(device) {
  assert(offset == 0 && "offset must be 0");
  type = BufferType(elementType, shape, strides, addressSpace, offset);
}

int64_t MemRefValue::getTotalFootprintInBytes() const {
  return type.getFootprintSizeInBytes();
}

//===----------------------------------------------------------------------===//
// RuntimeClient
//===----------------------------------------------------------------------===//

namespace {
class DefaultClientAllocator final : public RuntimeClientAllocator {
public:
  DefaultClientAllocator(RuntimeClient &client)
      : RuntimeClientAllocator(client) {}

  StatusOr<Ref<MemRefStorage>> allocate(PointerType type, uint64_t size,
                                        std::optional<uint32_t> alignment,
                                        Device *device,
                                        Ref<Stream> stream) final;

  StatusOr<Ref<MemRefStorage>> takeOwnership(uintptr_t ptr, PointerType type,
                                             Device *device,
                                             Ref<Stream> stream) final;

  Status deallocate(MemRefStorage &storage) final;
};
} // namespace

StatusOr<Ref<MemRefStorage>>
DefaultClientAllocator::allocate(PointerType type, uint64_t size,
                                 std::optional<uint32_t> alignment,
                                 Device *device, Ref<Stream> stream) {
  if (type == PointerType::host) {
    assert(alignment && !stream &&
           "expected alignment, no stream for host allocation");
    // Alignment has to be at a multiple of `size`. For small size
    // allocations, make sure to adjust size upward. The frontend may
    // request e.g. 4 bytes aligned to 16 byte boundary because it chose
    // some minimum alignment dumbly.
    alignment = std::max<uint32_t>(*alignment, alignof(std::max_align_t));
    size = llvm::alignTo(size, *alignment);
    void *mem = std::aligned_alloc(*alignment, size);
    if (mem == 0)
      return mtrt::getInternalErrorStatus("failed to allocate memory on host");
    MTRT_DBGF(
        "DefaultClientAllocator::allocate: Allocated %lu host bytes at %p",
        size, mem);

    return getOwnedMemRefStorage(reinterpret_cast<uintptr_t>(mem),
                                 PointerType::host, nullptr,
                                 Ref<RuntimeClient>(&client), nullptr);
  }

  if (type == PointerType::device) {
    assert(device && "expected valid device");
    size = std::max<size_t>(size, 16);
    MTRT_ASSIGN_OR_RETURN(auto deviceGuard, device->createDeviceGuard());
    if (!stream) {
      MTRT_ASSIGN_OR_RETURN(uintptr_t devPtr, mallocCUDA(size));
      return getOwnedMemRefStorage(devPtr, PointerType::device, device,
                                   Ref<RuntimeClient>(&client), nullptr);
    }
    MTRT_DBG("DefaultClientAllocator::allocate: Allocating {0} device bytes at "
             "{1:x}",
             size, stream.get());
    assert(stream && stream->getCUDAHandle() && "expected valid stream");
    MTRT_ASSIGN_OR_RETURN(uintptr_t devPtr,
                          mallocCUDAAsync(size, stream->getCUDAHandle()));
    return getOwnedMemRefStorage(devPtr, PointerType::device, device,
                                 Ref<RuntimeClient>(&client), stream);
  }
  if (type == PointerType::unified) {
    if (stream)
      return getInvalidArgStatus(
          "stream is not allowed when using unified memory allocator");
    MTRT_ASSIGN_OR_RETURN(uintptr_t managedPtr, mallocCUDAManaged(size));
    return getOwnedMemRefStorage(managedPtr, PointerType::unified, nullptr,
                                 Ref<RuntimeClient>(&client), nullptr);
  }
  if (type == PointerType::pinned_host) {
    MTRT_ASSIGN_OR_RETURN(uintptr_t hostPtr, mallocCUDAPinnedHost(size));
    return getOwnedMemRefStorage(hostPtr, PointerType::pinned_host, nullptr,
                                 Ref<RuntimeClient>(&client), nullptr);
  }
  return getStatusWithMsg(
      mtrt::StatusCode::InvalidArgument,
      "DeviceClientAllocator::allocate unimplemented allocation type: ",
      stringifyPointerType(type));
}

StatusOr<Ref<MemRefStorage>>
DefaultClientAllocator::takeOwnership(uintptr_t ptr, PointerType type,
                                      Device *device, Ref<Stream> stream) {
  return getOwnedMemRefStorage(ptr, type, device, Ref<RuntimeClient>(&client),
                               stream);
}

Status DefaultClientAllocator::deallocate(MemRefStorage &storage) {
  PointerType pointerType = storage.getMemorySpace();
  Ref<Stream> stream = storage.getStream();
  uintptr_t ptr = storage.getPtr();

  if (pointerType == PointerType::host) {
    MTRT_DBGF("Freeing host memory %lx", ptr);
    std::free(reinterpret_cast<void *>(ptr));
    return Status::getOk();
  }

  if (pointerType == PointerType::device ||
      pointerType == PointerType::unified) {
    if (stream) {
      MTRT_RETURN_IF_ERROR(freeCUDAAsync(
          ptr, reinterpret_cast<uintptr_t>(stream->getCUDAHandle())));
    } else {
      MTRT_RETURN_IF_ERROR(freeCUDA(ptr));
    }
    return Status::getOk();
  }

  if (pointerType == PointerType::pinned_host) {
    MTRT_RETURN_IF_ERROR(freeCUDAPinnedHost(ptr));
    return Status::getOk();
  }

  return mtrt::getInternalErrorStatus("unhandled allocation type");
}

static Status parseDebugFlags() {
  std::vector<const char *> argv = {"mlir-tensorrt-runtime"};
  std::string error;
  llvm::raw_string_ostream ss(error);
  if (!llvm::cl::ParseCommandLineOptions(argv.size(), argv.data(),
                                         "MLIR-TRT Runtime flags", &ss,
                                         "MTRT_FLAGS", false)) {
    ss.flush();
    return getInternalErrorStatus("Failed to parse MTRT_FLAGS options: {0}",
                                  error.c_str());
  }
  return Status::getOk();
}

// Populate the devices with CUDA devices. Note that failure to enumerate CUDA
// devices is not treated as an error here.
static mtrt::Status
populateDevices(llvm::SmallVectorImpl<std::unique_ptr<Device>> &devices) {
  StatusOr<int32_t> deviceCount = getCUDADeviceCount();
  if (!deviceCount.isOk())
    return getOkStatus();
  devices.reserve(*deviceCount);
  for (int32_t i = 0; i < *deviceCount; ++i) {
    MTRT_ASSIGN_OR_RETURN(std::unique_ptr<Device> device,
                          CUDADevice::create(HardwareId(i)));
    devices.push_back(std::move(device));
  }
  return getOkStatus();
}

StatusOr<Ref<RuntimeClient>> RuntimeClient::create() {
  static llvm::once_flag onceFlag{};
  llvm::call_once(onceFlag, parseDebugFlags);

  // Setup device objects. Create a view of the device pointers.
  llvm::SmallVector<std::unique_ptr<Device>> devices;
  mtrt::Status status = populateDevices(devices);
  if (!status.isOk()) {
    // TODO: we should emit a warning here.
  }

  auto client = Ref<RuntimeClient>(new RuntimeClient(std::move(devices)));
  auto defaultAllocator = std::make_unique<DefaultClientAllocator>(*client);
  client->setAllocator(std::move(defaultAllocator));
  return client;
}

llvm::ArrayRef<std::unique_ptr<Device>> RuntimeClient::getDevices() const {
  return devices;
}

StatusOr<std::unique_ptr<MemRefValue>>
RuntimeClient::allocateMemRef(const BufferType &type, Device *device,
                              Ref<Stream> stream, bool assertCanonicalStrides) {
  if (type.getAddressSpace() == PointerType::device ||
      type.getAddressSpace() == PointerType::unified) {
    if (!device)
      return getInvalidArgStatus("a specific device must be specified when "
                                 "creating a device buffer");
  }

  int64_t bitsPerElement = type.getElementType().getBitWidth();

  // Allocate required memory.
  StatusOr<int64_t> allocationSizeBytes = getFootprintInBytes(
      type.getShape(), type.getLayout().getStrides(), bitsPerElement);
  if (!allocationSizeBytes.isOk())
    return allocationSizeBytes.getStatus();

  StatusOr<Ref<MemRefStorage>> storage =
      allocator->allocate(type.getAddressSpace(), *allocationSizeBytes,
                          /*alignment=*/16, device, stream);
  if (!storage.isOk())
    return storage.getStatus();

  // Create the descriptor.
  StatusOr<std::unique_ptr<MemRefValue>> bufferImpl = MemRefValue::create(
      type.getAddressSpace(), type.getElementType(), std::move(*storage),
      type.getLayout().getOffset(), type.getShape(),
      type.getLayout().getStrides(), device, assertCanonicalStrides);
  if (bufferImpl.isError())
    return bufferImpl.getStatus();

  return std::move(*bufferImpl);
}

StatusOr<std::unique_ptr<MemRefValue>>
RuntimeClient::createExternalMemRef(const BufferType &type, uintptr_t ptr,
                                    Device *device, bool assertCanonicalStrides,
                                    std::function<void()> destroyCallback) {
  StatusOr<Ref<MemRefStorage>> storage = getViewMemRefStorage(
      ptr, type.getAddressSpace(), nullptr, std::move(destroyCallback),
      Ref<RuntimeClient>(this));
  if (!storage.isOk())
    return storage.getStatus();

  // Create the descriptor.
  StatusOr<std::unique_ptr<MemRefValue>> memref = MemRefValue::create(
      type.getAddressSpace(), type.getElementType(), std::move(*storage),
      type.getLayout().getOffset(), type.getShape(),
      type.getLayout().getStrides(), device, assertCanonicalStrides);
  if (!memref.isOk())
    return memref.getStatus();

  return memref;
}

StatusOr<std::unique_ptr<MemRefValue>>
RuntimeClient::copyHostToHost(const MemRefValue &hostBuffer) {
  if (!isHostVisible(hostBuffer.getAddressSpace()))
    return getInvalidArgStatus("to create a new host copy, data of the source "
                               "MemRef must reside in an address space "
                               "that the host can access");
  // Get the total buffer footprint.
  int64_t totalBufferSize = hostBuffer.getTotalFootprintInBytes();

  // Allocate a new host MemRef
  StatusOr<std::unique_ptr<MemRefValue>> hostMemRef =
      allocateMemRef(hostBuffer.getType(), /*device=*/nullptr,
                     /*stream=*/nullptr);
  if (!hostMemRef.isOk())
    return hostMemRef.getStatus();

  // Fill new buffer with the data from the source buffer.
  std::memcpy((*hostMemRef)->getVoidPtr(), hostBuffer.getVoidPtr(),
              totalBufferSize);
  return std::move(*hostMemRef);
}

/// Copy the `sourceBuffer` (on host) to the `stagingBuffer` (on host). If the
/// buffers types are equal and are one of the canonical packed layouts, then
/// we use a single memcpy. Otherwise, we execute a generic strided copy. The
/// strided copy is not efficient at all, but it is also very uncommon.
static void copyHostBufferToStagingBuffer(const MemRefValue &sourceBuffer,
                                          PinnedMemoryBlock &stagingBuffer) {
  uint64_t totalBufferSize = sourceBuffer.getTotalFootprintInBytes();
  if (totalBufferSize == 0)
    return;

  BufferType stagingBufferType = BufferType::createWithCanonicalLayout(
      sourceBuffer.getScalarType(), sourceBuffer.getShape(),
      mtrt::PointerType::pinned_host);

  if (sourceBuffer.getType().getLayout() == stagingBufferType.getLayout()) {
    std::memcpy(reinterpret_cast<void *>(stagingBuffer.ptr),
                sourceBuffer.getVoidPtr(), totalBufferSize);
    return;
  }

  MTRT_DBG("executing strided copy from {0} to {1:x}",
           sourceBuffer.getVoidPtr(), stagingBuffer.ptr);

  mtrt::executeStridedByteCopy(
      reinterpret_cast<uintptr_t>(sourceBuffer.getVoidPtr()), 0,
      sourceBuffer.getType().getShape(),
      sourceBuffer.getType().getByteStrides(),
      reinterpret_cast<uintptr_t>(stagingBuffer.ptr), 0,
      stagingBufferType.getShape(), stagingBufferType.getByteStrides(),
      llvm::divideCeil(sourceBuffer.getScalarType().getBitWidth(), 8),
      [](void *dst, void *src, size_t size) { std::memcpy(dst, src, size); });
}

StatusOr<std::unique_ptr<MemRefValue>>
RuntimeClient::copyToDevice(const MemRefValue &hostBufferImpl, Device &device,
                            Ref<Stream> cudaStream,
                            std::unique_ptr<Event> *doneWithHostBuffer) {
  if (!isHostVisible(hostBufferImpl.getAddressSpace()))
    return getInvalidArgStatus("to copy a MemRef to a device its data must "
                               "reside in an address space "
                               "that the host can access");

  // Get the total buffer footprint.
  int64_t totalBufferSize = hostBufferImpl.getTotalFootprintInBytes();

  // Set the CUDA device.
  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<DeviceGuard> deviceGuard,
                        device.createDeviceGuard());

  // Allocate device memory. If the source host buffer has a "packed" layout,
  // then use that same layout for the device buffer. But if it has a strided
  // layout that has padding, use a canonical layout for the device buffer.
  BufferType deviceBufferType = BufferType::createWithCanonicalLayout(
      hostBufferImpl.getScalarType(), hostBufferImpl.getShape(),
      mtrt::PointerType::device);
  StatusOr<std::unique_ptr<MemRefValue>> deviceMemRef =
      allocateMemRef(deviceBufferType, &device, cudaStream);
  if (!deviceMemRef.isOk())
    return deviceMemRef.getStatus();

  // If stream is given, copy the host memory to device asynchronously
  // (using staging buffer), otherwise execute a synchronous CUDA memcpy.
  if (!cudaStream) {
    cudaStream = device.getStream();
    assert(cudaStream && "expected valid stream");
  }

  // Allocate a staging buffer and copy host memory to the staging buffer.
  MTRT_ASSIGN_OR_RETURN(
      PinnedMemoryBlock pinnedMemory,
      this->getPinnedMemoryAllocator().allocate(totalBufferSize));
  copyHostBufferToStagingBuffer(hostBufferImpl, pinnedMemory);
  if (doneWithHostBuffer)
    *doneWithHostBuffer = Event::createReadyEvent();

  MTRT_DBG("copying {0} to {1} size={2} bytes asynchronously via pinned "
           "staging buffer at {3:x}",
           hostBufferImpl.getVoidPtr(), (*deviceMemRef)->getVoidPtr(),
           totalBufferSize, pinnedMemory.ptr);

  MTRT_RETURN_IF_ERROR(copyCUDAHostToDeviceAsync(
      (*deviceMemRef)->getVoidPtr(), reinterpret_cast<void *>(pinnedMemory.ptr),
      totalBufferSize, cudaStream->getCUDAHandle()));

  // Free pinned host memory asynchronously.
  mtrt::logUnhandledErrors(getPinnedMemoryAllocator().freeAsync(
                               pinnedMemory.ptr, cudaStream->getCUDAHandle()),
                           llvm::errs());

  return std::move(*deviceMemRef);
}

StatusOr<std::unique_ptr<MemRefValue>>
RuntimeClient::copyToHost(const MemRefValue &deviceMemRef,
                          Ref<Stream> cudaStream) {
  if (!isDeviceVisible(deviceMemRef.getAddressSpace()))
    return getInvalidArgStatus("to copy a MemRef to the host from a device, "
                               "its data must reside in an address space "
                               "that the host can access");

  int64_t copySizeInBytes = deviceMemRef.getTotalFootprintInBytes();

  StatusOr<Ref<MemRefStorage>> storage = allocator->allocate(
      PointerType::host, copySizeInBytes, /*alignment=*/16, nullptr, nullptr);
  if (!storage.isOk())
    return storage.getStatus();

  // Create the host MemRefValue..
  StatusOr<std::unique_ptr<MemRefValue>> hostMemRef = MemRefValue::create(
      PointerType::host, deviceMemRef.getScalarType(), std::move(*storage), 0,
      deviceMemRef.getShape(), deviceMemRef.getStrides(), {});
  if (!hostMemRef.isOk())
    return hostMemRef.getStatus();

  if (!cudaStream) {
    if (Device *device = deviceMemRef.getDevice())
      cudaStream = device->getStream();
    if (!cudaStream)
      return getInvalidArgStatus("to copy a MemRef to the host from a device, "
                                 "a stream must be provided");
    MTRT_RETURN_IF_ERROR(copyCUDADeviceToHostAsync(
        (*hostMemRef)->getVoidPtr(), deviceMemRef.getVoidPtr(), copySizeInBytes,
        /*stream=*/cudaStream->getCUDAHandle()));
  } else {
    MTRT_RETURN_IF_ERROR(copyCUDADeviceToHostAsync(
        (*hostMemRef)->getVoidPtr(), deviceMemRef.getVoidPtr(), copySizeInBytes,
        reinterpret_cast<uintptr_t>(cudaStream->getCUDAHandle())));
  }
  return std::move(*hostMemRef);
}

Status RuntimeClient::copyToHost(const MemRefValue &deviceMemRef,
                                 MemRefValue &hostMemRef, Ref<Stream> stream) {
  if (!isDeviceVisible(deviceMemRef.getAddressSpace()))
    return getInvalidArgStatus(
        "to copy a MemRef to the host from a device, "
        "its data must reside in an address space "
        "that the host can access but got {0}",
        mtrt::flat::EnumNamePointerType(deviceMemRef.getAddressSpace()));
  if (!isHostVisible(hostMemRef.getAddressSpace()))
    return getInvalidArgStatus(
        "to copy a MemRef to an existing host MemRef, the destination "
        "MemRef's "
        "address space must be host-visible but got address space {0}",
        mtrt::flat::EnumNamePointerType(hostMemRef.getAddressSpace()));
  if (deviceMemRef.getElementBitWidth() != hostMemRef.getElementBitWidth())
    return getInvalidArgStatus(
        "copying device MemRef to host MemRef requires that the element "
        "type "
        "bit-widths match, but got source bitwidth={0} and destination "
        "bitwidth={1}",
        deviceMemRef.getElementBitWidth(), hostMemRef.getElementBitWidth());

  if (deviceMemRef.getShape() != hostMemRef.getShape() ||
      deviceMemRef.getLayout() != hostMemRef.getLayout())
    return getInvalidArgStatus(
        "copying device MemRef to host MemRef requires the shape and "
        "strides "
        "to match, "
        " but the source MemRef has shape=({0,$[, ]}) strides=({1,$[, ]}) "
        "and "
        "the destination MemRef has shape=({2,$[, ]}) strides=({3,$[, ]})",
        deviceMemRef.getShape(), deviceMemRef.getStrides(),
        hostMemRef.getShape(), hostMemRef.getStrides());

  int64_t copySizeInBytes = deviceMemRef.getTotalFootprintInBytes();
  if (!stream) {
    MTRT_RETURN_IF_ERROR(copyCUDADeviceToHostAsync(
        hostMemRef.getVoidPtr(), deviceMemRef.getVoidPtr(), copySizeInBytes,
        /*stream=*/0));
    MTRT_RETURN_IF_ERROR(synchronizeCUDAStream(/*stream=*/0));
  } else {
    MTRT_RETURN_IF_ERROR(copyCUDADeviceToHostAsync(
        hostMemRef.getVoidPtr(), deviceMemRef.getVoidPtr(), copySizeInBytes,
        reinterpret_cast<uintptr_t>(stream->getCUDAHandle())));
  }
  return getOkStatus();
}

RuntimeClient::~RuntimeClient() = default;

StatusOr<Ref<Stream>> RuntimeClient::getCurrentDeviceStream() const {
  MTRT_ASSIGN_OR_RETURN(int32_t deviceId, getCurrentCUDADevice());
  auto it = llvm::find_if(devices, [&](const auto &device) {
    return device->getDeviceNumber() == deviceId;
  });
  if (it == devices.end())
    return getInvalidArgStatus("device not found");
  return (*it)->getStream();
}

StatusOr<std::unique_ptr<MemRefValue>>
RuntimeClient::copyDeviceBufferToOtherDevice(
    const MemRefValue &sourceBuffer, Device &dstDevice,
    std::unique_ptr<Event> &copyDoneEvent) {
  if (!isDeviceVisible(sourceBuffer.getAddressSpace()))
    return getInvalidArgStatus("source buffer is not device visible");
  if (!sourceBuffer.getDevice())
    return getInvalidArgStatus("source buffer is not associated with a device");

  Ref<Stream> destStream = dstDevice.getStream();
  Ref<Stream> sourceStream = sourceBuffer.getDevice()->getStream();

  // Create a new event.
  std::unique_ptr<mtrt::Event> sourceReadyEvent{nullptr};
  {
    // Record the event onto the source buffer's stream so that we can wait
    // until
    // the source buffer is ready. This assumes the definition of the source
    // buffer is on the source stream, at or prior to the head of the stream.
    MTRT_ASSIGN_OR_RETURN(auto devGuard,
                          sourceBuffer.getDevice()->createDeviceGuard());
    MTRT_ASSIGN_OR_RETURN(sourceReadyEvent, mtrt::Event::create(sourceStream));
  }

  MTRT_ASSIGN_OR_RETURN(auto devGuard, dstDevice.createDeviceGuard());
  const BufferType &type = sourceBuffer.getType();

  MTRT_ASSIGN_OR_RETURN(
      Ref<MemRefStorage> destStorage,
      allocator->allocate(PointerType::device,
                          sourceBuffer.getTotalFootprintInBytes(),
                          /*alignment=*/16, &dstDevice, destStream));

  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<MemRefValue> dstBuffer,
                        MemRefValue::create(PointerType::device, type,
                                            std::move(destStorage),
                                            /*device=*/&dstDevice,
                                            /*assertCanonicalStrides=*/false));

  // Wait until the source buffer is ready on the destination device stream and
  // then copy from source to destination device using the destination device
  // stream.

  // There are two options for doing this: use the CUDA event handle to
  // place a wait on the destination stream or invoke the copy from a
  // completion callback added to the 'sourceReady' event.
  //
  // Since we are assuming CUDA implementation, go ahead and use the CUDA
  // specific option.
  Status waitStatus = mtrt::waitCUDAEventOnStream(
      destStream->getCUDAHandle(), sourceReadyEvent->getCudaHandle());
  if (!waitStatus.isOk())
    return mtrt::getInternalErrorStatus("failed to wait on CUDA event: {0}",
                                        waitStatus.getString());
  Status copyStatus = mtrt::copyCUDAPeerAsync(
      dstBuffer->getVoidPtr(), dstDevice.getDeviceNumber(),
      sourceBuffer.getVoidPtr(), sourceBuffer.getDevice()->getDeviceNumber(),
      sourceBuffer.getType().getFootprintSizeInBytes(),
      destStream->getCUDAHandle());
  if (!copyStatus.isOk())
    return mtrt::getInternalErrorStatus(
        "failed to copy CUDA device-to-device: {0}", copyStatus.getString());

  MTRT_ASSIGN_OR_RETURN(copyDoneEvent, Event::create(destStream));

  // Move the "sourceReadyEvent" into a completion callback of the copy so its
  // lifetime exceeds the point where the raw cuda handle is needed above.
  copyDoneEvent->addReadyCallback(
      [](Status status, void *userData) {
        auto sourceReadyEvent =
            std::unique_ptr<Event>(reinterpret_cast<Event *>(userData));
        MTRT_DBG("device-to-device copy completion callback invoked, "
                 "sourceReadyEvent={0}",
                 sourceReadyEvent.get());
        Event::releaseWhenReady(std::move(sourceReadyEvent));
      },
      sourceReadyEvent.release());

  return std::move(dstBuffer);
}

StatusOr<std::unique_ptr<RuntimeSession>>
RuntimeClient::createSession(llvm::StringRef kind,
                             RuntimeSessionOptions options,
                             ExecutableView executable) {
  auto it = sessionFactories.find(kind);
  if (it == sessionFactories.end())
    return getInvalidArgStatus("no session factory registered for the name {0}",
                               kind);
  return it->second(Ref<RuntimeClient>(this), options, executable);
}

void RuntimeClient::registerSessionFactory(llvm::StringRef kind,
                                           RuntimeSessionFactory factory) {
  if (sessionFactories.count(kind) > 0) {
    // Duplicate registration is a fatal error.
    mtrt::cantFail(getInvalidArgStatus("session factory already registered for "
                                       "the name {0}",
                                       kind));
  }
  sessionFactories[kind] = factory;
}

//===----------------------------------------------------------------------===//
// NCCL Support Functions
//===----------------------------------------------------------------------===//

StatusOr<std::string> mtrt::getCommunicatorUniqueId() {
#ifdef MLIR_TRT_ENABLE_NCCL
  ncclUniqueId id;
  RETURN_ERROR_IF_NCCL_ERROR(ncclGetUniqueId(&id), nullptr);
  std::string asString = std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
  MTRT_DBGF("NCCL unique id: %s", asString.c_str());
  return asString;
#else
  return std::string{};
#endif
}

//===----------------------------------------------------------------------===//
// Print Utilities
//===----------------------------------------------------------------------===//

/// Define a couple type traits to help with static SFINAE switch below.
template <typename S, typename T, typename = void>
struct has_stream_printer : std::false_type {};
template <typename S, typename T>
struct has_stream_printer<
    S, T, std::void_t<decltype(std::declval<S &>() << std::declval<T>())>>
    : std::true_type {};
template <typename S, typename T, typename = void>
struct has_print_func : std::false_type {};
template <typename S, typename T>
struct has_print_func<
    S, T,
    std::void_t<decltype(mtrt::print(std::declval<S &>(), std::declval<T>()))>>
    : std::true_type {};

template <typename T, typename Callable>
static llvm::raw_ostream &interleave(llvm::raw_ostream &os, llvm::ArrayRef<T> x,
                                     Callable callable,
                                     std::string_view delim) {
  if (x.empty())
    return os;
  unsigned lim = x.size();
  for (unsigned i = 0, e = lim; i < e; i++) {
    callable(os, x[i]);
    if (i < lim - 1)
      os << delim;
  }
  return os;
}

#define LAMBDAF(x) [&]() { x; }

template <typename T,
          typename std::enable_if_t<has_print_func<llvm::raw_ostream, T>::value,
                                    void *> = nullptr>
static llvm::raw_ostream &interleaveComma(llvm::raw_ostream &os,
                                          llvm::ArrayRef<T> x) {
  interleave(
      os, x, [](llvm::raw_ostream &os, const auto &x) { mtrt::print(os, x); },
      ", ");
  return os;
}
template <typename T,
          typename std::enable_if_t<has_print_func<llvm::raw_ostream, T>::value,
                                    void *> = nullptr>
static llvm::raw_ostream &interleaveComma(llvm::raw_ostream &os,
                                          const llvm::SmallVectorImpl<T> &x) {
  return interleaveComma(os, llvm::ArrayRef(x));
}

template <typename Callable>
static llvm::raw_ostream &squareBraces(llvm::raw_ostream &os, Callable c) {
  os << "[";
  c();
  return os << "]";
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const TypeUnionView &arg) {
  if (arg.isa<MemRefTypeView>())
    return print(os, arg.get<MemRefTypeView>());
  if (arg.isa<ScalarTypeView>())
    return print(os, arg.get<ScalarTypeView>());
  return os << "UNK";
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const DimensionBoundsView &exe) {
  os << "dim_bounds<min = [";
  interleave(
      os, exe.getMin(), [](llvm::raw_ostream &os, auto x) { os << x; }, ",");
  os << "], max = [";
  interleave(
      os, exe.getMax(), [](llvm::raw_ostream &os, auto x) { os << x; }, ",");
  return os << "]>";
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const ValueBoundsView &exe) {
  os << "value_bounds<min = [";
  interleave(
      os, exe.getMin(), [](llvm::raw_ostream &os, auto x) { os << x; }, ",");
  os << "], max = [";
  interleave(
      os, exe.getMax(), [](llvm::raw_ostream &os, auto x) { os << x; }, ",");
  return os << "]>";
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const BoundsUnionView &bounds) {
  if (bounds.isa<DimensionBoundsView>())
    return print(os, bounds.get<DimensionBoundsView>());
  if (bounds.isa<ValueBoundsView>())
    return print(os, bounds.get<ValueBoundsView>());
  return os << "UNK";
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os, const Executable &exe) {
  os << "RuntimeExecutable<name=" << exe.getName() << ",";
  os << "functions=";
  squareBraces(os, LAMBDAF(interleaveComma(os, exe.getFunctions())));
  os << "data_segments=";
  squareBraces(os, LAMBDAF(interleaveComma(os, exe.getDataSegments());));
  os << ",source=";
  squareBraces(os, LAMBDAF(os << exe.getCode().size() << " bytes";));
  return os << ">";
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const DataSegmentInfo &segment) {
  os << llvm::formatv(
      "DataSegment<{0}, size={1}, alignment={2}, constant={3}, "
      "uninitialized={4}, address_space={5}>",
      segment.getName(), segment.size(), segment.getAlignment(),
      segment.isConstant(), segment.isUninitialized(),
      mtrt::flat::EnumNamePointerType(segment.getAddressSpace()));
  return os;
}
llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const MemRefTypeView &exe) {
  auto handleDimOrStride = [](llvm::raw_ostream &os, int64_t x) {
    if (x != std::numeric_limits<int64_t>::min())
      os << x;
    else
      os << "?";
  };

  os << "MemRef<";
  interleave(os, exe.getShape(), handleDimOrStride, "x");
  if (!exe.getShape().empty())
    os << "x";
  os << mtrt::flat::EnumNameScalarTypeCode(exe.getElementType());
  if (!exe.getStrides().empty()) {
    os << ",";
    interleave(os, exe.getStrides(), handleDimOrStride, "x");
  }
  os << "," << exe.getAddressSpace();
  return os << ">";
}
llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const ScalarTypeView &scalarType) {
  return os << mtrt::flat::EnumNameScalarTypeCode(scalarType);
}
llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const FunctionSignatureView &signature) {
  llvm::SmallVector<TypeUnionView> args = signature.getArgs();
  llvm::SmallVector<TypeUnionView> results = signature.getResults();
  llvm::SmallVector<BoundsUnionView> arg_bounds = signature.getArgBounds();
  llvm::SmallVector<BoundsUnionView> result_bounds =
      signature.getResultBounds();
  os << "Signature<args=";
  squareBraces(os, LAMBDAF(interleaveComma(os, llvm::ArrayRef(args))));
  os << ", results=";
  squareBraces(os, LAMBDAF(interleaveComma(os, llvm::ArrayRef(results))));
  os << ", num_output_args=" << signature.getNumOutputArgs();
  os << ", arg_bounds=";
  squareBraces(os, LAMBDAF(interleaveComma(os, llvm::ArrayRef(arg_bounds))));
  os << ", result_bounds=";
  squareBraces(os, LAMBDAF(interleaveComma(os, llvm::ArrayRef(result_bounds))));
  os << ", cconv=";
  os << mtrt::flat::EnumNameCallingConvention(signature.getCConv());
  os << ">";
  return os;
}

llvm::raw_ostream &mtrt::print(llvm::raw_ostream &os,
                               const FunctionView &func) {
  os << "Function<" << func.getName() << ", ";
  print(os, func.getSignature());
  return os << ">";
}
