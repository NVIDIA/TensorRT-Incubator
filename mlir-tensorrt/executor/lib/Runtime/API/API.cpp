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
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "mlir-executor/Support/Allocators.h"
#include "mlir-executor/Support/CUDAWrappers.h"
#include "mlir-executor/Support/Status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Threading.h"
#include <cstdlib>

#ifdef MLIR_EXECUTOR_ENABLE_NCCL
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
#endif //  MLIR_EXECUTOR_ENABLE_NCCL

using namespace mlirtrt;
namespace rt = mlirtrt::runtime;
using namespace rt;

//===----------------------------------------------------------------------===//
// Scalar Type
//===----------------------------------------------------------------------===//

ScalarTypeCode rt::parseElementType(std::string_view str) {
  const char *const *names = rt::impl::EnumNamesScalarTypeCode();
  const ScalarTypeCode *values = rt::impl::EnumValuesScalarTypeCode();
  // Flatbuffers' 'enum::MAX' is inclusive (equals largest value).
  constexpr unsigned maxEnum = static_cast<unsigned>(impl::ScalarTypeCode::MAX);
  for (unsigned i = 0; i <= maxEnum; i++) {
    if (str == names[i])
      return values[i];
  }
  return ScalarTypeCode::unknown;
}

int64_t rt::getBitsPerElement(ScalarTypeCode elType) {
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

StatusOr<ScalarType> rt::ScalarType::fromString(std::string_view str) {
  auto code = parseElementType(str);
  assert(code != ScalarTypeCode::unknown && "expected known element type code");
  if (code != ScalarTypeCode::unknown)
    return ScalarType(code);
  return getStatusWithMsg(StatusCode::InvalidArgument, "unknown element type (",
                          str, ")");
}

int64_t rt::ScalarType::getBitWidth() const {
  int64_t result = getBitsPerElement(code);
  assert(result != 0 && "expected positive bitwidth");
  return result;
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

PointerType rt::parsePointerType(std::string_view str) {
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

llvm::raw_ostream &rt::operator<<(llvm::raw_ostream &os, PointerType ptrType) {
  return os << rt::impl::EnumNamePointerType(ptrType);
}

std::string_view rt::stringifyPointerType(PointerType ptrType) {
  return rt::impl::EnumNamePointerType(ptrType);
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

FunctionView ExecutableView::getFunction(std::string_view name) const {
  const flatbuffers::Vector<flatbuffers::Offset<impl::Function>> &functions =
      *view->functions();
  auto it = std::find_if(functions.begin(), functions.end(),
                         [&](const impl::Function *x) {
                           return x->name()->string_view() == name;
                         });
  assert(it != view->functions()->end());
  return FunctionView(*it);
}

llvm::SmallVector<ConstantView> ExecutableView::getConstants() const {
  llvm::SmallVector<ConstantView> views;
  views.reserve(view->constants()->size());
  for (unsigned i = 0; i < view->constants()->size(); i++)
    views.push_back(view->constants()->Get(i));
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
  this->view = impl::GetExecutable(this->storage->data());
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

  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t *>(result->getStorage()->data()),
      result->getStorage()->size());
  if (!impl::VerifyExecutableBuffer(verifier))
    return getStatusWithMsg(StatusCode::InvalidArgument,
                            "failed to verify that the provided file contains "
                            "a valid MLIR-TRT Executable");
  return result;
}

StatusOr<std::unique_ptr<Executable>>
Executable::loadFromBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer) {
  auto result = std::make_unique<Executable>(
      std::make_unique<ExecutableStorageMemBuffer>(std::move(buffer)));
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t *>(result->getStorage()->data()),
      result->getStorage()->size());
  if (!impl::VerifyExecutableBuffer(verifier))
    return getStatusWithMsg(
        StatusCode::InvalidArgument,
        "failed to verify that the provided buffer contains "
        "a valid MLIR-TRT Executable");
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

  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t *>(result->getStorage()->data()),
      result->getStorage()->size());
  if (!impl::VerifyExecutableBuffer(verifier))
    return getStatusWithMsg(
        StatusCode::InvalidArgument,
        "failed to verify that the provided buffer contains "
        "a valid MLIR-TRT Executable");
  return result;
}

rt::Executable::Executable(std::unique_ptr<ExecutableStorage> storage_)
    : ExecutableView(nullptr), storage(std::move(storage_)) {
  assert(this->storage && "expected valid storage pointer");
  this->view = impl::GetExecutable(this->storage->data());
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

//===----------------------------------------------------------------------===//
// RuntimeSessionOptions
//===----------------------------------------------------------------------===//

StatusOr<RuntimeSessionOptions>
RuntimeSessionOptions::createUsingSingleHostMpi() {
#ifdef MLIR_EXECUTOR_ENABLE_NCCL
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
#else  // MLIR_EXECUTOR_ENABLE_NCCL
  return getInternalErrorStatus(
      "MLIR-TensorRT was not configured and built with MPI and NCCL support");
#endif // MLIR_EXECUTOR_ENABLE_NCCL
}

//===----------------------------------------------------------------------===//
// RuntimeSession
//===----------------------------------------------------------------------===//

RuntimeSession::RuntimeSession(RuntimeSessionOptions options,
                               ExecutableView exe)
    : options(std::move(options)), executable(exe),
      pinnedMemoryAllocator(std::make_unique<PinnedMemoryAllocator>()),
      allocTracker(std::make_unique<AllocTracker>()),
      resourceTracker(std::make_unique<ResourceTracker>()) {}

//===----------------------------------------------------------------------===//
// AllocTracker
//===----------------------------------------------------------------------===//

AllocTracker::~AllocTracker() {
  MTRT_DBGF("checking %u allocations", map.size());
  llvm::SmallVector<PointerInfo> ptrsToFree;
  ptrsToFree.reserve(map.size());
  for (const auto &[ptrVal, metadata] : map) {
    if (metadata->info.isInternallyManaged() &&
        metadata->externalReferenceCount.load() == 0) {
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

void AllocTracker::markReleasedInternally(uintptr_t ptr) {
  assert(llvm::is_contained(map, ptr) &&
         llvm::formatv("Untracked pointer {0}", ptr).str().c_str());
  std::unique_ptr<Metadata> const &metadata = map.at(ptr);
  metadata->releasedInternally = true;
}

bool AllocTracker::isReleasedInternally(uintptr_t ptr) const {
  assert(llvm::is_contained(map, ptr) &&
         llvm::formatv("Untracked pointer {0}", ptr).str().c_str());
  std::unique_ptr<Metadata> const &metadata = map.at(ptr);
  return metadata->releasedInternally;
}

void AllocTracker::incrementExternalCount(uintptr_t ptr) {
  assert(llvm::is_contained(map, ptr) &&
         llvm::formatv("Untracked pointer {0}", ptr).str().c_str());
  std::unique_ptr<Metadata> const &metadata = map.at(ptr);
  int32_t ref = ++metadata->externalReferenceCount;
  MTRT_DBG("Incremented external reference for pointer %d to %d", ptr, ref);
}

void AllocTracker::decrementExternalCount(uintptr_t ptr) {
  assert(llvm::is_contained(map, ptr) &&
         llvm::formatv("Untracked pointer {0}", ptr).str().c_str());
  std::unique_ptr<Metadata> const &metadata = map.at(ptr);
  int32_t ref = --metadata->externalReferenceCount;
  assert(ref >= 0 &&
         llvm::formatv("External reference count cannot be negative: {0}", ref)
             .str()
             .c_str());
  MTRT_DBG("Decremented external reference for pointer %d to %d", ptr, ref);
  if (ref == 0 && metadata->releasedInternally) {
    MTRT_DBG("External reference to an internally released pointer %d is 0, "
             "try deallocating pointer memory of size %lu",
             ptr, ref, metadata->info.size);
    Status s = safeDeallocate(*this, metadata->info.ptr);
    if (!s.isOk())
      MTRT_DBGF("error while deallocating dangling memory: %s",
                s.getString().c_str());
  }
}

int32_t AllocTracker::getExternalReferenceCount(uintptr_t ptr) const {
  assert(llvm::is_contained(map, ptr) &&
         llvm::formatv("Untracked pointer {0}", ptr).str().c_str());
  std::unique_ptr<Metadata> const &metadata = map.at(ptr);
  return metadata->externalReferenceCount.load();
}

void AllocTracker::track(PointerInfo info) {
  if (info.isInternallyManaged()) {
    // We issue an assertion error if we somehow have double-tracked this
    // pointer (perhaps the `untrack` was not correctly called). Note that we
    // may have previously tracked this pointer as an externally managed pointer
    // (e.g. function argument), in which case it may have been deallocated,
    // allowing an internal allocator to pick up that same address. That case is
    // not an error.
    assert((!contains(info.ptr) || get(info.ptr).isExternallyManaged()) &&
           "an internally managed pointer should not already be tracked");
  }
  MTRT_DBGF("AllocTracker is now tracking 0x%lx size=%lu space=%s ownership=%s",
            info.ptr, info.size, runtime::impl::EnumNamePointerType(info.type),
            runtime::impl::EnumNamePointerOwner(info.owner));
  auto value = std::make_unique<Metadata>();
  value->externalReferenceCount.store(0);
  value->releasedInternally = false;
  value->info = info;
  if (!contains(info.ptr)) {
    map.insert(std::make_pair(info.ptr, std::move(value)));
    return;
  }
  untrack(info.ptr);
  map.insert(std::make_pair(info.ptr, std::move(value)));
}

void AllocTracker::untrack(uintptr_t ptr) {
  assert(llvm::is_contained(map, ptr) &&
         llvm::formatv("Untracked pointer {0}", ptr).str().c_str());
  map.erase(map.find(ptr));
}

bool AllocTracker::contains(uintptr_t ptr) const { return map.contains(ptr); }

const PointerInfo &AllocTracker::get(uintptr_t ptr) const {
  auto it = map.find(ptr);
  assert(it != map.end() && "expected valid pointer info");
  return map.at(ptr)->info;
}

PointerInfo AllocTracker::lookupOrDefault(uintptr_t ptr) const {
  if (!contains(ptr))
    return PointerInfo{ptr, PointerInfo::kUnknownSize, PointerType::unknown,
                       PointerOwner::unknown};
  return map.at(ptr)->info;
}

StatusOr<PointerInfo> runtime::allocate(AllocTracker &tracker, PointerType type,
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
    size = llvm::alignTo(size, *alignment);
    uintptr_t mem =
        reinterpret_cast<uintptr_t>(::aligned_alloc(*alignment, size));
    if (mem == 0)
      return mlirtrt::getInternalErrorStatus(
          "failed to allocate memory on host");
    MTRT_DBGF("Allocated %lu host bytes at 0x%lx", size, mem);
    PointerInfo info{mem, size, type, PointerOwner::internal};
    tracker.track(info);
    return info;
  }

#ifdef MLIR_EXECUTOR_ENABLE_CUDA
  if (type == PointerType::device) {
    size = std::max<size_t>(size, 16);
    if (!stream) {
      void *alloc{nullptr};
      RETURN_ERROR_IF_CUDART_ERROR(cudaMalloc(&alloc, size));
      MTRT_DBGF("Allocated %lu device bytes 0x%lx", size,
                reinterpret_cast<uintptr_t>(alloc));
      PointerInfo info{reinterpret_cast<uintptr_t>(alloc), size, type,
                       PointerOwner::internal};
      tracker.track(info);
      return info;
    }
    void *alloc{nullptr};
    RETURN_ERROR_IF_CUDART_ERROR(
        cudaMallocAsync(&alloc, size, reinterpret_cast<cudaStream_t>(*stream)));
    MTRT_DBGF("Allocated %lu device bytes 0x%lx", size,
              reinterpret_cast<uintptr_t>(alloc));
    PointerInfo info{reinterpret_cast<uintptr_t>(alloc), size, type,
                     PointerOwner::internal};
    tracker.track(info);
    return info;
  }
  if (type == PointerType::unified) {
    MTRT_DBGF("Allocating %lu unified bytes", size);
    assert(!stream &&
           "async stream is not allowed when using unified memory allocator");
    void *alloc{nullptr};
    RETURN_ERROR_IF_CUDART_ERROR(cudaMallocManaged(&alloc, size));
    PointerInfo info{reinterpret_cast<uintptr_t>(alloc), size, type,
                     PointerOwner::internal};
    tracker.track(info);
    return info;
  }
#endif
  return getStatusWithMsg(
      mlirtrt::StatusCode::InvalidArgument,
      "unimplemented allocation type: ", stringifyPointerType(type));
}

mlirtrt::Status runtime::safeDeallocate(AllocTracker &tracker, uintptr_t ptr,
                                        std::optional<CudaStream> stream) {
  if (!tracker.contains(ptr)) {
    MTRT_DBGF("ignoring ptr 0x%lx because it was either already freed, "
              "externally managed, or has an external reference",
              ptr);
    return mlirtrt::Status::getOk();
  }

  if (tracker.getExternalReferenceCount(ptr) > 0) {
    // Destructor for external reference should truly free or delete this.
    // Defer safeDeallocate call until then.
    MTRT_DBGF("Defer freeing ptr 0x%lx and mark it as released internally as "
              "ir has an external reference. "
              "It is responsibility of the external reference counting "
              "mechanism to ensure safeDellaocate is called.",
              ptr);
    tracker.markReleasedInternally(ptr);
    return mlirtrt::Status::getOk();
  }

  PointerInfo obj = tracker.get(ptr);
  if (obj.owner == PointerOwner::external) {
    MTRT_DBGF("Untracking externally managed pointer 0x%lx", ptr);
    tracker.untrack(obj.ptr);
    return mlirtrt::Status::getOk();
  }

  if (obj.type == PointerType::host) {
    MTRT_DBGF("Freeing host memory %lx", ptr);
    std::free(reinterpret_cast<void *>(obj.ptr));
    tracker.untrack(ptr);
    return Status::getOk();
  }

#ifdef MLIR_EXECUTOR_ENABLE_CUDA
  if (obj.type == PointerType::device || obj.type == PointerType::unified) {
    if (stream && *stream != 0) {
      MTRT_DBGF("Asynchronously freeing CUDA device/pinned host memory 0x%lx "
                "type %d on stream %lx",
                ptr, static_cast<int32_t>(obj.type),
                reinterpret_cast<uintptr_t>(*stream));
      RETURN_ERROR_IF_CUDART_ERROR(
          cudaFreeAsync(reinterpret_cast<void *>(obj.ptr),
                        reinterpret_cast<cudaStream_t>(*stream)));
    } else {
      MTRT_DBGF(
          "Synchronously freeing CUDA device/pinned host memory %lx type %d",
          ptr, static_cast<int32_t>(obj.type));
      RETURN_ERROR_IF_CUDART_ERROR(cudaFree(reinterpret_cast<void *>(obj.ptr)));
    }
    tracker.untrack(ptr);
    return Status::getOk();
  }

  if (obj.type != PointerType::pinned_host) {
    MTRT_DBG("Synchronously freeing pinned host memory {0}",
             reinterpret_cast<void *>(obj.ptr));
    RETURN_ERROR_IF_CUDART_ERROR(
        cudaFreeHost(reinterpret_cast<void *>(obj.ptr)));
    return Status::getOk();
  }

#endif

  return mlirtrt::getInternalErrorStatus("unhandled allocation type");
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
  tracker.insert(std::make_pair(ptr, deleter));
}

void ResourceTracker::untrack(uintptr_t ptr) { tracker.erase(ptr); }

//===----------------------------------------------------------------------===//
// Device
//===----------------------------------------------------------------------===//

StatusOr<std::unique_ptr<Device>> Device::create(int32_t deviceNumber) {
  return std::unique_ptr<Device>(new Device(deviceNumber));
}

//===----------------------------------------------------------------------===//
// MemRefValue
//===----------------------------------------------------------------------===//

static StatusOr<int64_t> getFootprintInBytes(llvm::ArrayRef<int64_t> shape,
                                             llvm::ArrayRef<int64_t> strides,
                                             int64_t bitsPerElement) {
  assert(shape.size() == strides.size() &&
         "expected equal rank shape and strides");
  if (llvm::any_of(strides, [](int64_t x) { return x < 0; }))
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
    // This accounts for discrepancies in how different frameworks handle these
    // cases
    if (stride[i] != expectedStride[i] && shape[i] != 0 && shape[i] != 1)
      return false;

  return true;
}

StatusOr<std::unique_ptr<MemRefValue>> MemRefValue::create(
    RuntimeClient *client, mlirtrt::runtime::PointerType addressSpace,
    int64_t bitsPerElement, uintptr_t ptr, int64_t offset,
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
    std::optional<const Device *> device, std::optional<ScalarType> scalarType,
    std::optional<bool> assertCanonicalStrides) {
  if (!client)
    return getInvalidArgStatus("a valid RuntimeClient must be provided to "
                               "create a tracked MemRef object");
  if (!::getFootprintInBytes(shape, strides, bitsPerElement).isOk())
    return getInvalidArgStatus(
        "only memrefs with non-negative strides are allowed");
  if (!ptr)
    return getInvalidArgStatus(
        "MemRef objects must be created with a valid pointer");

  if (isDeviceVisible(addressSpace) && (!device || !*device))
    return getInvalidArgStatus("a specific device must be provided for MemRefs "
                               "that are device-visible");

  // Check if given strides match canonical stride
  if (assertCanonicalStrides && *assertCanonicalStrides) {
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
      new MemRefValue(client, addressSpace, bitsPerElement, ptr, offset, shape,
                      strides, device, scalarType));
}

MemRefValue::MemRefValue(RuntimeClient *client,
                         mlirtrt::runtime::PointerType addressSpace,
                         int64_t bitsPerElement, uintptr_t ptr, int64_t offset,
                         llvm::ArrayRef<int64_t> shape,
                         llvm::ArrayRef<int64_t> strides,
                         std::optional<const Device *> device,
                         std::optional<ScalarType> scalarType)
    : RuntimeValue(Kind::MemRef), client(client), addressSpace(addressSpace),
      bitsPerElement(bitsPerElement), ptr(ptr), offsetInBytes(offset),
      shape(shape.begin(), shape.end()),
      strides(strides.begin(), strides.end()), device(device),
      scalarType(scalarType) {}

int64_t MemRefValue::getTotalFootprintInBytes() const {
  StatusOr<int64_t> result =
      getFootprintInBytes(shape, strides, bitsPerElement);
  // Failing the calculation at this point is unexpected and not recoverable.
  if (!result.isOk())
    llvm_unreachable("invalid stride specification in memref");
  return *result;
}

//===----------------------------------------------------------------------===//
// RuntimeClient
//===----------------------------------------------------------------------===//

static Status parseDebugFlags() {
  std::vector<const char *> argv = {"mlir-tensorrt-runtime"};
  std::string error;
  llvm::raw_string_ostream ss(error);
  if (!llvm::cl::ParseCommandLineOptions(argv.size(), argv.data(),
                                         "MLIR-TRT Runtime flags", &ss,
                                         "MTRT_FLAGS", false)) {
    ss.flush();
    getInternalErrorStatus("Failed to parse MTRT_FLAGS options: {0}",
                           error.c_str());
  }
  return Status::getOk();
}

static mlirtrt::Status
populateDevices(llvm::SmallVectorImpl<std::unique_ptr<Device>> &devices) {
#ifdef MLIR_EXECUTOR_ENABLE_CUDA
  int32_t numDevices = 0;
  // Find the number of devices. In single-process mode, the "addressable
  // devices" is equivalent to any devices the process can view, but this
  // is not true in multi-process mode.
  RETURN_ERROR_IF_CUDART_ERROR(cudaGetDeviceCount(&numDevices));

  devices.reserve(numDevices);
  for (int32_t i = 0; i < numDevices; ++i) {
    StatusOr<std::unique_ptr<Device>> device = Device::create(i);
    if (!device.isOk())
      return device.getStatus();
    devices.push_back(std::move(*device));
  }
#endif
  return getOkStatus();
}

StatusOr<std::unique_ptr<RuntimeClient>> RuntimeClient::create() {
  static llvm::once_flag onceFlag{};
  llvm::call_once(onceFlag, parseDebugFlags);

  // Setup device objects. Create a view of the device pointers.
  llvm::SmallVector<std::unique_ptr<Device>> devices;
  mlirtrt::Status status = populateDevices(devices);
  if (!status.isOk())
    return status;

  return std::unique_ptr<RuntimeClient>(new RuntimeClient(std::move(devices)));
}

llvm::ArrayRef<std::unique_ptr<Device>> RuntimeClient::getDevices() const {
  return devices;
}

StatusOr<std::unique_ptr<MemRefValue>> RuntimeClient::allocateMemRef(
    PointerType addressSpace, int64_t bitsPerElement,
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
    std::optional<const Device *> device, std::optional<CudaStream> stream,
    std::optional<ScalarType> scalarType,
    std::optional<bool> assertCanonicalStrides) {
  if (addressSpace == PointerType::device ||
      addressSpace == PointerType::unified) {
    if (!device || !*device)
      return getInvalidArgStatus(
          "a specific device must be specified when creating a device buffer");
  }

  // Allocate required memory.
  StatusOr<int64_t> allocationSizeBytes =
      getFootprintInBytes(shape, strides, bitsPerElement);
  if (!allocationSizeBytes.isOk())
    return allocationSizeBytes.getStatus();

  StatusOr<PointerInfo> allocation =
      runtime::allocate(allocTracker, addressSpace, *allocationSizeBytes,
                        /*alignment=*/16, stream);
  if (!allocation.isOk())
    return allocation.getStatus();

  // Create the descriptor.
  StatusOr<std::unique_ptr<MemRefValue>> bufferImpl = MemRefValue::create(
      this, addressSpace, bitsPerElement, allocation->ptr, 0, shape, strides,
      device, scalarType, assertCanonicalStrides);
  if (bufferImpl.isError())
    return bufferImpl.getStatus();

  return std::move(*bufferImpl);
}

StatusOr<std::unique_ptr<MemRefValue>> RuntimeClient::createExternalMemRef(
    PointerType addressSpace, int64_t bitsPerElement, uintptr_t ptr,
    int64_t offset, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> strides, std::optional<const Device *> device,
    std::optional<ScalarType> scalarType,
    std::optional<bool> assertCanonicalStrides) {
  // Create the descriptor.
  StatusOr<std::unique_ptr<MemRefValue>> memref = MemRefValue::create(
      this, addressSpace, bitsPerElement, ptr, offset, shape, strides, device,
      scalarType, assertCanonicalStrides);
  if (!memref.isOk())
    return memref.getStatus();

  allocTracker.track((*memref)->getPointerInfo(PointerOwner::external));

  return memref;
}

Status RuntimeClient::deallocate(std::unique_ptr<MemRefValue> value,
                                 std::optional<CudaStream> stream) {
  return safeDeallocate(
      allocTracker, reinterpret_cast<uintptr_t>(value->getMemory()), stream);
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
      allocateMemRef(PointerType::host, hostBuffer.getElementBitWidth(),
                     hostBuffer.getShape(), hostBuffer.getStrides(),
                     std::nullopt, std::nullopt, hostBuffer.getScalarType());
  if (!hostMemRef.isOk())
    return hostMemRef.getStatus();

  // Fill new buffer with the data from the source buffer.
  std::memcpy((*hostMemRef)->getVoidPtr(), hostBuffer.getVoidPtr(),
              totalBufferSize);
  return std::move(*hostMemRef);
}

StatusOr<std::unique_ptr<MemRefValue>>
RuntimeClient::copyToDevice(const MemRefValue &hostBufferImpl,
                            const Device &device,
                            std::optional<CudaStream> cudaStream) {
#ifdef MLIR_EXECUTOR_ENABLE_CUDA
  if (!isHostVisible(hostBufferImpl.getAddressSpace()))
    return getInvalidArgStatus(
        "to copy a MemRef to a device its data must reside in an address space "
        "that the host can access");

  // Get the total buffer footprint.
  int64_t totalBufferSize = hostBufferImpl.getTotalFootprintInBytes();

  // Set the CUDA device.
  RETURN_ERROR_IF_CUDART_ERROR(cudaSetDevice(device.getDeviceNumber()));

  // Allocate device memory
  StatusOr<std::unique_ptr<MemRefValue>> deviceMemRef =
      allocateMemRef(PointerType::device, hostBufferImpl.getElementBitWidth(),
                     hostBufferImpl.getShape(), hostBufferImpl.getStrides(),
                     &device, cudaStream, hostBufferImpl.getScalarType());
  if (!deviceMemRef.isOk())
    return deviceMemRef.getStatus();

  // If stream is given, copy the host memory to device asynchronously
  // (using staging buffer), otherwise execute a synchronous CUDA memcpy.
  if (cudaStream) {

    // Allocate a staging buffer and copy host memory to the staging buffer.
    // TODO: Currently, this implementation supports only row major packed
    // canonical layout (no padding).
    StatusOr<mlirtrt::PinnedMemoryBlock> pinnedMemory =
        this->getPinnedMemorAllocator().allocate(totalBufferSize);
    if (!pinnedMemory.isOk())
      return pinnedMemory.getStatus();

    void *stagingHostMemPtr = reinterpret_cast<void *>(pinnedMemory->ptr);
    std::memcpy(stagingHostMemPtr, hostBufferImpl.getVoidPtr(),
                totalBufferSize);

    MTRT_DBG("copying {0} to {1} size={2} bytes asynchronously via pinned "
             "staging buffer at {3}",
             hostBufferImpl.getVoidPtr(), (*deviceMemRef)->getVoidPtr(),
             totalBufferSize, stagingHostMemPtr);

    RETURN_ERROR_IF_CUDART_ERROR(
        cudaMemcpyAsync((*deviceMemRef)->getVoidPtr(), stagingHostMemPtr,
                        totalBufferSize, cudaMemcpyKind::cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(*cudaStream)));

    // Free pinned host memory asynchronously.
    getPinnedMemorAllocator().freeAsync(pinnedMemory->ptr, *cudaStream);
  } else {
    MTRT_DBG("synchronously copying {0} (host) to {1} (device), size={2} bytes",
             hostBufferImpl.getVoidPtr(), (*deviceMemRef)->getVoidPtr(),
             totalBufferSize);
    RETURN_ERROR_IF_CUDART_ERROR(
        cudaMemcpy((*deviceMemRef)->getVoidPtr(), hostBufferImpl.getVoidPtr(),
                   totalBufferSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
  }
  return std::move(*deviceMemRef);
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
#endif
}

StatusOr<std::unique_ptr<MemRefValue>>
RuntimeClient::copyToHost(const MemRefValue &deviceMemRef,
                          std::optional<CudaStream> cudaStream) {
#ifdef MLIR_EXECUTOR_ENABLE_CUDA
  if (!isDeviceVisible(deviceMemRef.getAddressSpace()))
    return getInvalidArgStatus("to copy a MemRef to the host from a device, "
                               "its data must reside in an address space "
                               "that the host can access");

  int64_t copySizeInBytes = deviceMemRef.getTotalFootprintInBytes();

  // Allocate the host buffer.
  StatusOr<PointerInfo> allocation =
      runtime::allocate(this->getAllocTracker(), PointerType::host,
                        copySizeInBytes, 16, std::nullopt);
  if (!allocation.isOk())
    return allocation.getStatus();

  // Create the device buffer descriptor.
  StatusOr<std::unique_ptr<MemRefValue>> hostMemRef = MemRefValue::create(
      this, allocation->type, deviceMemRef.getElementBitWidth(),
      allocation->ptr, 0, deviceMemRef.getShape(), deviceMemRef.getStrides(),
      nullptr, deviceMemRef.getScalarType());
  if (!hostMemRef.isOk())
    return hostMemRef.getStatus();

  if (!cudaStream) {
    RETURN_ERROR_IF_CUDART_ERROR(
        cudaMemcpy((*hostMemRef)->getVoidPtr(), deviceMemRef.getVoidPtr(),
                   copySizeInBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
  } else {
    RETURN_ERROR_IF_CUDART_ERROR(
        cudaMemcpyAsync((*hostMemRef)->getVoidPtr(), deviceMemRef.getVoidPtr(),
                        copySizeInBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost,
                        reinterpret_cast<cudaStream_t>(*cudaStream)));
  }

  return std::move(*hostMemRef);
#endif
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
}

Status RuntimeClient::copyToHost(const MemRefValue &deviceMemRef,
                                 MemRefValue &hostMemRef,
                                 std::optional<CudaStream> stream) {
#ifdef MLIR_EXECUTOR_ENABLE_CUDA
  if (!isDeviceVisible(deviceMemRef.getAddressSpace()))
    return getInvalidArgStatus(
        "to copy a MemRef to the host from a device, "
        "its data must reside in an address space "
        "that the host can access but got {0}",
        impl::EnumNamePointerType(deviceMemRef.getAddressSpace()));
  if (!isHostVisible(hostMemRef.getAddressSpace()))
    return getInvalidArgStatus(
        "to copy a MemRef to an existing host MemRef, the destination MemRef's "
        "address space must be host-visible but got address space {0}",
        impl::EnumNamePointerType(hostMemRef.getAddressSpace()));
  if (deviceMemRef.getElementBitWidth() != hostMemRef.getElementBitWidth())
    return getInvalidArgStatus(
        "copying device MemRef to host MemRef requires that the element type "
        "bit-widths match, but got source bitwidth={0} and destination "
        "bitwidth={1}",
        deviceMemRef.getElementBitWidth(), hostMemRef.getElementBitWidth());

  if (deviceMemRef.getShape() != hostMemRef.getShape() ||
      deviceMemRef.getStrides() != hostMemRef.getStrides())
    return getInvalidArgStatus(
        "copying device MemRef to host MemRef requires the shape and strides "
        "to match, "
        " but the source MemRef has shape=({0,$[, ]}) strides=({1,$[, ]}) and "
        "the destination MemRef has shape=({2,$[, ]}) strides=({3,$[, ]})",
        deviceMemRef.getShape(), deviceMemRef.getStrides(),
        hostMemRef.getShape(), hostMemRef.getStrides());

  int64_t copySizeInBytes = deviceMemRef.getTotalFootprintInBytes();
  if (!stream) {
    RETURN_ERROR_IF_CUDART_ERROR(
        cudaMemcpy(hostMemRef.getVoidPtr(), deviceMemRef.getVoidPtr(),
                   copySizeInBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
  } else {
    RETURN_ERROR_IF_CUDART_ERROR(
        cudaMemcpyAsync(hostMemRef.getVoidPtr(), deviceMemRef.getVoidPtr(),
                        copySizeInBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost,
                        reinterpret_cast<cudaStream_t>(*stream)));
  }
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA enabled");
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
struct has_print_func<S, T,
                      std::void_t<decltype(mlirtrt::runtime::print(
                          std::declval<S &>(), std::declval<T>()))>>
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
      os, x, [](llvm::raw_ostream &os, const auto &x) { rt::print(os, x); },
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

llvm::raw_ostream &rt::print(llvm::raw_ostream &os, const TypeUnionView &arg) {
  if (arg.isa<MemrefTypeView>())
    return print(os, arg.get<MemrefTypeView>());
  if (arg.isa<ScalarTypeView>())
    return print(os, arg.get<ScalarTypeView>());
  if (arg.isa<ExternalOpaqueTypeView>())
    return print(os, arg.get<ExternalOpaqueTypeView>());
  return os << "UNK";
}

llvm::raw_ostream &rt::print(llvm::raw_ostream &os,
                             const DimensionBoundsView &exe) {
  os << "dim_bounds<min = [";
  interleave(
      os, exe.getMin(), [](llvm::raw_ostream &os, auto x) { os << x; }, ",");
  os << "], max = [";
  interleave(
      os, exe.getMax(), [](llvm::raw_ostream &os, auto x) { os << x; }, ",");
  return os << "]>";
}

llvm::raw_ostream &rt::print(llvm::raw_ostream &os,
                             const ValueBoundsView &exe) {
  os << "value_bounds<min = [";
  interleave(
      os, exe.getMin(), [](llvm::raw_ostream &os, auto x) { os << x; }, ",");
  os << "], max = [";
  interleave(
      os, exe.getMax(), [](llvm::raw_ostream &os, auto x) { os << x; }, ",");
  return os << "]>";
}

llvm::raw_ostream &rt::print(llvm::raw_ostream &os,
                             const BoundsUnionView &bounds) {
  if (bounds.isa<DimensionBoundsView>())
    return print(os, bounds.get<DimensionBoundsView>());
  if (bounds.isa<ValueBoundsView>())
    return print(os, bounds.get<ValueBoundsView>());
  return os << "UNK";
}

llvm::raw_ostream &rt::print(llvm::raw_ostream &os, const Executable &exe) {
  os << "RuntimeExecutable<name=" << exe.getName() << ",";
  os << "functions=";
  squareBraces(os, LAMBDAF(interleaveComma(os, exe.getFunctions())));
  os << "constants=";
  squareBraces(os, LAMBDAF(interleaveComma(os, exe.getConstants());));
  os << ",source=";
  squareBraces(os, LAMBDAF(os << exe.getCode().size() << " bytes";));
  return os << ">";
}

llvm::raw_ostream &rt::print(llvm::raw_ostream &os,
                             const ConstantView &constant) {
  os << "Constant<" << constant.getName() << ", " << constant.size() << " bytes"
     << ">";
  return os;
}
llvm::raw_ostream &rt::print(llvm::raw_ostream &os, const MemrefTypeView &exe) {

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
  os << mlirtrt::runtime::impl::EnumNameScalarTypeCode(exe.getElementType());
  if (!exe.getStrides().empty()) {
    os << ",";
    interleave(os, exe.getStrides(), handleDimOrStride, "x");
  }
  os << "," << exe.getAddressSpace();
  return os << ">";
}
llvm::raw_ostream &rt::print(llvm::raw_ostream &os,
                             const ScalarTypeView &scalarType) {
  return os << impl::EnumNameScalarTypeCode(scalarType);
}
llvm::raw_ostream &rt::print(llvm::raw_ostream &os,
                             const ExternalOpaqueTypeView &type) {
  return os << "ExternalOpaqueRefType<"
            << impl::EnumNameExternalOpaqueRefKind(type) << ">";
}
llvm::raw_ostream &rt::print(llvm::raw_ostream &os,
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
  os << ">";
  return os;
}

llvm::raw_ostream &rt::print(llvm::raw_ostream &os, const FunctionView &func) {
  os << "Function<" << func.getName() << ", ";
  print(os, func.getSignature());
  return os << ">";
}
