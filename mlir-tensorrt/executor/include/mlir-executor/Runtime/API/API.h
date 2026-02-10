//===- API.h ----------------------------------------------------*- C++ -*-===//
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
/// Definitions and utilities for the runtime interface that are critical for
/// the compielr-runtime or runtime-user interface.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_RUNTIME_API_API
#define MLIR_EXECUTOR_RUNTIME_API_API

#include "mlir-executor/Runtime/API/Executable.h"
#include "mlir-executor/Runtime/FFI/FFI.h"
#include "mlir-executor/Runtime/Support/Allocators.h"
#include "mlir-executor/Runtime/Support/CUDAEventPool.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ThreadPool.h"
#include <complex>
#include <memory>
#include <mutex>
#include <variant>

namespace mtrt {

class RuntimeClient;

template <typename T>
using RefCounted = llvm::ThreadSafeRefCountedBase<T>;

template <typename T>
using Ref = llvm::IntrusiveRefCntPtr<T>;

/// CRTP template for creating integer ID type wrappers.
template <typename Derived>
struct StrongTypeWrapper {
  explicit StrongTypeWrapper(int value) : value(value) {}
  bool operator==(const StrongTypeWrapper &other) const {
    return value == other.value;
  }
  bool operator!=(const StrongTypeWrapper &other) const {
    return value != other.value;
  }
  operator int() const { return value; }

private:
  int value;
};

/// Strong type wrapper to represent a hardware id. This is essentially the CUDA
/// ordinal.
struct HardwareId : public StrongTypeWrapper<HardwareId> {
  using StrongTypeWrapper::StrongTypeWrapper;
};

/// Strong type wrapper to represent a host id. This identifies each host in a
/// multi-host execution environment.
struct HostId : public StrongTypeWrapper<HostId> {
  using StrongTypeWrapper::StrongTypeWrapper;
};

///===---------------------------------------------------------------------===//
// Device
//===----------------------------------------------------------------------===//

/// DeviceGuard is an abstract RAII handle that scopes a temporary activation
/// of device-specific state (for example, making a device current). Concrete
/// backends create instances via `Device::createDeviceGuard()`. When a guard
/// is destroyed, the backend must restore the previous device/context.
class DeviceGuard {
public:
  DeviceGuard() = default;
  virtual ~DeviceGuard() = default;

  DeviceGuard(DeviceGuard &&) = delete;
  DeviceGuard(const DeviceGuard &) = delete;
  DeviceGuard &operator=(DeviceGuard &&) = delete;
  DeviceGuard &operator=(const DeviceGuard &) = delete;
};

class Stream;

/// Specifies kinds of devices.
enum class DeviceKind { GPU, GreenContext };

/// Return the string representation of the device kind.
llvm::StringRef getDeviceKindString(DeviceKind kind);

/// A device description holds metadata information about a device. Each
/// device has a globally unique integer ID within each device kind type class
/// (e.g. GPU, DLA, etc). For CUDA GPU devices on a single host system, this is
/// just the CUDA device ordinal exposed via the CUDA Runtime API. IDs must be
/// contiguous so that they completely fill the range `[0, NumDevices)`.
class DeviceDescription {
protected:
  DeviceDescription(DeviceKind kind, HostId hostId)
      : kind(kind), hostId(hostId) {}

public:
  virtual ~DeviceDescription() = default;
  virtual int32_t getUniqueID() const = 0;

  /// Return the kind of the device.
  DeviceKind getKind() const { return kind; }

  /// Return a human-readable description of the device. If verbose is true,
  /// return a verbose description with as much information as possible (e.g.
  /// for debugging purposes).
  virtual llvm::StringRef getString(bool verbose = false) const = 0;

  /// Return the ID of the host that this device is attached to.
  HostId getHostID() const { return hostId; }

  using AttributeValue =
      std::variant<std::string, bool, int32_t, std::vector<int32_t>, float>;

  /// Return the attribute dictionary.
  const llvm::StringMap<AttributeValue> &getAttributes() const {
    return attributes;
  }
  /// Return the mutable attribute dictionary.
  llvm::StringMap<AttributeValue> &getAttributes() { return attributes; }

protected:
  DeviceKind kind;
  HostId hostId;

  /// Contains attributes of the device.
  using AttributeMap = llvm::StringMap<AttributeValue>;
  AttributeMap attributes;
};

/// Device description for a CUDA GPU device.
class GPUDeviceDescription final : public DeviceDescription {
private:
  GPUDeviceDescription(HostId hostId, HardwareId deviceNumber);

public:
  ~GPUDeviceDescription() = default;

  /// Create a CUDA device description for a given host and device number.
  /// Note that this assumes the `HostId` refers to the current process'
  /// host ID; the information about the device is going to be populated
  /// from the CUDA runtime API.
  /// TODO: currently only a single host (hostId = 0) is supported. To expand
  /// support for multiple hosts, we need to have a way to create the unique ID
  /// (e.g need to know number of devices per host).
  static StatusOr<std::unique_ptr<GPUDeviceDescription>>
  createFromLocal(HostId hostId, HardwareId deviceNumber);

  llvm::StringRef getString(bool verbose = false) const final;
  int32_t getUniqueID() const final { return deviceNumber; }
  HardwareId getDeviceNumber() const { return deviceNumber; }

protected:
  HardwareId deviceNumber;
  std::string description;
  std::string debugString;
};

/// Device is an abstract handle describing a compute device visible to the
/// runtime (e.g., a CUDA GPU). It exposes a stable numeric identifier, a
/// backend kind string, and a factory for acquiring a `DeviceGuard` that makes
/// the device current for the guard's lifetime.
/// Devices are always uniquely owned by the RuntimeClient, so the
/// RuntimeClient's lifetime should always exceed the lifetime of any object
/// that takes a reference to a Device.
class Device {
public:
  virtual ~Device() = default;

  /// Return a backend-specific, zero-based device ordinal that uniquely
  /// identifies this device within the process.
  virtual HardwareId getDeviceNumber() const = 0;

  /// Create an RAII guard that activates this device for the duration of the
  /// guard's lifetime. Implementations must restore any prior device/context on
  /// guard destruction.
  virtual StatusOr<std::unique_ptr<DeviceGuard>> createDeviceGuard() const = 0;

  /// Return the stream associated with the device.
  virtual Ref<Stream> getStream() const = 0;

  /// Return the thread pool associated with the device.
  virtual llvm::ThreadPoolInterface &getThreadPool() = 0;

  /// Return the device description.
  virtual const DeviceDescription &getDescription() const = 0;
};

/// A CUDADevice represents a CUDA GPU.
class CUDADevice final : public Device {
protected:
  CUDADevice(std::unique_ptr<GPUDeviceDescription> description,
             Ref<Stream> stream)
      : description(std::move(description)), stream(std::move(stream)),
        threadPool() {}

public:
  /// Create a CUDA device from the local host and device number.
  static StatusOr<std::unique_ptr<Device>>
  createFromLocal(HostId hostId, HardwareId deviceNumber);

  Ref<Stream> getStream() const final { return stream; }
  llvm::ThreadPoolInterface &getThreadPool() final { return threadPool; }
  StatusOr<std::unique_ptr<DeviceGuard>> createDeviceGuard() const final;
  HardwareId getDeviceNumber() const final {
    return description->getDeviceNumber();
  }
  const DeviceDescription &getDescription() const final { return *description; }

private:
  std::unique_ptr<GPUDeviceDescription> description;
  Ref<Stream> stream;
  llvm::StdThreadPool threadPool;
};

//===----------------------------------------------------------------------===//
// Stream
//===----------------------------------------------------------------------===//

/// A stream is a sequence of operations that execute on GPU in the order in
/// which they are issued by the host.
class Stream : public llvm::ThreadSafeRefCountedBase<Stream> {
public:
  virtual ~Stream() = default;
  /// Get the underlying CUDA stream handle.
  virtual CudaStream getCUDAHandle() const = 0;
  /// Get the device that this stream is associated with.
  virtual Device *getDevice() const = 0;
  /// Synchronize the stream with host.
  virtual Status sync() = 0;
};

//===----------------------------------------------------------------------===//
// Event
//===----------------------------------------------------------------------===//

/// A functional type representing a generic callback that should be invoked
/// when an event is ready.
using OnReadyCallback = std::function<void(Status status, void *userData)>;

/// This class implements a future interface built on top of CUDA runtime. The
/// event is created on the head of a Stream and becomes ready when the Stream's
/// execution reaches that point.
///
/// ## Callbacks
///
/// Callbacks which are invoked when the Stream becomes ready can
/// be added via `addReadyCallback`. It is safe to perform CUDA operations in
/// the callbacks. Callbacks are invoked on the thread pool owned by the
/// Stream's associated Device.
///
/// ## Lifecycle
///
/// Events are managed by unique_ptr. Ownership of an event can be relinquished
/// by invoking `Stream::releaseWhenReady`, which transfers the ownership to an
/// on-ready callback, ensuring that the Event and its resources (e.g. CUDA
/// handle) outlive any other callbacks.
///
/// ## CUDA Interop
///
/// The Event class provides future-like interface. The underlying mechanism
/// uses a CUDA stream and `cudaLaunchHostFunc` to signal the event
/// ready state. Technically, a CUDA Event Handle is not needed to implement
/// this mechanism, but it remains useful to expose a CUDA Event associated
/// with the with the point on the stream timeline immediately prior to
/// signaling the ready state. This allows for synchronization with other
/// CUDA streams. However, care must be taken when using `Event::getCudaHandle`
/// to ensure that the lifetime of the owning `unique_ptr<Event>` outlives
/// any use of the raw CUDA handle.
///
/// TODO: consider renaming this "Future" and make "Event <=> cudaEvent_t"
/// reference counted.
class Event {
public:
  /// Create a new event associated with a CUDA event that is recorded onto the
  /// given stream. Then event will become ready when the associated CUDA event
  /// is ready.
  static StatusOr<std::unique_ptr<Event>> create(Ref<Stream> stream);

  /// Create a trivially ready event.
  static std::unique_ptr<Event> createReadyEvent();

  /// Destroys the event. Joins `thread` if possible to ensure the Event has
  /// lapsed and callbacks are invoked.
  ~Event();

  /// Returns true if this event is ready.
  bool checkIsReady();

  /// Set the event to 'ready' and invoke all callbacks.
  void setReady(Status s);

  /// Add `callback` to the callback queue. If the event is ready, then
  /// `callback` is invoked immediately on the current thread. Otherwise,
  /// callbacks are invoked asynchronously on the Device thread pool
  /// associated with the event's Stream.
  void addReadyCallback(OnReadyCallback callback, void *userData);

  /// Get the status of the event.
  Status getStatus();

  /// Wait for the event to be ready.
  Status waitForReady(
      std::chrono::microseconds timeout = std::chrono::microseconds::max());

  /// Get the raw CUDA event handle associated with the timepoint immediately
  /// prior to signaling the ready state.
  CudaEvent getCudaHandle() const { return cudaEventHandle; }

  /// Complete the lifecycle of an Event by appending an on-ready callback that
  /// deletes the given `event`. Note that the event may be deleted immediately
  /// on the current thread if the event is already ready, otherwise the
  /// deletion will occur on whatever thread executes the callbacks.
  static void releaseWhenReady(std::unique_ptr<Event> event);

private:
  Event(bool ready, Status status, CudaEvent cudaEventHandle);

  /// Set the internal status.
  void setStatus(Status status);

  CudaEvent cudaEventHandle;

  /// Lock to protect state.
  std::mutex lock;

  /// Tracks the events ready state.
  bool ready;

  /// The set of callbacks which will be called asynchronously by a work in
  /// `threadPool` when the event becomes ready.
  std::vector<std::pair<OnReadyCallback, void *>> callbacks;

  /// An internal status code for the error. When the event completion callbacks
  /// are invoked, the status will be passed to the callbacks so that they can
  /// be notified of any errors.
  Status status{Status::getOk()};
};

//===----------------------------------------------------------------------===//
// PointerInfo
//===----------------------------------------------------------------------===//

/// PointerInfo contains metadata about an allocated region of memory,
/// which may be externally or internally managed.
struct PointerInfo {
  PointerInfo(uintptr_t ptr, uint64_t size = kUnknownSize,
              PointerType type = PointerType::unknown,
              PointerOwner owner = PointerOwner::internal)
      : ptr(ptr), size(size), type(type), owner(owner) {}

  PointerInfo() = default;

  /// Actual pointer value.
  uintptr_t ptr{0};

  /// Size of the allocation in bytes.
  uint64_t size{0};

  /// Type of pointer/allocation (e.g. host, device)
  PointerType type{PointerType::unknown};

  /// Whether or not the pointer is an externally managed reference.
  PointerOwner owner{PointerOwner::unknown};

  /// The sentical value that indicates `size` is unkown. This is common when we
  /// import an external pointer.
  static constexpr uint64_t kUnknownSize = std::numeric_limits<uint64_t>::max();

  bool isHostVisible() const {
    return type == PointerType::host || type == PointerType::pinned_host ||
           type == PointerType::unified;
  }

  bool isDeviceVisible() const {
    return type == PointerType::device || type == PointerType::pinned_host ||
           type == PointerType::unified;
  }

  bool hasKnownSize() const { return size != PointerInfo::kUnknownSize; }

  template <typename T>
  T *getPtrToOffset(size_t byteOffset) {
    uintptr_t adjusted = ptr + byteOffset;
    assert((!hasKnownSize() || adjusted < ptr + size) &&
           "attempted out-of-bounds access");
    return reinterpret_cast<T *>(adjusted);
  }

  bool isInternallyManaged() const { return owner == PointerOwner::internal; }
  bool isExternallyManaged() const { return owner == PointerOwner::external; }
};

//===----------------------------------------------------------------------===//
// RuntimeValue
//===----------------------------------------------------------------------===//

class RuntimeValue {
public:
  virtual ~RuntimeValue() = default;

  enum class Kind { Scalar, MemRef };
  RuntimeValue(Kind kind) : kind(kind) {}
  Kind getKind() const { return kind; }

private:
  Kind kind;
};

//===----------------------------------------------------------------------===//
// ScalarValue
//===----------------------------------------------------------------------===//

class ScalarValue : public RuntimeValue {
public:
  union Storage {
    uint64_t real;
    void *complex;
  };

  template <typename T>
  ScalarValue(T data_, ScalarType type)
      : RuntimeValue(Kind::Scalar), type(type) {
    static_assert(sizeof(T) <= sizeof(uint64_t) &&
                      alignof(T) <= alignof(uint64_t),
                  "expected scalar type size to be <= 8 bytes");
    *reinterpret_cast<T *>(&data.real) = data_;
  }

  template <typename T>
  ScalarValue(T real_, T imag_, ScalarType type)
      : RuntimeValue(Kind::Scalar), type(type) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "complex constructor only valid for float and double types.");
    static_assert(sizeof(T) <= sizeof(uint64_t) &&
                      alignof(T) <= alignof(uint64_t),
                  "expected scalar type size to be <= 8 bytes.");
    assert(isComplex() &&
           "complex value constructor used for non-complex scalar type.");
    data.complex = std::make_unique<std::complex<T>>(real_, imag_).release();
  }

  /// Create a new scalar value where the storage is zero-initialized.
  static std::unique_ptr<ScalarValue> createUndef(ScalarType type);

  // Delete copy constructors.
  ScalarValue(const ScalarValue &other) = delete;
  ScalarValue &operator=(const ScalarValue &other) = delete;

  // Move constructors.
  ScalarValue(ScalarValue &&other) noexcept;
  ScalarValue &operator=(ScalarValue &&other) noexcept;

  ~ScalarValue();

  ScalarType getType() const { return type; }

  template <typename T>
  T get() const {
    static_assert(sizeof(T) <= sizeof(uint64_t),
                  "expected scalar type size to be <= 8 bytes.");
    assert(!isComplex() && "use `getComplex()` for complex scalar.");
    return *reinterpret_cast<const T *>(&data.real);
  }

  template <typename T>
  std::complex<T> getComplex() const {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "getComplex() only supports float and double types.");
    static_assert(sizeof(T) <= sizeof(uint64_t) &&
                      alignof(T) <= alignof(uint64_t),
                  "expected scalar type size to be <= 8 bytes.");
    assert(isComplex() &&
           "complex value constructor used for non-complex scalar type.");
    if constexpr (std::is_same_v<T, float>) {
      assert(
          type.getCode() == ScalarTypeCode::complex32 &&
          "Type mismatch: expected scalar type code of complex32 for float.");
    } else {
      assert(
          type.getCode() == ScalarTypeCode::complex64 &&
          "Type mismatch: expected scalar type code of complex64 for double.");
    }
    return *static_cast<const std::complex<T> *>(data.complex);
  }

  bool isComplex() const {
    return type.getCode() == ScalarTypeCode::complex32 ||
           type.getCode() == ScalarTypeCode::complex64;
  }

  void *getRaw() { return isComplex() ? data.complex : &data.real; }

  static bool classof(const RuntimeValue *v) {
    return v->getKind() == Kind::Scalar;
  }

private:
  ScalarValue(ScalarType type, Storage data)
      : RuntimeValue(Kind::Scalar), data(std::move(data)), type(type) {}

  void cleanup();
  Storage data;
  ScalarType type;
};

//===----------------------------------------------------------------------===//
// MemRefValue
//===----------------------------------------------------------------------===//

class RuntimeClientAllocator;

/// MemRefStorage is an abstract base class that owns the underlying buffer
/// associated with a MemRefValue. It is reference counted to enable shared
/// ownership across the C++ runtime API and other external users (e.g. C API,
/// Python API).
class MemRefStorage : public RefCounted<MemRefStorage> {
public:
  virtual ~MemRefStorage() {}

  uintptr_t getPtr() const { return ptr; }

  virtual PointerType getMemorySpace() const = 0;
  virtual Ref<Stream> getStream() const { return nullptr; }

  Ref<RuntimeClient> getClient() const { return client; }

  Device *getDevice() const { return device; }

protected:
  MemRefStorage(uintptr_t ptr, Device *device, Ref<RuntimeClient> client)
      : ptr(ptr), device(device), client(std::move(client)) {}

  uintptr_t ptr;

  Device *device;

  /// Reference back to RuntimeClient. This provides concrete subclasses access
  /// to the client's allocator object. It also ensures that the client is not
  /// destroyed while any users of the storage are still alive.
  Ref<RuntimeClient> client;
};

//===----------------------------------------------------------------------===//
// BufferStridedLayout
//===----------------------------------------------------------------------===//

class BufferStridedLayout {
public:
  BufferStridedLayout() = default;
  BufferStridedLayout(llvm::ArrayRef<int64_t> strides, int64_t offset)
      : strides(strides), offset(offset) {}

  llvm::ArrayRef<int64_t> getStrides() const { return strides; }
  int64_t getOffset() const { return offset; }

  /// Create a canonical row major layout.
  /// We follow the PyTorch convention that unit dimensions have unit stride.
  static BufferStridedLayout
  createCanonicalRowMajor(llvm::ArrayRef<int64_t> shape);

  /// Return whether the layout is equal to another layout.
  bool operator==(const BufferStridedLayout &other) const {
    return strides == other.strides && offset == other.offset;
  }
  bool operator!=(const BufferStridedLayout &other) const {
    return !(*this == other);
  }

  /// Return debug string representation of the layout.
  std::string toString() const;

  friend class BufferType;

private:
  std::vector<int64_t> strides;
  int64_t offset{0};
};

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

/// Encapsulates the information about a runtime buffer (MemRefValue)'s logical
/// type: scalar element type, shape, strides, offset, and address space.
class BufferType {
public:
  BufferType() = default;

  BufferType(ScalarType elementType, const std::vector<int64_t> &shape,
             const std::vector<int64_t> &strides,
             mtrt::PointerType addressSpace, int64_t offset)
      : elementType(elementType), shape(shape), layout(strides, offset),
        addressSpace(addressSpace) {}
  BufferType(ScalarType elementType, const std::vector<int64_t> &shape,
             BufferStridedLayout layout, mtrt::PointerType addressSpace)
      : elementType(elementType), shape(shape), layout(layout),
        addressSpace(addressSpace) {}

  static BufferType
  createWithByteStrides(ScalarType elementType,
                        const std::vector<int64_t> &shape,
                        const std::vector<int64_t> &byteStrides,
                        mtrt::PointerType addressSpace, int64_t offset);

  static BufferType
  createWithElementStrides(ScalarType elementType,
                           const std::vector<int64_t> &shape,
                           const std::vector<int64_t> &elementStrides,
                           mtrt::PointerType addressSpace, int64_t offset);

  static BufferType createWithCanonicalLayout(ScalarType elementType,
                                              const std::vector<int64_t> &shape,
                                              mtrt::PointerType addressSpace);

  /// Creates a BufferType the flatbuffers' MemRefTypeView.
  static BufferType getFromSerializedType(const MemRefTypeView &type);

  /// Return whether the shape is static.
  bool hasStaticShape() const;

  /// Return the number of elements in the buffer (volume of shape).
  int64_t getNumElements() const;

  /// Return the number of bytes occupied by this buffer type, taking into
  /// account the strides, which may have padding.
  size_t getFootprintSizeInBytes() const;

  int64_t getRank() const { return static_cast<int64_t>(shape.size()); }

  /// Return the shape of the buffer.
  llvm::ArrayRef<int64_t> getShape() const {
    return llvm::ArrayRef<int64_t>(shape.data(), shape.size());
  }

  ScalarType getElementType() const { return ScalarType(elementType); }

  /// Return the strides of the buffer in terms of bytes.
  std::vector<int64_t> getByteStrides() const;

  /// Return the layout of the buffer.
  const BufferStridedLayout &getLayout() const { return layout; }

  /// Compare the type to another type.
  bool operator==(const BufferType &other) const {
    return other.elementType == elementType && other.shape == shape &&
           other.layout == layout && other.addressSpace == addressSpace;
  }
  bool operator!=(const BufferType &other) const { return !(*this == other); }

  /// Returns a string representation of the buffer type.
  std::string toString() const;

  /// Return the address space of the buffer.
  PointerType getAddressSpace() const { return addressSpace; }

  /// Returns true if the buffer layout has a canonical "row major" layout,
  /// meaning that the dimensions are ordered major-to-minor in terms of strides
  /// and the buffer is fully packed and contiguous (no padding).
  bool isCanonicalPacked() const;

  /// Returns true if the buffer layout (shape + strides) has the
  /// canonical layout.
  bool isCanonicalRowMajor() const;

  /// Returns true if the buffer layout has a canonical "column major" layout,
  /// meaning that the dimensions are ordered minor-to-major in terms of strides
  /// and the buffer is fully packed and contiguous (no padding).
  bool isCanonicalColMajor() const;

private:
  ScalarTypeCode elementType{ScalarTypeCode::unknown};
  std::vector<int64_t> shape;
  BufferStridedLayout layout;
  mtrt::PointerType addressSpace{mtrt::PointerType::unknown};
};

std::ostream &operator<<(std::ostream &os, const BufferType &t);

//===----------------------------------------------------------------------===//
// MemRefValue
//===----------------------------------------------------------------------===//

/// A MemRefValue encapsulates a reference to MemRefStorage along with logical
/// buffer type information (e.g. scalar value type, shape, strides, etc).
class MemRefValue : public RuntimeValue {
public:
  /// Create a new MemRef descriptor. All size quantities are in "units of
  /// elements" unless otherwise noted.
  static mtrt::StatusOr<std::unique_ptr<MemRefValue>>
  create(mtrt::PointerType addressSpace, ScalarTypeCode elementType,
         Ref<MemRefStorage> storage, int64_t offset,
         llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
         Device *device = nullptr, bool assertCanonicalStrides = false);

  static mtrt::StatusOr<std::unique_ptr<MemRefValue>>
  create(mtrt::PointerType addressSpace, const BufferType &type,
         Ref<MemRefStorage> storage, Device *device = nullptr,
         bool assertCanonicalStrides = false) {
    return create(addressSpace, type.getElementType(), std::move(storage),
                  type.getLayout().getOffset(), type.getShape(),
                  type.getLayout().getStrides(), device,
                  assertCanonicalStrides);
  }

  mtrt::PointerType getBufferKind() { return type.getAddressSpace(); }
  int64_t getElementBitWidth() const {
    return type.getElementType().getBitWidth();
  }
  /// Return the layout of the buffer.
  const BufferStridedLayout &getLayout() const { return type.getLayout(); }
  llvm::ArrayRef<int64_t> getShape() const { return type.getShape(); }
  llvm::ArrayRef<int64_t> getStrides() const {
    return type.getLayout().getStrides();
  }
  int64_t getRank() const { return type.getRank(); }
  int64_t getTotalFootprintInBytes() const;
  uintptr_t getMemory() const { return storage->getPtr(); }
  void *getVoidPtr() const { return reinterpret_cast<void *>(getMemory()); }
  Device *getDevice() const { return device; }
  PointerInfo getPointerInfo(PointerOwner ownership) const {
    return PointerInfo(getMemory(), getTotalFootprintInBytes(),
                       type.getAddressSpace(), ownership);
  }
  PointerType getAddressSpace() const { return type.getAddressSpace(); }

  static bool classof(const RuntimeValue *v) {
    return v->getKind() == Kind::MemRef;
  }

  ScalarType getScalarType() const { return type.getElementType(); }

  Ref<RuntimeClient> getClient() const { return storage->getClient(); }

  /// Return the reference count of the underlying storage.
  unsigned getStorageRefCount() const { return storage->UseCount(); }

  /// Return a new MemRefValue that references the same storage as this one.
  /// The reference count of the storage is incremented.
  std::unique_ptr<MemRefValue> createRef() const {
    return std::unique_ptr<MemRefValue>(
        new MemRefValue(type.getAddressSpace(), type.getElementType(), storage,
                        type.getLayout().getOffset(), type.getShape(),
                        type.getLayout().getStrides(), device));
  }

  /// Return a reference to the underlying storage.
  Ref<MemRefStorage> getStorageRef() const { return storage; }

  /// Return the type of the buffer.
  const BufferType &getType() const { return type; }

private:
  MemRefValue(mtrt::PointerType addressSpace, ScalarTypeCode elementType,
              Ref<MemRefStorage> storage, int64_t offset,
              llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
              Device *device);

  /// Holds the underlying storage object.
  Ref<MemRefStorage> storage;

  /// The logical type of the buffer.
  BufferType type;

  /// Non-owned view to the associated device if the address space is a device
  /// address. This may be nullptr in the case of a host buffer or an externally
  /// managed buffer.
  Device *device;
};

//===----------------------------------------------------------------------===//
// RuntimeSessionOptions
//===----------------------------------------------------------------------===//

/// RuntimeSessionOptions encapsulates the required information for creating
/// a RuntimeSession context from a Executable. Executables that have been
/// compiled for a multi-process parallel execution environment must be provided
/// with the appropriate runtime information such as the process rank , the
/// number of total processes in the grid, and the NCCL UUID for initial
/// communicator creation.
class RuntimeSessionOptions {
public:
  /// Construct session options by directly specifying the device ID, number of
  /// devices, and NCCL UUID. Single-device sessions can use the default
  /// options.
  static RuntimeSessionOptions getSPMDOptions(int32_t numDevices = 1,
                                              int32_t deviceId = 0,
                                              llvm::StringRef ncclUuid = "");

  /// Enable the specified features for the runtime session.
  void enableFeatures(llvm::ArrayRef<std::string> features);

  /// Returns true if the given feature is enabled.
  bool isFeatureEnabled(llvm::StringRef feature) const;

  /// Populates the runtime session options using the MPI calls. Each MPI
  /// process is expected to be associated with a CUDA device associated with
  /// the corresponding rank id.
  static StatusOr<RuntimeSessionOptions> createUsingSingleHostMpi();

  /// Return the number of devices (ranks) in the process grid.
  int32_t getNumDevices() const { return numDevicesGlobally; }

  /// Return the number of devices per program.
  int32_t getNumDevicesPerProgram() const { return numDevicesPerProgram; }

  /// Return the rank of the worker in the process grid that the runtime session
  /// should correspond to. This is only valid for SPMD mode on a single host.
  /// Currently this should correspond to CUDA deviceId, but in the future we
  /// will need to maintain a lookup table from rank to local device ID and
  /// vice-versa.
  StatusOr<int32_t> getSpmdDeviceId() const;

  /// Return the UUID that should be used to create the top-level NCCL
  /// communicator for each device. This should only be empty if there is only
  /// one device.a
  llvm::StringRef getNcclUuid() const { return ncclUuid; }

  /// Return the set of features that are enabled for this session.
  const llvm::StringSet<> &getEnabledFeatures() const { return features; }

  /// Return the map from logical device ID to CUDA ordinal.
  const std::vector<int32_t> &getLogicalDeviceIdToCUDAOrdinal() const {
    return logicalDeviceIdToCUDAOrdinal;
  }

  /// Set the path to the crash reproducer file.
  void setCrashReproducerPath(llvm::StringRef path) {
    crashReproducerPath = path.str();
  }

  /// Return the path to the crash reproducer file.
  llvm::StringRef getCrashReproducerPath() const { return crashReproducerPath; }

  /// Return true if the crash reproducer should be tested by writing the Lua
  bool getTestCrashReproducer() const;

private:
  RuntimeSessionOptions(int32_t numDevices, int32_t numDevicesPerProgram,
                        std::vector<int32_t> logicalDeviceIdToCUDAOrdinal,
                        llvm::StringRef ncclUuid);

  int32_t numDevicesGlobally;
  /// The number of devices per program. This is normally 1 (SPMD mode), but in
  /// the future it could be more than 1.
  int32_t numDevicesPerProgram;

  /// Maps the logical device ID (accessible within the program module) to the
  /// CUDA ordinal. The size of this vector is 1 for SPMD mode, larger than 1 in
  /// MPMD mode.
  std::vector<int32_t> logicalDeviceIdToCUDAOrdinal;

  /// The UUID that should be used to create the top-level NCCL communicator.
  std::string ncclUuid;

  /// A list of features names (e.g. module names) that should be enabled for
  /// this session.
  llvm::StringSet<> features;

  /// The path to a crash reproducer file for code from an executable (e.g. a
  /// Lua script that fails to load).
  std::string crashReproducerPath;
};

//===----------------------------------------------------------------------===//
// AllocTracker
//===----------------------------------------------------------------------===//

/// AllocTracker is a simple wrapper around a map from pointers to
/// additional metadata information (`PointerInfo`) containing the
/// pointer kind, pointer size, and whether or not the pointer is
/// owned by the runtime or is an external reference.
///
/// The CoreModule instantiates an AllocTracker in each Lua context.
/// Users in other modules can access the tracker by calling
/// `AllocTracker::get` and passing a `sol::state_view`.
class AllocTracker {
public:
  /// AllocTracker's destructor will free any tracked non-external allocations.
  ~AllocTracker();

  /// Track the provided pointer.
  void track(PointerInfo info);

  /// Stop tracking the provided ptr.
  void untrack(uintptr_t ptr);

  /// Retrive the information for the provided pointer. Asserts that the pointer
  /// is tracked.
  const PointerInfo &get(uintptr_t ptr) const;

  /// Lookup the provided pointer. If the pointer is not tracked, then the
  /// `PointerInfo` fields besides `PointerInfo::ptr` will be filled with
  /// default values (e.g. unknown size, unknown type).
  PointerInfo lookupOrDefault(uintptr_t ptr) const;

  /// Return true if the tracker's map contains `ptr`.
  bool contains(uintptr_t ptr) const;

private:
  struct Metadata {
    PointerInfo info;
  };

  using MapType = llvm::DenseMap<uintptr_t, std::unique_ptr<Metadata>>;

public:
  auto find(uintptr_t ptr) { return map.find(ptr); }
  auto erase(MapType::iterator it) { map.erase(it); }
  auto end() const { return map.end(); }
  auto begin() const { return map.begin(); }

private:
  llvm::DenseMap<uintptr_t, std::unique_ptr<Metadata>> map;
};

/// A helper that allocates buffers based on the provided pointer type. The
/// AllocTracker will be updated so that it is aware of the allocation. The
/// allocation size (in bytes) and alignment (optional, in bytes) are specified
/// by the caller. For CUDA allocations, a stream may optionally be provided.
StatusOr<PointerInfo> allocate(AllocTracker &tracker, PointerType type,
                               uint64_t size, std::optional<uint32_t> alignment,
                               std::optional<CudaStream> stream);

/// Synchronously dealloates the specified pointer. The PointerInfo from
/// `tracker` is used to determine the method of dceallocation (e.g. `std::free`
/// vs `cudaFree`). The AllocTracker is updated to remove the allocation from
/// its tracking set. If the pointer is absent from `tracker`, an error is
/// returned.
Status safeDeallocate(AllocTracker &tracker, uintptr_t ptr,
                      std::optional<CudaStream> stream = {});

//===----------------------------------------------------------------------===//
// ResourceTracker
//===----------------------------------------------------------------------===//

/// ResourceTracker tracks pools of arbitrary allocated objects paired with
/// deleter methods. It acts as a global context associated with a
/// RuntimeSession and allows runtime backends to pass around objects by
/// pointer. At the end of the session, all the objects will be automatically
/// cleaned up by calling their deleter method. These resources may include
/// things like CUDA runtime objects. Use sparingly.
class ResourceTracker {
public:
  ResourceTracker() = default;
  ~ResourceTracker();

  using Deleter = llvm::function_ref<void(uintptr_t)>;

  void track(uintptr_t ptr, Deleter deleter);
  void untrack(uintptr_t ptr);

private:
  llvm::MapVector<uintptr_t, llvm::function_ref<void(uintptr_t)>> tracker;
};

//===----------------------------------------------------------------------===//
// RuntimeSession
//===----------------------------------------------------------------------===//

/// `RuntimeSession` is the interface that wraps the interpreter execution
/// context. It must be associated with a single CUDA device (equivalently a
/// single rank in the process grid if the Executable is meant to be executed
/// over a process grid). The `RuntimeSession` accepts options that must be
/// provided for functions that need to be executed over a process grid (e.g.
/// NCCL UUID). A single Executable can be used to create multiple
/// RuntimeSessions. A RuntimeSession does not own the Executable, it only has a
/// read-only view to the Executable's storage (e.g. constant data, code, etc).
/// The Executable must outlive any RuntimeSessions that are created from it
/// (and currently no reference counting is implemented).
/// TODO: methods for accessing/setting default stream should be moved here.
class RuntimeSession {
public:
  RuntimeSession(RuntimeSessionOptions options, ExecutableView executable,
                 Ref<RuntimeClient> client);
  virtual ~RuntimeSession() {}

  Ref<RuntimeClient> getClient() const { return client; }

  ExecutableView getExecutable() const { return executable; }

  PinnedMemoryAllocator &getPinnedMemoryAllocator() {
    return pinnedMemoryAllocator;
  }

  AllocTracker &getAllocTracker() { return *allocTracker; }

  ResourceTracker &getResourceTracker() { return *resourceTracker; }

  /// Returns the options used to construct the session.
  const RuntimeSessionOptions &getOptions() { return options; }

  /// Execute the session function using the given arguments.
  virtual StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
  executeFunction(llvm::StringRef name, llvm::ArrayRef<RuntimeValue *> inputs,
                  llvm::ArrayRef<RuntimeValue *> outArgs) = 0;

  /// Return the stream on which work for this session will be launched.
  Ref<Stream> getStream() const;

  /// Set the stream on which work for this session will be launched.
  /// Note that the device for this session is set when it is created.
  /// Therefore, the stream must be associated with the same device, otherwise
  /// an error will be returned.
  ///
  /// It is only valid to pass `nullptr` here if the session was created
  /// without a device and therefore does not have the CUDA feature enabled.
  virtual Status setStream(Ref<Stream> stream);

  /// Return the CUDA event pool for this session.
  CudaEventPool &getCUDAEventPool() { return *cudaEventPool; }

protected:
  /// Called when the stream for this session is changed.
  virtual Status onStreamChanged(Ref<Stream> oldStream, Ref<Stream> newStream);

  Ref<RuntimeClient> client;
  RuntimeSessionOptions options;
  ExecutableView executable;
  PinnedMemoryAllocator &pinnedMemoryAllocator;
  std::unique_ptr<AllocTracker> allocTracker;
  std::unique_ptr<ResourceTracker> resourceTracker;
  Ref<Stream> stream;
  std::unique_ptr<CudaEventPool> cudaEventPool;
};

/// Register the default runtime session LLVM CommandLine options.
void registerGlobalRuntimeSessionCLOptions();

//===----------------------------------------------------------------------===//
// RuntimeClient
//===----------------------------------------------------------------------===//

/// RuntimeClientAllocator is the allocation interface for the RuntimeClient.
/// It differs from the RuntimeSession allocator in that it yields ref-counted
/// MemRefStorage objects rather than raw pointers.
/// TODO: In the future, deallocation will  be enqueued on a separate callback
/// thread pool managed by the client or device.
class RuntimeClientAllocator {
public:
  virtual ~RuntimeClientAllocator() = default;

  virtual StatusOr<Ref<MemRefStorage>>
  allocate(PointerType pointerType, uint64_t size,
           std::optional<uint32_t> alignemnt, Device *device,
           Ref<Stream> stream) = 0;

  /// Use the given pointer as the storage for the MemRefValue.
  virtual StatusOr<Ref<MemRefStorage>> takeOwnership(uintptr_t ptr,
                                                     PointerType pointerType,
                                                     Device *device,
                                                     Ref<Stream> stream) = 0;

  virtual Status deallocate(MemRefStorage &storage) = 0;

protected:
  RuntimeClientAllocator(RuntimeClient &client) : client(client) {}

  RuntimeClient &client;
};

/// The `RuntimeClient` provides a convenient way for users of the C++ API
/// to perform memory allocations and create other resources. The specifics
/// of the ownership semantics for each resource creation are described by the
/// methods below. The `RuntimeClient` is not associated with any particular
/// `Executable` or `RuntimeSession`, rather it is a standalone resource manager
/// that is not required for use of the other parts of the runtime API.
///
/// The RuntimeClient may track resources internally and attempt to free them
/// or warn the user about the dangling resources when it is destructed.
/// Therefore, any resource created by the RuntimeClient should outlive the
/// RuntimeClient and be destroyed/deallocated through the appropriate method.
class RuntimeClient : public RefCounted<RuntimeClient> {
public:
  ~RuntimeClient();

  /// Creates the client. Enumerates CUDA devices on the machine. Creates the
  /// internal allocators.
  static StatusOr<Ref<RuntimeClient>> create();

  llvm::ArrayRef<std::unique_ptr<Device>> getDevices() const;

  // Allocates a new MemRefValue with the buffer type and layout.
  StatusOr<std::unique_ptr<MemRefValue>>
  allocateMemRef(const BufferType &type, Device *device = nullptr,
                 Ref<Stream> stream = nullptr,
                 bool assertCanonicalStrides = false);

  StatusOr<std::unique_ptr<MemRefValue>> createExternalMemRef(
      const BufferType &type, uintptr_t ptr, Device *device = nullptr,
      bool assertCanonicalStrides = false, std::function<void()> = nullptr);

  // Allocates a new host buffer and fills it with data present in the
  // `hostBuffer`.
  StatusOr<std::unique_ptr<MemRefValue>>
  copyHostToHost(const MemRefValue &hostBuffer);

  /// Allocates a new device buffer and fills it with data present on the
  /// host in the specified buffer. The allocation and copy are performed on
  /// the given stream.
  StatusOr<std::unique_ptr<MemRefValue>>
  copyToDevice(const MemRefValue &hostBuffer, Device &device,
               Ref<Stream> stream, std::unique_ptr<Event> *doneWithHostBuffer);

  /// Allocates a new host buffer and fills it with data present on the device
  /// in the specified buffer. The allocation and copy are performed on the
  /// given stream.
  StatusOr<std::unique_ptr<MemRefValue>>
  copyToHost(const MemRefValue &deviceMemRef, Ref<Stream> stream);

  /// Copy from the given device MemRefValue to an existing MemRefValue on the
  /// host.
  Status copyToHost(const MemRefValue &deviceMemRef, MemRefValue &hostMemRef,
                    Ref<Stream> stream);

  /// Copy the given device buffer to another device. The copy occurs
  /// asynchronously on the destination Device's stream. The event associated
  /// with copy completion (which currently also signals done with use of source
  /// buffer), is returned in `copyDoneEvent`.
  StatusOr<std::unique_ptr<MemRefValue>>
  copyDeviceBufferToOtherDevice(const MemRefValue &sourceBuffer,
                                Device &dstDevice,
                                std::unique_ptr<Event> &copyDoneEvent);

  /// Returns the ResourceTracker.
  ResourceTracker &getResourceTracker() { return resourceTracker; }

  /// Return the PinnedMemoryAllocator.
  PinnedMemoryAllocator &getPinnedMemoryAllocator() {
    return pinnedMemoryAllocator;
  }

  RuntimeClientAllocator &getAllocator() { return *allocator; }

  /// Return the current stream of the current CUDA device.
  StatusOr<Ref<Stream>> getCurrentDeviceStream() const;

  //===----------------------------------------------------------------------===//
  // Session Management
  //===----------------------------------------------------------------------===//

  using RuntimeSessionFactory =
      std::function<StatusOr<std::unique_ptr<RuntimeSession>>(
          Ref<RuntimeClient> client, RuntimeSessionOptions options,
          ExecutableView executable)>;

  /// Construct a new RuntimeSession of the given kind.
  StatusOr<std::unique_ptr<RuntimeSession>>
  createSession(llvm::StringRef kind, RuntimeSessionOptions options,
                ExecutableView executable);

  /// Register a new RuntimeSession factory.
  void registerSessionFactory(llvm::StringRef kind,
                              RuntimeSessionFactory factory);

  mtrt::PluginRegistry &getPluginRegistry();

private:
  void setAllocator(std::unique_ptr<RuntimeClientAllocator> allocator) {
    this->allocator = std::move(allocator);
  }

  RuntimeClient(llvm::SmallVector<std::unique_ptr<Device>> devices);

  llvm::SmallVector<std::unique_ptr<Device>> devices;
  PinnedMemoryAllocator pinnedMemoryAllocator;
  ResourceTracker resourceTracker;
  std::unique_ptr<RuntimeClientAllocator> allocator;

  /// Session factory registry.
  llvm::StringMap<RuntimeSessionFactory> sessionFactories;

  mtrt::PluginRegistry pluginRegistry;
};

//===----------------------------------------------------------------------===//
// NCCL Support functions
//===----------------------------------------------------------------------===//

/// Return the NCCL unique communicator ID as a string if the project was
/// configured with NCCL enabled. If the project was not configured with NCCL
/// enabled, then returns an empty string.
StatusOr<std::string> getCommunicatorUniqueId();

} // namespace mtrt

#endif // MLIR_EXECUTOR_RUNTIME_API_API
