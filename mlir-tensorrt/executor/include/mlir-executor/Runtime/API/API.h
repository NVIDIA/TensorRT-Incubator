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
#include "mlir-executor/Support/Allocators.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include <atomic>
#include <complex>
#include <memory>
#include <string_view>

namespace mlirtrt::runtime {

class RuntimeClient;

//===----------------------------------------------------------------------===//
// Intrusive Reference Counting Classes
//===----------------------------------------------------------------------===//

/// Example usage:
///
/// ```cpp
/// #include <iostream>
/// class MyObject : public RefCounted<MyObject> {
/// public:
///     MyObject(int v) : value(v) {
///         std::cout << "MyObject(" << value << ") constructed\n";
///     }
///     ~MyObject() {
///         std::cout << "MyObject(" << value << ") destroyed\n";
///     }
///     void greet() const {
///         std::cout << "Hello from " << value << "\n";
///     }
/// private:
///     int value;
/// };
/// int main() {
///     Ref<MyObject> a(new MyObject(42));
///     {
///         Ref<MyObject> b = a;
///         b->greet(); // "Hello from 42"
///     } // b goes out of scope
///     a->greet();   // still alive
/// } // a goes out of scope, MyObject destroyed
/// ```

template <typename Derived>
class RefCounted {
public:
  void incRef() noexcept { ref_count_.fetch_add(1, std::memory_order_relaxed); }

  void decRef() noexcept {
    if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete static_cast<Derived *>(this);
    }
  }

  unsigned getRefCount() const {
    return ref_count_.load(std::memory_order_relaxed);
  }

protected:
  RefCounted() noexcept : ref_count_(0) {}
  ~RefCounted() = default;

private:
  std::atomic<int> ref_count_;
};

template <typename T>
class Ref {
public:
  explicit Ref(T *ptr = nullptr) noexcept : ptr_(ptr) {
    if (ptr_)
      ptr_->incRef();
  }

  Ref(const Ref &other) noexcept : ptr_(other.ptr_) {
    if (ptr_)
      ptr_->incRef();
  }

  Ref(Ref &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

  Ref &operator=(const Ref &other) noexcept {
    if (this != &other) {
      if (ptr_)
        ptr_->decRef();
      ptr_ = other.ptr_;
      if (ptr_)
        ptr_->incRef();
    }
    return *this;
  }

  Ref &operator=(Ref &&other) noexcept {
    if (this != &other) {
      if (ptr_)
        ptr_->decRef();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  ~Ref() {
    if (ptr_)
      ptr_->decRef();
  }

  T *get() const noexcept { return ptr_; }
  T &operator*() const noexcept { return *ptr_; }
  T *operator->() const noexcept { return ptr_; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
  T *ptr_;
};

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

/// Parse element type code from string.
ScalarTypeCode parseElementType(std::string_view str);

/// Return number of bits in the given element type.
int64_t getBitsPerElement(ScalarTypeCode elType);

/// ScalarTypeCode wrapper to make API easier to use.
class ScalarType {
public:
  ScalarType(ScalarTypeCode code);
  ScalarType() = delete;

  /// Parse an element type from a string.
  static StatusOr<ScalarType> fromString(std::string_view str);
  /*implicit*/ operator ScalarTypeCode() { return code; }

  /// Return the *storage* bit width of the type.
  int64_t getBitWidth() const;

  /// Return the underlying ScalarTypeCode enum value.
  ScalarTypeCode getCode() const { return code; }

  /// Get the human-readable string representation of this type.
  llvm::StringRef getStrRef() const {
    return impl::EnumNameScalarTypeCode(code);
  }

  bool operator==(const ScalarType &other) const {
    return other.code == this->code;
  }
  bool operator!=(const ScalarType &other) const {
    return other.code != this->code;
  }

private:
  ScalarTypeCode code;
};

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

/// Parse pointer type from string.
PointerType parsePointerType(std::string_view str);

/// Print pointer type enum name to string.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, PointerType ptrType);

/// Return the string representation of `ptrType`.
std::string_view stringifyPointerType(PointerType ptrType);

//===----------------------------------------------------------------------===//
// TypeView
// This section includes classes that form the TypeUnion:
// MemRefTypeView, ScalarTypeView
//===----------------------------------------------------------------------===//

/// Base class for all the below classes that provide flatbuffer-view wrappers
/// for flatbuffer tables that comprise the `Type` union in the schema.
template <typename T, impl::Type ObjType>
struct FlatbufferTypeObjectView {
  FlatbufferTypeObjectView(const T *view) : view(view) {}

  static constexpr impl::Type type = ObjType;
  const T *view;
};

/// A wrapper around the generated `impl::ScalarTypeView`.  It does not own any
/// memory; it only provides a read-only view into the buffer.
class ScalarTypeView : public FlatbufferTypeObjectView<impl::ScalarType,
                                                       impl::Type::ScalarType> {
public:
  using FlatbufferTypeObjectView::FlatbufferTypeObjectView;
  operator impl::ScalarTypeCode() const { return view->type(); }
};

/// A constant representing a dynamic size. This mirrors the same value as MLIR
/// 'ShapedType::kDynamic'.
static constexpr int64_t kDynamicSize = std::numeric_limits<int64_t>::min();

/// A wrapper around `impl::MemRefTypeT` to provide additional convenience
/// utilities.  It does not own any memory; it only
// provides a read-only view into the buffer.
class MemRefTypeView : public FlatbufferTypeObjectView<impl::MemRefType,
                                                       impl::Type::MemRefType> {
public:
  MemRefTypeView(const impl::MemRefType *view)
      : FlatbufferTypeObjectView(view) {}

  int64_t getRank() const { return view->shape()->size(); }

  /// Return the scalar type code of the memref.
  ScalarType getElementType() const { return view->element_type(); }

  llvm::ArrayRef<int64_t> getShape() const {
    return llvm::ArrayRef<int64_t>(view->shape()->data(),
                                   view->shape()->size());
  }
  llvm::ArrayRef<int64_t> getStrides() const {
    return llvm::ArrayRef<int64_t>(view->strides()->data(),
                                   view->strides()->size());
  }
  PointerType getAddressSpace() const {
    return PointerType(view->address_space());
  }
};

/// A wrapper equivalent to the flatbuffer-generated TypeUnion object. The
/// `view` object may be a `impl::MemRef|impl::ScalarType` and
/// `type` is the tag indicating the pointer type.
struct TypeUnionView {
  impl::Type type;
  const void *view;

  template <typename T>
  bool isa() const {
    return type == T::type;
  }

  template <typename T>
  T get() const {
    assert(isa<T>() && "invalid type");
    return T(reinterpret_cast<decltype(T::view)>(view));
  }
};

/// Base class for all the below classes that provide flatbuffer-view wrappers
/// for flatbuffer tables that comprise the `Bounds` union in the schema.
template <typename T, impl::Bounds ObjType>
struct FlatbufferBoundsObjectView {
  FlatbufferBoundsObjectView(const T *view) : view(view) {}

  static constexpr impl::Bounds bound = ObjType;
  const T *view;
};

class DimensionBoundsView
    : public FlatbufferBoundsObjectView<impl::DimensionBounds,
                                        impl::Bounds::DimensionBounds> {
public:
  DimensionBoundsView(const impl::DimensionBounds *view)
      : FlatbufferBoundsObjectView(view) {}

  llvm::ArrayRef<int64_t> getMin() const {
    return llvm::ArrayRef<int64_t>(view->min()->data(), view->min()->size());
  }
  llvm::ArrayRef<int64_t> getMax() const {
    return llvm::ArrayRef<int64_t>(view->max()->data(), view->max()->size());
  }
};

class ValueBoundsView
    : public FlatbufferBoundsObjectView<impl::ValueBounds,
                                        impl::Bounds::ValueBounds> {
public:
  ValueBoundsView(const impl::ValueBounds *view)
      : FlatbufferBoundsObjectView(view) {}

  llvm::ArrayRef<int64_t> getMin() const {
    return llvm::ArrayRef<int64_t>(view->min()->data(), view->min()->size());
  }
  llvm::ArrayRef<int64_t> getMax() const {
    return llvm::ArrayRef<int64_t>(view->max()->data(), view->max()->size());
  }
};

/// A wrapper equivalent to the flatbuffer-generated BoundsUnion object. The
/// `view` object may be a `impl::DimensionBounds|impl::ValueBounds` and
/// `bound` is the tag indicating the bound type.
struct BoundsUnionView {
  impl::Bounds bound;
  const void *view;

  template <typename T>
  bool isa() const {
    return bound == T::bound;
  }

  template <typename T>
  T get() const {
    assert(isa<T>() && "invalid bound attr");
    return T(reinterpret_cast<decltype(T::view)>(view));
  }
};

//===----------------------------------------------------------------------===//
// FunctionSignatureView
//===----------------------------------------------------------------------===//
class FunctionSignatureView;

/// Print a text summary of the signature to the stream.
llvm::raw_ostream &print(llvm::raw_ostream &os,
                         const FunctionSignatureView &sig);

/// A wrapper around the flatbuffer-generated FunctionSignature to provide
/// additional convenience utilities. It does not own any memory; it only
/// provides a read-only view into the buffer.
class FunctionSignatureView {
public:
  FunctionSignatureView(const impl::FunctionSignature *view) : view(view) {
    assert(view != nullptr && "expected valid view");
  }

  uint32_t getNumArgs() const {
    return view->args() ? view->args()->size() : 0;
  }
  uint32_t getNumResults() const {
    return view->results() ? view->results()->size() : 0;
  }
  uint32_t getNumInputArgs() const {
    assert(getNumArgs() >= getNumOutputArgs() &&
           "invalid number of output arguments specified");
    return getNumArgs() - getNumOutputArgs();
  }
  uint32_t getNumOutputArgs() const { return view->num_output_args(); }
  uint32_t getNumArgBounds() const {
    return view->arg_bounds() ? view->arg_bounds()->size() : 0;
  }
  uint32_t getNumResBounds() const {
    return view->result_bounds() ? view->result_bounds()->size() : 0;
  }

  TypeUnionView getArg(int64_t idx) const {
    assert(idx < getNumArgs() && "expected valid argument index");
    return TypeUnionView{view->args_type()->Get(idx), view->args()->Get(idx)};
  }

  TypeUnionView getResult(int64_t idx) const {
    assert(idx < getNumResults() && "expected valid result index");
    return TypeUnionView{view->results_type()->Get(idx),
                         view->results()->Get(idx)};
  }

  BoundsUnionView getArgBound(int64_t idx) const {
    assert(idx < getNumArgBounds() && "expected valid argument bounds index");
    return BoundsUnionView{view->arg_bounds_type()->Get(idx),
                           view->arg_bounds()->Get(idx)};
  }

  BoundsUnionView getResultBound(int64_t idx) const {
    assert(idx < getNumResBounds() && "expected valid result bounds index");
    return BoundsUnionView{view->result_bounds_type()->Get(idx),
                           view->result_bounds()->Get(idx)};
  }

  TypeUnionView getOutputArg(int64_t idx) const {
    assert(idx < getNumOutputArgs() && "expected valid output argument index");
    unsigned offset = getNumInputArgs() + idx;
    return TypeUnionView{view->args_type()->Get(offset),
                         view->args()->Get(offset)};
  }
  int64_t isOutputArg(int64_t argIdx) const {
    assert(argIdx < getNumArgs() && "expected valid argument index");
    return argIdx < (getNumArgs() - getNumOutputArgs());
  }

  llvm::SmallVector<TypeUnionView> getArgs() const {
    llvm::SmallVector<TypeUnionView> args;
    unsigned numArgs = getNumArgs();
    args.reserve(numArgs);
    for (unsigned i = 0; i < numArgs; i++)
      args.push_back(getArg(i));
    return args;
  }

  llvm::SmallVector<TypeUnionView> getResults() const {
    llvm::SmallVector<TypeUnionView> args;
    unsigned numArgs = getNumResults();
    args.reserve(numArgs);
    for (unsigned i = 0; i < numArgs; i++)
      args.push_back(getResult(i));
    return args;
  }

  llvm::SmallVector<BoundsUnionView> getArgBounds() const {
    llvm::SmallVector<BoundsUnionView> args;
    unsigned numArgs = getNumArgBounds();
    args.reserve(numArgs);
    for (unsigned i = 0; i < numArgs; i++)
      args.push_back(getArgBound(i));
    return args;
  }

  llvm::SmallVector<BoundsUnionView> getResultBounds() const {
    llvm::SmallVector<BoundsUnionView> args;
    unsigned numArgs = getNumResBounds();
    args.reserve(numArgs);
    for (unsigned i = 0; i < numArgs; i++)
      args.push_back(getResultBound(i));
    return args;
  }

  std::optional<std::string_view> getShapeFunctionName() const {
    const flatbuffers::String *name = view->shape_function_name();
    if (!name || name->size() == 0)
      return std::nullopt;
    return view->shape_function_name()->string_view();
  }

  /// Returns the calling convention associated with this function.
  CallingConvention getCConv() const { return view->calling_convention(); }

  const impl::FunctionSignature *view;
};

/// A FunctionView is a thin wrapper around a flatbuffer Function object. It
/// does not own any memory; it only provides a read-only view into the buffer.
class FunctionView {
public:
  FunctionView(const impl::Function *view) : view(view) {
    assert(view != nullptr);
  }
  FunctionView() : view(nullptr) {}

  FunctionSignatureView getSignature() const {
    return FunctionSignatureView(view->signature());
  }

  std::string_view getName() const { return view->name()->string_view(); }

  operator bool() const { return view != nullptr; }

  operator const impl::Function *() const { return view; }

private:
  const impl::Function *view;
};

/// A DataSegmentInfo is a thin wrapper around a flatbuffer DataSegment object.
/// It does not own any memory; it only provides a read-only view into the
/// buffer.
class DataSegmentInfo {
public:
  DataSegmentInfo(const impl::DataSegment *view) : view(view) {}

  std::string_view getName() const { return view->name()->string_view(); }

  const int8_t *data() const {
    return view->data() ? view->data()->data() : nullptr;
  }
  size_t size() const {
    return view->data() ? view->data()->size() : getUninitializedSize();
  }
  uint32_t getAlignment() const { return view->alignment(); }
  bool isConstant() const { return view->constant(); }
  bool isUninitialized() const { return view->uninitialized_size() > 0; }
  uint64_t getUninitializedSize() const { return view->uninitialized_size(); }
  PointerType getAddressSpace() const { return view->address_space(); }

private:
  const impl::DataSegment *view;
};

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

/// `ExecutableView` is simply a wrapper around the low-level Flatbuffer
/// API for accessing an Executable object serialized into a flatbuffer.
class ExecutableView {
public:
  ExecutableView(const impl::Executable *view) : view(view) {}

  std::string_view getCode() const { return view->source()->string_view(); }

  size_t getNumFunctions() const { return view->functions()->size(); }

  FunctionView getFunction(int64_t idx) const {
    return FunctionView(view->functions()->Get(idx));
  }

  /// Return a function by name. This asserts that the function with the given
  /// name exists.
  StatusOr<FunctionView> getFunction(std::string_view name) const;

  size_t getNumDataSegments() const {
    if (!view || !view->data_segments())
      return 0;
    return view->data_segments()->size();
  }

  DataSegmentInfo getDataSegments(int64_t idx) const {
    assert(view->data_segments() && "expected valid data segment pointer");
    return view->data_segments()->Get(idx);
  }

  std::string_view getName() const {
    if (!view->name())
      return "unnamed-executable";
    return view->name()->string_view();
  }

  llvm::ArrayRef<uint32_t> getProcessorGridShape() const {
    assert(view->process_grid_shape() && "expected valid process grid shape");
    return llvm::ArrayRef<uint32_t>(view->process_grid_shape()->data(),
                                    view->process_grid_shape()->size());
  }

  /// Return a vector of DataSegmentInfos.
  llvm::SmallVector<DataSegmentInfo> getDataSegments() const;

  /// Return a vector of FunctionViews.
  llvm::SmallVector<FunctionView> getFunctions() const;

  /// Allow contextual conversion to bool for checking validity.
  operator bool() const { return view != nullptr; }

protected:
  const impl::Executable *view;
};

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

/// Class that wraps executable. It owns storage of the flatbuffer. Access is
/// provided through flatbuffer view overlay.
class Executable : public ExecutableView {
public:
  virtual ~Executable();

  Executable(std::unique_ptr<ExecutableStorage> storage);
  Executable(Executable &&other);

  std::unique_ptr<Executable> getCopy() const;

  ExecutableView getView() const { return ExecutableView(this->view); }

  /// Read the binary data from the given file and return a deserialized
  /// Executable.
  static StatusOr<std::unique_ptr<Executable>>
  loadFromFile(std::string_view path);

  /// Load executable from file. The file may be mmap'd, decision is
  /// made by `llvm::MemoryBuffer`.
  static StatusOr<std::unique_ptr<Executable>>
  loadFromBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer);

  /// Load from a stroage view. This allocates data to hold the buffer
  /// and copies from `data`. This method should be used when it is
  /// when it is not known that `data` has the proper alignment, otherwise
  /// use create a llvm::MemoryBuffer view and use `loadFromBuffer`.
  static StatusOr<std::unique_ptr<Executable>>
  loadFromUnalignedRef(llvm::ArrayRef<char> data);

  /// Return the underlying storage object.
  const std::unique_ptr<ExecutableStorage> &getStorage() const {
    return storage;
  }

  /// Verify the underlying Flatbuffer object.
  Status verify() const;

private:
  std::unique_ptr<ExecutableStorage> storage;
};

///===---------------------------------------------------------------------===//
// Device
//===----------------------------------------------------------------------===//

class Device {
public:
  /// Creates a new device. Verifies that the CUDA device ordinal is valid for
  /// the current system.
  static StatusOr<std::unique_ptr<Device>> create(int32_t deviceNumber);

  int32_t getDeviceNumber() const { return deviceNumber; }

private:
  Device(int32_t deviceNumber) : deviceNumber(deviceNumber) {}

  int32_t deviceNumber;
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
  void cleanup();
  Storage data;
  ScalarType type;
};

//===----------------------------------------------------------------------===//
// MemRefValue
//===----------------------------------------------------------------------===//

class RuntimeClientAllocator;

class MemRefStorage : public RefCounted<MemRefStorage> {
public:
  virtual ~MemRefStorage() {}

  uintptr_t getPtr() const { return ptr; }

  virtual PointerType getMemorySpace() const = 0;
  virtual std::optional<CudaStream> getStream() const { return {}; }

protected:
  MemRefStorage(uintptr_t ptr) : ptr(ptr) {}

  uintptr_t ptr;
};

class MemRefValue : public RuntimeValue {
public:
  /// Create a new MemRef descriptor. All size quantities are in "units of
  /// elements" unless otherwise noted.
  static mlirtrt::StatusOr<std::unique_ptr<MemRefValue>>
  create(RuntimeClient *client, mlirtrt::runtime::PointerType addressSpace,
         int64_t bitsPerElement, Ref<MemRefStorage> storage, int64_t offset,
         llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
         std::optional<const Device *> device,
         std::optional<ScalarType> scalarType,
         std::optional<bool> assertCanonicalStrides = {});

  mlirtrt::runtime::PointerType getBufferKind() { return addressSpace; }
  int64_t getElementBitWidth() const { return bitsPerElement; }
  int64_t getOffset() const { return offsetInBytes; }
  llvm::ArrayRef<int64_t> getShape() const { return shape; }
  llvm::ArrayRef<int64_t> getStrides() const { return strides; }
  int64_t getRank() const { return shape.size(); }
  int64_t getTotalFootprintInBytes() const;
  uintptr_t getMemory() const { return storage->getPtr(); }
  void *getVoidPtr() const { return reinterpret_cast<void *>(getMemory()); }
  std::optional<const Device *> getDevice() const { return device; }
  mlirtrt::runtime::PointerInfo
  getPointerInfo(mlirtrt::runtime::PointerOwner ownership) const {
    return mlirtrt::runtime::PointerInfo(
        getMemory(), getTotalFootprintInBytes(), addressSpace, ownership);
  }
  PointerType getAddressSpace() const { return addressSpace; }

  static bool classof(const RuntimeValue *v) {
    return v->getKind() == Kind::MemRef;
  }

  const std::optional<ScalarType> &getScalarType() const { return scalarType; }

  RuntimeClient *getClient() const { return client; }

  /// Return the reference count of the underlying storage.
  unsigned getStorageRefCount() const { return storage->getRefCount(); }

  /// Return a new MemRefValue that references the same storage as this one.
  /// The reference count of the storage is incremented.
  std::unique_ptr<MemRefValue> createRef() const {
    return std::unique_ptr<MemRefValue>(
        new MemRefValue(client, addressSpace, bitsPerElement, storage,
                        offsetInBytes, shape, strides, device, scalarType));
  }

private:
  MemRefValue(RuntimeClient *client, mlirtrt::runtime::PointerType addressSpace,
              int64_t bitsPerElement, Ref<MemRefStorage> storage,
              int64_t offset, llvm::ArrayRef<int64_t> shape,
              llvm::ArrayRef<int64_t> strides,
              std::optional<const Device *> device,
              std::optional<ScalarType> scalarType);

  /// Non-owned reference back to RuntimeClient that tracks this MemRef.
  RuntimeClient *client;
  /// Address space for the pointer.
  mlirtrt::runtime::PointerType addressSpace;
  int64_t bitsPerElement;
  /// Holds the underlying storage object.
  Ref<MemRefStorage> storage;
  int64_t offsetInBytes;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> strides;
  /// Non-owned view to the associated device if the address space is a device
  /// address.
  std::optional<const Device *> device;
  std::optional<ScalarType> scalarType{};
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
  RuntimeSessionOptions(int32_t numDevices = 1, int32_t deviceId = 0,
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
  int32_t getNumDevices() const { return numDevices; }

  /// Return the rank of the worker in the process grid that the runtime session
  /// should correspond to.
  /// Currently this should correspond to CUDA deviceId, but in the future we
  /// will need to maintain a lookup table from rank to local device ID and
  /// vice-versa.
  int32_t getDeviceId() const { return deviceId; }

  /// Return the UUID that should be used to create the top-level NCCL
  /// communicator for each device. This should only be empty if there is only
  /// one device.a
  llvm::StringRef getNcclUuid() const { return ncclUuid; }

  /// Return the set of features that are enabled for this session.
  const llvm::StringSet<> &getEnabledFeatures() const { return features; }

private:
  int32_t numDevices;
  int32_t deviceId;
  std::string ncclUuid;

  /// A list of features names (e.g. module names) that should be enabled for
  /// this session.
  llvm::StringSet<> features;
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
  RuntimeSession(RuntimeSessionOptions options, ExecutableView executable);
  virtual ~RuntimeSession() {}

  ExecutableView getExecutable() const { return executable; }

  PinnedMemoryAllocator &getPinnedMemoryAllocator() {
    return *pinnedMemoryAllocator;
  }

  AllocTracker &getAllocTracker() { return *allocTracker; }

  ResourceTracker &getResourceTracker() { return *resourceTracker; }

  /// Returns the options used to construct the session.
  const RuntimeSessionOptions &getOptions() { return options; }

protected:
  RuntimeSessionOptions options;

  ExecutableView executable;

  std::unique_ptr<PinnedMemoryAllocator> pinnedMemoryAllocator;
  std::unique_ptr<AllocTracker> allocTracker;
  std::unique_ptr<ResourceTracker> resourceTracker;
};

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
           std::optional<uint32_t> alignemnt,
           std::optional<CudaStream> stream) = 0;

  /// Use the given pointer as the storage for the MemRefValue.
  virtual StatusOr<Ref<MemRefStorage>>
  takeOwnership(uintptr_t ptr, PointerType pointerType,
                std::optional<CudaStream> stream) = 0;

  virtual Status deallocate(MemRefStorage &storage) = 0;
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
class RuntimeClient {
public:
  ~RuntimeClient();

  /// Creates the client. Enumerates CUDA devices on the machine. Creates the
  /// internal allocators.
  static StatusOr<std::unique_ptr<RuntimeClient>> create();

  llvm::ArrayRef<std::unique_ptr<Device>> getDevices() const;

  StatusOr<std::unique_ptr<MemRefValue>>
  allocateMemRef(PointerType addressSpace, int64_t bitsPerElement,
                 llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
                 std::optional<const Device *> device = {},
                 std::optional<CudaStream> stream = {},
                 std::optional<ScalarType> scalarType = {},
                 std::optional<bool> assertCanonicalStrides = {});

  StatusOr<std::unique_ptr<MemRefValue>>
  createExternalMemRef(PointerType addressSpace, int64_t bitsPerElement,
                       uintptr_t ptr, int64_t offset,
                       llvm::ArrayRef<int64_t> shape,
                       llvm::ArrayRef<int64_t> strides,
                       std::optional<const Device *> device = {},
                       std::optional<ScalarType> scalarType = {},
                       std::optional<bool> assertCanonicalStrides = {},
                       std::function<void()> = nullptr);

  // Allocates a new host buffer and fills it with data present in the
  // `hostBuffer`.
  StatusOr<std::unique_ptr<MemRefValue>>
  copyHostToHost(const MemRefValue &hostBuffer);

  /// Allocates a new device buffer and fills it with data present on the
  /// host in the specified buffer. The allocation and copy are performed on
  /// the given stream.
  StatusOr<std::unique_ptr<MemRefValue>>
  copyToDevice(const MemRefValue &hostBuffer, const Device &device,
               std::optional<CudaStream> stream);

  /// Allocates a new host buffer and fills it with data present on the device
  /// in the specified buffer. The allocation and copy are performed on the
  /// given stream.
  StatusOr<std::unique_ptr<MemRefValue>>
  copyToHost(const MemRefValue &deviceMemRef, std::optional<CudaStream> stream);

  /// Copy from the given device MemRefValue to an existing MemRefValue on the
  /// host.
  Status copyToHost(const MemRefValue &deviceMemRef, MemRefValue &hostMemRef,
                    std::optional<CudaStream> stream);

  /// Returns the ResourceTracker.
  ResourceTracker &getResourceTracker() { return resourceTracker; }

  /// Return the PinnedMemoryAllocator.
  PinnedMemoryAllocator &getPinnedMemoryAllocator() {
    return pinnedMemoryAllocator;
  }

  RuntimeClientAllocator &getAllocator() { return *allocator; }

private:
  RuntimeClient(llvm::SmallVector<std::unique_ptr<Device>> devices,
                std::unique_ptr<RuntimeClientAllocator> allocator)
      : devices(std::move(devices)), allocator(std::move(allocator)) {}

  llvm::SmallVector<std::unique_ptr<Device>> devices;
  PinnedMemoryAllocator pinnedMemoryAllocator;
  ResourceTracker resourceTracker;
  std::unique_ptr<RuntimeClientAllocator> allocator;
};

//===----------------------------------------------------------------------===//
// NCCL Support functions
//===----------------------------------------------------------------------===//

/// Return the NCCL unique communicator ID as a string if the project was
/// configured with NCCL enabled. If the project was not configured with NCCL
/// enabled, then returns an empty string.
StatusOr<std::string> getCommunicatorUniqueId();

//===----------------------------------------------------------------------===//
// Debug Print Utilities
//===----------------------------------------------------------------------===//

/// Print a text summary of the executable to the stream.
llvm::raw_ostream &print(llvm::raw_ostream &os, const Executable &exe);
/// Print a text summary of the constant to the stream.
llvm::raw_ostream &print(llvm::raw_ostream &os,
                         const DataSegmentInfo &constant);
/// Print a text summary of the type to the stream.
llvm::raw_ostream &print(llvm::raw_ostream &os, const MemRefTypeView &type);
/// Print a text summary of the type to the stream.
llvm::raw_ostream &print(llvm::raw_ostream &os, const ScalarTypeView &type);
/// Print a text summary of the signature to the stream.
llvm::raw_ostream &print(llvm::raw_ostream &os,
                         const FunctionSignatureView &sig);
/// Print a text summary of the argument to the stream.
llvm::raw_ostream &print(llvm::raw_ostream &os, const TypeUnionView &arg);
/// Print a text summary of the function to the stream.
llvm::raw_ostream &print(llvm::raw_ostream &os, const FunctionView &func);
/// Print a text summary of the bounds to the stream.
llvm::raw_ostream &print(llvm::raw_ostream &os, const BoundsUnionView &bounds);
/// Print a text summary of the dim bounds to the stream.
llvm::raw_ostream &print(llvm::raw_ostream &os, const DimensionBoundsView &dim);
/// Print a text summary of the value bounds to the stream.
llvm::raw_ostream &print(llvm::raw_ostream &os, const ValueBoundsView &val);

/// Print a text summary of the unique ptr's underlying object to the stream.
template <typename T>
inline llvm::raw_ostream &print(llvm::raw_ostream &os,
                                const std::unique_ptr<T> &obj) {
  return print(os, *obj);
}

} // namespace mlirtrt::runtime

#endif // MLIR_EXECUTOR_RUNTIME_API_API
