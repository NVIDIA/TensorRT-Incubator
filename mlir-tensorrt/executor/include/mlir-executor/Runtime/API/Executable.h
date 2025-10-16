//===- Executable.h --------------------------------------------*- C++ -*-===//
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
/// Declares the generated Flatbuffer serialization/deserialization and
/// other helper routines related to the RuntimeExecutable format.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_RUNTIME_API_EXECUTABLE
#define MLIR_EXECUTOR_RUNTIME_API_EXECUTABLE

#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/MemoryBuffer.h"

#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#endif
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace mtrt {
// Alias some objects from the generated Flatbuffer object API class instead of
// using them directly.
using ScalarTypeCode = mtrt::flat::ScalarTypeCode;
using PointerType = mtrt::flat::PointerType;
using PointerOwner = mtrt::flat::PointerOwner;
using TypeCode = mtrt::flat::Type;
using CallingConvention = mtrt::flat::CallingConvention;

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
    return mtrt::flat::EnumNameScalarTypeCode(code);
  }

  /// Get the equivalent ScalarTypeCode integer value that has the given number
  /// of bits.
  static StatusOr<ScalarTypeCode> getIntegerTypeWithBitWidth(int64_t bitWidth);

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
PointerType parsePointerType(llvm::StringRef str);

/// Print pointer type enum name to string.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, PointerType ptrType);

/// Return the string representation of `ptrType`.
llvm::StringRef stringifyPointerType(PointerType ptrType);

/// Return true if the PointerType is a host-visible address space.
bool isHostVisible(PointerType pointerType);

/// Return true if the PointerType is a device-visible address space.
bool isDeviceVisible(PointerType pointerType);

//===----------------------------------------------------------------------===//
// TypeView
// This section includes classes that form the TypeUnion:
// MemRefTypeView, ScalarTypeView
//===----------------------------------------------------------------------===//

/// Base class for all the below classes that provide flatbuffer-view wrappers
/// for flatbuffer tables that comprise the `Type` union in the schema.
template <typename T, mtrt::flat::Type ObjType>
struct FlatbufferTypeObjectView {
  FlatbufferTypeObjectView(const T *view) : view(view) {}

  static constexpr mtrt::flat::Type type = ObjType;
  const T *view;
};

/// A wrapper around the generated `mtrt::flat::ScalarTypeView`.  It does not
/// own any memory; it only provides a read-only view into the buffer.
class ScalarTypeView
    : public FlatbufferTypeObjectView<mtrt::flat::ScalarType,
                                      mtrt::flat::Type::ScalarType> {
public:
  using FlatbufferTypeObjectView::FlatbufferTypeObjectView;
  operator mtrt::ScalarTypeCode() const { return view->type(); }
};

/// A constant representing a dynamic size. This mirrors the same value as MLIR
/// 'ShapedType::kDynamic'.
static constexpr int64_t kDynamicSize = std::numeric_limits<int64_t>::min();

/// A wrapper around `mtrt::flat::MemRefType` to provide additional
/// convenience utilities.  It does not own any memory; it only
// provides a read-only view into the buffer.
class MemRefTypeView
    : public FlatbufferTypeObjectView<mtrt::flat::MemRefType,
                                      mtrt::flat::Type::MemRefType> {
public:
  MemRefTypeView(const mtrt::flat::MemRefType *view);

  int64_t getRank() const;

  /// Return the scalar type code of the memref.
  ScalarType getElementType() const;

  llvm::ArrayRef<int64_t> getShape() const;

  llvm::ArrayRef<int64_t> getStrides() const;

  PointerType getAddressSpace() const;
};

/// A wrapper equivalent to the flatbuffer-generated TypeUnion object. The
/// `view` object may be a `mtrt::flat::MemRef|mtrt::flat::ScalarType` and
/// `type` is the tag indicating the pointer type.
struct TypeUnionView {
  mtrt::flat::Type type;
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
template <typename T, mtrt::flat::Bounds ObjType>
struct FlatbufferBoundsObjectView {
  FlatbufferBoundsObjectView(const T *view) : view(view) {}

  static constexpr mtrt::flat::Bounds bound = ObjType;
  const T *view;
};

class DimensionBoundsView
    : public FlatbufferBoundsObjectView<mtrt::flat::DimensionBounds,
                                        mtrt::flat::Bounds::DimensionBounds> {
public:
  DimensionBoundsView(const mtrt::flat::DimensionBounds *view);

  llvm::ArrayRef<int64_t> getMin() const;

  llvm::ArrayRef<int64_t> getMax() const;
};

class ValueBoundsView
    : public FlatbufferBoundsObjectView<mtrt::flat::ValueBounds,
                                        mtrt::flat::Bounds::ValueBounds> {
public:
  ValueBoundsView(const mtrt::flat::ValueBounds *view);

  llvm::ArrayRef<int64_t> getMin() const;

  llvm::ArrayRef<int64_t> getMax() const;
};

/// A wrapper equivalent to the flatbuffer-generated BoundsUnion object. The
/// `view` object may be a
/// `mtrt::flat::DimensionBounds|mtrt::flat::ValueBounds` and `bound` is the
/// tag indicating the bound type.
struct BoundsUnionView {
  mtrt::flat::Bounds bound;
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
  /// Construct a FunctionSignatureView from a flatbuffer FunctionSignature
  /// pointer.
  FunctionSignatureView(const mtrt::flat::FunctionSignature *view);

  /// Return the total number of arguments (input args + output args).
  uint32_t getNumArgs() const;

  /// Return the number of result values.
  uint32_t getNumResults() const;

  /// Return the number of input arguments (excludes output arguments).
  uint32_t getNumInputArgs() const;

  /// Return the number of output arguments.
  uint32_t getNumOutputArgs() const;

  /// Return the type of the argument at index \p idx.
  TypeUnionView getArg(int64_t idx) const;

  /// Return the type of the result at index \p idx.
  TypeUnionView getResult(int64_t idx) const;

  /// Return the bounds for the argument at index \p idx, if any.
  BoundsUnionView getArgBound(int64_t idx) const;

  /// Return the bounds for the result at index \p idx, if any.
  BoundsUnionView getResultBound(int64_t idx) const;

  /// Return the type of the output argument at index \p idx.
  TypeUnionView getOutputArg(int64_t idx) const;

  /// Return a vector containing all argument types.
  llvm::SmallVector<TypeUnionView> getArgs() const;

  /// Return a vector containing all result types.
  llvm::SmallVector<TypeUnionView> getResults() const;

  /// Return a vector containing bounds for all arguments.
  llvm::SmallVector<BoundsUnionView> getArgBounds() const;

  /// Return a vector containing bounds for all results.
  llvm::SmallVector<BoundsUnionView> getResultBounds() const;

  /// Return the name of the associated shape function, if any.
  std::optional<std::string_view> getShapeFunctionName() const;

  /// Returns the calling convention associated with this function.
  CallingConvention getCConv() const;

  /// Returns an array of indicators, one for each result, indicating whether
  /// the result may be undefined. Only valid for ABI version >= 1.
  llvm::ArrayRef<uint8_t> getUndef() const;

  /// Return the ABI version associated with this function.
  uint32_t getAbiVersion() const;

  const mtrt::flat::FunctionSignature *view;
};

/// A FunctionView is a thin wrapper around a flatbuffer Function object. It
/// does not own any memory; it only provides a read-only view into the buffer.
class FunctionView {
public:
  /// Construct a FunctionView from a flatbuffer Function pointer.
  FunctionView(const mtrt::flat::Function *view);

  /// Construct an empty FunctionView with null view pointer.
  FunctionView();

  /// Return the signature of this function.
  FunctionSignatureView getSignature() const;

  /// Return the name of this function.
  std::string_view getName() const;

  /// Allow contextual conversion to bool for checking validity.
  operator bool() const;

  /// Allow implicit conversion to the underlying flatbuffer Function pointer.
  operator const mtrt::flat::Function *() const;

  /// Return the ABI version associated with this function.
  uint32_t getAbiVersion() const;

private:
  const mtrt::flat::Function *view;
};

/// A DataSegmentInfo is a thin wrapper around a flatbuffer DataSegment object.
/// It does not own any memory; it only provides a read-only view into the
/// buffer.
class DataSegmentInfo {
public:
  DataSegmentInfo(const mtrt::flat::DataSegment *view);

  std::string_view getName() const;

  const int8_t *data() const;

  size_t size() const;

  uint32_t getAlignment() const;

  bool isConstant() const;

  bool isUninitialized() const;

  uint64_t getUninitializedSize() const;

  PointerType getAddressSpace() const;

private:
  const mtrt::flat::DataSegment *view;
};

//===----------------------------------------------------------------------===//
// ExecutableView
//===----------------------------------------------------------------------===//

/// `ExecutableView` is simply a wrapper around the low-level Flatbuffer
/// API for accessing an Executable object serialized into a flatbuffer.
class ExecutableView {
public:
  ExecutableView(const mtrt::flat::Executable *view);

  std::string_view getCode() const;

  size_t getNumFunctions() const;

  FunctionView getFunction(int64_t idx) const;

  /// Return a function by name. This asserts that the function with the given
  /// name exists.
  StatusOr<FunctionView> getFunction(std::string_view name) const;

  size_t getNumDataSegments() const;

  DataSegmentInfo getDataSegments(int64_t idx) const;

  std::string_view getName() const;

  llvm::ArrayRef<uint32_t> getProcessorGridShape() const;

  /// Return a vector of DataSegmentInfos.
  llvm::SmallVector<DataSegmentInfo> getDataSegments() const;

  /// Return a vector of FunctionViews.
  llvm::SmallVector<FunctionView> getFunctions() const;

  /// Return the ABI version for this executable.
  uint32_t getAbiVersion() const;

  /// Allow contextual conversion to bool for checking validity.
  operator bool() const;

protected:
  const mtrt::flat::Executable *view;
};

//===----------------------------------------------------------------------===//
// ExecutableStorage
//===----------------------------------------------------------------------===//

/// A ExecutableStorage manages storage for the executable. Different concrete
/// implementations may choose to manage the storage using e.g.
/// `llvm::MemoryBuffer` or via a just-encoded flatbuffer-allocated buffer.
class ExecutableStorage {
public:
  ExecutableStorage() = default;
  virtual ~ExecutableStorage() {}
  ExecutableStorage(const ExecutableStorage &) = delete;
  ExecutableStorage &operator=(const ExecutableStorage &) = delete;

  virtual std::unique_ptr<ExecutableStorage> getCopy() const = 0;

  virtual const void *data() const = 0;
  virtual size_t size() const = 0;
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

  /// Load from a storage view. This allocates data to hold the buffer
  /// and copies from `data`. This method should be used when it is
  /// not known that `data` has the proper alignment, otherwise
  /// use `loadFromBuffer` to create a `llvm::MemoryBuffer` view.
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

/// Print the description for unique_ptr<T> if T has a print function.
template <typename T>
inline llvm::raw_ostream &print(llvm::raw_ostream &os,
                                const std::unique_ptr<T> &obj) {
  return print(os, *obj);
}

struct format_shape : public llvm::FormatAdapter<llvm::ArrayRef<int64_t>> {
  format_shape(llvm::ArrayRef<int64_t> &&N)
      : llvm::FormatAdapter<llvm::ArrayRef<int64_t>>(std::move(N)) {}

  void format(llvm::raw_ostream &os, llvm::StringRef style) override {
    llvm::interleave(
        this->Item, os,
        [&](int64_t x) {
          if (x == kDynamicSize)
            os << "?";
          else
            os << x;
        },
        style);
  }
};

} // namespace mtrt

namespace llvm {
template <typename T>
struct format_provider<
    T, typename std::enable_if_t<
           std::is_same_v<decltype(mtrt::print(std::declval<raw_ostream &>(),
                                               std::declval<const T &>())),
                          raw_ostream &>,
           void>> {
  static void format(const T &V, raw_ostream &Stream, StringRef Style) {
    mtrt::print(Stream, V);
  }
};
} // namespace llvm

#endif // MLIR_EXECUTOR_RUNTIME_API_EXECUTABLE
