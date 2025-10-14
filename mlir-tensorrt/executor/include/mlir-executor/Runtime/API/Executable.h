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
  MemRefTypeView(const mtrt::flat::MemRefType *view)
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
  DimensionBoundsView(const mtrt::flat::DimensionBounds *view)
      : FlatbufferBoundsObjectView(view) {}

  llvm::ArrayRef<int64_t> getMin() const {
    return llvm::ArrayRef<int64_t>(view->min()->data(), view->min()->size());
  }
  llvm::ArrayRef<int64_t> getMax() const {
    return llvm::ArrayRef<int64_t>(view->max()->data(), view->max()->size());
  }
};

class ValueBoundsView
    : public FlatbufferBoundsObjectView<mtrt::flat::ValueBounds,
                                        mtrt::flat::Bounds::ValueBounds> {
public:
  ValueBoundsView(const mtrt::flat::ValueBounds *view)
      : FlatbufferBoundsObjectView(view) {}

  llvm::ArrayRef<int64_t> getMin() const {
    return llvm::ArrayRef<int64_t>(view->min()->data(), view->min()->size());
  }
  llvm::ArrayRef<int64_t> getMax() const {
    return llvm::ArrayRef<int64_t>(view->max()->data(), view->max()->size());
  }
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
  FunctionSignatureView(const mtrt::flat::FunctionSignature *view)
      : view(view) {
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
    assert(idx < getNumArgs() && "expected valid argument index");
    int32_t boundsIdx = view->arg_bounds_indices()->Get(idx);
    if (boundsIdx < 0)
      return BoundsUnionView{mtrt::flat::Bounds::NONE, nullptr};
    return BoundsUnionView{view->bounds_values_type()->Get(boundsIdx),
                           view->bounds_values()->Get(boundsIdx)};
  }

  BoundsUnionView getResultBound(int64_t idx) const {
    assert(idx < getNumResults() && "expected valid result index");
    int32_t boundsIdx = view->result_bounds_indices()->Get(idx);
    if (boundsIdx < 0)
      return BoundsUnionView{mtrt::flat::Bounds::NONE, nullptr};
    return BoundsUnionView{view->bounds_values_type()->Get(boundsIdx),
                           view->bounds_values()->Get(boundsIdx)};
  }

  TypeUnionView getOutputArg(int64_t idx) const {
    assert(idx < getNumOutputArgs() && "expected valid output argument index");
    unsigned offset = getNumInputArgs() + idx;
    return TypeUnionView{view->args_type()->Get(offset),
                         view->args()->Get(offset)};
  }
  int64_t isOutputArg(int64_t argIdx) const {
    assert(argIdx < getNumArgs() && "expected valid argument index");
    return argIdx >= (getNumArgs() - getNumOutputArgs());
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
    unsigned numArgs = getNumArgs();
    args.reserve(numArgs);
    for (unsigned i = 0; i < numArgs; i++)
      args.push_back(getArgBound(i));
    return args;
  }

  llvm::SmallVector<BoundsUnionView> getResultBounds() const {
    llvm::SmallVector<BoundsUnionView> args;
    unsigned numArgs = getNumResults();
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

  const mtrt::flat::FunctionSignature *view;
};

/// A FunctionView is a thin wrapper around a flatbuffer Function object. It
/// does not own any memory; it only provides a read-only view into the buffer.
class FunctionView {
public:
  FunctionView(const mtrt::flat::Function *view) : view(view) {
    assert(view != nullptr);
  }
  FunctionView() : view(nullptr) {}

  FunctionSignatureView getSignature() const {
    return FunctionSignatureView(view->signature());
  }

  std::string_view getName() const { return view->name()->string_view(); }

  operator bool() const { return view != nullptr; }

  operator const mtrt::flat::Function *() const { return view; }

private:
  const mtrt::flat::Function *view;
};

/// A DataSegmentInfo is a thin wrapper around a flatbuffer DataSegment object.
/// It does not own any memory; it only provides a read-only view into the
/// buffer.
class DataSegmentInfo {
public:
  DataSegmentInfo(const mtrt::flat::DataSegment *view) : view(view) {}

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
  const mtrt::flat::DataSegment *view;
};

//===----------------------------------------------------------------------===//
// ExecutableView
//===----------------------------------------------------------------------===//

/// `ExecutableView` is simply a wrapper around the low-level Flatbuffer
/// API for accessing an Executable object serialized into a flatbuffer.
class ExecutableView {
public:
  ExecutableView(const mtrt::flat::Executable *view) : view(view) {}

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
