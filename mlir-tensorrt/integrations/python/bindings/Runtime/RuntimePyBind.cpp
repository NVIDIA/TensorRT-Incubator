//===- Client.cpp  --------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of Python bindings for the mlir-tensorrt-runtime library.
///
//===----------------------------------------------------------------------===//
#include "../Utils.h"
#include "dlpack/dlpack.h"
#include "mlir-executor-c/Common/Common.h"
#include "mlir-executor-c/Runtime/Runtime.h"
#include "mlir-tensorrt-common-c/Support/Status.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include <memory>
#include <stdexcept>
#include <string_view>

namespace py = pybind11;
using namespace mtrt;

//===----------------------------------------------------------------------===//
// MTRT_* <-> PyCapsule utilities.
// These are only needed in the case where we want to use implicitly cast the
// PyBind11 object to the original C API type. This  is required to use
// `std::optional<...>` of the original C API type as an argument type in the
// functions bound to Python through Pybind11 below.
//===----------------------------------------------------------------------===//

MTRT_DEFINE_RUNTIME_INLINE_PY_CAPSULE_CASTER_FUNCS(Device)
MTRT_DEFINE_RUNTIME_INLINE_PY_CAPSULE_CASTER_FUNCS(Stream)

namespace pybind11::detail {
MTRT_DEFINE_PYBIND_CASTER(Device, MTRT_Device);
MTRT_DEFINE_PYBIND_CASTER(Stream, MTRT_Stream);
} // namespace pybind11::detail

namespace {

class PyRuntimeClient;

/// Encapsulates setting llvm::debugflag and current debug types from MTRT
/// runtime
struct PyGlobalDebugFlag {
  static void set(py::object &, bool enable) {
    MTRT_Status s = mtrtEnableGlobalDebug(enable);
    THROW_IF_MTRT_ERROR(s);
  }

  static bool get(const py::object &) {
    bool enabled;
    MTRT_Status s = mtrtIsGlobalDebugEnabled(&enabled);
    THROW_IF_MTRT_ERROR(s);
    return enabled;
  }

  static void set_types(const std::string &type) {
    MTRT_Status s = mtrtSetGlobalDebugType(type.c_str());
    THROW_IF_MTRT_ERROR(s);
  }

  static void set_types(const std::vector<std::string> &types) {
    std::vector<const char *> pointers;
    pointers.reserve(types.size());
    for (const std::string &str : types)
      pointers.push_back(str.c_str());
    MTRT_Status s = mtrtSetGlobalDebugTypes(pointers.data(), pointers.size());
    THROW_IF_MTRT_ERROR(s);
  }
};

/// Python wrapper around MTRT_Event
class PyStream : public PyMTRTWrapper<PyStream, MTRT_Stream> {
public:
  using Base::Base;
  DECLARE_WRAPPER_CONSTRUCTORS(PyStream);
  static constexpr auto kMethodTable = CAPITable<MTRT_Stream>{
      mtrtStreamIsNull, mtrtStreamDestroy, mtrtPythonCapsuleToStream,
      mtrtPythonStreamToCapsule};
};

/// Python wrapper around MTRT_Device.
class PyDevice : public PyMTRTWrapper<PyDevice, MTRT_Device> {
public:
  using Base::Base;
  DECLARE_WRAPPER_CONSTRUCTORS(PyDevice);
  static constexpr auto kMethodTable = CAPITable<MTRT_Device>{
      mtrtDeviceIsNull,
      +[](MTRT_Device device) {
        (void)device;
        return mtrtStatusGetOk();
      },
      mtrtPythonCapsuleToDevice, mtrtPythonDeviceToCapsule};
};

/// Python object of wrapper for `MTRT_RuntimeValue`.
class PyRuntimeValue {
public:
  PyRuntimeValue(MTRT_RuntimeValue value) : value(value) {}

  PyRuntimeValue(const PyRuntimeValue &other) = delete;

  virtual ~PyRuntimeValue() {
    if (!mtrtRuntimeValueIsNull(value))
      mtrtRuntimeValueDestroy(value);
  }

  MTRT_RuntimeValue getRuntimeValue() { return value; }

protected:
  MTRT_RuntimeValue value;
};

/// Python wrapper around MTRT_ScalarValue.
class PyScalarValue : public PyRuntimeValue {
public:
  PyScalarValue(MTRT_RuntimeValue value) : PyRuntimeValue(value) {
    if (!mtrtRuntimeValueIsScalar(value))
      throw std::invalid_argument("value is not a scalar value");
  }

  PyScalarValue(const PyScalarValue &other) = delete;

  PyScalarValue(MTRT_ScalarValue value)
      : PyRuntimeValue(mtrtScalarValueCastToRuntimeValue(value)) {}

  operator MTRT_ScalarValue() { return mtrtRuntimeValueDynCastToScalar(value); }
};

/// Python wrapper around MTRT_MemRefValue.
class PyMemRefValue : public PyRuntimeValue {
public:
  PyMemRefValue(MTRT_RuntimeValue value) : PyRuntimeValue(value) {
    if (!mtrtRuntimeValueIsMemRef(value))
      throw std::invalid_argument("value is not a memref value");
  }
  PyMemRefValue(MTRT_MemRefValue value)
      : PyRuntimeValue(mtrtMemRefCastToRuntimeValue(value)) {}

  PyMemRefValue(const PyMemRefValue &other) = delete;
  PyMemRefValue &operator=(const PyMemRefValue &other) = delete;

  operator MTRT_MemRefValue() { return mtrtRuntimeValueDynCastToMemRef(value); }

  py::object storage;
};

/// Python object type wrapper for `MTRT_StableHLOToExecutableOptions`.
class PyRuntimeSessionOptions
    : public PyMTRTWrapper<PyRuntimeSessionOptions,
                           MTRT_RuntimeSessionOptions> {
public:
  using Base::Base;
  DECLARE_WRAPPER_CONSTRUCTORS(PyRuntimeSessionOptions);

  static constexpr auto kMethodTable = CAPITable<MTRT_RuntimeSessionOptions>{
      mtrtRuntimeSessionOptionsIsNull, mtrtRuntimeSessionOptionsDestroy};
};

/// Python wrapper around MTRT_RuntimeSession.
class PyRuntimeSession
    : public PyMTRTWrapper<PyRuntimeSession, MTRT_RuntimeSession>,
      public std::enable_shared_from_this<PyRuntimeSession> {
public:
  using Base::Base;
  DECLARE_WRAPPER_CONSTRUCTORS(PyRuntimeSession)

  static constexpr auto kMethodTable = mtrt::CAPITable<MTRT_RuntimeSession>{
      mtrtRuntimeSessionIsNull, mtrtRuntimeSessionDestroy};
};

/// Python wrapper around MTRT_RuntimeClient.
class PyRuntimeClient {
public:
  PyRuntimeClient(MTRT_RuntimeClient client) : client(client) {}
  ~PyRuntimeClient() {
    sessions.clear();
    mtrtRuntimeClientDestroy(client);
  }

  operator MTRT_RuntimeClient() { return client; }

  std::unordered_set<std::shared_ptr<PyRuntimeSession>> &getSessionsSet() {
    return sessions;
  }

private:
  MTRT_RuntimeClient client;
  std::unordered_set<std::shared_ptr<PyRuntimeSession>> sessions;
};

} // namespace

//===----------------------------------------------------------------------===//
// Utilities for buffer protocol interop.
// The functions in this section were based on the utility functions in
// `third_party/llvm-project/mlir/lib/Bindings/Python/IRAttributes.cpp`, which
// has the Apache License v2.0 with LLVM Exceptions. (See
// https://llvm.org/LICENSE.txt), They are modified here to suite our purposes.
//===----------------------------------------------------------------------===//

static bool isUnsignedIntegerFormat(std::string_view format) {
  if (format.empty())
    return false;
  char code = format[0];
  return code == 'I' || code == 'B' || code == 'H' || code == 'L' ||
         code == 'Q';
}

static bool isSignedIntegerFormat(std::string_view format) {
  if (format.empty())
    return false;
  char code = format[0];
  return code == 'i' || code == 'b' || code == 'h' || code == 'l' ||
         code == 'q';
}

static MTRT_ScalarTypeCode
getScalarTypeCodeFromPyBufferProtocolFormat(llvm::StringRef format,
                                            const Py_buffer &view) {
  MTRT_ScalarTypeCode elementType =
      MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_unknown;
  if (format == "f") {
    // f32
    assert(view.itemsize == 4 && "mismatched array itemsize");
    elementType = MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_f32;
  } else if (format == "d") {
    // f64
    assert(view.itemsize == 8 && "mismatched array itemsize");
    elementType = MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_f64;
  } else if (format == "e") {
    // f16
    assert(view.itemsize == 2 && "mismatched array itemsize");
    elementType = MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_f16;
  } else if (format == "?") {
    // bool / i1
    assert(view.itemsize == 1 && "mismatched array itemsize");
    elementType = MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_i1;
  } else if (isSignedIntegerFormat(format) || isUnsignedIntegerFormat(format)) {
    if (view.itemsize == 4) {
      // i32
      elementType = MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_i32;
    } else if (view.itemsize == 8) {
      // i64
      elementType = MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_i64;
    } else if (view.itemsize == 1) {
      // i8
      elementType = MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_i8;
    } else if (view.itemsize == 2) {
      // i16
      elementType = MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_i16;
    }
  } else if (format == "Zf") {
    elementType = MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_complex32;
  } else if (format == "Zd") {
    elementType = MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_complex64;
  }
  if (elementType == MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_unknown)
    throw std::invalid_argument(
        std::string("unimplemented array format conversion from format: ") +
        std::string(format));

  return elementType;
}

static std::unique_ptr<PyMemRefValue> createMemRef(
    PyRuntimeClient &client, std::vector<int64_t> shape,
    MTRT_ScalarTypeCode dtype, std::optional<MTRT_Device> device,
    std::optional<MTRT_Stream> stream,
    MTRT_PointerType addressSpace = MTRT_PointerType::MTRT_PointerType_device) {
  llvm::SmallVector<int64_t> strides;
  strides.resize(shape.size(), 1);
  for (int64_t idx = shape.size() - 2; idx >= 0; idx--)
    strides[idx] = strides[idx + 1] * shape[idx + 1];

  MTRT_MemRefValue result{nullptr};
  MTRT_Status s =
      mtrtMemRefCreate(client, addressSpace, dtype, shape.size(), shape.data(),
                       strides.data(), device ? *device : mtrtDeviceGetNull(),
                       stream ? *stream : mtrtStreamGetNull(), &result);
  THROW_IF_MTRT_ERROR(s);
  return std::make_unique<PyMemRefValue>(result);
}

static std::unique_ptr<PyMemRefValue> getMemRefFromHostBufferProtocol(
    PyRuntimeClient &client, py::buffer array,
    std::optional<std::vector<int64_t>> explicitShape,
    std::optional<MTRT_ScalarTypeCode> dtype, std::optional<MTRT_Device> device,
    std::optional<MTRT_Stream> stream,
    MTRT_PointerType addressSpace = MTRT_PointerType::MTRT_PointerType_device) {
  // Request a contiguous view. In exotic cases, this will cause a copy.
  int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;

  Py_buffer view;
  if (PyObject_GetBuffer(array.ptr(), &view, flags) != 0) {
    throw py::error_already_set();
  }
  auto freeBuffer = llvm::make_scope_exit([&]() { PyBuffer_Release(&view); });
  llvm::SmallVector<int64_t> shape, strides;
  if (explicitShape)
    llvm::append_range(shape, *explicitShape);
  else
    shape.append(view.shape, view.shape + view.ndim);

  /// Detect format codes that are suitable for bulk loading. This includes
  /// all byte aligned integer and floating point types up to 8 bytes.
  /// Notably, this excludes, bool (which needs to be bit-packed) and
  /// other exotics which do not have a direct representation in the buffer
  /// protocol (i.e. complex, etc).
  /// TODO: Until we have a mechanism to preserve the staging buffer until
  /// asynchronous copy is done, if a `stream` is provided then we must perform
  /// a synchronization before freeing the staging buffer.
  MTRT_ScalarTypeCode elementType =
      dtype ? *dtype
            : getScalarTypeCodeFromPyBufferProtocolFormat(view.format, view);

  int64_t bytesPerElement = view.itemsize;
  if (dtype) {
    int64_t bitSize{0};
    MTRT_Status s = mtrtScalarTypeCodeBitsPerElement(elementType, &bitSize);
    THROW_IF_MTRT_ERROR(s);
    bytesPerElement = llvm::divideCeil(bitSize, 8);
  }

  strides.reserve(shape.size());
  if (!explicitShape) {
    for (auto [idx, dim] : llvm::enumerate(shape)) {
      if (view.strides[idx] % bytesPerElement != 0)
        throw std::invalid_argument("expected each dimension stride to be "
                                    "divisible by item size in bytes");
      strides.push_back(view.strides[idx] / bytesPerElement);
    }
  } else {
    strides.resize(shape.size(), 1);
    for (int64_t idx = shape.size() - 2; idx >= 0; idx--)
      strides[idx] = strides[idx + 1] * shape[idx + 1];
  }

  MTRT_MemRefValue hostView{nullptr};
  MTRT_Status s = mtrtMemRefCreateExternal(
      client, MTRT_PointerType_host, elementType,
      reinterpret_cast<uintptr_t>(view.buf), 0, shape.size(), shape.data(),
      strides.data(), mtrtDeviceGetNull(), &hostView);
  THROW_IF_MTRT_ERROR(s);
  auto cleanupView = llvm::make_scope_exit([&]() {
    MTRT_Status s = mtrtMemRefValueDestroy(hostView);
    THROW_IF_MTRT_ERROR(s);
  });

  if (addressSpace == MTRT_PointerType::MTRT_PointerType_host) {
    MTRT_MemRefValue result{nullptr};
    s = mtrtCopyFromHostToHost(hostView, &result);
    THROW_IF_MTRT_ERROR(s);
    return std::make_unique<PyMemRefValue>(result);
  }

  MTRT_MemRefValue result{nullptr};
  s = mtrtCopyFromHostToDevice(hostView, device ? *device : mtrtDeviceGetNull(),
                               !stream ? mtrtStreamGetNull() : *stream,
                               &result);
  THROW_IF_MTRT_ERROR(s);

  if (stream) {
    s = mtrtStreamSynchronize(stream ? *stream : mtrtStreamGetNull());
    THROW_IF_MTRT_ERROR(s);
  }

  return std::make_unique<PyMemRefValue>(result);
}

static std::unique_ptr<PyMemRefValue> getMemRefViewWithCContiguousLayout(
    PyRuntimeClient &client, uintptr_t ptr, llvm::ArrayRef<int64_t> shape,
    MTRT_ScalarTypeCode scalarType, std::optional<MTRT_Device> device,
    MTRT_PointerType addressSpace) {
  // This assumes that the data is C-contiguous.
  llvm::SmallVector<int64_t> strides(shape.size(), 1);
  if (!shape.empty()) {
    strides.back() = 1;
    for (int64_t idx = static_cast<int64_t>(strides.size()) - 2; idx >= 0;
         idx--)
      strides[idx] = strides[idx + 1] * shape[idx + 1];
  }

  MTRT_MemRefValue view{nullptr};
  MTRT_Status s = mtrtMemRefCreateExternal(
      client, addressSpace, scalarType, ptr, 0, shape.size(), shape.data(),
      strides.data(), device ? *device : mtrtDeviceGetNull(), &view);
  THROW_IF_MTRT_ERROR(s);

  return std::make_unique<PyMemRefValue>(view);
}

template <typename Type>
py::buffer_info
getMemRefBufferInfo(const MTRT_MemRefValueInfo &info,
                    std::optional<llvm::StringRef> explicitFormat = {}) {

  if (info.bitsPerElement < 8)
    throw std::runtime_error(
        "buffer protocol for packed sub-byte dtypes is not supported");

  // Prepare the data for the buffer_info.
  // Buffer is configured for read-only access below.
  Type *data = reinterpret_cast<Type *>(info.ptr + info.offset);
  // Prepare the shape for the buffer_info.
  llvm::SmallVector<intptr_t, 4> shape(info.shape, info.shape + info.rank);

  // Prepare the strides for the buffer_info.
  llvm::SmallVector<intptr_t, 4> strides;

  for (unsigned i = 0; i < info.rank; i++)
    strides.push_back(sizeof(Type) * info.strides[i]);

  std::string format;
  if (explicitFormat)
    format = *explicitFormat;
  else
    format = py::format_descriptor<Type>::format();
  return py::buffer_info(data, sizeof(Type), format, info.rank, shape, strides,
                         /*readonly=*/true);
}

static py::buffer_info
getPyBufferProtocolInfoFromMemRef(MTRT_MemRefValue memref) {
  MTRT_MemRefValueInfo info;
  MTRT_Status s = mtrtMemRefValueGetInfo(memref, &info);
  THROW_IF_MTRT_ERROR(s);
  switch (info.scalarType) {
  case MTRT_ScalarTypeCode_f32:
    return getMemRefBufferInfo<float>(info);
  case MTRT_ScalarTypeCode_f64:
    return getMemRefBufferInfo<double>(info);
  case MTRT_ScalarTypeCode_f16:
    return getMemRefBufferInfo<uint16_t>(info, "e");
  case MTRT_ScalarTypeCode_bf16:
    return getMemRefBufferInfo<uint16_t>(info);
  case MTRT_ScalarTypeCode_f8e4m3fn:
    return getMemRefBufferInfo<uint8_t>(info);
  case MTRT_ScalarTypeCode_i64:
    return getMemRefBufferInfo<int64_t>(info);
  case MTRT_ScalarTypeCode_i32:
    return getMemRefBufferInfo<int32_t>(info);
  case MTRT_ScalarTypeCode_i8:
    return getMemRefBufferInfo<int8_t>(info);
  case MTRT_ScalarTypeCode_i4:
    return getMemRefBufferInfo<int8_t>(info);
  case MTRT_ScalarTypeCode_i1:
    return getMemRefBufferInfo<bool>(info);
  case MTRT_ScalarTypeCode_i16:
    return getMemRefBufferInfo<int16_t>(info);
  default:
    break;
  }

  // TODO: Currently crashes the program.
  // Reported as https://github.com/pybind/pybind11/issues/3336
  throw std::invalid_argument(
      "unsupported data type for conversion to Python buffer");
}

//===----------------------------------------------------------------------===//
// Helpers for executable enqueue call and argument conversions
//===----------------------------------------------------------------------===//

using FuncArgUnion = std::variant<MTRT_MemRefValue, MTRT_ScalarValue>;

//===----------------------------------------------------------------------===//
// RAII Global Setup Object
//===----------------------------------------------------------------------===//

namespace {
struct ExecutorRuntimeGlobalSetup {
  ExecutorRuntimeGlobalSetup() { mtrtRuntimeInitialize(); }
  ~ExecutorRuntimeGlobalSetup() { mtrtRuntimeShutdown(); }
};

//===----------------------------------------------------------------------===//
// DLPack Utilities
//===----------------------------------------------------------------------===//

/// Wrapper around a DLManagedTensor object. A pointer to the allocated memory
/// for this object is placed into `tensor->manager_ctx`, enabling destruction
/// from the capsule deleter.
struct DLPackManagerContext {

  /// The `buffer_reference` holds a reference to the Python object.
  // py::object buffer_reference;

  // Hold a reference to the underlying memref value.
  std::unique_ptr<PyMemRefValue> memref;

  // Storage for the shape information. Referenced by fields in `tensor`.
  std::vector<int64_t> shape;
  // Storage for the stride information. Referenced by fields in `tensor`.
  std::vector<int64_t> strides;

  // The DLPack managed tensor object.
  DLManagedTensor tensor;
};
} // namespace

/// Deleter for a DLPack managed tensor object. This deleter is called from the
/// DLPack tensor capsule's destructor.
static void DLPackTensorDeleter(DLManagedTensor *t) {
  // NOTE: the PY_IsInitialized() check below comes from the NumPy DLPack
  // reference implementation.
  if (!Py_IsInitialized()) {
    // Leak the Python object if the Python runtime is not available.
    // This can happen if the DLPack consumer destroys the tensor late
    // after Python runtime finalization (for example in case the tensor
    // was indirectly kept alive by a C++ static variable).
    return;
  }
  if (t) {
    PyGILState_STATE state = PyGILState_Ensure();
    //
    delete static_cast<DLPackManagerContext *>(t->manager_ctx);
    PyGILState_Release(state);
  }
}

/// Convert a MTRT scalar type code to a DLPack data type code.
static DLDataTypeCode toDLPackDataTypeCode(MTRT_ScalarTypeCode type) {
  switch (type) {
  case MTRT_ScalarTypeCode_i1:
    return DLDataTypeCode::kDLBool;
  case MTRT_ScalarTypeCode_i4:
  case MTRT_ScalarTypeCode_i8:
  case MTRT_ScalarTypeCode_i16:
  case MTRT_ScalarTypeCode_i32:
  case MTRT_ScalarTypeCode_i64:
    return DLDataTypeCode::kDLInt;
  case MTRT_ScalarTypeCode_ui8:
    return DLDataTypeCode::kDLUInt;
  case MTRT_ScalarTypeCode_f8e4m3fn:
  case MTRT_ScalarTypeCode_f16:
  case MTRT_ScalarTypeCode_f32:
  case MTRT_ScalarTypeCode_f64:
    return DLDataTypeCode::kDLFloat;
  case MTRT_ScalarTypeCode_bf16:
    return DLDataTypeCode::kDLBfloat;
  case MTRT_ScalarTypeCode_complex32:
  case MTRT_ScalarTypeCode_complex64:
    return DLDataTypeCode::kDLComplex;
  default:
    throw std::invalid_argument(
        "Scalar type code conversion to DLPackDataTypeCode is not supported.");
  }
  return DLDataTypeCode::kDLFloat;
}

static void memrefViewDLPackDestructionCallback(void *userData) {
  auto *pack = static_cast<DLManagedTensor *>(userData);
  pack->deleter(pack);
}

static std::unique_ptr<PyMemRefValue>
createMemRefViewFromDLPack(PyRuntimeClient &client, py::capsule capsule,
                           std::optional<bool> assertCanonicalStrides) {
  if (std::string_view(capsule.name()) != "dltensor") {
    throw std::invalid_argument(
        "DLPack tensor must be a capsule with name \"dltensor\", got \"" +
        std::string(capsule.name()) +
        "\". "
        "Note that a DLPack tensor may be consumed at most once.");
  }

  DLManagedTensor *managedTensor =
      static_cast<DLManagedTensor *>(capsule.get_pointer());

  MTRT_MemRefValue result{nullptr};

  // Extract the necessary information from the DLManagedTensor
  void *data = managedTensor->dl_tensor.data;
  int64_t *shape = managedTensor->dl_tensor.shape;
  int64_t *strides = managedTensor->dl_tensor.strides;

  // Create a suffix product stride array in the event that the DLPack object's
  // stride array is set to `null`
  std::vector<int64_t> stridesArray;
  if (!strides) {
    int32_t ndim = managedTensor->dl_tensor.ndim;
    stridesArray.resize(ndim);
    if (ndim > 0) {
      stridesArray[ndim - 1] = 1;
      for (int i = ndim - 2; i >= 0; i--) {
        stridesArray[i] = shape[i + 1] * stridesArray[i + 1];
      }
    }
    strides = stridesArray.data();
  }

  int64_t offset = managedTensor->dl_tensor.byte_offset;
  int rank = managedTensor->dl_tensor.ndim;
  DLDataType dtype = managedTensor->dl_tensor.dtype;
  DLDeviceType device_type = managedTensor->dl_tensor.device.device_type;
  int device_id = managedTensor->dl_tensor.device.device_id;

  MTRT_ScalarTypeCode elementType;
  MTRT_Status s;
  s = mtrtGetScalarTypeCodeFromDLDataType(dtype, &elementType);
  THROW_IF_MTRT_ERROR(s);

  int64_t bitsPerElementExpected;
  s = mtrtScalarTypeCodeBitsPerElement(elementType, &bitsPerElementExpected);
  THROW_IF_MTRT_ERROR(s);
  if (dtype.bits != bitsPerElementExpected) {
    throw std::invalid_argument("DLPack tensor has unexpected bit width: " +
                                std::to_string(dtype.bits) + " expected: " +
                                std::to_string(bitsPerElementExpected));
  }

  if (dtype.lanes != 1) {
    throw std::invalid_argument(
        "DLPack tensor has unsupported number of lanes: " +
        std::to_string(dtype.lanes) + "; only lanes = 1 is supported");
  }

  MTRT_PointerType addressSpace;
  s = mtrtGetPointerTypeFromDLDeviceType(device_type, &addressSpace);
  THROW_IF_MTRT_ERROR(s);

  MTRT_Device device{nullptr};
  if (addressSpace == MTRT_PointerType_device) {
    s = mtrtRuntimeClientGetDevice(client, device_id, &device);
    THROW_IF_MTRT_ERROR(s);
  }

  // Per the DLPack specification, we must "consume" capsule by changing its
  // name.
  capsule.set_name("used_dltensor");
  PyCapsule_SetDestructor(capsule.ptr(), nullptr);

  if (data) {
    s = mtrtMemRefCreateExternal(
        client, addressSpace, elementType, reinterpret_cast<uintptr_t>(data),
        offset, rank, shape, strides, device, &result,
        assertCanonicalStrides ? *assertCanonicalStrides : false,
        /*destroyCallback=*/
        MTRT_MemRefDestroyCallback{reinterpret_cast<void *>(managedTensor),
                                   memrefViewDLPackDestructionCallback});
  } else {
    s = mtrtMemRefCreate(client, addressSpace, elementType, rank, shape,
                         strides, device, mtrtStreamGetNull(), &result,
                         assertCanonicalStrides ? *assertCanonicalStrides
                                                : false);
  }

  THROW_IF_MTRT_ERROR(s);
  return std::make_unique<PyMemRefValue>(result);
}

static py::capsule pyMemRefValueToDLPackCapsule(py::handle obj,
                                                int32_t /*stream*/) {
  // This cast should always succeed since this is a member function
  // of PyMemRefValue.
  PyMemRefValue &buffer = py::cast<PyMemRefValue &>(obj);

  MTRT_MemRefValue ref = mtrtMemRefCreateRef(buffer);

  auto pack = std::make_unique<DLPackManagerContext>();
  pack->memref = std::make_unique<PyMemRefValue>(ref);
  pack->tensor.manager_ctx = pack.get();
  pack->tensor.deleter = DLPackTensorDeleter;

  DLDeviceType device_type;
  int32_t device_id;
  MTRT_Status s =
      mtrtMemRefValueGetDLPackDevice(buffer, &device_type, &device_id);
  THROW_IF_MTRT_ERROR(s);

  MTRT_MemRefValueInfo info;
  s = mtrtMemRefValueGetInfo(buffer, &info);
  THROW_IF_MTRT_ERROR(s);

  // Generally we shouldn't be creating DLPack tensors with offsets.
  if (info.offset != 0)
    throw std::invalid_argument(
        "conversion to DLPack is only supported for 0-offset "
        "MemRefValues");

  DLTensor &dt = pack->tensor.dl_tensor;
  dt.data = reinterpret_cast<void *>(info.ptr);
  dt.device.device_type = device_type;
  dt.device.device_id = device_id;
  dt.ndim = info.rank;

  DLDataType dtype;
  dtype.code = toDLPackDataTypeCode(info.scalarType);

  int64_t bitsPerElement;
  s = mtrtScalarTypeCodeBitsPerElement(info.scalarType, &bitsPerElement);
  THROW_IF_MTRT_ERROR(s);
  dtype.bits = bitsPerElement;
  dtype.lanes = 1;
  dt.dtype = dtype;

  pack->shape = std::vector<int64_t>(info.shape, info.shape + info.rank);
  pack->strides = std::vector<int64_t>(info.strides, info.strides + info.rank);

  dt.shape = reinterpret_cast<std::int64_t *>(pack->shape.data());
  dt.strides = reinterpret_cast<std::int64_t *>(pack->strides.data());
  dt.byte_offset = 0;

  // We cannot use pybind's capsule object constructor because we
  // need to detect if the capsule name has been changed in the
  // deleter, but pybind hides the underlying Python object from the
  // deleter.
  py::capsule capsule = py::capsule(
      &pack.release()->tensor, "dltensor", [](PyObject *obj) noexcept {
        DLManagedTensor *dlmt = static_cast<DLManagedTensor *>(
            PyCapsule_GetPointer(obj, "dltensor"));
        if (dlmt) {
          DLPackTensorDeleter(dlmt);
        } else {
          // The tensor has been deleted. Clear any
          // error from PyCapsule_GetPointer.
          PyErr_Clear();
        }
      });
  return capsule;
}

//===----------------------------------------------------------------------===//
// Core Binding Definitions
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_api, m) {
  static ExecutorRuntimeGlobalSetup globalSetup;

  py::register_exception<MTRTException>(m, "MTRTException");

  populateCommonBindingsInModule(m);

  py::class_<PyDevice>(m, "Device", py::module_local())
      .def_property_readonly(MTRT_PYTHON_CAPI_PTR_ATTR, &PyDevice::getCapsule)
      .def("get_name", [](PyDevice &self) -> py::str {
        int32_t index;
        MTRT_Status s = mtrtDeviceGetIndex(self, &index);
        THROW_IF_MTRT_ERROR(s);
        std::string deviceName = "cuda:" + std::to_string(index);
        return py::str(deviceName.c_str());
      });

  py::class_<PyRuntimeValue>(m, "RuntimeValue", py::module_local())
      .def(py::init<>([](PyRuntimeClient *client, int64_t scalar) {
             MTRT_RuntimeValue value;
             MTRT_Status s = mtrtRuntimeValueScalarCreate(
                 scalar, MTRT_ScalarTypeCode_i64, &value);
             THROW_IF_MTRT_ERROR(s);
             return new PyRuntimeValue(value);
           }),
           py::arg("client"), py::arg("scalar_int"));

  py::class_<PyScalarValue, PyRuntimeValue>(
      m, "ScalarValue", py::module_local(), py::buffer_protocol())
      .def_property_readonly("type",
                             [](PyScalarValue &self) {
                               MTRT_ScalarTypeCode code;
                               MTRT_Status s =
                                   mtrtScalarValueGetType(self, &code);
                               THROW_IF_MTRT_ERROR(s);
                               return code;
                             })
      .def_property_readonly("data", [](PyScalarValue &self) {
        int64_t data;
        MTRT_Status s = mtrtScalarValueGet(self, &data);
        THROW_IF_MTRT_ERROR(s);
        return data;
      });
  py::class_<PyMemRefValue, PyRuntimeValue>(
      m, "MemRefValue", py::module_local(), py::buffer_protocol())
      .def_property_readonly("ptr",
                             [](PyMemRefValue &self) {
                               MTRT_MemRefValueInfo info;
                               MTRT_Status s =
                                   mtrtMemRefValueGetInfo(self, &info);
                               THROW_IF_MTRT_ERROR(s);
                               return info.ptr;
                             })
      .def_property_readonly(
          "shape",
          [](PyMemRefValue &self) {
            MTRT_MemRefValueInfo info;
            MTRT_Status s = mtrtMemRefValueGetInfo(self, &info);
            THROW_IF_MTRT_ERROR(s);
            return std::vector<int64_t>(info.shape, info.shape + info.rank);
          })
      .def_property_readonly(
          "strides",
          [](PyMemRefValue &self) {
            MTRT_MemRefValueInfo info;
            MTRT_Status s = mtrtMemRefValueGetInfo(self, &info);
            THROW_IF_MTRT_ERROR(s);
            return std::vector<int64_t>(info.strides, info.strides + info.rank);
          })
      .def_property_readonly("dtype",
                             [](PyMemRefValue &self) {
                               MTRT_MemRefValueInfo info;
                               MTRT_Status s =
                                   mtrtMemRefValueGetInfo(self, &info);
                               THROW_IF_MTRT_ERROR(s);
                               return info.scalarType;
                             })
      .def_property_readonly("address_space",
                             [](PyMemRefValue &self) {
                               MTRT_MemRefValueInfo info;
                               MTRT_Status s =
                                   mtrtMemRefValueGetInfo(self, &info);
                               THROW_IF_MTRT_ERROR(s);
                               return info.addressSpace;
                             })
      .def("__dlpack__", pyMemRefValueToDLPackCapsule, py::arg("stream") = 0)
      .def("__dlpack_device__",
           [](PyMemRefValue &self) {
             DLDeviceType device_type;
             int32_t device_id;
             MTRT_Status s =
                 mtrtMemRefValueGetDLPackDevice(self, &device_type, &device_id);
             THROW_IF_MTRT_ERROR(s);
             return py::make_tuple(static_cast<int>(device_type), device_id);
           })
      .def_buffer([](PyMemRefValue &self) {
        return getPyBufferProtocolInfoFromMemRef(self);
      })
      .def("ref_count",
           [](PyMemRefValue &self) { return mtrtMemRefReferenceCount(self); });

  py::class_<PyStream>(m, "Stream", py::module_local())
      .def_property_readonly(MTRT_PYTHON_CAPI_PTR_ATTR, &PyStream::getCapsule)
      .def("sync",
           [](PyStream &stream) {
             MTRT_Status s = mtrtStreamSynchronize(stream);
             THROW_IF_MTRT_ERROR(s);
           })
      .def("__str__",
           [](PyStream &self) {
             auto callback = [](MlirStringRef data, void *initialString) {
               *reinterpret_cast<std::string *>(initialString) +=
                   llvm::StringRef(data.data, data.length);
             };

             std::string result;
             mtrtStreamPrint(self, callback, &result);
             return result;
           })
      .def_property_readonly("ptr", [](PyStream &self) {
        uintptr_t ptr;
        MTRT_Status s = mtrtStreamGetPointer(self, &ptr);
        THROW_IF_MTRT_ERROR(s);
        return ptr;
      });

  py::class_<PyRuntimeClient, std::shared_ptr<PyRuntimeClient>>(
      m, "RuntimeClient", py::module_local())
      .def(py::init<>([]() {
        MTRT_RuntimeClient client{nullptr};
        MTRT_Status s = mtrtRuntimeClientCreate(&client);
        THROW_IF_MTRT_ERROR(s);
        return std::make_shared<PyRuntimeClient>(client);
      }))
      /// Return the devices accessible by the client.
      .def("get_devices",
           [](PyRuntimeClient &self) {
             std::vector<PyDevice *> devices;
             int32_t numDevices = 0;
             MTRT_Status s = mtrtRuntimeClientGetNumDevices(self, &numDevices);
             THROW_IF_MTRT_ERROR(s);
             devices.reserve(numDevices);
             for (int32_t i = 0; i < numDevices; i++) {
               MTRT_Device device{nullptr};
               s = mtrtRuntimeClientGetDevice(self, i, &device);
               THROW_IF_MTRT_ERROR(s);
               devices.push_back(new PyDevice(device));
             }
             return devices;
           })
      .def(
          "create_scalar",
          [](PyRuntimeClient &self, py::object data,
             std::optional<MTRT_ScalarTypeCode> scalarTypeCode) {
            MTRT_RuntimeValue value;
            int64_t idata;
            if (scalarTypeCode) {
              // dispatch based on explicit code, (we just need to handle
              // f32, f64, i32 and i64 for now, otherwise raise exception).
              // `data` should be py::int_ in the case of integer, py::float_
              // otherwise Try casting float object to int
              switch (*scalarTypeCode) {
              case MTRT_ScalarTypeCode_f32: {
                if (!py::isinstance<py::float_>(data))
                  throw std::runtime_error("Python object must represent float "
                                           "for a ScalarTypeCode.fp32");
                float fdata = py::cast<float>(data);
                std::memcpy(&idata, &fdata, sizeof(float));
                break;
              }
              case MTRT_ScalarTypeCode_f64: {
                if (!py::isinstance<py::float_>(data))
                  throw std::runtime_error("Python object must represent float "
                                           "for a ScalarTypeCode.fp64");
                double fdata = py::cast<double>(data);
                std::memcpy(&idata, &fdata, sizeof(double));
                break;
              }
              case MTRT_ScalarTypeCode_i32:
              case MTRT_ScalarTypeCode_i64:
                if (!py::isinstance<py::int_>(data))
                  throw std::runtime_error(
                      "Python object must represent an integer for a "
                      "ScalarTypeCode.i32 or ScalarTypeCode.i64");
                // Try casting to int
                idata = py::cast<int>(data);
                break;
              default:
                throw std::runtime_error("Unsupported scalar type code");
              }

              if (!py::isinstance<py::int_>(data) &&
                  !py::isinstance<py::float_>(data))
                throw std::runtime_error("Unsupported type: Only int and float "
                                         "data type are supported.");

              // Finally create a scalar value ScalarTypeCode::i64 and cast to
              // RuntimeValue.
              MTRT_Status s =
                  mtrtRuntimeValueScalarCreate(idata, *scalarTypeCode, &value);
              THROW_IF_MTRT_ERROR(s);

              // Cast from RuntimeValue to ScalarValue and return PyScalarValue.
              return new PyScalarValue(mtrtRuntimeValueDynCastToScalar(value));
            }

            // Other path when user provides a Python int object but no type
            // code, we can default to i64
            if (py::isinstance<py::int_>(data)) {
              idata = py::cast<int>(data);
              // Finally create a scalar value ScalarTypeCode::i64 and cast to
              // RuntimeValue.
              MTRT_Status s = mtrtRuntimeValueScalarCreate(
                  idata, MTRT_ScalarTypeCode_i64, &value);
              THROW_IF_MTRT_ERROR(s);
              // Cast from RuntimeValue to ScalarValue and return PyScalarValue.
              return new PyScalarValue(mtrtRuntimeValueDynCastToScalar(value));
            }

            throw std::runtime_error("Unsupported scalar type!");
          },
          py::arg("scalar_value"), py::arg("type_code"),
          "creates a runtime ScalarValue from the provided Python object; an "
          "explicit type "
          "may be provided, otherwise defaults to i64 for Python integers and "
          "f32 for Python floats")

      .def(
          "create_memref",
          [](PyRuntimeClient &self, py::buffer array,
             std::optional<std::vector<int64_t>> shape,
             std::optional<MTRT_ScalarTypeCode> dtype,
             std::optional<MTRT_Device> device,
             std::optional<MTRT_Stream> stream) {
            MTRT_PointerType addressSpace =
                !device ? MTRT_PointerType::MTRT_PointerType_host
                        : MTRT_PointerType::MTRT_PointerType_device;
            return getMemRefFromHostBufferProtocol(self, array, shape, dtype,
                                                   device, stream, addressSpace)
                .release();
          },
          py::arg("array"), py::pos_only(), py::arg("shape") = py::none(),
          py::arg("dtype") = py::none(), py::arg("device") = py::none(),
          py::arg("stream") = py::none(), py::keep_alive<0, 1>())
      .def(
          "create_memref",
          [](PyRuntimeClient &self, std::vector<int64_t> shape,
             MTRT_ScalarTypeCode dtype, std::optional<MTRT_Device> device,
             std::optional<MTRT_Stream> stream) {
            MTRT_PointerType addressSpace =
                !device ? MTRT_PointerType::MTRT_PointerType_host
                        : MTRT_PointerType::MTRT_PointerType_device;
            return createMemRef(self, shape, dtype, device, stream,
                                addressSpace)
                .release();
          },
          py::arg("shape"), py::arg("dtype"), py::arg("device") = py::none(),
          py::arg("stream") = py::none(), py::keep_alive<0, 1>(),
          "returns a new memref and allocates uninitialized backing storage")
      .def(
          "create_memref_view_from_dlpack",
          [](PyRuntimeClient &self, py::capsule capsule,
             std::optional<bool> assertCanonicalStrides) {
            return createMemRefViewFromDLPack(self, capsule,
                                              assertCanonicalStrides);
          },
          py::arg("dltensor") = py::none(),
          py::arg("assert_canonical_strides") = py::none(),
          // keep the client alive
          py::keep_alive<0, 1>(),
          // keep the capsule alive.
          py::keep_alive<0, 2>())
      .def(
          "from_dlpack",
          [](PyRuntimeClient &self, py::object capsule) {
            if (!py::hasattr(capsule, "__dlpack__"))
              throw std::invalid_argument(
                  "object must have a \"__dlpack__\" method");
            py::object dlPackManagedTensor =
                py::getattr(capsule, "__dlpack__")();
            std::unique_ptr<PyMemRefValue> memRef =
                createMemRefViewFromDLPack(self, dlPackManagedTensor,
                                           /*assertCanonicalStrides=*/false);
            return memRef;
          },
          py::arg("dltensor") = py::none(),
          // Keep the client alive
          py::keep_alive<0, 1>())
      .def(
          "create_memref_view_from_dlpack",
          [](PyRuntimeClient &self, py::capsule capsule,
             std::optional<bool> assertCanonicalStrides) {
            return createMemRefViewFromDLPack(self, capsule,
                                              assertCanonicalStrides)
                .release();
          },
          py::arg("dltensor") = py::none(),
          py::arg("assert_canonical_strides") = py::none(),
          // returned memrefvalue keeps the client alive
          py::keep_alive<0, 1>(),
          // returned memrefvalue keeeps the dlpack capsule alive
          py::keep_alive<0, 2>())
      .def(
          "create_device_memref_view",
          [](PyRuntimeClient &self, uintptr_t ptr, std::vector<int64_t> shape,
             MTRT_ScalarTypeCode scalarType, PyDevice &device) {
            MTRT_PointerType addressSpace = MTRT_PointerType_device;
            return getMemRefViewWithCContiguousLayout(
                       self, ptr, shape, scalarType, device, addressSpace)
                .release();
          },
          py::arg("ptr"), py::arg("shape"), py::arg("dtype"), py::arg("device"),
          py::keep_alive<0, 1>())
      .def(
          "create_host_memref_view",
          [](PyRuntimeClient &self, uintptr_t ptr, std::vector<int64_t> shape,
             MTRT_ScalarTypeCode scalarType) {
            MTRT_PointerType addressSpace = MTRT_PointerType_host;
            return getMemRefViewWithCContiguousLayout(
                       self, ptr, shape, scalarType, {}, addressSpace)
                .release();
          },
          py::arg("ptr"), py::arg("shape"), py::arg("dtype") = py::none(),
          py::keep_alive<0, 1>())
      .def(
          "create_stream",
          [](PyRuntimeClient &self) {
            MTRT_Stream stream{nullptr};
            MTRT_Status s = mtrtStreamCreate(&stream);
            THROW_IF_MTRT_ERROR(s);
            return PyStream(stream);
          },
          py::keep_alive<0, 1>())
      .def(
          "copy_to_device",
          [](PyRuntimeClient &self, PyMemRefValue &hostMemRef, PyDevice &device,
             std::optional<MTRT_Stream> stream) {
            MTRT_MemRefValue deviceMemRef{nullptr};
            MTRT_Status s = mtrtCopyFromHostToDevice(
                hostMemRef, device, stream ? *stream : mtrtStreamGetNull(),
                &deviceMemRef);
            THROW_IF_MTRT_ERROR(s);
            return new PyMemRefValue(deviceMemRef);
          },
          py::arg("host_memref"), py::arg("device"),
          py::arg("stream") = py::none(), py::keep_alive<0, 1>())
      .def(
          "copy_to_host",
          [](PyRuntimeClient &self, PyMemRefValue &deviceMemRef,
             std::optional<MTRT_Stream> stream) {
            MTRT_MemRefValue hostMemRef{nullptr};
            MTRT_Status s = mtrtCopyFromDeviceToNewHostMemRef(
                deviceMemRef, stream ? *stream : mtrtStreamGetNull(),
                &hostMemRef);
            THROW_IF_MTRT_ERROR(s);
            return new PyMemRefValue(hostMemRef);
          },
          py::arg("device_memref"), py::arg("stream") = py::none(),
          py::keep_alive<0, 1>())
      .def(
          "copy_to_host",
          [](PyRuntimeClient &self, PyMemRefValue &deviceMemRef,
             PyMemRefValue &hostMemRef, std::optional<MTRT_Stream> stream) {
            MTRT_Status s = mtrtCopyFromDeviceToExistingHostMemRef(
                deviceMemRef, hostMemRef,
                stream ? *stream : mtrtStreamGetNull());
            THROW_IF_MTRT_ERROR(s);
          },
          py::arg("device_memref"), py::arg("existing_host_memref"),
          py::arg("stream") = py::none());

  py::class_<PyRuntimeSessionOptions>(m, "RuntimeSessionOptions",
                                      py::module_local())
      .def(py::init<>([](int32_t numDevices, int32_t deviceId,
                         std::string ncclUuid,
                         std::optional<std::vector<std::string>> features)
                          -> PyRuntimeSessionOptions * {
             MTRT_RuntimeSessionOptions options;
             MTRT_Status s = mtrtRuntimeSessionOptionsCreate(
                 numDevices, deviceId,
                 MTRT_StringView{ncclUuid.data(), ncclUuid.size()}, &options);
             THROW_IF_MTRT_ERROR(s);

             if (features) {
               for (const std::string &feature : *features)
                 mtrtRuntimeSessionOptionsEnableFeature(
                     options, MTRT_StringView{feature.data(), feature.size()});
             } else {
               std::array<llvm::StringRef, 3> defaultFeatures = {"core", "cuda",
                                                                 "tensorrt"};
               // Enable all the default features.
               for (const auto &feature : defaultFeatures)
                 mtrtRuntimeSessionOptionsEnableFeature(
                     options, MTRT_StringView{feature.data(), feature.size()});
             }
             return new PyRuntimeSessionOptions(options);
           }),
           py::arg("num_devices") = 1, py::arg("device_id") = 0,
           py::arg("nccl_uuid") = py::str(""),
           py::arg("features") = py::none());

  py::class_<PyRuntimeSession, std::shared_ptr<PyRuntimeSession>>(
      m, "RuntimeSession", py::module_local())
      .def(py::init<>([](PyRuntimeSessionOptions &options, PyExecutable &exe) {
             MTRT_RuntimeSession session;
             MTRT_Status s = mtrtRuntimeSessionCreate(options, exe, &session);
             THROW_IF_MTRT_ERROR(s);

             return std::make_shared<PyRuntimeSession>(session);
           }),
           py::arg("options"), py::arg("executable"))
      .def(
          "execute_function",
          [](PyRuntimeSession &self, std::string name,
             std::vector<PyRuntimeValue *> inArgs,
             std::optional<std::vector<PyMemRefValue *>> outArgs,
             std::optional<MTRT_Stream> stream,
             PyRuntimeClient *client = nullptr) {
            MTRT_StringView nameRef{name.data(), name.size()};

            int64_t numResults;
            MTRT_Status s =
                mtrtRuntimeSessionGetNumResults(self, nameRef, &numResults);
            THROW_IF_MTRT_ERROR(s);

            llvm::SmallVector<MTRT_RuntimeValue> inArgsGeneric =
                llvm::map_to_vector(
                    inArgs, [](PyRuntimeValue *arg) -> MTRT_RuntimeValue {
                      return arg->getRuntimeValue();
                    });
            llvm::SmallVector<MTRT_RuntimeValue> outArgsGeneric =
                outArgs ? llvm::map_to_vector(*outArgs,
                                              [](PyMemRefValue *arg) {
                                                return arg->getRuntimeValue();
                                              })
                        : llvm::SmallVector<MTRT_RuntimeValue>{};

            std::vector<MTRT_RuntimeValue> resultsGeneric(numResults);

            s = mtrtRuntimeSessionExecuteFunction(
                self, nameRef, inArgsGeneric.data(), inArgsGeneric.size(),
                outArgsGeneric.data(), outArgsGeneric.size(),
                resultsGeneric.data(), stream ? *stream : mtrtStreamGetNull(),
                client ? MTRT_RuntimeClient(*client)
                       : mtrtRuntimeClientGetNull());
            THROW_IF_MTRT_ERROR(s);

            std::vector<std::unique_ptr<PyRuntimeValue>> resultPyObject;
            if (numResults > 0) {
              if (!client)
                throw std::invalid_argument(
                    "client must be provided when there are returned results");

              for (const MTRT_RuntimeValue &arg : resultsGeneric)
                if (mtrtRuntimeValueIsMemRef(arg))
                  resultPyObject.push_back(std::unique_ptr<PyRuntimeValue>(
                      new PyMemRefValue(mtrtRuntimeValueDynCastToMemRef(arg))));
                else
                  resultPyObject.push_back(std::unique_ptr<PyRuntimeValue>(
                      new PyScalarValue(mtrtRuntimeValueDynCastToScalar(arg))));
            }

            if (client) {
              std::shared_ptr<PyRuntimeSession> sessionRef =
                  self.shared_from_this();
              if (client->getSessionsSet().count(sessionRef) == 0)
                client->getSessionsSet().insert(sessionRef);
            }

            return resultPyObject;
          },
          py::arg("name"), py::arg("in_args"), py::arg("out_args") = py::none(),
          py::arg("stream") = py::none(), py::arg("client") = nullptr,
          "Execute a function given input and optional output arguments. "
          "Return optional results as a Python object if output arguments are "
          "not present.");

  py::class_<PyGlobalDebugFlag>(m, "GlobalDebug", py::module_local())
      .def_property_static("flag", &PyGlobalDebugFlag::get,
                           &PyGlobalDebugFlag::set, "LLVM-wide debug flag")
      .def_static(
          "set_types",
          py::overload_cast<const std::string &>(&PyGlobalDebugFlag::set_types),
          "Sets specific debug type to be produced by LLVM")
      .def_static("set_types",
                  py::overload_cast<const std::vector<std::string> &>(
                      &PyGlobalDebugFlag::set_types),
                  "Sets specific debug types to be produced by LLVM");
}
