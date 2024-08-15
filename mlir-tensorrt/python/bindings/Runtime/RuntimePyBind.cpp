//===- Client.cpp  --------------------------------------------------------===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#include "mlir-executor-c/Support/Status.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <exception>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace py = pybind11;
using namespace mlirtrt;

//===----------------------------------------------------------------------===//
// MTRT_* <-> PyCapsule utilities.
//===----------------------------------------------------------------------===//

MTRT_DEFINE_RUNTIME_INLINE_PY_CAPSULE_CASTER_FUNCS(Device)
MTRT_DEFINE_RUNTIME_INLINE_PY_CAPSULE_CASTER_FUNCS(Stream)
MTRT_DEFINE_RUNTIME_INLINE_PY_CAPSULE_CASTER_FUNCS(RuntimeValue)
MTRT_DEFINE_RUNTIME_INLINE_PY_CAPSULE_CASTER_FUNCS(MemRefValue)
MTRT_DEFINE_RUNTIME_INLINE_PY_CAPSULE_CASTER_FUNCS(ScalarValue)

//===----------------------------------------------------------------------===//
// PyBind Casters
//===----------------------------------------------------------------------===//
static py::object mtrtApiObjectToCapsule(py::handle apiObject) {
  if (PyCapsule_CheckExact(apiObject.ptr()))
    return py::reinterpret_borrow<py::object>(apiObject);
  if (!py::hasattr(apiObject, MTRT_PYTHON_CAPI_PTR_ATTR)) {
    auto repr = py::repr(apiObject).cast<std::string>();
    throw py::type_error(
        (llvm::Twine("Expected an MLIR-TensorRT object (got ") + repr + ").")
            .str());
  }
  return apiObject.attr(MTRT_PYTHON_CAPI_PTR_ATTR);
}

namespace pybind11::detail {
/// Casts object (capsule) -> MTRT_Device
template <>
struct type_caster<MTRT_Device> {
  PYBIND11_TYPE_CASTER(MTRT_Device, _("MTRT_Device"));
  bool load(handle src, bool) {
    py::object capsule = mtrtApiObjectToCapsule(src);
    value = mtrtPythonCapsuleToDevice(capsule.ptr());
    return !mtrtDeviceIsNull(value);
  }
};

/// Casts object (capsule) -> MTRT_Stream
template <>
struct type_caster<MTRT_Stream> {
  PYBIND11_TYPE_CASTER(MTRT_Stream, _("MTRT_Stream"));
  bool load(handle src, bool) {
    py::object capsule = mtrtApiObjectToCapsule(src);
    value = mtrtPythonCapsuleToStream(capsule.ptr());
    return !mtrtStreamIsNull(value);
  }
};

/// Casts object (capsule) -> MTRT_RuntimeValue
template <>
struct type_caster<MTRT_RuntimeValue> {
  PYBIND11_TYPE_CASTER(MTRT_RuntimeValue, _("MTRT_RuntimeValue"));
  bool load(handle src, bool) {
    py::object capsule = mtrtApiObjectToCapsule(src);
    value = mtrtPythonCapsuleToRuntimeValue(capsule.ptr());
    return !mtrtRuntimeValueIsNull(value);
  }
};

/// Casts object (capsule) -> MTRT_ScalarValue
template <>
struct type_caster<MTRT_ScalarValue> {
  PYBIND11_TYPE_CASTER(MTRT_ScalarValue, _("MTRT_ScalarValue"));
  bool load(handle src, bool) {
    py::object capsule = mtrtApiObjectToCapsule(src);
    value = mtrtPythonCapsuleToScalarValue(capsule.ptr());
    return !mtrtScalarValueIsNull(value);
  }
};

/// Casts object (capsule) -> MTRT_MemRefValue
template <>
struct type_caster<MTRT_MemRefValue> {
  PYBIND11_TYPE_CASTER(MTRT_MemRefValue, _("MTRT_MemRefValue"));
  bool load(handle src, bool) {
    py::object capsule = mtrtApiObjectToCapsule(src);
    value = mtrtPythonCapsuleToMemRefValue(capsule.ptr());
    return !mtrtMemRefValueIsNull(value);
  }
};

} // namespace pybind11::detail

namespace {

class PyRuntimeClient;

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

/// Python wrapper around MTRT_ScalarValue.
class PyScalarValue : public PyMTRTWrapper<PyScalarValue, MTRT_ScalarValue> {
public:
  using Base::Base;
  DECLARE_WRAPPER_CONSTRUCTORS(PyScalarValue);

  static constexpr auto kMethodTable = CAPITable<MTRT_ScalarValue>{
      mtrtScalarValueIsNull,
      [](MTRT_ScalarValue value) {
        (void)value;
        return mtrtStatusGetOk();
      },
      mtrtPythonCapsuleToScalarValue, mtrtPythonScalarValueToCapsule};
};

/// Python wrapper around MTRT_MemRefValue.
class PyMemRefValue : public PyMTRTWrapper<PyMemRefValue, MTRT_MemRefValue> {
public:
  using Base::Base;
  DECLARE_WRAPPER_CONSTRUCTORS(PyMemRefValue);

  static constexpr auto kMethodTable = CAPITable<MTRT_MemRefValue>{
      mtrtMemRefValueIsNull, mtrtMemRefValueDestroy,
      mtrtPythonCapsuleToMemRefValue, mtrtPythonMemRefValueToCapsule};

  MTRT_RuntimeClient getClient() { return mtrtMemRefGetClient(*this); }
};

/// Python object of wrapper for `MTRT_RuntimeValue`.
class PyRuntimeValue : public PyMTRTWrapper<PyRuntimeValue, MTRT_RuntimeValue> {
public:
  using Base::Base;
  DECLARE_WRAPPER_CONSTRUCTORS(PyRuntimeValue);

  static constexpr auto kMethodTable = CAPITable<MTRT_RuntimeValue>{
      mtrtRuntimeValueIsNull, mtrtRuntimeValueDestroy,
      mtrtPythonCapsuleToRuntimeValue, mtrtPythonRuntimeValueToCapsule};
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
    : public PyMTRTWrapper<PyRuntimeSession, MTRT_RuntimeSession> {
public:
  using Base::Base;
  DECLARE_WRAPPER_CONSTRUCTORS(PyRuntimeSession)

  static constexpr auto kMethodTable = mlirtrt::CAPITable<MTRT_RuntimeSession>{
      mtrtRuntimeSessionIsNull, mtrtRuntimeSessionDestroy};
};

/// Python wrapper around MTRT_RuntimeClient.
class PyRuntimeClient
    : public PyMTRTWrapper<PyRuntimeClient, MTRT_RuntimeClient> {
public:
  using Base::Base;
  DECLARE_WRAPPER_CONSTRUCTORS(PyRuntimeClient);

  static constexpr auto kMethodTable = CAPITable<MTRT_Runtimeclient>{
      mtrtRuntimeClientIsNull, mtrtRuntimeClientDestroy};
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
  int64_t bitSize{0};
  MTRT_Status s = mtrtScalarTypeCodeBitsPerElement(dtype, &bitSize);
  THROW_IF_MTRT_ERROR(s);
  int64_t bytesPerElement = llvm::divideCeil(bitSize, 8);

  strides.resize(shape.size(), 1);
  for (int64_t idx = shape.size() - 2; idx >= 0; idx--)
    strides[idx] = strides[idx + 1] * shape[idx + 1];

  MTRT_MemRefValue result{nullptr};
  s = mtrtMemRefCreate(client, addressSpace, bytesPerElement * 8, shape.size(),
                       shape.data(), strides.data(),
                       device ? *device : mtrtDeviceGetNull(),
                       stream ? *stream : mtrtStreamGetNull(), dtype, &result);
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

  // Detect format codes that are suitable for bulk loading. This includes
  // all byte aligned integer and floating point types up to 8 bytes.
  // Notably, this excludes, bool (which needs to be bit-packed) and
  // other exotics which do not have a direct representation in the buffer
  // protocol (i.e. complex, etc).
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
      client, MTRT_PointerType_host, bytesPerElement * 8,
      reinterpret_cast<uintptr_t>(view.buf), 0, shape.size(), shape.data(),
      strides.data(), mtrtDeviceGetNull(), elementType, &hostView);
  THROW_IF_MTRT_ERROR(s);

  if (addressSpace == MTRT_PointerType::MTRT_PointerType_host) {
    MTRT_MemRefValue result{nullptr};
    s = mtrtCopyFromHostToHost(hostView, &result);
    THROW_IF_MTRT_ERROR(s);
    return std::make_unique<PyMemRefValue>(result);
  }

  MTRT_MemRefValue result{nullptr};
  s = mtrtCopyFromHostToDevice(hostView, device ? *device : mtrtDeviceGetNull(),
                               stream ? *stream : mtrtStreamGetNull(), &result);
  THROW_IF_MTRT_ERROR(s);
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

  int64_t bitWidth{0};
  MTRT_Status s = mtrtScalarTypeCodeBitsPerElement(scalarType, &bitWidth);
  THROW_IF_MTRT_ERROR(s);

  MTRT_MemRefValue view{nullptr};
  s = mtrtMemRefCreateExternal(client, addressSpace, bitWidth, ptr, 0,
                               shape.size(), shape.data(), strides.data(),
                               device ? *device : mtrtDeviceGetNull(),
                               scalarType, &view);
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

/// Above replace `convertArgType` with this (go from `py::object`)
static MTRT_RuntimeValue convertArgType(py::object obj) {
  if (py::isinstance<PyMemRefValue>(obj)) {
    return mtrtMemRefCastToRuntimeValue(py::cast<PyMemRefValue &>(obj).get());
  }
  if (py::isinstance<PyScalarValue>(obj)) {
    return mtrtScalarValueCastToRuntimeValue(
        py::cast<PyScalarValue &>(obj).get());
  }
  throw std::runtime_error("argument must be MemRef or scalar");
}

//===----------------------------------------------------------------------===//
// Declare the bindings.
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_api, m) {
  py::register_exception<MTRTException>(m, "MTRTException");

  populateCommonBindingsInModule(m);

  py::class_<PyDevice>(m, "Device", py::module_local())
      .def_property_readonly(MTRT_PYTHON_CAPI_PTR_ATTR, &PyDevice::getCapsule);
  py::class_<PyScalarValue>(m, "ScalarValue", py::module_local(),
                            py::buffer_protocol())
      .def_property_readonly(MTRT_PYTHON_CAPI_PTR_ATTR,
                             &PyScalarValue::getCapsule)
      .def_property_readonly("type", [](PyScalarValue &self) {
        MTRT_ScalarTypeCode code;
        MTRT_Status s = mtrtScalarValueGetType(self, &code);
        THROW_IF_MTRT_ERROR(s);
        return code;
      });
  py::class_<PyMemRefValue>(m, "MemRefValue", py::module_local(),
                            py::buffer_protocol())
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
      .def_property_readonly(MTRT_PYTHON_CAPI_PTR_ATTR,
                             &PyMemRefValue::getCapsule)
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
      .def(
          "__dlpack__",
          [](PyMemRefValue &self, int32_t /*stream*/) {
            MTRT_DLPackManagedTensor tensor;
            MTRT_Status s =
                mtrtMemRefValueGetDLPackManagedTensor(self, &tensor);
            THROW_IF_MTRT_ERROR(s);
            assert(tensor.ptr != nullptr &&
                   "expected valid MTRT_DLPackManagedTensor");
            return py::capsule(reinterpret_cast<DLManagedTensor *>(tensor.ptr),
                               "dltensor");
          },
          py::arg("stream") = 0)
      .def("__dlpack_device__",
           [](PyMemRefValue &self) {
             int32_t device_type;
             int32_t device_id;
             MTRT_Status s =
                 mtrtMemRefValueGetDLPackDevice(self, &device_type, &device_id);
             THROW_IF_MTRT_ERROR(s);
             return py::make_tuple(device_type, device_id);
           })
      .def_buffer([](PyMemRefValue &self) {
        return getPyBufferProtocolInfoFromMemRef(self);
      });

  py::class_<PyStream>(m, "Stream", py::module_local())
      .def_property_readonly(MTRT_PYTHON_CAPI_PTR_ATTR, &PyStream::getCapsule)
      .def("sync", [](PyStream &stream) {
        MTRT_Status s = mtrtStreamSynchronize(stream);
        THROW_IF_MTRT_ERROR(s);
      });

  py::class_<PyRuntimeClient>(m, "RuntimeClient", py::module_local())
      .def(py::init<>([]() {
        MTRT_RuntimeClient client{nullptr};
        MTRT_Status s = mtrtRuntimeClientCreate(&client);
        THROW_IF_MTRT_ERROR(s);
        return new PyRuntimeClient(client);
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
            int idata;
            if (scalarTypeCode) {
              // dispatch based on explicit code, (we just need to handle i32,
              // i64 and for now, otherwise raise exception). `data` should be
              // py::int_ in the case of integer, py::float_ otherwise Try
              // casting float object to int
              switch (*scalarTypeCode) {
              case MTRT_ScalarTypeCode_f32:
                if (!py::isinstance<py::float_>(data))
                  throw std::runtime_error("Python object must represent floa "
                                           "for a ScalarTypeCode.fp32");
                // Try casting to int
                idata = py::cast<int>(data);
                break;
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
              MTRT_Status s = mtrtRuntimeValueScalarI64Create(idata, &value);
              THROW_IF_MTRT_ERROR(s);

              // Cast from RuntimeValue to ScalarValue and return PyScalarValue.
              return PyScalarValue(mtrtRuntimeValueDynCastToScalar(value));
            }

            // Other path when user provides a Python int object but no type
            // code, we can default to i64
            if (py::isinstance<py::int_>(data)) {
              idata = py::cast<int>(data);
              // Finally create a scalar value ScalarTypeCode::i64 and cast to
              // RuntimeValue.
              MTRT_Status s = mtrtRuntimeValueScalarI64Create(idata, &value);
              THROW_IF_MTRT_ERROR(s);
              // Cast from RuntimeValue to ScalarValue and return PyScalarValue.
              return PyScalarValue(mtrtRuntimeValueDynCastToScalar(value));
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
          py::arg("stream") = py::none())
      .def("external_reference_count",
           [](PyRuntimeClient &self, uintptr_t ptr) {
             int32_t externalRefCount;
             MTRT_Status s =
                 mtrtMemRefReferenceCount(self, ptr, &externalRefCount);
             THROW_IF_MTRT_ERROR(s);
             return externalRefCount;
           })
      .def("is_released_internally", [](PyRuntimeClient &self, uintptr_t ptr) {
        bool isReleasedInternally;
        MTRT_Status s =
            mtrtMemRefIsReleasedInternally(self, ptr, &isReleasedInternally);
        THROW_IF_MTRT_ERROR(s);
        return isReleasedInternally;
      });

  py::class_<PyRuntimeValue>(m, "RuntimeValue", py::module_local())
      .def_property_readonly(MTRT_PYTHON_CAPI_PTR_ATTR,
                             &PyRuntimeValue::getCapsule)
      .def(py::init<>([](int64_t scalar) {
             MTRT_RuntimeValue value;
             MTRT_Status s = mtrtRuntimeValueScalarI64Create(scalar, &value);
             THROW_IF_MTRT_ERROR(s);
             return new PyRuntimeValue(value);
           }),
           py::arg("scalar_int"));

  py::class_<PyRuntimeSessionOptions>(m, "RuntimeSessionOptions",
                                      py::module_local())
      .def(py::init<>([](int32_t numDevices, int32_t deviceId,
                         std::string ncclUuid) -> PyRuntimeSessionOptions * {
             MTRT_RuntimeSessionOptions options;
             MTRT_Status s = mtrtRuntimeSessionOptionsCreate(
                 numDevices, deviceId,
                 MTRT_StringView{ncclUuid.data(), ncclUuid.size()}, &options);
             THROW_IF_MTRT_ERROR(s);
             return new PyRuntimeSessionOptions(options);
           }),
           py::arg("num_devices") = 1, py::arg("device_id") = 0,
           py::arg("nccl_uuid") = py::str(""));

  py::class_<PyRuntimeSession>(m, "RuntimeSession", py::module_local())
      .def(py::init<>([](PyRuntimeSessionOptions &options, PyExecutable &exe) {
             MTRT_RuntimeSession session;
             MTRT_Status s = mtrtRuntimeSessionCreate(options, exe, &session);
             THROW_IF_MTRT_ERROR(s);
             return new PyRuntimeSession(session);
           }),
           py::arg("options"), py::arg("executable"))
      .def(
          "execute_function",
          [](PyRuntimeSession &self, std::string name,
             std::vector<py::object> inArgs, std::vector<py::object> outArgs,
             std::optional<MTRT_Stream> stream) {
            MTRT_StringView nameRef{name.data(), name.size()};

            auto inArgsGeneric = llvm::map_to_vector(inArgs, convertArgType);
            auto outArgsGeneric = llvm::map_to_vector(outArgs, convertArgType);

            MTRT_Status s = mtrtRuntimeSessionExecuteFunction(
                self, nameRef, inArgsGeneric.data(), inArgsGeneric.size(),
                outArgsGeneric.data(), outArgsGeneric.size(),
                stream ? *stream : mtrtStreamGetNull());
            THROW_IF_MTRT_ERROR(s);
          },
          py::arg("name"), py::arg("in_args"), py::arg("out_args"),
          py::arg("stream") = py::none());
}
