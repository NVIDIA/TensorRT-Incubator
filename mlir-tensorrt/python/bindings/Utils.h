//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Internal helpers for python bindings of the Compiler and Runtime pybind
/// modules that interface with MLIR-TensorRT's C API.
///
/// The design of MLIR-TensorRT's C API and the Python bindings under
/// RuntimePyBind.cpp and CompilerPyBind.cpp is inspired by the MLIR C API and
/// associated Python bindings. The high-level overview is that all the C API
/// objects (e.g. `MTRT_SomeName` structs) are simple wrappers around an opaque
/// `void*`. The py bindings construct special Python types that wrap
/// these structs (and thus also the opaque pointer), and the Python runtime
/// will reference-count the objects created and call the appropriate
/// destruction methods (set by us to the apropriate C API function) when they
/// go out of scope.
///
/// In the CompilerPyBind.cpp and RuntimePyBind.cpp modules, you will see
/// many `Py[ObjName]` classes that derive from `PyMTRTWrapper`. These all
/// correspond 1-1 with `MTRT_[ObjName]` structs in the C API.
///
/// Each of the `Py[ObjName]` classes defines a constexpr table of functions
/// `kMethodTable` that specifies which C API functions to call for the wrapper
/// to check for nullity and to deallocate the resources held by the MTRT
/// struct. It may (optionally) also provide methods for converting the `MTRT_`
/// struct to and from a Python capsule object. For classes that implement this
/// capsule conversion, py will recognize that `MTRT_[ObjName]` arguments
/// in other functions can be converted to a Python object. Therefore, objects
/// like `MTRT_Executable` which are wrapped by module-local `PyExecutable`
/// objects in separate packages like the Executable and Runtime packages, can
/// be shared across the boundary even though they are technically different
/// Python types.
///
//===----------------------------------------------------------------------===//
#ifndef BINDINGS_UTILS
#define BINDINGS_UTILS

#include "CPyBindInterop.h"
#include "mlir-executor-c/Common/Common.h"
#include "mlir-executor-c/Support/Status.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "llvm/ADT/Twine.h"
#include <memory>
#include <pybind11/attr.h>
#include <sstream>
#include <string>
#include <string_view>

namespace py = pybind11;

namespace mlirtrt {

class MTRTException : public std::exception {
public:
  explicit MTRTException(const std::string &msg) : msg(msg) {}
  MTRTException(const char *str, size_t len) : msg(str, len) {}

  virtual const char *what() const throw() override { return msg.c_str(); }

private:
  std::string msg;
};

#define THROW_IF_MTRT_ERROR(status)                                            \
  do {                                                                         \
    MTRT_Status err = (status);                                                \
    if (!mtrtStatusIsOk(err)) {                                                \
      ::mlirtrt::PyMTRTCError e(err);                                          \
      throw ::mlirtrt::MTRTException(e.getMessage());                          \
    }                                                                          \
  } while (false)

/// Python wrapper around MTRT_Status.
class PyMTRTCError {
public:
  PyMTRTCError(MTRT_Status status) : status(status) {}
  ~PyMTRTCError() {
    if (mtrtStatusIsNull(status))
      return;
    mtrtStatusDestroy(status);
  }

  std::string getMessage() const {
    const char *msg = "";
    if (!mtrtStatusIsOk(status))
      mtrtStatusGetMessage(status, &msg);
    return std::string(msg);
  }

  MTRT_Status status;
};

//===----------------------------------------------------------------------===//
// CAPITable
//===----------------------------------------------------------------------===//

/// A CAPITable is a constexpr-constructable aggregate of function pointers
/// for CAPI object `T`. The function pointers describe which functions to use
/// for common operations such as checking nullity, destruction, conversion
/// to/from Python capsule, and so on. This helps us move most boilerplate for
/// PyBind type wrappers to a single template class `PyMTRTWrapper`.
template <typename T>
class CAPITable {
public:
  constexpr CAPITable(bool (*isNull)(T), MTRT_Status (*destroy)(T),
                      T (*capsuleToCApi)(PyObject *) = nullptr,
                      PyObject *(*cApiToCapsule)(T) = nullptr)
      : isNull(isNull), destroy(destroy), cApiToCapsule(cApiToCapsule),
        capsuleToCApi(capsuleToCApi) {}

  bool (*isNull)(T);
  MTRT_Status (*destroy)(T);
  PyObject *(*cApiToCapsule)(T);
  T (*capsuleToCApi)(PyObject *);
};

//===----------------------------------------------------------------------===//
// PyMTRTWrapper
//===----------------------------------------------------------------------===//

#define DECLARE_WRAPPER_CONSTRUCTORS(Derived)                                  \
  Derived(const Derived &) = delete;                                           \
  Derived &operator=(const Derived &) = delete;                                \
  Derived(Derived &&other) noexcept : PyMTRTWrapper(std::move(other)) {}

/// CRTP Wrapper class for classes that wrap `MTRT_*` C API types for generation
/// of Python bindings.
template <typename Derived, typename CType>
class PyMTRTWrapper {
public:
  using Base = PyMTRTWrapper<Derived, CType>;

  PyMTRTWrapper(CType obj) : obj(obj) {}
  PyMTRTWrapper(PyMTRTWrapper &&other) noexcept : obj(other.obj) {
    other.release();
  }
  PyMTRTWrapper(const PyMTRTWrapper &) = delete;
  PyMTRTWrapper &operator=(const PyMTRTWrapper &) = delete;
  ~PyMTRTWrapper() {
    if (cFuncTable.isNull(obj))
      return;
    // Invoke destruction function if present. Some objects are pure views and
    // therefore don't have a destruction function.
    if (cFuncTable.destroy != nullptr)
      cFuncTable.destroy(obj);
  }

  CType get() { return obj; }
  operator CType() { return obj; }

  void release() { obj.ptr = nullptr; }

  py::object getCapsule() {
    if constexpr (cFuncTable.cApiToCapsule == nullptr) {
      throw py::value_error("object cannot be converted to opaque capsule");
    } else {
      return py::reinterpret_steal<py::object>(cFuncTable.cApiToCapsule(get()));
    }
  }

  static py::object createFromCapsule(py::object capsule) {
    if constexpr (cFuncTable.capsuleToCApi == nullptr) {
      throw py::value_error("object cannot be converted from opaque capsule");
    } else {
      CType cObj = cFuncTable.capsuleToCApi(capsule.ptr());
      return py::cast(Derived(cObj), py::return_value_policy::move);
    }
  }

protected:
  constexpr static CAPITable<CType> cFuncTable = Derived::kMethodTable;
  CType obj;
};

/// CRTP Wrapper class for classes that wrap `MTRT_*` C API types for generation
/// of Python bindings.
template <typename Derived, typename CType>
class PySharedMTRTWrapper {
public:
  using Base = PySharedMTRTWrapper<Derived, CType>;
  PySharedMTRTWrapper(CType obj) : obj(std::make_shared<CType>(obj)) {}

  CType get() { return *obj; }
  operator CType() { return *obj; }

  py::object getCapsule() {
    if constexpr (cFuncTable.cApiToCapsule == nullptr) {
      throw py::value_error("object cannot be converted to opaque capsule");
    } else {
      return py::reinterpret_steal<py::object>(cFuncTable.cApiToCapsule(get()));
    }
  }

  static py::object createFromCapsule(py::object capsule) {
    if constexpr (cFuncTable.capsuleToCApi == nullptr) {
      throw py::value_error("boject cannot be converted from opaque capsule");
    } else {
      CType cObj = cFuncTable.capsuleToCApi(capsule.ptr());
      return py::cast(Derived(cObj), py::return_value_policy::move);
    }
  }

protected:
  constexpr static CAPITable<CType> cFuncTable = Derived::kMethodTable;
  std::shared_ptr<CType> obj;
};

/// This is a simple wrapper around std::stringstream for interfacing with
/// the MTRT C-API functions that require `MTRT_PrintCallBackInfo` for
/// printing/formatting purposes.
class StringStreamAdaptor {
public:
  StringStreamAdaptor() = default;

  /// Forward to the arguments to the `std::stringstream` stream operator.
  template <typename T>
  auto &operator<<(T &&arg) {
    ss << std::forward<T>(arg);
    return *this;
  }

  /// Return the MTRT callback information. This passes a self-reference to the
  /// user data since we need a raw C function pointer for the callback.
  MTRT_PrintCallbackInfo getCallback() {
    auto callback = +[](MTRT_StringView part, void *userData) {
      StringStreamAdaptor *printAccum =
          reinterpret_cast<StringStreamAdaptor *>(userData);
      *printAccum << std::string_view(part.data, part.length);
    };
    return MTRT_PrintCallbackInfo{callback, this};
  }

  /// Returns a new C++ STL string from the current state of the stringstream.
  std::string str() const { return ss.str(); }

private:
  std::stringstream ss;
};
} // namespace mlirtrt

//===----------------------------------------------------------------------===//
// Shared Bindings
//===----------------------------------------------------------------------===//
MTRT_DEFINE_COMPILER_INLINE_PY_CAPSULE_CASTER_FUNCS(Executable)
MTRT_DEFINE_COMPILER_INLINE_PY_CAPSULE_CASTER_FUNCS(Type)
MTRT_DEFINE_COMPILER_INLINE_PY_CAPSULE_CASTER_FUNCS(Bounds)

namespace mlirtrt {
/// Python type wrapper for `MTRT_Executable`. This declaration is provided in
/// the header here so that it can be repurposed accross multiple modules
/// (compiler, runtime).
class PyExecutable : public PyMTRTWrapper<PyExecutable, MTRT_Executable> {
public:
  using PyMTRTWrapper::PyMTRTWrapper;
  DECLARE_WRAPPER_CONSTRUCTORS(PyExecutable);

  static constexpr auto kMethodTable = CAPITable<MTRT_Executable>{
      mtrtExecutableIsNull, mtrtExecutableDestroy,
      mtrtPythonCapsuleToExecutable, mtrtPythonExecutableToCapsule};

  MTRT_FunctionSignature getFunctionSignature(std::string name) {
    return mtrtGetFunctionSignature(*this, name.c_str());
  }
};

/// Python type wrapper for `MTRT_FunctionSignature`. This declaration is
/// provided in the header here so that it can be repurposed accross multiple
/// modules (compiler, runtime).
class PyFunctionSignature
    : public PyMTRTWrapper<PyFunctionSignature, MTRT_FunctionSignature> {
public:
  using PyMTRTWrapper::PyMTRTWrapper;
  DECLARE_WRAPPER_CONSTRUCTORS(PyFunctionSignature);

  static constexpr auto kMethodTable =
      CAPITable<MTRT_FunctionSignature>{mtrtFunctionSignatureIsNull, nullptr};
};

class PyBounds : public PyMTRTWrapper<PyBounds, MTRT_Bounds> {
public:
  using Base::Base;

  static constexpr auto kMethodTable = CAPITable<MTRT_Bounds>{
      mtrtBoundsIsNull, mtrtBoundsDestroy, mtrtPythonCapsuleToBounds,
      mtrtPythonBoundsToCapsule};
};

class PyType : public PySharedMTRTWrapper<PyType, MTRT_Type> {
public:
  using Base::Base;

  static constexpr auto kMethodTable =
      CAPITable<MTRT_Type>{mtrtTypeIsNull, mtrtTypeDestroy,
                           mtrtPythonCapsuleToType, mtrtPythonTypeToCapsule};

  static void bind(py::module &m) {
    py::class_<PyType> cls(m, "Type", py::module_local());
    cls.def(py::init<PyType &>(), py::arg("cast_from_type"));
  }
};

/// CRTP base classes for Python attributes that subclass Type and should
/// be castable from it (i.e. via something like ScalarType(type)).
template <typename DerivedTy>
class PyConcreteType : public PyType {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = py::class_<DerivedTy, PyType>;
  using IsAFunctionTy = bool (*)(MTRT_Type);

  PyConcreteType() = default;
  PyConcreteType(MTRT_Type attr) : PyType(attr) {}
  PyConcreteType(PyType &orig) : PyConcreteType(castFrom(orig)) {}

  static MTRT_Type castFrom(PyType &orig) {
    if (!DerivedTy::isaFunction(orig)) {
      auto origRepr = py::repr(py::cast(orig)).cast<std::string>();
      throw py::value_error((llvm::Twine("Cannot cast Type to ") +
                             DerivedTy::pyClassName + " (from " + origRepr +
                             ")")
                                .str());
    }
    return orig.get();
  }

  static void bind(py::module &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName, py::module_local());
    // Binds the constructor `PyConcereteType(PyType&)` as the Python
    // class constructor. Derived types must use static method to override other
    // construction functions.
    cls.def(py::init<PyType &>(), py::arg("cast_from_type"));
    cls.def_static(
        "isinstance",
        [](PyType &otherAttr) -> bool {
          return DerivedTy::isaFunction(otherAttr);
        },
        py::arg("other"));
    cls.def("__repr__", [](DerivedTy &self) {
      return std::string(DerivedTy::pyClassName);
    });

    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

class PyScalarType : public PyConcreteType<PyScalarType> {
public:
  using PyConcreteType::PyConcreteType;

  static constexpr IsAFunctionTy isaFunction = mtrtTypeIsaScalarType;
  static constexpr const char *pyClassName = "ScalarType";

  static std::string dunderStr(PyScalarType &self) { return ""; }

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MTRT_ScalarTypeCode code) -> PyScalarType {
          MTRT_Type result{nullptr};
          MTRT_Status s = mtrtScalarTypeCreate(code, &result);
          THROW_IF_MTRT_ERROR(s);
          assert(mtrtTypeIsaScalarType(result));
          return result;
        },
        py::arg("type_code"));
    c.def(
        "__str__",
        [](PyScalarType &self) {
          assert(!mtrtTypeIsNull(self.get()) && "expected valid type");
          MTRT_ScalarType scalarType = mtrtTypeGetScalarType(self);
          MTRT_ScalarTypeCode code = mtrtScalarTypeGetCode(scalarType);
          return (llvm::Twine(pyClassName) + "(" +
                  mtrtScalarTypeCodeGetString(code) + ")")
              .str();
        },
        "returns string representation of the type");
  }
};

class PyMemRefType : public PyConcreteType<PyMemRefType> {
public:
  using PyConcreteType::PyConcreteType;

  static constexpr IsAFunctionTy isaFunction = mtrtTypeIsaMemRefType;
  static constexpr const char *pyClassName = "MemRefType";

  static std::string dunderStr(PyMemRefType &self) { return ""; }

  static void bindDerived(ClassTy &c) {
    // TODO: Add a separate creation method which allows setting strides.
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, MTRT_ScalarTypeCode elementType,
           MTRT_PointerType addressSpace) -> PyMemRefType {
          MTRT_Type result{nullptr};
          MTRT_Status s = mtrtMemRefTypeCreate(
              shape.size(), shape.data(), elementType, addressSpace, &result);
          THROW_IF_MTRT_ERROR(s);
          assert(mtrtTypeIsaMemRefType(result));
          return result;
        },
        py::arg("shape"), py::arg("elementType"), py::arg("addressSpace"),
        "construct a memref type");
    c.def(
        "__str__",
        [](PyMemRefType &self) {
          assert(!mtrtTypeIsNull(self) && "expected valid type");
          // TODO: Add how to a memref type is printed.
          return (llvm::Twine(pyClassName) + "()").str();
        },
        "returns string representation of memref type");
    c.def_property_readonly("shape", [](PyMemRefType &self) {
      MTRT_MemRefTypeInfo info;
      MTRT_Status s = mtrtMemRefTypeGetInfo(self, &info);
      THROW_IF_MTRT_ERROR(s);
      return std::vector<int64_t>(info.shape, info.shape + info.rank);
    });
    c.def_property_readonly("strides", [](PyMemRefType &self) {
      MTRT_MemRefTypeInfo info;
      MTRT_Status s = mtrtMemRefTypeGetInfo(self, &info);
      THROW_IF_MTRT_ERROR(s);
      return std::vector<int64_t>(info.strides, info.strides + info.rank);
    });
    c.def_property_readonly(MTRT_PYTHON_CAPI_PTR_ATTR,
                            &PyMemRefType::getCapsule);
    c.def_property_readonly("dtype", [](PyMemRefType &self) {
      MTRT_MemRefTypeInfo info;
      MTRT_Status s = mtrtMemRefTypeGetInfo(self, &info);
      THROW_IF_MTRT_ERROR(s);
      return info.elementType;
    });
    c.def_property_readonly("address_space", [](PyMemRefType &self) {
      MTRT_MemRefTypeInfo info;
      MTRT_Status s = mtrtMemRefTypeGetInfo(self, &info);
      THROW_IF_MTRT_ERROR(s);
      return info.addressSpace;
    });
  }
};

static inline void populateExecutableBindingInModule(py::module &m) {
  py::class_<PyExecutable> executable(m, "Executable", py::module_local());
  executable.def(py::init<>([](std::string buffer) -> PyExecutable * {
                   MTRT_Executable executable;
                   MTRT_StringView bufferStr =
                       mtrtStringViewCreate(buffer.data(), buffer.size());
                   MTRT_Status s = mtrtExecutableCreate(bufferStr, &executable);
                   THROW_IF_MTRT_ERROR(s);
                   return new PyExecutable(executable);
                 }),
                 py::arg("buffer"),
                 "constructs an executable from a bytes buffer.");
  executable.def(
      "get_signature",
      [](PyExecutable &self, std::string name) {
        return PyFunctionSignature(self.getFunctionSignature(name));
      },
      py::keep_alive<0,
                     1>()); // Ensure that PyFunctionSignature is kept
                            // alive as long as PyExecutable is alive.
  executable.def(
      "serialize",
      [](PyExecutable &self) -> std::optional<py::bytes> {
        MTRT_StringView buffer;
        MTRT_Status s = mtrtExecutableGetStorageView(self, &buffer, nullptr);
        THROW_IF_MTRT_ERROR(s);
        return py::bytes(buffer.data, buffer.length);
      },
      "returns serialized executable in `bytes`");
}

static inline void populateFunctionBindingInModule(py::module &m) {
  py::class_<PyFunctionSignature> signature(m, "PyFunctionSignature",
                                            py::module_local());
  signature.def(
      "__str__",
      [](PyFunctionSignature &self) {
        assert(!mtrtFunctionSignatureIsNull(self) && "expected valid ptr");
        StringStreamAdaptor ss;
        ss << "FunctionSignature(";
        MTRT_Status s = mtrtFunctionSignatureGetString(self, ss.getCallback());
        THROW_IF_MTRT_ERROR(s);
        ss << ")";
        return ss.str();
      },
      "returns string representation of function signature type");
  signature.def("get_num_args", [](PyFunctionSignature &self) {
    int64_t numArgs = 0;
    MTRT_Status s = mtrtFunctionSignatureGetNumArgs(self, &numArgs);
    THROW_IF_MTRT_ERROR(s);
    return numArgs;
  });
  signature.def("get_num_results", [](PyFunctionSignature &self) {
    int64_t numResults = 0;
    MTRT_Status s = mtrtFunctionSignatureGetNumResults(self, &numResults);
    THROW_IF_MTRT_ERROR(s);
    return numResults;
  });
  signature.def("get_num_input_args", [](PyFunctionSignature &self) {
    int64_t numInputArgs = 0;
    MTRT_Status s = mtrtFunctionSignatureGetNumInputArgs(self, &numInputArgs);
    THROW_IF_MTRT_ERROR(s);
    return numInputArgs;
  });
  signature.def("get_num_output_args", [](PyFunctionSignature &self) {
    int64_t numOutputArgs = 0;
    MTRT_Status s = mtrtFunctionSignatureGetNumOutputArgs(self, &numOutputArgs);
    THROW_IF_MTRT_ERROR(s);
    return numOutputArgs;
  });
  signature.def("get_arg", [](PyFunctionSignature &self, int index) {
    MTRT_Type type;
    MTRT_Status s = mtrtFunctionSignatureGetArg(self, index, &type);
    THROW_IF_MTRT_ERROR(s);
    return PyType(type);
  });
  signature.def("get_result", [](PyFunctionSignature &self, int index) {
    MTRT_Type type;
    MTRT_Status s = mtrtFunctionSignatureGetResult(self, index, &type);
    THROW_IF_MTRT_ERROR(s);
    return PyType(type);
  });
  signature.def("get_num_arg_bounds", [](PyFunctionSignature &self) {
    int64_t numArgBounds = 0;
    MTRT_Status s = mtrtFunctionSignatureGetNumArgBounds(self, &numArgBounds);
    THROW_IF_MTRT_ERROR(s);
    return numArgBounds;
  });
  signature.def("get_num_res_bounds", [](PyFunctionSignature &self) {
    int64_t numResultBounds = 0;
    MTRT_Status s =
        mtrtFunctionSignatureGetNumResBounds(self, &numResultBounds);
    THROW_IF_MTRT_ERROR(s);
    return numResultBounds;
  });
  signature.def("get_arg_bound", [](PyFunctionSignature &self, int index) {
    MTRT_Bounds bounds;
    MTRT_Status s = mtrtFunctionSignatureGetArgBound(self, index, &bounds);
    THROW_IF_MTRT_ERROR(s);
    return PyBounds(bounds);
  });
  signature.def("get_res_bound", [](PyFunctionSignature &self, int index) {
    MTRT_Bounds bounds;
    MTRT_Status s = mtrtFunctionSignatureGetResultBound(self, index, &bounds);
    THROW_IF_MTRT_ERROR(s);
    return PyBounds(bounds);
  });
  signature.def(
      "get_shape_func_name",
      [](PyFunctionSignature &self) -> std::optional<std::string> {
        MTRT_StringView name;
        MTRT_Status s = mtrtFunctionSignatureGetShapeFuncName(self, &name);
        THROW_IF_MTRT_ERROR(s);
        if (name.length == 0)
          return std::nullopt;
        return std::string(name.data, name.length);
      },
      "returns the name of the MLIR-TensorRT function in the same executable "
      "that computes the result shapes from "
      "the input shapes if available, otherwise it runs None");
}

/// This function declares a `PyExecutable` on the given Python module object.
/// This type corresponds to MTRT_Executable. Both the `mlir_tensorrt.runtime`
/// and `mlir_tensorrt.compiler` declare their own module-local "Executable"
/// types. They are effectively the same implementation (constructed through
/// this function), although Python will see them as distinct types.
/// Nevertheless, one can pass them back and forth from compiler to runtime
/// package and vice-versa since we provide the appropriate py opaque
/// caster utilities in the `CPyBindInterop.h` header.
static inline void populateCommonBindingsInModule(py::module &m) {
  py::enum_<MTRT_ScalarTypeCode>(m, "ScalarTypeCode", py::module_local())
      .value("f8e4m3fn", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_f8e4m3fn)
      .value("f16", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_f16)
      .value("bf16", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_bf16)
      .value("f32", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_f32)
      .value("f64", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_f64)
      .value("i1", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_i1)
      .value("i4", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_i4)
      .value("i8", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_i8)
      .value("ui8", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_ui8)
      .value("i16", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_i16)
      .value("i32", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_i32)
      .value("i64", MTRT_ScalarTypeCode::MTRT_ScalarTypeCode_i64)
      .export_values();

  py::enum_<MTRT_PointerType>(m, "PointerType", py::module_local())
      .value("host", MTRT_PointerType::MTRT_PointerType_host)
      .value("pinned_host", MTRT_PointerType::MTRT_PointerType_pinned_host)
      .value("device", MTRT_PointerType::MTRT_PointerType_device)
      .value("unified", MTRT_PointerType::MTRT_PointerType_unified)
      .value("unknown", MTRT_PointerType::MTRT_PointerType_unknown)
      .export_values();

  // Creating Type bindings.
  PyType::bind(m);
  PyScalarType::bind(m);
  PyMemRefType::bind(m);

  py::class_<PyBounds> bounds(m, "PyBounds", py::module_local());
  bounds.def("min", [](PyBounds &self) {
    MTRT_ArrayRefI64 minBounds = mtrtArrayRefI64GetEmpty();
    MTRT_Status s = mtrtBoundsGetSize(self, &minBounds);
    THROW_IF_MTRT_ERROR(s);
    std::vector<int64_t> bounds(minBounds.size, 0);
    if (minBounds.size == 0)
      return bounds;
    minBounds.ptr = bounds.data();
    s = mtrtBoundsGetMin(self, &minBounds);
    THROW_IF_MTRT_ERROR(s);
    return bounds;
  });
  bounds.def("max", [](PyBounds &self) {
    MTRT_ArrayRefI64 maxBounds = mtrtArrayRefI64GetEmpty();
    MTRT_Status s = mtrtBoundsGetSize(self, &maxBounds);
    THROW_IF_MTRT_ERROR(s);
    std::vector<int64_t> bounds(maxBounds.size, 0);
    if (maxBounds.size == 0)
      return bounds;
    maxBounds.ptr = bounds.data();
    s = mtrtBoundsGetMax(self, &maxBounds);
    THROW_IF_MTRT_ERROR(s);
    return bounds;
  });

  populateExecutableBindingInModule(m);
  populateFunctionBindingInModule(m);
}

} // namespace mlirtrt

#endif // BINDINGS_UTILS
