
//===- TensorRTModule.cpp -------------------------------------------------===//
//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of pybind11 based python bindings for TensorRT dialect.
///
//===----------------------------------------------------------------------===//
#include "mlir-c/IR.h"
#include "mlir-tensorrt-dialect-c/TensorRTAttributes.h"
#include "mlir-tensorrt-dialect-c/TensorRTDialect.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#define ADD_PYTHON_ATTRIBUTE_ADAPTOR(attrName)                                 \
  mlir::python::adaptors::mlir_attribute_subclass(m, #attrName "Attr",         \
                                                  tensorrtIs##attrName##Attr)  \
      .def_classmethod(                                                        \
          "get",                                                               \
          [](py::object cls, const std::string &value, MlirContext ctx) {      \
            return cls(tensorrt##attrName##AttrGet(                            \
                ctx, mlirStringRefCreate(value.c_str(), value.size())));       \
          },                                                                   \
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none(),   \
          "Creates a " #attrName "attribute with the given value.")            \
      .def_property_readonly("value", [](MlirAttribute self) {                 \
        return toPyString(tensorrt##attrName##AttrGetValue(self));             \
      });

namespace py = pybind11;

static auto toPyString(MlirStringRef mlirStringRef) {
  return py::str(mlirStringRef.data, mlirStringRef.length);
}

PYBIND11_MODULE(_tensorrt, m) {
  m.doc() = "MLIR TensorRT python extension";

  //
  // Dialects.
  //
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__tensorrt__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  //
  // Passes.
  //

  m.def("register_passes", []() { mlirRegisterTensorRTPasses(); });

  //
  // Attributes
  //
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(ActivationType)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(PaddingMode)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(PoolingType)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(ElementWiseOperation)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(GatherMode)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(UnaryOperation)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(ReduceOperation)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(SliceMode)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(TopKOperation)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(MatrixOperation)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(ResizeMode)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(ResizeCoordinateTransformation)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(ResizeSelector)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(ResizeRoundMode)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(LoopOutput)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(TripLimit)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(FillOperation)
  ADD_PYTHON_ATTRIBUTE_ADAPTOR(ScatterMode)
}
