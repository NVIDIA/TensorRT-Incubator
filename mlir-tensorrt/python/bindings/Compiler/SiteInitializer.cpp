//===- SiteInitializer.cpp  -----------------------------------------------===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// This library produces a PyBind11 module that is embedded with the compiler
/// python package, `mlir_tensorrt.compiler`. When it is imported, the
/// `register_dialects` method will automatically be invoked.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-c/Compiler/Registration/RegisterAllDialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#define REGISTER_DIALECT(name)                                                 \
  MlirDialectHandle name##_dialect = mlirGetDialectHandle__##name##__();       \
  mlirDialectHandleInsertDialect(name##_dialect, registry)

PYBIND11_MODULE(_site_initialize_0, m) {
  m.doc() = "MLIR all MLIR-TensorRT related dialects and passes";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirTensorRTRegisterAllDialects(registry);
  });

  mlirTensorRTRegisterAllPasses();
}
