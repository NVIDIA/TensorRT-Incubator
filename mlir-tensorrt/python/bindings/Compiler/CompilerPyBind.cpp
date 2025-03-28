//===- CompilerPyBind.cpp -------------------------------------------------===//
//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
//===----------------------------------------------------------------------===//
///
/// Implementation of a PyBind11 module for the high-level compiler API.
///
//===----------------------------------------------------------------------===//
#include "../CPyBindInterop.h"
#include "../Utils.h"
#include "NvInferRuntime.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"
#include "mlir-executor-c/Common/Common.h"
#include "mlir-executor-c/Support/Status.h"
#include "mlir-executor-c/Target/ExecutorTranslations.h"
#include "mlir-tensorrt-c/Compiler/Compiler.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "pybind11/pybind11.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include <pybind11/attr.h>

#ifdef MLIR_TRT_TARGET_TENSORRT
#include "mlir-tensorrt-dialect/Utils/NvInferAdaptor.h"
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
#include "nvinfer/trt_plugin_python.h"
#endif // MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
#endif

namespace py = pybind11;
using namespace mlirtrt;

namespace {

//===----------------------------------------------------------------------===//
// Python Wrapper Classes
//===----------------------------------------------------------------------===//

/// Python object type wrapper for `MTRT_CompilerClient`.
class PyCompilerClient
    : public PyMTRTWrapper<PyCompilerClient, MTRT_CompilerClient> {
public:
  using PyMTRTWrapper::PyMTRTWrapper;
  DECLARE_WRAPPER_CONSTRUCTORS(PyCompilerClient);

  static constexpr auto kMethodTable = CAPITable<MTRT_CompilerClient>{
      mtrtCompilerClientIsNull, mtrtCompilerClientDestroy};
};

/// Python object type wrapper for `MlirPassManager`.
class PyPassManagerReference
    : public PyMTRTWrapper<PyPassManagerReference, MlirPassManager> {
public:
  using PyMTRTWrapper::PyMTRTWrapper;
  DECLARE_WRAPPER_CONSTRUCTORS(PyPassManagerReference);

  static constexpr auto kMethodTable =
      CAPITable<MlirPassManager>{mtrtPassManagerReferenceIsNull, nullptr};
};
} // namespace

//===----------------------------------------------------------------------===//
// TensorRT Plugin Utilities
// We only support interop with TensorRT plugin objects for TRT >= 10.0.
//===----------------------------------------------------------------------===//
#ifdef MLIR_TRT_TARGET_TENSORRT
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
namespace {
/// A `PyPluginFieldInfo` corresponds to `nvinfer1::PluginField`, which
/// is a uniform array of one or more objects of a particular type.
class PyPluginFieldInfo {
public:
  PyPluginFieldInfo(nvinfer1::PluginFieldType type, int32_t length)
      : type(type), length(length) {}

  /// The type of the field.
  nvinfer1::PluginFieldType type;

  /// The number of elements of type `type` expected by the field.
  int32_t length;
};
} // namespace

/// Queries the global TensorRT plugin registry for a creator for a plugin of
/// the given name, version, and namespace. It then queries the plugin creator
/// for the expected PluginField information.
MTRT_Status getTensorRTPluginFieldSchema(
    std::string name, std::string version, std::string pluginNamespace,
    std::optional<std::string> dsoPath,
    std::unordered_map<std::string, PyPluginFieldInfo> *result) {

  nvinfer1::IPluginRegistry *pluginRegistry = getPluginRegistry();
  assert(pluginRegistry && "invalid plugin registry");

  nvinfer1::IPluginCreatorInterface *creator = pluginRegistry->getCreator(
      name.c_str(), version.c_str(),
      pluginNamespace.c_str() ? pluginNamespace.c_str() : "");

  std::stringstream ss;

  // Creator is not yet registered with plugin registry. Try loading plugin
  // library to register creators.
  if (!creator && dsoPath.has_value()) {
    // If a dynamic shared object (DSO) path is provided, handle plugin
    // registration as follows:
    // 1. If plugins are registered via the REGISTER_TENSORRT_PLUGIN interface,
    // simply load the library using dlopen.
    // 2. If the library implements getCreators or getPluginCreators, register
    // all creators from the library into the TensorRT plugin registry.
    std::string errMsg;
    auto dylibHandle = llvm::sys::DynamicLibrary::getPermanentLibrary(
        dsoPath->c_str(), &errMsg);
    if (!dylibHandle.isValid()) {
      ss << "failed to load TensorRT plugin library (" << *dsoPath
         << ") due to error: " << errMsg;
      return mtrtStatusCreate(MTRT_StatusCode_InvalidArgument,
                              ss.str().c_str());
    }

    void *getCreatorsSym = dylibHandle.getAddressOfSymbol("getCreators");
    if (!getCreatorsSym)
      getCreatorsSym = dylibHandle.getAddressOfSymbol("getPluginCreators");

    if (getCreatorsSym) {
      nvinfer1::IPluginRegistry::PluginLibraryHandle handle =
          pluginRegistry->loadLibrary(dsoPath->c_str());
      if (!handle) {
        ss << "failed to load and register a shared library of plugins from "
              "dso "
              "path: "
           << *dsoPath << "\n";
        return mtrtStatusCreate(MTRT_StatusCode_InternalError,
                                ss.str().c_str());
      }

      creator = pluginRegistry->getCreator(
          name.c_str(), version.c_str(),
          pluginNamespace.c_str() ? pluginNamespace.c_str() : "");
    }
  }

  if (!creator) {
    ss << "failed to get a registered plugin creator from plugin registry "
          "with "
          "plugin name: "
       << name << ", namespace: " << pluginNamespace
       << ", and version: " << version;
    return mtrtStatusCreate(MTRT_StatusCode_InternalError, ss.str().c_str());
  }

  const nvinfer1::PluginFieldCollection *pluginFieldCollection;

  nvinfer1::InterfaceInfo creatorInfo = creator->getInterfaceInfo();
  if (!std::strcmp(creatorInfo.kind, "PLUGIN CREATOR_V3ONE"))
    pluginFieldCollection =
        static_cast<nvinfer1::IPluginCreatorV3One *>(creator)->getFieldNames();
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
  else if (!std::strcmp(creatorInfo.kind, "PLUGIN CREATOR_V3QUICK"))
    pluginFieldCollection =
        static_cast<nvinfer1::IPluginCreatorV3Quick *>(creator)
            ->getFieldNames();
#endif // MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
  else
    pluginFieldCollection =
        static_cast<nvinfer1::IPluginCreator *>(creator)->getFieldNames();

  std::for_each(pluginFieldCollection->fields,
                pluginFieldCollection->fields + pluginFieldCollection->nbFields,
                [&result](nvinfer1::PluginField const pluginField) {
                  result->emplace(
                      pluginField.name,
                      PyPluginFieldInfo{pluginField.type, pluginField.length});
                });

  return mtrtStatusGetOk();
}

static void bindTensorRTPluginAdaptorObjects(py::module m) {
  // Bind a Python-style enum corresponding to nvinfer1::IPluginFieldType
  py::enum_<nvinfer1::PluginFieldType>(m, "PluginFieldType", py::module_local())
      .value("FLOAT16", nvinfer1::PluginFieldType::kFLOAT16)
      .value("FLOAT32", nvinfer1::PluginFieldType::kFLOAT32)
      .value("FLOAT64", nvinfer1::PluginFieldType::kFLOAT64)
      .value("INT8", nvinfer1::PluginFieldType::kINT8)
      .value("INT16", nvinfer1::PluginFieldType::kINT16)
      .value("INT32", nvinfer1::PluginFieldType::kINT32)
      .value("CHAR", nvinfer1::PluginFieldType::kCHAR)
      .value("DIMS", nvinfer1::PluginFieldType::kDIMS)
      .value("UNKNOWN", nvinfer1::PluginFieldType::kUNKNOWN)
      .value("BF16", nvinfer1::PluginFieldType::kBF16)
      .value("INT64", nvinfer1::PluginFieldType::kINT64)
      .value("FP8", nvinfer1::PluginFieldType::kFP8)
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 1, 0)
      .value("INT4", nvinfer1::PluginFieldType::kINT4)
#endif
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
      .value("FP4", nvinfer1::PluginFieldType::kFP4)
#endif
      ;

  py::class_<PyPluginFieldInfo>(m, "PluginFieldInfo", py::module_local())
      .def_readonly("type", &PyPluginFieldInfo::type)
      .def_readonly("length", &PyPluginFieldInfo::length);

  m.def(
      "get_tensorrt_plugin_field_schema",
      [](std::string name, std::string version, std::string plugin_namespace,
         std::optional<std::string> dso_path) {
        MTRT_Status s;
        std::unordered_map<std::string, PyPluginFieldInfo> result;
        s = getTensorRTPluginFieldSchema(name, version, plugin_namespace,
                                         dso_path, &result);
        THROW_IF_MTRT_ERROR(s);
        return result;
      },
      py::arg("name"), py::arg("version"), py::arg("plugin_namespace"),
      py::arg("dso_path"),
      "Queries the global TensorRT plugin registry for a creator for a "
      "plugin of the given name, version, and namespace. It then queries the "
      "plugin creator for the expected PluginField information.");
}
#endif
#endif

PYBIND11_MODULE(_api, m) {

  populateCommonBindingsInModule(m);

  m.def("translate_mlir_to_executable", [](MlirOperation op) {
    MTRT_Executable exe{nullptr};
    MTRT_Status status = translateToRuntimeExecutable(op, &exe);
    THROW_IF_MTRT_ERROR(status);
    return new PyExecutable(exe);
  });

  py::class_<PyCompilerClient>(m, "CompilerClient", py::module_local())
      .def(py::init<>([](MlirContext context) -> PyCompilerClient * {
        MTRT_CompilerClient client;
        MTRT_Status s = mtrtCompilerClientCreate(context, &client);
        THROW_IF_MTRT_ERROR(s);
        return new PyCompilerClient(client);
      }))
      .def(
          "get_compilation_task",
          [](PyCompilerClient &self, const std::string &mnemonic,
             const std::vector<std::string> &args) -> PyPassManagerReference * {
            std::vector<MlirStringRef> refs(args.size());
            for (unsigned i = 0; i < args.size(); i++)
              refs[i] = mlirStringRefCreate(args[i].data(), args[i].size());

            MlirPassManager pm{nullptr};
            MTRT_Status status = mtrtCompilerClientGetCompilationTask(
                self, mlirStringRefCreate(mnemonic.data(), mnemonic.size()),
                refs.data(), refs.size(), &pm);
            THROW_IF_MTRT_ERROR(status);
            return new PyPassManagerReference(pm);
          });

  py::class_<PyPassManagerReference>(m, "PassManagerReference",
                                     py::module_local())
      .def("run", [](PyPassManagerReference &self, MlirOperation op) {
        MlirLogicalResult result = mlirPassManagerRunOnOp(self.get(), op);
        if (mlirLogicalResultIsFailure(result))
          throw MTRTException("failed to run pass pipeline");
      });

#ifdef MLIR_TRT_TARGET_TENSORRT
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  bindTensorRTPluginAdaptorObjects(m);
#endif
#endif
}
