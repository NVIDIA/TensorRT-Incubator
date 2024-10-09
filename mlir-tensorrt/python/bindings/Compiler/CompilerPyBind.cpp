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
#include "mlir-c/Support.h"
#include "mlir-executor-c/Common/Common.h"
#include "mlir-executor-c/Support/Status.h"
#include "mlir-tensorrt-c/Compiler/Compiler.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "pybind11/pybind11.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include <pybind11/attr.h>

#ifdef MLIR_TRT_TARGET_TENSORRT
#include "mlir-tensorrt-dialect/Utils/NvInferAdaptor.h"
#endif

namespace py = pybind11;
using namespace mlirtrt;

///===----------------------------------------------------------------------===//
// CPython <-> CAPI utilities
//===----------------------------------------------------------------------===//

MTRT_DEFINE_COMPILER_INLINE_PY_CAPSULE_CASTER_FUNCS(
    StableHLOToExecutableOptions)

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

/// Python object type wrapper for `MTRT_StableHLOToExecutableOptions`.
class PyStableHLOToExecutableOptions
    : public PyMTRTWrapper<PyStableHLOToExecutableOptions,
                           MTRT_StableHLOToExecutableOptions> {
public:
  using PyMTRTWrapper::PyMTRTWrapper;
  DECLARE_WRAPPER_CONSTRUCTORS(PyStableHLOToExecutableOptions);

  static constexpr auto kMethodTable =
      CAPITable<MTRT_StableHLOToExecutableOptions>{
          mtrtStableHloToExecutableOptionsIsNull,
          mtrtStableHloToExecutableOptionsDestroy,
          mtrtPythonCapsuleToStableHLOToExecutableOptions,
          mtrtPythonStableHLOToExecutableOptionsToCapsule};

  // We need this member so we can keep the Python callback alive long enough.
  std::function<std::string(MlirOperation)> callback;

  ~PyStableHLOToExecutableOptions() { callback = nullptr; }
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
  if (std::string{creatorInfo.kind} == "PLUGIN CREATOR_V3ONE")
    pluginFieldCollection =
        static_cast<nvinfer1::IPluginCreatorV3One *>(creator)->getFieldNames();
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

  py::class_<PyCompilerClient>(m, "CompilerClient", py::module_local())
      .def(py::init<>([](MlirContext context) -> PyCompilerClient * {
        MTRT_CompilerClient client;
        MTRT_Status s = mtrtCompilerClientCreate(context, &client);
        THROW_IF_MTRT_ERROR(s);
        return new PyCompilerClient(client);
      }));

  py::class_<PyStableHLOToExecutableOptions>(m, "StableHLOToExecutableOptions",
                                             py::module_local())
      .def(py::init<>([](PyCompilerClient &client,
                         const std::vector<std::string> &args)
                          -> PyStableHLOToExecutableOptions * {
             std::vector<MlirStringRef> refs(args.size());
             for (unsigned i = 0; i < args.size(); i++)
               refs[i] = mlirStringRefCreate(args[i].data(), args[i].size());

             MTRT_StableHLOToExecutableOptions options;
             MTRT_Status s = mtrtStableHloToExecutableOptionsCreateFromArgs(
                 client, &options, refs.data(), refs.size());
             THROW_IF_MTRT_ERROR(s);
             return new PyStableHLOToExecutableOptions(options);
           }),
           py::arg("client"), py::arg("args"))
      .def(
          "set_debug_options",
          [](PyStableHLOToExecutableOptions &self, bool enabled,
             std::vector<std::string> debugTypes,
             std::optional<std::string> dumpIrTreeDir,
             std::optional<std::string> dumpTensorRTDir) {
            // The strings are copied by the CAPI call, so we just need to
            // refence the C-strings temporarily.
            std::vector<const char *> literals;
            for (const std::string &str : debugTypes)
              literals.push_back(str.c_str());
            THROW_IF_MTRT_ERROR(mtrtStableHloToExecutableOptionsSetDebugOptions(
                self, enabled, literals.data(), literals.size(),
                dumpIrTreeDir ? dumpIrTreeDir->c_str() : nullptr,
                dumpTensorRTDir ? dumpTensorRTDir->c_str() : nullptr));
          },
          py::arg("enabled"),
          py::arg("debug_types") = std::vector<std::string>{},
          py::arg("dump_ir_tree_dir") = py::none(),
          py::arg("dump_tensorrt_dir") = py::none())

#ifdef MLIR_TRT_TARGET_TENSORRT
      .def(
          "set_tensorrt_translation_metadata_callback",
          [](PyStableHLOToExecutableOptions &self,
             std::function<std::string(MlirOperation)> pyCallback) {
            // Since we're constructing a C callback, our closures must not
            // capture. We can pass in the Python callback via the userData
            // argument.
            auto callback = [](MlirOperation op, MlirStringCallback append,
                               void *appendCtx, void *userDataVoid) {
              auto &pyCallback =
                  *static_cast<std::function<std::string(MlirOperation)> *>(
                      userDataVoid);

              if (!pyCallback)
                return;

              std::string result;
              try {
                result = pyCallback(op);
              } catch (const std::exception &e) {
                llvm::errs() << e.what() << '\n';
              }

              append(MlirStringRef{result.data(), result.size()}, appendCtx);
            };

            self.callback = pyCallback;
            THROW_IF_MTRT_ERROR(
                mtrtStableHloToExecutableOptionsSetTensorRTTranslationMetadataCallback(
                    self, callback, reinterpret_cast<void *>(&self.callback)));
          },
          py::arg("callback"), py::keep_alive<1, 2>{})
#endif
      ;

  m.def(
      "compiler_stablehlo_to_executable",
      [](PyCompilerClient &client, MlirOperation module,
         PyStableHLOToExecutableOptions &options) {
        MTRT_Executable exe{nullptr};
        MTRT_Status status =
            mtrtCompilerStableHLOToExecutable(client, module, options, &exe);
        THROW_IF_MTRT_ERROR(status);
        return new PyExecutable(exe);
      },
      py::arg("client"), py::arg("module"), py::arg("options"));

  m.def(
      "get_stablehlo_program_refined_signature",
      [](PyCompilerClient &client, MlirOperation module, std::string funcName) {
        MlirType signature{nullptr};
        MTRT_StableHLOProgramSignatureRefinementOptions options{nullptr};
        MTRT_Status status =
            mtrtStableHloProgramSignatureRefinementOptionsCreate(
                mtrtStringViewCreate(funcName.c_str(), funcName.size()),
                &options);
        THROW_IF_MTRT_ERROR(status);
        status = mtrtGetStableHloProgramRefinedSignature(client, module,
                                                         options, &signature);
        THROW_IF_MTRT_ERROR(status);
        status = mtrtStableHloProgramSignatureRefinementOptionsDestroy(options);
        THROW_IF_MTRT_ERROR(status);
        return signature;
      },
      py::arg("client"), py::arg("module"), py::arg("func_name"));

#ifdef MLIR_TRT_TARGET_TENSORRT
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  bindTensorRTPluginAdaptorObjects(m);
#endif
#endif
}
