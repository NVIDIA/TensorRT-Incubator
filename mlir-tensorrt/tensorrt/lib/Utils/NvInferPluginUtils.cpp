//===- NvInferPluginUtils.cpp ---------------------------------------------===//
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
/// Implementation of utilities for TensorRT plugins.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-dialect/Utils/NvInferPluginUtils.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTBase.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include <limits>

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
#include "nvinfer/trt_plugin_python.h"
#endif // MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)

#if defined(__GNUC__) || defined(__clang__)
// Ignore deprecated declarations in this file; we use a lot of them to support
// older TRT versions.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace mlir;
using namespace mlir::tensorrt;

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
using TRTIntT = int64_t;
#else
using TRTIntT = int32_t;
#endif

FailureOr<nvinfer1::DataType> mlir::tensorrt::getNvInferDataType(Location loc,
                                                                 Type t) {
  Type elType = mlir::getElementTypeOrSelf(t);
  if (elType.isF32())
    return nvinfer1::DataType::kFLOAT;
  if (elType.isF16())
    return nvinfer1::DataType::kHALF;
  if (elType.isInteger(32))
    return nvinfer1::DataType::kINT32;
  if (isTensorRTInt8Type(elType))
    return nvinfer1::DataType::kINT8;
  if (elType.isInteger(1))
    return nvinfer1::DataType::kBOOL;
  if (elType.isUnsignedInteger(8))
    return nvinfer1::DataType::kUINT8;

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(9, 1, 0)
  if (isa<Float8E4M3FNType>(elType))
    return nvinfer1::DataType::kFP8;
  if (elType.isBF16())
    return nvinfer1::DataType::kBF16;
  if (elType.isInteger(64))
    return nvinfer1::DataType::kINT64;
#endif

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  if (elType.isInteger(4))
    return nvinfer1::DataType::kINT4;
#endif

  return emitError(loc) << "MLIR Type " << t
                        << " can't be converted to TensorRT type";
}

namespace {
/// Manages the serialization of an MLIR DictionaryAttr to a
/// PluginFieldCollection. This class owns the storage for the
/// PluginFieldCollection, so it must remain live until a plugin no longer needs
/// the plugin field collection data (generally just initialization).
class PluginParams {
public:
  PluginParams() = default;

  /// PluginParams is not copyable because the internal pointers held by the
  /// elements in `params` refer to data held in the storage vectors.
  PluginParams(const PluginParams &) = delete;
  PluginParams &operator=(const PluginParams &) = delete;

  /// Instantiate a PluginParams by serializing the `creatorParams` against the
  /// provided schema. This object can then be casted to a
  /// PluginFieldCollection.
  static FailureOr<std::unique_ptr<PluginParams>>
  create(DictionaryAttr creatorParams,
         const nvinfer1::PluginFieldCollection &schema) {
    auto p = std::make_unique<PluginParams>();
    p->params.reserve(creatorParams.size());
    for (const nvinfer1::PluginField &schemaField :
         llvm::make_range(schema.fields, schema.fields + schema.nbFields)) {
      std::string name(schemaField.name);
      std::optional<NamedAttribute> attr = creatorParams.getNamed(name);

      // Not all plugin fields are required, but we have no way to know which
      // ones are, so treat them all as optional.
      if (attr) {
        p->params.emplace_back();
        nvinfer1::PluginField &field = p->params.back();
        field.name = schemaField.name;
        field.type = schemaField.type;

        if (failed(p->serializeAttribute(field.type, attr->getValue(), field)))
          return failure();
      }
    }
    return p;
  }

  operator nvinfer1::PluginFieldCollection() const {
    nvinfer1::PluginFieldCollection result;
    result.nbFields = params.size();
    result.fields = params.data();
    return result;
  }

private:
  const void *addScalar(FloatAttr attr) {
    scalarStorage.push_back(attr.getValue().bitcastToAPInt());
    return reinterpret_cast<const void *>(scalarStorage.back().getRawData());
  }
  const void *addScalar(IntegerAttr attr) {
    scalarStorage.push_back(attr.getValue());
    return reinterpret_cast<const void *>(scalarStorage.back().getRawData());
  }
  FailureOr<const void *> addShape(DenseI64ArrayAttr attr) {
    if (attr.size() > nvinfer1::Dims::MAX_DIMS)
      return emitError(UnknownLoc::get(attr.getContext()))
             << "cannot convert DenseI64ArrayAttr with greater than 8 elements "
                "to an 'nvinfer1::Dims' object";

    dimsStorage.push_back(nvinfer1::Dims{});
    dimsStorage.back().nbDims = attr.size();

    auto const arrayRef = attr.asArrayRef();
    if (std::is_same_v<int32_t, decltype(*dimsStorage.back().d)>) {
      if (!std::all_of(arrayRef.begin(), arrayRef.end(), [](int64_t val) {
            return val >= std::numeric_limits<int32_t>::min() &&
                   val <= std::numeric_limits<int32_t>::max();
          })) {
        return emitError(UnknownLoc::get(attr.getContext()),
                         "All values in shape must be in range of int32");
      }
    }

    llvm::copy(arrayRef, dimsStorage.back().d);
    return reinterpret_cast<const void *>(&dimsStorage.back());
  }
  FailureOr<const void *> addShapes(ArrayAttr attr) {
    const void *start = nullptr;
    for (auto denseArray : attr.getAsRange<DenseI64ArrayAttr>()) {
      FailureOr<const void *> shapePtr = addShape(denseArray);
      if (failed(shapePtr))
        return failure();
      if (!start)
        start = *shapePtr;
    }
    return start;
  }
  const void *addString(StringAttr attr) {
    /// MLIRContext owns the storage here, so nothing to append to storage.
    return reinterpret_cast<const void *>(attr.strref().data());
  }
  const void *addElements(DenseElementsAttr attr) {
    /// MLIRContext owns the storage here, so nothing to append to storage.
    return reinterpret_cast<const void *>(attr.getRawData().data());
  }

  LogicalResult serializeFloatAttribute(Attribute attr,
                                        nvinfer1::PluginField &field,
                                        Type expectedType) {
    auto error = [&]() {
      return emitError(UnknownLoc::get(attr.getContext()))
             << "failed to serialize plugin creator field " << attr
             << ", expected element type " << expectedType;
    };

    if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
      if (floatAttr.getType() != expectedType)
        return error();
      field.data = addScalar(floatAttr);
      field.length = 1;
      return success();
    }
    if (auto els = dyn_cast<DenseElementsAttr>(attr)) {
      if (els.getType().getElementType() != expectedType)
        return error();
      field.data = addElements(els);
      field.length = els.getType().getNumElements();
      return success();
    }
    return error();
  }
  LogicalResult serializeIntAttribute(Attribute attr,
                                      nvinfer1::PluginField &field,
                                      Type expectedType) {
    auto error = [&]() {
      return emitError(UnknownLoc::get(attr.getContext()))
             << "failed to serialize plugin creator field " << attr
             << ", expected element type " << expectedType;
    };

    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      if (intAttr.getType() != expectedType)
        return error();
      field.data = addScalar(intAttr);
      field.length = 1;
      return success();
    }
    if (auto els = dyn_cast<DenseElementsAttr>(attr)) {
      if (els.getType().getElementType() != expectedType)
        return error();
      if (auto elsSplat = dyn_cast<SplatElementsAttr>(attr)) {
        if (!isa<IntegerType, FloatType>(elsSplat.getElementType()))
          return failure();
        // Only handle types width the logical width of at least CHAR_BIT
        // and divisible by CHAR_BIT
        if (elsSplat.getElementType().getIntOrFloatBitWidth() % CHAR_BIT != 0U)
          return failure();
        unsigned const byteWidth = els.getRawData().size();
        splatsStorage.emplace_back(
            llvm::WritableMemoryBuffer::getNewUninitMemBuffer(
                elsSplat.getNumElements() * byteWidth, "",
                llvm::Align(byteWidth)));
        auto bufferData = splatsStorage.back()->getBufferStart();
        if (els.getType().getElementType().isInteger(32))
          std::fill_n(reinterpret_cast<int32_t *>(bufferData),
                      elsSplat.getNumElements(),
                      elsSplat.getSplatValue<int32_t>());

        else if (els.getType().getElementType().isInteger(64))
          std::fill_n(reinterpret_cast<int64_t *>(bufferData),
                      elsSplat.getNumElements(),
                      elsSplat.getSplatValue<int64_t>());

        else if (els.getType().getElementType().isInteger(16))
          std::fill_n(reinterpret_cast<int16_t *>(bufferData),
                      elsSplat.getNumElements(),
                      elsSplat.getSplatValue<int16_t>());

        else if (els.getType().getElementType().isInteger(8))
          std::fill_n(reinterpret_cast<int8_t *>(bufferData),
                      elsSplat.getNumElements(),
                      elsSplat.getSplatValue<int8_t>());
        else
          return failure();

        field.data = bufferData;
        field.length = elsSplat.getNumElements();
        return success();
      }
      field.data = addElements(els);
      field.length = els.getType().getNumElements();
      return success();
    }
    return error();
  }

  LogicalResult serializeAttribute(nvinfer1::PluginFieldType type,
                                   Attribute attr,
                                   nvinfer1::PluginField &field) {
    auto error = [&]() {
      return emitError(UnknownLoc::get(attr.getContext()))
             << "failed to serialize plugin creator field " << attr
             << " for nvinfer1::PluginFieldType=" << static_cast<int>(type);
    };

    OpBuilder b(attr.getContext());
    switch (type) {
    case nvinfer1::PluginFieldType::kFLOAT16:
      return serializeFloatAttribute(attr, field, b.getF16Type());
    case nvinfer1::PluginFieldType::kFLOAT32:
      return serializeFloatAttribute(attr, field, b.getF32Type());
    case nvinfer1::PluginFieldType::kFLOAT64:
      return serializeFloatAttribute(attr, field, b.getF64Type());
    case nvinfer1::PluginFieldType::kINT8:
      return serializeIntAttribute(attr, field, b.getIntegerType(8));
    case nvinfer1::PluginFieldType::kINT16:
      return serializeIntAttribute(attr, field, b.getIntegerType(16));
    case nvinfer1::PluginFieldType::kINT32:
      return serializeIntAttribute(attr, field, b.getIntegerType(32));
    case nvinfer1::PluginFieldType::kCHAR: {
      auto strAttr = dyn_cast<StringAttr>(attr);
      if (!strAttr)
        return error();
      field.data = addString(strAttr);
      field.length = strAttr.strref().size();
      return success();
    }
    case nvinfer1::PluginFieldType::kDIMS: {
      if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
        FailureOr<const void *> ptr = addShapes(arrayAttr);
        if (failed(ptr))
          return failure();
        field.data = *ptr;
        field.length = arrayAttr.size();
        return success();
      }
      if (auto shapeAttr = dyn_cast<DenseI64ArrayAttr>(attr)) {
        FailureOr<const void *> ptr = addShape(shapeAttr);
        if (failed(ptr))
          return failure();
        field.data = *ptr;
        field.length = 1;
        return success();
      }
      return error();
    }
    case nvinfer1::PluginFieldType::kUNKNOWN:
      return error();
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(9, 2, 0)
    case nvinfer1::PluginFieldType::kBF16:
      return serializeFloatAttribute(attr, field, b.getBF16Type());
    case nvinfer1::PluginFieldType::kINT64:
      return serializeIntAttribute(attr, field, b.getI64Type());
    case nvinfer1::PluginFieldType::kFP8:
      return serializeFloatAttribute(attr, field, b.getF64Type());
#endif
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 1, 0)
    case nvinfer1::PluginFieldType::kINT4:
      return serializeIntAttribute(attr, field, b.getIntegerType(4));
#endif
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
    case nvinfer1::PluginFieldType::kFP4:
      return serializeFloatAttribute(attr, field,
                                     Float4E2M1FNType::get(b.getContext()));
#endif
    }

    llvm_unreachable("unhandled plugin field type enumeration value");
  }

private:
  SmallVector<nvinfer1::PluginField> params;
  SmallVector<APInt> scalarStorage;
  SmallVector<nvinfer1::Dims> dimsStorage;
  SmallVector<std::unique_ptr<llvm::WritableMemoryBuffer>> splatsStorage;
};
} // namespace

// The logic below is complicated by the fact that the IPluginCreatorInterface
// base class did not exist before TRT 10.0. After TRT 10.0, IPluginCreator is
// a child of that class.
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
using PluginCreatorInterface = nvinfer1::IPluginCreatorInterface;
#else
using PluginCreatorInterface = nvinfer1::IPluginCreator;
#endif

/// Return the plugin creator based on the interface type.
static std::unique_ptr<PluginInterfaceBase>
makePluginCreatorBase(PluginCreatorInterface *creator, bool owning) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  if (llvm::StringRef(creator->getInterfaceInfo().kind) ==
      "PLUGIN CREATOR_V3ONE") {
    return std::make_unique<PluginCreator<nvinfer1::IPluginCreatorV3One>>(
        static_cast<nvinfer1::IPluginCreatorV3One *>(creator), owning);
  }
#endif
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
  if (llvm::StringRef(creator->getInterfaceInfo().kind) ==
      "PLUGIN CREATOR_V3QUICK") {
    return std::make_unique<PluginCreator<nvinfer1::IPluginCreatorV3Quick>>(
        static_cast<nvinfer1::IPluginCreatorV3Quick *>(creator), owning);
  }
#endif
  return std::make_unique<PluginCreator<nvinfer1::IPluginCreator>>(
      static_cast<nvinfer1::IPluginCreator *>(creator), owning);
}

static PluginCreatorInterface *
getPluginCreatorInterface(nvinfer1::IPluginRegistry *registry, const char *name,
                          const char *version, const char *pluginNamespace) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  return registry->getCreator(name, version, pluginNamespace);
#else
  return registry->getPluginCreator(name, version, pluginNamespace);
#endif
}

//===----------------------------------------------------------------------===//
// PluginManager
//===----------------------------------------------------------------------===//

FailureOr<PluginInterfaceBase *> PluginManager::getPluginCreator(
    Location loc, StringRef name, StringRef version, StringRef pluginNamespace,
    std::optional<StringRef> dsoPath, std::optional<StringRef> creatorFunc) {

  std::string key =
      llvm::formatv("{0}::{1}::{2}", name, version, pluginNamespace);
  if (pluginCreatorViews.contains(key)) {
    return pluginCreatorViews.at(key).get();
  }

  if (pluginCreators.contains(key)) {
    return pluginCreators.at(key).get();
  }

  if (creatorFunc && !dsoPath)
    return emitError(loc)
           << "expected DSO path to be provided for TensorRT Plugin " << name;

  llvm::sys::DynamicLibrary dylibHandle;
  if (dsoPath) {
    std::string errMsg;
    dylibHandle = llvm::sys::DynamicLibrary::getPermanentLibrary(
        dsoPath->str().c_str(), &errMsg);
    if (!dylibHandle.isValid())
      return emitError(loc) << "failed to load TensorRT plugin library ("
                            << *dsoPath << ") due to error: " << errMsg;
  }

  if (creatorFunc) {
    assert(dylibHandle.isValid() && "dynamic lib handle invalid.");
    void *symbolPtr =
        dylibHandle.getAddressOfSymbol(std::string(*creatorFunc).c_str());
    if (!symbolPtr)
      return emitError(loc)
             << "external TensorRT plugin library (" << *dsoPath
             << ") does not contain a symbol with name \"" << *creatorFunc
             << "\"; make sure that the plugin library exports that function "
                "with C-style linkage and function type 'void(*)()";

    PluginCreatorInterface *creator =
        reinterpret_cast<PluginCreatorInterface *(*)()>(symbolPtr)();
    if (!creator)
      return emitError(loc)
             << "failed to create plugin creator for plugin " << name;
    pluginCreators.try_emplace(key, makePluginCreatorBase(creator, true));
    return pluginCreators.at(key).get();
  }

  nvinfer1::IPluginRegistry *registry = getPluginRegistry();
  if (!registry)
    return emitError(loc) << "plugin registry not found";

  PluginCreatorInterface *creator = getPluginCreatorInterface(
      registry, name.str().c_str(), version.str().c_str(),
      pluginNamespace.str().c_str());

  // Creator is not yet registered with plugin registry. Try loading plugin
  // library to register creators.
  if (!creator && dsoPath) {
    assert(dylibHandle.isValid() && "dynamic lib handle invalid.");
    // If a dynamic shared object (DSO) path is provided, handle plugin
    // registration as follows:
    // 1. If plugins are registered via the REGISTER_TENSORRT_PLUGIN
    // interface, simply load the library using dlopen.
    // 2. If the library implements getCreators or getPluginCreators, register
    // all creators from the library into the TensorRT plugin registry.
    void *getCreatorsSym = dylibHandle.getAddressOfSymbol("getCreators");
    if (!getCreatorsSym)
      getCreatorsSym = dylibHandle.getAddressOfSymbol("getPluginCreators");

    if (getCreatorsSym) {
      nvinfer1::IPluginRegistry::PluginLibraryHandle handle =
          registry->loadLibrary(dsoPath->str().c_str());
      if (!handle)
        return emitError(loc)
               << llvm::formatv("failed to load and register a shared "
                                "library of plugins from {0}",
                                dsoPath->str());
      creator = getPluginCreatorInterface(registry, name.str().c_str(),
                                          version.str().c_str(),
                                          pluginNamespace.str().c_str());
    }
  }

  if (!creator) {
    return emitError(loc) << llvm::formatv(
               "failed to get a registered plugin creator from plugin "
               "registry with name {0}, namespace {1}, and version {2}",
               name.str(), pluginNamespace.str(), version.str());
  }

  pluginCreatorViews.try_emplace(key, makePluginCreatorBase(creator, false));
  return pluginCreatorViews.at(key).get();
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

FailureOr<PluginInterfaceBase *> PluginManager::getExternalPlugin(
    Location loc, StringRef name, StringRef version, StringRef pluginNamespace,
    const DictionaryAttr &creatorParams, StringRef layerName,
    std::optional<StringRef> dsoPath, std::optional<StringRef> creatorFunc) {
  FailureOr<PluginInterfaceBase *> creatorBase = getPluginCreator(
      loc, name, version, pluginNamespace, dsoPath, creatorFunc);
  if (failed(creatorBase))
    return failure();

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  auto creatorIface =
      static_cast<PluginCreator<nvinfer1::IPluginCreatorInterface> *>(
          *creatorBase);

  nvinfer1::InterfaceInfo const ifaceInfo =
      creatorIface->ptr->getInterfaceInfo();
  if (llvm::StringRef(ifaceInfo.kind) == "PLUGIN CREATOR_V3ONE") {
    auto creator = static_cast<PluginCreator<nvinfer1::IPluginCreatorV3One> *>(
        *creatorBase);
    return createPluginFromCreator<nvinfer1::IPluginCreatorV3One>(
        loc, creatorParams, layerName, creator->ptr);
  }
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
  if (llvm::StringRef(ifaceInfo.kind) == "PLUGIN CREATOR_V3QUICK") {
    auto creator =
        static_cast<PluginCreator<nvinfer1::IPluginCreatorV3Quick> *>(
            *creatorBase);
    std::string pluginIdString =
        std::string(name) + "::" + std::string(pluginNamespace);
    StringRef pluginId{pluginIdString.data(), pluginIdString.size()};
    return createPluginFromCreator<nvinfer1::IPluginCreatorV3Quick>(
        loc, creatorParams, pluginId, creator->ptr);
  }
#endif // MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
#endif // MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  auto creator =
      static_cast<PluginCreator<nvinfer1::IPluginCreator> *>(*creatorBase);
  return createPluginFromCreator<nvinfer1::IPluginCreator>(
      loc, creatorParams, layerName, creator->ptr);
}

template <typename PluginCreatorT>
FailureOr<PluginInterfaceBase *> PluginManager::createPluginFromCreator(
    Location loc, const DictionaryAttr &creatorParams, StringRef layerName,
    PluginCreatorT *creator) {

  const nvinfer1::PluginFieldCollection *pluginFieldSchema =
      creator->getFieldNames();
  FailureOr<std::unique_ptr<PluginParams>> pluginParamsStorage =
      PluginParams::create(creatorParams, *pluginFieldSchema);
  if (failed(pluginParamsStorage))
    return emitError(loc)
           << "failed to serialize plugin creator parameters into a "
              "PluginFieldCollection";

  nvinfer1::PluginFieldCollection serializedParams(**pluginParamsStorage);

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  // If the plugin creator type is V3One, then we use the below method to create
  // the V3 plugin.
  if constexpr (std::is_same_v<PluginCreatorT, nvinfer1::IPluginCreatorV3One>) {
    nvinfer1::IPluginV3 *plugin =
        creator->createPlugin(std::string(layerName).c_str(), &serializedParams,
                              nvinfer1::TensorRTPhase::kBUILD);
    if (!plugin)
      return emitError(loc) << "failed to create plugin";

    plugins.emplace_back(new Plugin<nvinfer1::IPluginV3>(plugin));
    return plugins.back().get();
  }
#endif

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
  // If the plugin creator type is V3Quick, then we use the below method to
  // create the V3 plugin.
  if constexpr (std::is_same_v<PluginCreatorT,
                               nvinfer1::IPluginCreatorV3Quick>) {

    SmallVector<std::string> nameAndNamespace;
    for (llvm::StringRef s : llvm::split(layerName, "::"))
      nameAndNamespace.push_back(std::string(s));

    if (nameAndNamespace.size() != 2)
      return emitError(loc)
             << "failed to parse PluginCreatorV3Quick namespace and name";

    nvinfer1::IPluginV3 *plugin = creator->createPlugin(
        nameAndNamespace[0].c_str(), nameAndNamespace[1].c_str(),
        &serializedParams, nvinfer1::TensorRTPhase::kBUILD,
        nvinfer1::QuickPluginCreationRequest::kSTRICT_AOT);
    if (!plugin)
      return emitError(loc) << "failed to create plugin";

    plugins.emplace_back(new Plugin<nvinfer1::IPluginV3>(plugin));
    return plugins.back().get();
  }
#endif

  if constexpr (std::is_same_v<PluginCreatorT, nvinfer1::IPluginCreator>) {
    // This is an unsafe cast, but there is no way in the TRT API to
    // determine the actual type of a PluginV2.
    nvinfer1::IPluginV2DynamicExt *plugin =
        reinterpret_cast<nvinfer1::IPluginV2DynamicExt *>(
            creator->createPlugin(layerName.str().c_str(), &serializedParams));
    if (!plugin)
      return emitError(loc) << "failed to create plugin";
    plugins.emplace_back(new Plugin<nvinfer1::IPluginV2DynamicExt>(plugin));
  }

  return plugins.back().get();
}

//===----------------------------------------------------------------------===//
// Plugin DimExpr <-> MLIR interop
//===----------------------------------------------------------------------===//

namespace {

/// Implementation of `VDimensionExpr` that just wraps an MLIR Value. The
/// expression is a constant if it is defined by an `arith.constant`.
class DimExprImpl : public nvinfer1::apiv::VDimensionExpr {
public:
  DimExprImpl(Value value) : value(value) {}

  bool isConstant() const override {
    return value.getDefiningOp<arith::ConstantOp>();
  }

  TRTIntT getConstantValue() const override {
    arith::ConstantOp op = value.getDefiningOp<arith::ConstantOp>();
    // For dynamic dimensions, match TensorRT's behavior.
    if (!op)
      return std::numeric_limits<TRTIntT>::min();
    return cast<IntegerAttr>(op.getValue()).getInt();
  }

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  /// TODO: support size tensors in TensorRT plugins.
  bool isSizeTensor() const override { return false; }
#endif

private:
  Value value{nullptr};
  friend class DimExpr;
};

class DimExpr : public nvinfer1::IDimensionExpr {
public:
  DimExpr(Value value) { this->mImpl = new DimExprImpl(value); }
  ~DimExpr() { delete this->mImpl; }

  operator Value() const {
    auto v = reinterpret_cast<DimExprImpl *>(this->mImpl)->value;
    assert(v != nullptr);
    return v;
  }
};

/// Implements VExprBuilder. Creation of a DimensionExpr is just building
/// the corresponding MLIR op and returning the Value.
class ExprBuilderImpl : public nvinfer1::apiv::VExprBuilder {
public:
  ExprBuilderImpl(SmallVector<std::unique_ptr<DimExpr>> &storage,
                  ImplicitLocOpBuilder &builder)
      : storage(storage), builder(builder) {}

  const nvinfer1::IDimensionExpr *constant(TRTIntT value) override {
    return newExpr(
        builder.create<arith::ConstantOp>(builder.getI64IntegerAttr(value)));
  }

  const nvinfer1::IDimensionExpr *
  operation(nvinfer1::DimensionOperation op,
            const nvinfer1::IDimensionExpr &first,
            const nvinfer1::IDimensionExpr &second) override {

    const auto &lhs = reinterpret_cast<const DimExpr &>(first);
    const auto &rhs = reinterpret_cast<const DimExpr &>(second);

    using nvinfer1::DimensionOperation;
    switch (op) {
    case DimensionOperation::kSUM:
      return newExpr(builder.create<arith::AddIOp>(lhs, rhs));
    case DimensionOperation::kPROD:
      return newExpr(builder.create<arith::MulIOp>(lhs, rhs));
    case DimensionOperation::kMAX:
      return newExpr(builder.create<arith::MaxSIOp>(lhs, rhs));
    case DimensionOperation::kMIN:
      return newExpr(builder.create<arith::MinSIOp>(lhs, rhs));
    case DimensionOperation::kSUB:
      return newExpr(builder.create<arith::SubIOp>(lhs, rhs));
    case DimensionOperation::kEQUAL:
      return newExpr(
          builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, rhs));
    case DimensionOperation::kLESS:
      return newExpr(
          builder.create<arith::CmpIOp>(arith::CmpIPredicate::slt, lhs, rhs));
    case DimensionOperation::kFLOOR_DIV:
      return newExpr(builder.create<arith::FloorDivSIOp>(lhs, rhs));
    case DimensionOperation::kCEIL_DIV:
      return newExpr(builder.create<arith::CeilDivSIOp>(lhs, rhs));
    }
    llvm_unreachable("unhandled dimension operation type");
  }

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  /// TODO: support size tensors
  const nvinfer1::IDimensionExpr *
  declareSizeTensor(int32_t outputIndex, const nvinfer1::IDimensionExpr &opt,
                    const nvinfer1::IDimensionExpr &upper) override {
    llvm_unreachable("size tensor is unsupported");
    return nullptr;
  }
#endif

private:
  template <typename... Args>
  const nvinfer1::IDimensionExpr *newExpr(Args... args) {

    storage.emplace_back(
        std::make_unique<DimExpr>(std::forward<Args>(args)...));
    return storage.back().get();
  }

  SmallVector<std::unique_ptr<DimExpr>> &storage;
  ImplicitLocOpBuilder &builder;

  friend class ExprBuilder;
};

class ExprBuilder : public nvinfer1::IExprBuilder {
public:
  ExprBuilder(ImplicitLocOpBuilder &builder) {
    this->mImpl = new ExprBuilderImpl(storage, builder);
  }

  ~ExprBuilder() { delete this->mImpl; }

  /// Return an expression that wraps the given MLIR Value.
  const nvinfer1::IDimensionExpr *getExpr(Value v) {

    return reinterpret_cast<ExprBuilderImpl *>(mImpl)->newExpr(v);
  }

private:
  SmallVector<std::unique_ptr<DimExpr>> storage;
};

} // namespace

static LogicalResult
buildOutputExpressions(Operation *op, PluginInterfaceBase *pluginBase,
                       ArrayRef<nvinfer1::DimsExprs> inputExprs,
                       ExprBuilder &exprBuilder,
                       SmallVectorImpl<nvinfer1::DimsExprs> &outputExprs) {
#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)
  if (auto plugin = llvm::dyn_cast<Plugin<nvinfer1::IPluginV3>>(pluginBase)) {
    nvinfer1::IPluginCapability *iface = plugin->ptr->getCapabilityInterface(
        nvinfer1::PluginCapabilityType::kBUILD);
    nvinfer1::InterfaceInfo pluginIfaceInfo = iface->getInterfaceInfo();
    if (llvm::StringRef(pluginIfaceInfo.kind) == "PLUGIN_V3ONE_BUILD") {
      auto *buildPluginIface =
          static_cast<nvinfer1::IPluginV3OneBuild *>(iface);
      if (buildPluginIface->getOutputShapes(
              inputExprs.data(), inputExprs.size(), nullptr, 0,
              outputExprs.data(), outputExprs.size(), exprBuilder))
        return failure();

      return success();
    }

#if MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)
    if (llvm::StringRef(pluginIfaceInfo.kind) == "PLUGIN_V3QUICKAOT_BUILD") {
      SmallVector<nvinfer1::DataType> outputTypes(op->getNumResults(),
                                                  nvinfer1::DataType{});
      SmallVector<nvinfer1::DataType> inputTypes(op->getNumOperands());
      SmallVector<int32_t> inputRanks(inputTypes.size());
      for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
        RankedTensorType tensorType = cast<RankedTensorType>(operand.getType());
        FailureOr<nvinfer1::DataType> trtTypeFromMlirTensorType =
            getNvInferDataType(op->getLoc(), tensorType.getElementType());
        if (failed(trtTypeFromMlirTensorType))
          return failure();

        inputTypes[i] = trtTypeFromMlirTensorType.value();
        inputRanks[i] = tensorType.getRank();
      }
      auto *buildPluginIface =
          static_cast<nvinfer1::IPluginV3QuickAOTBuild *>(iface);

      if (buildPluginIface->getOutputDataTypes(
              outputTypes.data(), outputTypes.size(), inputTypes.data(),
              inputRanks.data(), inputTypes.size()))
        return failure();

      if (buildPluginIface->getOutputShapes(
              inputExprs.data(), inputExprs.size(), nullptr, 0,
              outputExprs.data(), outputExprs.size(), exprBuilder))
        return failure();
      return success();
    }
#endif // MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 9, 0)

    return failure();
  }
#endif // MLIR_TRT_COMPILE_TIME_TENSORRT_VERSION_GTE(10, 0, 0)

  auto *pluginV2 =
      static_cast<Plugin<nvinfer1::IPluginV2DynamicExt> *>(pluginBase);
  for (unsigned i = 0, e = outputExprs.size(); i < e; ++i)
    outputExprs[i] = pluginV2->ptr->getOutputDimensions(
        i, inputExprs.data(), inputExprs.size(), exprBuilder);

  return success();
}

LogicalResult tensorrt::buildPluginShapeRegion(
    Operation *op, PluginInterfaceBase *pluginBase,
    llvm::function_ref<void(OpBuilder &, Location loc, ArrayRef<Value>)>
        buildTerminator) {
  ImplicitLocOpBuilder builder(op->getLoc(), op);
  assert(op->getNumRegions() == 1);
  Block &block = op->getRegion(0).emplaceBlock();
  builder.setInsertionPointToStart(&block);
  ExprBuilder exprBuilder(builder);

  ValueRange operands = op->getOperands();
  int64_t numOperands = operands.size();
  SmallVector<nvinfer1::DimsExprs> inputExprs(numOperands);
  for (auto [i, operand] : llvm::enumerate(operands)) {
    RankedTensorType tensorType = cast<RankedTensorType>(operand.getType());
    inputExprs[i].nbDims = tensorType.getRank();
    // Construct expressions for each dimension of this input.
    for (auto [j, extent] : llvm::enumerate(tensorType.getShape())) {
      // We add a block argument for each dimension regardless of whether it
      // is a constant or not.
      BlockArgument arg = block.addArgument(builder.getI64Type(), op->getLoc());
      // If the input dimension is statically known, then just create a
      // constant in the region.
      if (!ShapedType::isDynamic(extent)) {
        inputExprs[i].d[j] = exprBuilder.constant(extent);
        continue;
      }
      inputExprs[i].d[j] = exprBuilder.getExpr(arg);
    }
  }

  SmallVector<nvinfer1::DimsExprs> outputExprs(
      op->getNumResults(), nvinfer1::DimsExprs{0, {nullptr}});
  if (failed(buildOutputExpressions(op, pluginBase, inputExprs, exprBuilder,
                                    outputExprs)))
    return emitError(op->getLoc())
           << "failed to build output shape expressions";

  // Retrieve the results expressions and forward the values to the
  // terminator.
  SmallVector<Value> results;
  for (nvinfer1::DimsExprs &exprs : outputExprs) {
    for (int64_t idx = 0; idx < exprs.nbDims; idx++) {
      assert(exprs.d[idx] != nullptr && "expected valid dims expression");
      results.push_back(*reinterpret_cast<const DimExpr *>(exprs.d[idx]));
    }
  }
  buildTerminator(builder, builder.getLoc(), results);
  return success();
}
