//===- NvInferPluginUtils.h  ----------------------------------------------===//
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
/// Utilities for loading and manipulating TensorRT plugins.
///
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_MLIR_TENSORRT_DIALECT_UTILS_NVINFERPLUGINUTILS
#define INCLUDE_MLIR_TENSORRT_DIALECT_UTILS_NVINFERPLUGINUTILS
#include <type_traits>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include "NvInferRuntime.h"
#include "NvInferRuntimePlugin.h"
#include "mlir-tensorrt-dialect/Utils/NvInferAdaptor.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::tensorrt {

/// Given type `t` return the corresponding TensorRT::DataType enum value of the
/// elementType of `t` (if it is a `ShapedType`) or `t` itself.
/// If there is no corresponding `nvinfer1::DataType` enum value, this function
/// will cause the program to abort. This is meant to simplify the usage API
/// below in the `buildLayer` dispatch function, so types should be
/// appropriately verified before using.
FailureOr<nvinfer1::DataType> getNvInferDataType(Location loc, Type t);

//===----------------------------------------------------------------------===//
// PluginManager
//===----------------------------------------------------------------------===//

enum class PluginKind {
  V2,
  V3,
};

class PluginInterfaceBase {
public:
  PluginKind getKind() const { return kind; }

  PluginInterfaceBase(PluginKind kind_) : kind(kind_) {}
  virtual ~PluginInterfaceBase() {}

private:
  const PluginKind kind;
};

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
template <typename T>
PluginKind PluginCreatorKindV =
    std::is_same_v<T, nvinfer1::IPluginCreator> ? PluginKind::V2
                                                : PluginKind::V3;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <typename CreatorT>
class PluginCreator : public PluginInterfaceBase {
public:
  PluginCreator(CreatorT *ptr_, bool owning_)
      : PluginInterfaceBase(PluginCreatorKindV<CreatorT>), ptr(ptr_),
        owning(owning_) {}

  ~PluginCreator() {
    if (ptr && owning) {
      delete ptr;
    }
    ptr = nullptr;
  }

  static bool classof(const PluginInterfaceBase *S) {
    return S->getKind() == PluginCreatorKindV<CreatorT>;
  }

  CreatorT *ptr;

private:
  bool owning;
};

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
template <typename T>
constexpr PluginKind PluginKindV =
    std::is_same_v<T, nvinfer1::IPluginV2DynamicExt> ? PluginKind::V2
                                                     : PluginKind::V3;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <typename PluginT>
class Plugin : public PluginInterfaceBase {
public:
  Plugin(PluginT *ptr_)
      : PluginInterfaceBase(PluginKindV<PluginT>), ptr(ptr_) {}

  ~Plugin() {
    if (ptr) {
      if constexpr (PluginKindV<PluginT> == PluginKind::V2) {
        ptr->destroy();
      } else {
        delete ptr;
      }
      ptr = nullptr;
    }
  }

  static bool classof(const PluginInterfaceBase *S) {
    return S->getKind() == PluginKindV<PluginT>;
  }

  PluginT *ptr;
};

class PluginManager {
public:
  /// Attempts to retrieve a plugin creator.
  /// If `dsoPath` is provided, the specified library is loaded prior to
  /// attempting to load the plugin creator.
  /// If `creatorFunc` is provided, it will be used to create the plugin
  /// creator. Otherwise, this method will perform a lookup in the TensorRT
  /// plugin registry.
  FailureOr<PluginInterfaceBase *>
  getPluginCreator(Location loc, StringRef pluginName, StringRef pluginVersion,
                   StringRef pluginNamespace, std::optional<StringRef> dsoPath,
                   std::optional<StringRef> creatorFunc);

  FailureOr<PluginInterfaceBase *>
  getExternalPlugin(Location loc, StringRef pluginName, StringRef pluginVersion,
                    StringRef pluginNamespace,
                    const DictionaryAttr &creatorParams, StringRef layerName,
                    std::optional<StringRef> dsoPath,
                    std::optional<StringRef> creatorFunc);

private:
  template <typename CreatorT>
  FailureOr<PluginInterfaceBase *>
  createPluginFromCreator(Location loc, const DictionaryAttr &creatorParams,
                          StringRef layerName, CreatorT *creator);

private:
  /// Maps plugin names to instantiated plugin creator instances.
  llvm::StringMap<std::unique_ptr<PluginInterfaceBase>> pluginCreators;
  /// Cached creator references that are stored in the TensorRT global creator
  /// registry.
  llvm::StringMap<std::unique_ptr<PluginInterfaceBase>> pluginCreatorViews;
  /// Keeps track of instantiated plugin objects. The lifetime of these objects
  /// (and thus PluginManager) must persist through the lifetime of the
  /// NvInferNetworkEncoder.
  llvm::SmallVector<std::unique_ptr<PluginInterfaceBase>> plugins;
};

/// Construct IR within the single region of `op` representing dynamic shape
/// calculations.
LogicalResult buildPluginShapeRegion(
    Operation *op, PluginInterfaceBase *pluginBase,
    llvm::function_ref<void(OpBuilder &, Location loc, ArrayRef<Value>)>
        buildTerminator);

} // namespace mlir::tensorrt

#endif // INCLUDE_MLIR_TENSORRT_DIALECT_UTILS_NVINFERPLUGINUTILS
