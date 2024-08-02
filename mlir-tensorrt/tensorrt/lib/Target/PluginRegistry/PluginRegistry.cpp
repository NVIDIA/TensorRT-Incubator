//===- PluginRegistration.cpp ---------------------------------------------===//
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
/// Implementation for TensorRT plugin registration methods.
///
//===----------------------------------------------------------------------===//
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "mlir-tensorrt-dialect/Target/PluginRegistry/PluginRegistry.h"
#include "NvInferRuntimePlugin.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <mutex>
#include <string>

using namespace mlir;
using namespace mlir::tensorrt;

static std::string getPluginDebugString(std::string_view pluginNamespace,
                                        std::string_view name,
                                        std::string_view version) {
  std::string result(pluginNamespace);
  result.reserve(result.size() + name.size() + version.size() + 3);
  result.append("::").append(name).append("@").append(version);
  return result;
}

static constexpr std::string_view kMlirTensorRTPluginNamespace =
    "mlir_tensorrt";

namespace {
class PluginCreatorRegistry {
public:
  void registerPluginCreator(
      std::unique_ptr<nvinfer1::IPluginCreator> pluginCreator) {
    pluginCreator->setPluginNamespace(kMlirTensorRTPluginNamespace.data());
    std::string pluginString = getPluginDebugString(
        pluginCreator->getPluginNamespace(), pluginCreator->getPluginName(),
        pluginCreator->getPluginVersion());
    if (registry.find(pluginString) == registry.end()) {
      if (!getPluginRegistry()->registerCreator(
              *pluginCreator, kMlirTensorRTPluginNamespace.data())) {
        llvm::errs() << "Failed to register " << pluginString << "\n";
        return;
      }
      registry[pluginString] = std::move(pluginCreator);
    }
  }

private:
  std::map<std::string, std::unique_ptr<nvinfer1::IPluginCreator>> registry;
};
} // namespace

void mlir::tensorrt::registerPluginCreator(
    std::function<std::unique_ptr<nvinfer1::IPluginCreator>()> func) {
  static std::mutex lock;
  std::lock_guard<std::mutex> g(lock);
  static auto registry = std::make_unique<PluginCreatorRegistry>();
  registry->registerPluginCreator(func());
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
