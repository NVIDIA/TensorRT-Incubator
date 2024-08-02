//===- PluginRegistry.h -----------------------------------------*- C++ -*-===//
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
/// Declarations for TensorRT plugin registration methods.
///
//===----------------------------------------------------------------------===//
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <NvInfer.h>
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <functional>
#include <memory>

namespace mlir::tensorrt {

/// Register a plugin creator. The `func` should be a callable that returns a
/// unique handle to the creator. This handle will then be inserted into the
/// internal registry (to avoid multiple registrations) and registered with
/// TensorRT's plugin creator registry. This function is thread-safe.
void registerPluginCreator(
    std::function<std::unique_ptr<nvinfer1::IPluginCreator>()> func);

} // namespace mlir::tensorrt
