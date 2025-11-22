//===- FFI.h ----------------------------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Declaration of APIs related to the FFI host API, including registration
/// of plugins, enumeration of plugins and dispatch utilities.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_RUNTIME_FFI_FFI_H
#define MLIR_EXECUTOR_RUNTIME_FFI_FFI_H

#include "mlir-tensorrt-common/Support/Status.h"
#include <memory>
#include <string>
#include <vector>

namespace mtrt {

/// An opaque handle to a dynamically loaded TVM-FFI module.
struct TVMFFILibraryHandle;

/// A wrapper around a loaded TVM-FFI function.
struct TVMFFICallableHandle;

/// A utility that dispatches calls to a loaded TVM-FFI function.
Status invokeTVMFFICallable(TVMFFICallableHandle *callable, uintptr_t stream,
                            uintptr_t argsArrayPtr, int64_t numArgs) noexcept;

/// Plugin registry is responsible for loading plugins and unloading plugins on
/// destruction.
class PluginRegistry {
public:
  PluginRegistry();
  ~PluginRegistry();

  /// Load a TVM-FFI library from a file path.
  StatusOr<TVMFFILibraryHandle *> loadTVMFFILibrary(const std::string &path);

  /// Create a TVM-FFI callable handle from a library name and function name.
  StatusOr<TVMFFICallableHandle *>
  createTVMFFICallable(const std::string &libName, const std::string &funcName);

private:
  std::vector<std::unique_ptr<TVMFFILibraryHandle>> tvmLibRefs;
  std::vector<std::unique_ptr<TVMFFICallableHandle>> tvmFuncRefs;
};

} // namespace mtrt

#endif // MLIR_EXECUTOR_RUNTIME_FFI_FFI_H
