//===- LuaErrorHandling.h ---------------------------------------*- C++ -*-===//
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
#ifndef MLIR_EXECUTOR_RUNTIME_BACKEND_LUA_LUAEXTENSIONREGISTRY
#define MLIR_EXECUTOR_RUNTIME_BACKEND_LUA_LUAEXTENSIONREGISTRY

#include "mlir-executor/Runtime/API/API.h"

struct lua_State;

namespace mtrt {

struct LuaRuntimeExtensionInitArgs {
  const RuntimeSessionOptions &options;
  lua_State *state;
  PinnedMemoryAllocator *pinnedMemoryAllocator;
  AllocTracker *allocTracker;
  ResourceTracker *resourceTracker;
  PluginRegistry &pluginRegistry;
};

struct LuaRuntimeExtension {
  std::function<void(const LuaRuntimeExtensionInitArgs &args)> populateLuaState;
};

void registerLuaRuntimeExtension(llvm::StringRef name,
                                 LuaRuntimeExtension extensionInfo);

/// Enable the Lua runtime extension modules that are specified in the features
/// of 'options'. If an enabled feature is not present in the registry, then an
/// error is returned.
Status populateRuntimeExtensions(const LuaRuntimeExtensionInitArgs &args);

} // namespace mtrt

#endif // MLIR_EXECUTOR_RUNTIME_BACKEND_LUA_LUAEXTENSIONREGISTRY
