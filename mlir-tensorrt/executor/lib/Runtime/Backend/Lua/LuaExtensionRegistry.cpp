//===- LuaExtensionRegistry.cpp
//--------------------------------------------===//
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
/// Registry for Lua runtime extensions.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensionRegistry.h"
#include "mlir-executor/Runtime/API/API.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlirtrt;
using namespace mlirtrt::runtime;

static llvm::ManagedStatic<llvm::StringMap<LuaRuntimeExtension>>
    extensionRegistry;

void runtime::registerLuaRuntimeExtension(llvm::StringRef name,
                                          LuaRuntimeExtension extensionInfo) {
  (*extensionRegistry)[name] = std::move(extensionInfo);
}

void runtime::populateRuntimeExtensions(
    const RuntimeSessionOptions &options, lua_State *state,
    PinnedMemoryAllocator *pinnedMemoryAllocator, AllocTracker *allocTracker,
    ResourceTracker *resourceTracker) {
  for (const auto &[key, ext] : *extensionRegistry)
    ext.populateLuaState(options, state, pinnedMemoryAllocator, allocTracker,
                         resourceTracker);
}
