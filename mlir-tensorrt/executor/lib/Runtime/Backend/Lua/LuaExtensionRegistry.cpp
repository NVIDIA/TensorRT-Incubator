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
#include "mlir-executor/Runtime/Support/Support.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mtrt;
using namespace mtrt;

static llvm::ManagedStatic<llvm::StringMap<LuaRuntimeExtension>>
    extensionRegistry;

void mtrt::registerLuaRuntimeExtension(llvm::StringRef name,
                                       LuaRuntimeExtension extensionInfo) {
  (*extensionRegistry)[name] = std::move(extensionInfo);
}

Status mtrt::populateRuntimeExtensions(
    const RuntimeSessionOptions &options, lua_State *state,
    PinnedMemoryAllocator *pinnedMemoryAllocator, AllocTracker *allocTracker,
    ResourceTracker *resourceTracker) {
  for (const auto &[key, ext] : *extensionRegistry) {
    if (options.isFeatureEnabled(key)) {
      MTRT_DBG("Enabling Lua runtime module: {0}", key);
      ext.populateLuaState(options, state, pinnedMemoryAllocator, allocTracker,
                           resourceTracker);
      continue;
    }
    MTRT_DBG("Disabling Lua runtime module: {0}", key);
  }

  // Check for features that are enabled but not supported by the runtime.
  for (const auto &feature : options.getEnabledFeatures()) {
    if (!extensionRegistry->contains(feature.getKey())) {
      return getInvalidArgStatus(
          "feature {0} is enabled but not supported by the runtime",
          feature.getKey());
    }
  }
  return getOkStatus();
}
