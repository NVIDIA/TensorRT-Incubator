//===- TensorRTModule.h -----------------------------------------*- C++ -*-===//
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
/// Declarations for the TensorRT runtime module of the Lua executor backend.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_RUNTIME_BACKEND_LUA_MODULES_TENSORRT_TENSORRTMODULE_H
#define MLIR_TENSORRT_RUNTIME_BACKEND_LUA_MODULES_TENSORRT_TENSORRTMODULE_H

#include "mlir-executor/Support/Allocators.h"

struct lua_State;

namespace mlirtrt::runtime {

class AllocTracker;
class ResourceTracker;

/// Register functions that implement the Executor TensorRT module in the given
/// Lua state.
void registerExecutorTensorRTModuleLuaRuntimeMethods(
    lua_State *luaState, PinnedMemoryAllocator *pinnedMemoryAllocator,
    AllocTracker *allocTracker, ResourceTracker *resourceTracker, OutputAllocatorTracker *outputAllocatorTracker);

} // namespace mlirtrt::runtime

#endif // MLIR_TENSORRT_RUNTIME_BACKEND_LUA_MODULES_TENSORRT_TENSORRTMODULE_H
