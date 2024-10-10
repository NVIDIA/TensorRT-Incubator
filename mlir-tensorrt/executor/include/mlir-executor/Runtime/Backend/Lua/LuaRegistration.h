//===- LuaRegistration.h ----------------------------------------*- C++ -*-===//
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
/// Registration for the Lua runtime methods.
///
//===----------------------------------------------------------------------===//

#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Support/Allocators.h"
#include <cstdint>
#include <string_view>

struct lua_State;

namespace mlirtrt::runtime {
/// Register various external functions with the given Lua state using a
/// directly specified device number, total device count, and a pre-determined
/// NCCL uuid.
void registerLuaRuntimeMethods(lua_State *state,
                               const RuntimeSessionOptions &options,
                               PinnedMemoryAllocator *pinnedMemoryAllocator,
                               AllocTracker *allocTracker,
                               ResourceTracker *resourceTracker, OutputAllocatorTracker *outputAllocatorTracker);

} // namespace mlirtrt::runtime
