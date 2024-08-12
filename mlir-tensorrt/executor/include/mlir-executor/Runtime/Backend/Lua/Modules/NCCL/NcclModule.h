//===- NcclModule.h ---------------------------------------------*- C++ -*-===//
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
/// Executor NCCL module runtime components.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_RUNTIME_LUARUNTIME_MODULES_NCCL_NCCLMODULE_H
#define MLIR_TENSORRT_RUNTIME_LUARUNTIME_MODULES_NCCL_NCCLMODULE_H

#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Support/Status.h"
#include <string>

struct lua_State;

namespace mlirtrt::runtime {

#ifdef MLIR_EXECUTOR_ENABLE_NCCL

/// Returns the ncclUniqueId as a string. If the project is not built with NCCL,
/// then this just returns an empty string.
StatusOr<std::string> getCommunicatorUniqueId();

/// Register various external functions with the given Lua state.
void registerExecutorNCCLModuleLuaRuntimeMethods(lua_State *state,
                                                 ResourceTracker *tracker);

/// Registers functions that are dependent on certain parameters like the
/// device number and ncclUniqueId. This is usually called late just before
/// execution.
void registerDeviceDependentNCCLMethods(lua_State *state, int32_t numDevices,
                                        int32_t deviceIdx,
                                        llvm::StringRef ncclUuid);

#else

/// Returns the ncclUniqueId as a string. If the project is not built with NCCL,
/// then this just returns an empty string.
inline static StatusOr<std::string> getCommunicatorUniqueId() {
  return std::string{};
}

#endif // MLIR_EXECUTOR_ENABLE_NCCL

} // namespace mlirtrt::runtime

#endif // MLIR_TENSORRT_RUNTIME_LUARUNTIME_MODULES_NCCL_NCCLMODULE_H
