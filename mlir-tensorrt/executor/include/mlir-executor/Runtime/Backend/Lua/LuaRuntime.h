//===- LuaRuntime.h ---------------------------------------------*- C++ -*-===//
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
/// Declarations for routines that enable Lua code execution.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_RUNTIME_BACKEND_LUA_LUARUNTIME_H
#define MLIR_TENSORRT_RUNTIME_BACKEND_LUA_LUARUNTIME_H

#include "cuda_runtime_api.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Support/Status.h"
#include <string_view>

struct lua_State;

namespace mlirtrt::runtime {
/// Convenience method that loads the given Lua script and then executes the
/// `main` function. It is assumed that `main` takes no arguments and returns an
/// integer result (which is returned if the execution is successful).
/// TODO: this should take a handle to a function for streaming output/errors.
StatusOr<int64_t> runExecutorLuaScript(std::string_view luaScript,
                                       GpuAllocator *allocator);

/// Synchronously run a serialized executor Executable one time. An `Executable`
/// is essentially a Lua script packaged with metadata and serialized constants
/// for things like weights data, etc. This method
/// may be asynchronous if the Lua script is asynchronous. It takes ownership of
/// the executable. It loads all globals into the Lua context and then executes
/// the `main` function of the embedded Lua script. It is assumed that `main`
/// takes no arguments and returns an integer result (which is returned if the
/// execution is successful).
/// TODO: this should take a handle to a function for
/// streaming output/errors.
StatusOr<int64_t>
runExecutorExecutable(std::unique_ptr<Executable> executable,
                      std::unique_ptr<GpuAllocator> allocator);

/// Create an execution state. This will setup a Lua environment and invoke
/// global initialization.
StatusOr<std::unique_ptr<RuntimeSession>>
createRuntimeSessionWithLuaBackend(ExecutableView executable,
                                   std::unique_ptr<GpuAllocator> allocator,
                                   const RuntimeSessionOptions &options);

/// Set the primary stream for the loaded executable to use.
Status setRuntimeSessionCudaStream(RuntimeSession &session,
                                   cudaStream_t stream);

/// Get the primary stream for the loaded executable to use.
cudaStream_t getRuntimeSessionCudaStream(RuntimeSession &session);

/// Execute a named function in the session with the specified input args and
/// output (destination args). Returns any results.
StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
executeFunctionWithLuaBackend(RuntimeSession &session, std::string_view name,
                              llvm::ArrayRef<RuntimeValue *> inputArgs,
                              llvm::ArrayRef<RuntimeValue *> outputArgs,
                              std::optional<cudaStream_t> stream = {});

} // namespace mlirtrt::runtime

#endif // MLIR_TENSORRT_RUNTIME_BACKEND_LUA_LUARUNTIME_H
