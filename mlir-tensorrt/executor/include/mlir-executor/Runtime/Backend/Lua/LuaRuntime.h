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

#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Lua/SolAdaptor.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include <functional>
#include <string_view>

struct lua_State;

namespace mtrt {

/// Implementation of the LuaRuntimeSession.
class LuaRuntimeSession : public RuntimeSession {
public:
  ~LuaRuntimeSession();

  /// Type of function for callbacks that register extra Lua modules.
  using LuaModuleRegistrationFunc =
      std::function<void(lua_State *, AllocTracker *, ResourceTracker *)>;

  /// Create a new LuaRuntimeSession using the provided options and executable.
  /// The optional `registerExtraLuaFunctions` allows registered additional
  /// backend modules besides the builtin ones (everything supported by the
  /// build settings is immediately registered).
  /// This will setup a Lua environment and invoke
  /// global initialization.
  /// TODO: add capabilities options to 'options' so that only modules
  /// specifically required are registered.
  static StatusOr<std::unique_ptr<LuaRuntimeSession>>
  create(RuntimeSessionOptions options, ExecutableView executable,
         LuaModuleRegistrationFunc registerExtraLuaFunctions = {});

  /// Return a reference to the Lua state. Note that `sol::state` or any other
  /// modification to Lua state is not thread-safe, see
  /// https://sol2.readthedocs.io/en/latest/threading.html.
  lua_State *getLuaState();

  /// Set the primary stream for the loaded executable to use.
  Status setCudaStream(CudaStream stream);

  /// Get the primary stream for the loaded executable to use.
  CudaStream getCudaStream();

private:
  LuaRuntimeSession(RuntimeSessionOptions options, ExecutableView executable);

  class Impl;
  std::unique_ptr<Impl> impl;
};

/// Convenience method that loads the given Lua script and then executes the
/// `main` function. It is assumed that `main` takes no arguments and returns an
/// integer result (which is returned if the execution is successful).
/// TODO: this should take a handle to a function for streaming output/errors.
StatusOr<int64_t> runExecutorLuaScript(
    RuntimeSessionOptions options, std::string_view luaScript,
    LuaRuntimeSession::LuaModuleRegistrationFunc registerExtraLuaFuncs = {});

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
StatusOr<int64_t> runExecutorExecutable(
    RuntimeSessionOptions options, std::unique_ptr<Executable> executable,
    LuaRuntimeSession::LuaModuleRegistrationFunc registerExtraLuaFuncs = {});

/// Execute a named function in the session with the specified input args and
/// output (destination args). Returns optional results supporting both DPS and
/// non-DPS style calling convention.
/// `client` argument is required in case the function return atleast one memref
/// value.
StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
executeFunctionWithLuaBackend(LuaRuntimeSession &session, std::string_view name,
                              llvm::ArrayRef<RuntimeValue *> inputArgs,
                              llvm::ArrayRef<RuntimeValue *> outputArgs,
                              std::optional<CudaStream> stream = {},
                              std::optional<RuntimeClient *> client = {});

} // namespace mtrt

#endif // MLIR_TENSORRT_RUNTIME_BACKEND_LUA_LUARUNTIME_H
