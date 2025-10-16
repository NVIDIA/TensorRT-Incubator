//===- LuaRuntimeCompat.h -------------------------------------------------===//
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
/// Compatibility layer for converting between MTRT Runtime values and Lua
/// objects. Supports multiple ABI versions for backward/forward compatibility.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_EXECUTOR_RUNTIME_BACKEND_LUA_LUARUNTIMECOMPAT
#define MLIR_EXECUTOR_RUNTIME_BACKEND_LUA_LUARUNTIMECOMPAT

#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Lua/SolAdaptor.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include <memory>

namespace mtrt {

// Forward declarations
class RuntimeValue;
class LuaRuntimeSession;

/// Convert a RuntimeValue (MemRef or Scalar) to a Lua object based on ABI
/// version.
///
/// @param value The RuntimeValue to convert (must be MemRef or Scalar)
/// @param lua The Lua state view
/// @param abiVersion The ABI version to use for conversion (default: 0)
/// @return A sol::object representing the value, or an error Status
StatusOr<sol::object> mtrtBoxValueToLua(RuntimeValue *value,
                                        sol::state_view &lua,
                                        uint32_t abiVersion = 0);

/// Convert a Lua object back to a RuntimeValue based on ABI version.
///
/// @param obj The Lua object to convert
/// @param type The expected type of the result
/// @param session The runtime session (for accessing client, allocator, etc.)
/// @param abiVersion The ABI version to use for conversion (default: 0)
/// @return A unique_ptr to RuntimeValue, or an error Status
StatusOr<std::unique_ptr<RuntimeValue>>
luaUnboxToMTRT(const sol::object &obj, const TypeUnionView &type,
               LuaRuntimeSession &session, uint32_t abiVersion = 0);

/// Invoke a Lua function and return the results.
StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
invokeLuaFunction(LuaRuntimeSession &session, FunctionView func,
                  llvm::ArrayRef<RuntimeValue *> args,
                  llvm::ArrayRef<RuntimeValue *> outArgs,
                  uint32_t abiVersion = 0);

} // namespace mtrt

#endif // MLIR_EXECUTOR_RUNTIME_BACKEND_LUA_LUARUNTIMECOMPAT
