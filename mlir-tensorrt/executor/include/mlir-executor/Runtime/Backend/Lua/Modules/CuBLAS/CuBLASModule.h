//===- CuBLASModule.h --------------------------------------------*- C++-*-===//
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
/// Executor CuBLAS module runtime components.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_RUNTIME_LUARUNTIME_MODULES_CUBLAS_CUBLASMODULE_H
#define MLIR_TENSORRT_RUNTIME_LUARUNTIME_MODULES_CUBLAS_CUBLASMODULE_H

#include "mlir-executor/Runtime/API/API.h"

struct lua_State;

namespace mlirtrt::runtime {

#ifdef MLIR_EXECUTOR_ENABLE_CUBLAS

/// Register various external functions with the given Lua state.
void registerExecutorCuBLASModuleLuaRuntimeMethods(
    lua_State *state, AllocTracker *allocTracker,
    ResourceTracker *resourceTracker);

#endif // MLIR_EXECUTOR_ENABLE_CUBLAS

} // namespace mlirtrt::runtime

#endif // MLIR_TENSORRT_RUNTIME_LUARUNTIME_MODULES_CUBLAS_CUBLASMODULE_H
