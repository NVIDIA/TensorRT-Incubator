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
///
/// Functions/macros that should be used within C++ functions we define that
/// are called by Lua.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_RUNTIME_BACKEND_LUA_LUAERRORHANDLING_H
#define MLIR_TENSORRT_RUNTIME_BACKEND_LUA_LUAERRORHANDLING_H

#define SET_LUA_ERROR_AND_RETURN_IF_CUDART_ERROR(x, lstate, ...)               \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      lua_State *L = lstate;                                                   \
      luaL_error(L, cudaGetErrorString(err));                                  \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  } while (false)

#define SET_LUA_ERROR_IF_CUDART_ERROR(x, lstate)                               \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      lua_State *L = lstate;                                                   \
      luaL_error(L, cudaGetErrorString(err));                                  \
    }                                                                          \
  } while (false)

#define SET_LUA_ERROR_AND_RETURN_IF_CUDA_ERROR(x, lstate, ...)                 \
  do {                                                                         \
    CUresult err = (x);                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      lua_State *L = lstate;                                                   \
      const char *msg = "";                                                    \
      cuGetErrorString(err, &msg);                                             \
      luaL_error(L, msg);                                                      \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  } while (false)

#define SET_LUA_ERROR_IF_CUDA_ERROR(x, lstate)                                 \
  do {                                                                         \
    CUresult err = (x);                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      lua_State *L = lstate;                                                   \
      const char *msg = "";                                                    \
      cuGetErrorString(err, &msg);                                             \
      luaL_error(L, msg);                                                      \
    }                                                                          \
  } while (false)

#define SET_LUA_ERROR_IF_NCCL_ERROR(x, lstate, comm)                           \
  do {                                                                         \
    ncclResult_t err = (x);                                                    \
    if (err != ncclSuccess && err != ncclInProgress) {                         \
      lua_State *L = lstate;                                                   \
      std::string msg = llvm::formatv(                                         \
          "{0}:{1} NCCL error [msg=\"{2}\" ncclGetLastError=\"{3}\"]",         \
          __FILE__, __LINE__, ncclGetErrorString(err),                         \
          comm ? ncclGetLastError(comm) : "");                                 \
      luaL_error(L, msg.c_str());                                              \
    }                                                                          \
  } while (false)

#define SET_LUA_ERROR_IF_MPI_ERROR(x, lstate)                                  \
  do {                                                                         \
    int err = (x);                                                             \
    if (err != MPI_SUCCESS) {                                                  \
      lua_State *L = lstate;                                                   \
      std::string msg = "MPI error: " + std::to_string(err);                   \
      luaL_error(L, msg.c_str());                                              \
    }                                                                          \
  } while (false)

#define SET_LUA_ERROR_IF_ERROR(x, lstate)                                      \
  do {                                                                         \
    if (!x.isOk()) {                                                           \
      lua_State *L = lstate;                                                   \
      luaL_error(L, std::string(x.getString()).c_str());                       \
    }                                                                          \
  } while (false)

#define SET_LUA_ERROR_AND_RETURN_IF_ERROR(x, lstate, ...)                      \
  do {                                                                         \
    if (!x.isOk()) {                                                           \
      lua_State *L = lstate;                                                   \
      luaL_error(L, std::string(x.getString()).c_str());                       \
      return __VA_ARGS__;                                                      \
    }                                                                          \
  } while (false)

#endif // MLIR_TENSORRT_RUNTIME_BACKEND_LUA_LUAERRORHANDLING_H
