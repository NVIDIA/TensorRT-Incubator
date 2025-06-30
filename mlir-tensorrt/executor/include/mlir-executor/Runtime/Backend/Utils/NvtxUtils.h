//===- NvtxUtils.h --------------------------------------------*- C++ -*-===//
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
/// Defines macros to add NVTX tracing to several Lua VM modules.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_RUNTIME_BACKEND_UTILS_NVTXUTILS_H
#define MLIR_TENSORRT_RUNTIME_BACKEND_UTILS_NVTXUTILS_H

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#endif
#include "nvtx3/nvtx3.hpp"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace mlirtrt::runtime {
struct RuntimeNvtxDomain {
  static constexpr char const *name{"MLIR TensorRT Lua Runtime"};
};
using NvtxRange = nvtx3::scoped_range_in<RuntimeNvtxDomain>;

namespace tracing {
constexpr inline nvtx3::rgb RuntimeColor() { return nvtx3::rgb{17, 122, 70}; }
constexpr inline nvtx3::rgb CoreModuleColor() {
  return nvtx3::rgb{188, 76, 58};
}
constexpr inline nvtx3::rgb TensorRTModuleColor() {
  return nvtx3::rgb{146, 132, 7};
}
constexpr inline nvtx3::rgb CudaModuleColor() {
  return nvtx3::rgb{61, 65, 176};
}
} // namespace tracing
} // namespace mlirtrt::runtime

#define ADD_RUNTIME_MODULE_RANGE(funcName)                                     \
  mlirtrt::runtime::NvtxRange r(mlirtrt::runtime::tracing::RuntimeColor(),     \
                                funcName)

#define ADD_CORE_MODULE_RANGE(funcName)                                        \
  mlirtrt::runtime::NvtxRange r(mlirtrt::runtime::tracing::CoreModuleColor(),  \
                                funcName)

#define ADD_TENSORRT_MODULE_RANGE(funcName)                                    \
  mlirtrt::runtime::NvtxRange r(                                               \
      mlirtrt::runtime::tracing::TensorRTModuleColor(), funcName)

#define ADD_CUDA_MODULE_RANGE(funcName)                                        \
  mlirtrt::runtime::NvtxRange r(mlirtrt::runtime::tracing::CudaModuleColor(),  \
                                funcName)

#endif // MLIR_TENSORRT_RUNTIME_BACKEND_UTILS_NVTXUTILS_H
