//===- NvPtxCompilerUtils.h -------------------------------------*- C++ -*-===//
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
/// Utilities wrapping the NvPtxCompiler library.
///
//===----------------------------------------------------------------------===//
#ifndef RUNTIME_BACKEND_LUA_MODULES_CUDA_NVPTXCOMPILERUTILS_H
#define RUNTIME_BACKEND_LUA_MODULES_CUDA_NVPTXCOMPILERUTILS_H

#include "mlir-executor/Support/Status.h"
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace mlirtrt::runtime {

/// A `CuBinWrapper` is just a serialized cubin object.
struct CuBinWrapper {
  std::vector<int8_t> data;
  CuBinWrapper(std::vector<int8_t> &&data) : data(data) {}
};

/// Compile PTX data to a cubin object.
mlirtrt::StatusOr<std::unique_ptr<CuBinWrapper>>
compilePtxToCuBin(const char *ptxData, size_t len, std::string_view arch);

} // namespace mlirtrt::runtime

#endif // RUNTIME_BACKEND_LUA_MODULES_CUDA_NVPTXCOMPILERUTILS_H
