//===- NvPtxCompilerUtils.cpp  --------------------------------------------===//
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
/// Implementation of utilities that wrap the [NvPtxCompiler
/// library](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html).
///
/// Note that we use NvPtxCompiler library because it is convenient, but we
/// could eliminate the dependency by relying on similar functionality in the
/// CUDA driver API to JIT compile PTX modules to CUbin files.
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/Backend/Common/NvPtxCompilerUtils.h"

#include "mlir-executor/Runtime/Backend/Common/Support.h"
#include "nvPTXCompiler.h"

namespace rt = mlirtrt::runtime;

#define NVPTXCOMPILER_SAFE_CALL(x)                                             \
  do {                                                                         \
    nvPTXCompileResult result = x;                                             \
    if (result != NVPTXCOMPILE_SUCCESS) {                                      \
      fprintf(stderr, "error: %s failed with error code %d\n", #x, result);    \
      return nullptr;                                                          \
    }                                                                          \
  } while (0)

std::unique_ptr<rt::CuBinWrapper>
rt::compilePtxToCuBin(const char *ptxData, size_t len, std::string_view arch) {
  nvPTXCompilerHandle compiler = nullptr;
  auto releaseCompiler =
      make_scope_exit([&]() { nvPTXCompilerDestroy(&compiler); });

  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(&compiler, len, ptxData));

  unsigned minorVer, majorVer;
  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetVersion(&majorVer, &minorVer));
  MTRT_DBGF("libptxcompiler version : %d.%d\n", majorVer, minorVer);

  std::string target = "--gpu-name=" + std::string(arch);
  std::vector<char const *> compileOptions = {target.c_str(), "--verbose"};
  auto status = nvPTXCompilerCompile(
      compiler, static_cast<int32_t>(compileOptions.size()),
      compileOptions.data());
  if (status != NVPTXCOMPILE_SUCCESS) {
    size_t errorSize;
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLogSize(compiler, &errorSize));
    if (errorSize != 0) {
      std::vector<char> errorLog(errorSize + 1);
      NVPTXCOMPILER_SAFE_CALL(
          nvPTXCompilerGetErrorLog(compiler, errorLog.data()));
      std::string_view logStr(errorLog.data(), errorLog.size());
      std::cerr << "NvPtxCompiler error log: " << logStr << std::endl;
    }
    return nullptr;
  }

  size_t elfSize;
  NVPTXCOMPILER_SAFE_CALL(
      nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));

  std::vector<int8_t> elfBuffer(elfSize);
  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(
      compiler, reinterpret_cast<void *>(elfBuffer.data())));

  return std::make_unique<CuBinWrapper>(std::move(elfBuffer));
}
