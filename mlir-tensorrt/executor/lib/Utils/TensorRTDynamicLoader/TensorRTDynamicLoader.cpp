//===- TensorRTDynamicLoader.cpp-------------------------------------------===//
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
/// Provides a set of stubs that dynamically load TensorRt symbols from
/// libninvfer.so when needed. This is required because TensorRt tends to
/// interfere with LLVM tooling setup if it is linked normally.
///
//===----------------------------------------------------------------------===//
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif // defined(__clang__)
#include "NvInferVersion.h"
#include <NvInfer.h>
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif // defined(__clang__)

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

extern "C" {

#define LOAD_LIB_OR_RETURN(varName, errorReturnValue, libName)                 \
  do {                                                                         \
    std::string errorMsg;                                                      \
    varName = llvm::sys::DynamicLibrary::getLibrary(libName, &errorMsg);       \
    if (!varName.isValid()) {                                                  \
      llvm::errs() << "failed to load libnvinfer.so: " << errorMsg << "\n";    \
      return errorReturnValue;                                                 \
    }                                                                          \
  } while (false)

#define LOAD_NVINFER_LIB_OR_RETURN(varName, errorReturnValue)                  \
  LOAD_LIB_OR_RETURN(varName, errorReturnValue,                                \
                     "libnvinfer.so." STR(NV_TENSORRT_MAJOR))
#define LOAD_NVINFER_LIB_OR_RETURN_NULLPTR(varName)                            \
  LOAD_NVINFER_LIB_OR_RETURN(varName, nullptr)

void *createInferBuilder_INTERNAL(void *logger, int version) noexcept {
  using SymbolPtrType = void *(*)(void *, int);
  llvm::sys::DynamicLibrary lib;
  LOAD_NVINFER_LIB_OR_RETURN_NULLPTR(lib);
  static SymbolPtrType funcPtr = reinterpret_cast<SymbolPtrType>(
      lib.getAddressOfSymbol("createInferBuilder_INTERNAL"));
  if (!funcPtr) {
    llvm::errs() << "failed to load symbol createInferBuilder_INTERNAL\n";
    return nullptr;
  }
  return funcPtr(logger, version);
}

void *createInferRefitter_INTERNAL(void *engine, void *logger,
                                   int version) noexcept {
  using SymbolPtrType = void *(*)(void *, void *, int);
  llvm::sys::DynamicLibrary lib;
  LOAD_NVINFER_LIB_OR_RETURN_NULLPTR(lib);
  static SymbolPtrType funcPtr = reinterpret_cast<SymbolPtrType>(
      lib.getAddressOfSymbol("createInferRefitter_INTERNAL"));
  if (!funcPtr) {
    llvm::errs() << "failed to load symbol createInferRefitter_INTERNAL\n";
    return nullptr;
  }
  return funcPtr(engine, logger, version);
}

void *createInferRuntime_INTERNAL(void *logger, int version) noexcept {
  using SymbolPtrType = void *(*)(void *, int);
  llvm::sys::DynamicLibrary lib;
  LOAD_NVINFER_LIB_OR_RETURN_NULLPTR(lib);
  static SymbolPtrType funcPtr = reinterpret_cast<SymbolPtrType>(
      lib.getAddressOfSymbol("createInferRuntime_INTERNAL"));
  if (!funcPtr) {
    llvm::errs() << "failed to load symbol createInferRuntime_INTERNAL\n";
    return nullptr;
  }
  return funcPtr(logger, version);
}

nvinfer1::ILogger *getLogger() noexcept {
  using SymbolPtrType = nvinfer1::ILogger *(*)();
  llvm::sys::DynamicLibrary lib;
  LOAD_NVINFER_LIB_OR_RETURN_NULLPTR(lib);
  static SymbolPtrType funcPtr =
      reinterpret_cast<SymbolPtrType>(lib.getAddressOfSymbol("getLogger"));
  if (!funcPtr) {
    llvm::errs() << "failed to load symbol getLogger\n";
    return nullptr;
  }
  return funcPtr();
}

int getInferLibVersion() noexcept {
  using SymbolPtrType = int (*)();
  llvm::sys::DynamicLibrary lib;
  LOAD_NVINFER_LIB_OR_RETURN(lib, 0);
  static SymbolPtrType funcPtr = reinterpret_cast<SymbolPtrType>(
      lib.getAddressOfSymbol("getInferLibVersion"));
  if (!funcPtr) {
    llvm::errs() << "failed to load symbol getInferLibVersion\n";
    return 0;
  }
  return funcPtr();
}

int getInferLibBuildVersion() noexcept {
  using SymbolPtrType = int (*)();
  llvm::sys::DynamicLibrary lib;
  LOAD_NVINFER_LIB_OR_RETURN(lib, 0);
  static SymbolPtrType funcPtr = reinterpret_cast<SymbolPtrType>(
      lib.getAddressOfSymbol("getInferLibBuildVersion"));
  if (!funcPtr) {
    llvm::errs() << "failed to load symbol getInferLibBuildVersion\n";
    return 0;
  }
  return funcPtr();
}

nvinfer1::IPluginRegistry *getPluginRegistry() noexcept {
  using SymbolPtrType = nvinfer1::IPluginRegistry *(*)();
  llvm::sys::DynamicLibrary lib;
  LOAD_NVINFER_LIB_OR_RETURN_NULLPTR(lib);
  static SymbolPtrType funcPtr = reinterpret_cast<SymbolPtrType>(
      lib.getAddressOfSymbol("getPluginRegistry"));
  if (!funcPtr) {
    llvm::errs() << "failed to load symbol getPluginRegistry\n";
    return nullptr;
  }
  return funcPtr();
}
}
