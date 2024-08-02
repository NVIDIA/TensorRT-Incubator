//===- CommonRuntime.h ------------------------------------------*- C++ -*-===//
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
/// Internal utilities common to any runtime implementation (not just Lua
/// runtime).
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_RUNTIME_COMMON_COMMONRUNTIME_H
#define MLIR_TENSORRT_RUNTIME_COMMON_COMMONRUNTIME_H

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "mlir-executor/Support/Status.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <iostream>

namespace mlirtrt::runtime {

/// Stream a CUDA driver error to an ostream.
std::ostream &operator<<(std::ostream &es, CUresult result);
/// Stream a CUDA runtime error to an ostream.
std::ostream &operator<<(std::ostream &es, cudaError_t error);

struct CudaStreamInfo {
  cudaStream_t stream{nullptr};
  bool isView{false};
};

template <typename T>
struct PointerWrapper {

  PointerWrapper(uintptr_t ptr) : ptr(ptr) {}
  PointerWrapper(T ptr) : ptr(reinterpret_cast<uintptr_t>(ptr)) {}

  operator T() { return reinterpret_cast<T>(ptr); }
  operator uintptr_t() { return ptr; }

  uintptr_t ptr;
};

struct CudaStreamPtr : public PointerWrapper<cudaStream_t> {
  using PointerWrapper<cudaStream_t>::PointerWrapper;
  static StatusOr<CudaStreamPtr> create(ResourceTracker &tracker);
};

struct CudaEventPtr : public PointerWrapper<cudaEvent_t> {
  using PointerWrapper::PointerWrapper;
  static StatusOr<CudaEventPtr> create(ResourceTracker &tracker);
};

struct CudaModulePtr : public PointerWrapper<CUmodule> {
  using PointerWrapper::PointerWrapper;
  static StatusOr<CudaModulePtr>
  create(ResourceTracker &tracker, llvm::StringRef ptx, llvm::StringRef arch);
};

struct CudaFunctionPtr : public PointerWrapper<CUfunction> {
  using PointerWrapper::PointerWrapper;
  static StatusOr<CudaFunctionPtr>
  create(ResourceTracker &tracker, CudaModulePtr module, llvm::StringRef name);
};

//===----------------------------------------------------------------------===//
// Copy utilities
//===----------------------------------------------------------------------===//

void executeStridedCopy(
    int64_t elemSize, uintptr_t src, int64_t srcOffset,
    const std::vector<int64_t> &srcShape, std::vector<int64_t> &srcStrides,
    uintptr_t dst, int64_t dstOffset, const std::vector<int64_t> &dstShape,
    std::vector<int64_t> &dstStrides,
    std::function<void(void *dst, void *src, size_t size)> memcpyFunc);

/// Execute a strided copy where the strides and offsets are given in bytes.
void executeStridedByteCopy(
    uintptr_t src, int64_t srcOffsetBytes, const std::vector<int64_t> &srcShape,
    const std::vector<int64_t> &srcByteStrides, uintptr_t dst,
    int64_t dstOffsetBytes, const std::vector<int64_t> &dstShape,
    const std::vector<int64_t> &dstByteStrides, size_t elemSizeBytes,
    std::function<void(void *dst, void *src, size_t size)> memcpyFunc);

} // namespace mlirtrt::runtime

#endif // MLIR_TENSORRT_RUNTIME_COMMON_COMMONRUNTIME_H
