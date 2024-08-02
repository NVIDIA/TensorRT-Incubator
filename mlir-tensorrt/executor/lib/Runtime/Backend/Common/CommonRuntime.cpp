//===- CommonRuntime.cpp  -------------------------------------------------===//
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
#include "mlir-executor/Runtime/Backend/Common/CommonRuntime.h"

#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/Backend/Common/NvPtxCompilerUtils.h"
#include "mlir-executor/Support/Status.h"
#include "llvm/ADT/StringRef.h"

using namespace mlirtrt;
namespace mrt = mlirtrt::runtime;
using namespace mrt;

std::ostream &mrt::operator<<(std::ostream &es, CUresult result) {
  const char *name = nullptr;
  cuGetErrorName(result, &name);
  es << (name != nullptr ? std::string_view(name) : std::string_view("UNKNOWN"))
     << " = ";
  cuGetErrorString(result, &name);
  es << (name != nullptr ? name : "unknown CUresult type");
  return es;
}

std::ostream &mrt::operator<<(std::ostream &es, cudaError_t error) {
  const char *name = cudaGetErrorName(error);
  es << (name != nullptr ? std::string_view(name) : std::string_view("UNKNOWN"))
     << " = ";
  cudaGetErrorString(error);
  es << (name != nullptr ? name : "unknown cudaError_t type");
  return es;
}

StatusOr<CudaStreamPtr> CudaStreamPtr::create(ResourceTracker &tracker) {
  cudaStream_t stream;
  RETURN_ERROR_IF_CUDART_ERROR(cudaStreamCreate(&stream));
  MTRT_DBGF("created cuda stream %lu", reinterpret_cast<uintptr_t>(stream));

  tracker.track(reinterpret_cast<uintptr_t>(stream), [](uintptr_t ptr) {
    if (ptr) {
      MTRT_DBGF("freeing cuda stream 0x%lx", ptr);
      cudaStreamDestroy(reinterpret_cast<cudaStream_t>(ptr));
    }
  });

  return CudaStreamPtr(stream);
}

StatusOr<CudaEventPtr> CudaEventPtr::create(ResourceTracker &tracker) {
  cudaEvent_t event;
  RETURN_ERROR_IF_CUDART_ERROR(cudaEventCreate(&event));
  MTRT_DBGF("created event 0x%lx", reinterpret_cast<uintptr_t>(event));
  tracker.track(reinterpret_cast<uintptr_t>(event), [](uintptr_t ptr) {
    if (ptr) {
      MTRT_DBGF("freeing cuda event 0x%lx", ptr);
      cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ptr));
    }
  });
  return CudaEventPtr(event);
}

StatusOr<CudaModulePtr> CudaModulePtr::create(ResourceTracker &tracker,
                                              llvm::StringRef ptxData,
                                              llvm::StringRef arch) {
  // JIT compile the PTX.
  assert(!ptxData.empty() && "expected ptx data with positive string length");
  std::unique_ptr<CuBinWrapper> cubin =
      compilePtxToCuBin(ptxData.data(), ptxData.size(), arch);
  if (cubin == nullptr)
    return getInternalErrorStatus("failed to load PTX to cubin");

  CUmodule module{nullptr};
  CUresult result = cuModuleLoadDataEx(
      &module, reinterpret_cast<const void *>(cubin->data.data()), 0, 0, 0);
  if (result != CUDA_SUCCESS) {
    const char *msg = "unknown error";
    cuGetErrorString(result, &msg);
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to load serialized CUDA module: ", msg);
  }
  tracker.track(reinterpret_cast<uintptr_t>(module), [](uintptr_t obj) {
    if (obj) {
      MTRT_DBGF("Unloading CUDA module: 0x%lx", obj);
      cuModuleUnload(reinterpret_cast<CUmodule>(obj));
    }
  });
  return CudaModulePtr(module);
}

StatusOr<CudaFunctionPtr> CudaFunctionPtr::create(ResourceTracker &tracker,
                                                  CudaModulePtr module,
                                                  llvm::StringRef name) {
  CUfunction func;
  CUresult result = cuModuleGetFunction(&func, module, name.str().c_str());
  if (result != CUDA_SUCCESS) {
    const char *msg = "unknown error";
    cuGetErrorString(result, &msg);
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to load function from CUDA module: ", msg);
  }

  return CudaFunctionPtr(func);
}

void mrt::executeStridedCopy(
    int64_t elemSize, uintptr_t src, int64_t srcOffset,
    const std::vector<int64_t> &srcShape, std::vector<int64_t> &srcStrides,
    uintptr_t dst, int64_t dstOffset, const std::vector<int64_t> &dstShape,
    std::vector<int64_t> &dstStrides,
    std::function<void(void *dst, void *src, size_t size)> memcpyFunc) {
  // Handle edge case of empty source tensor.
  if (std::all_of(srcShape.begin(), srcShape.end(),
                  [](int64_t dimSize) { return dimSize == 0; }))
    return;

  char *srcPtr = reinterpret_cast<char *>(src + srcOffset * elemSize);
  char *dstPtr = reinterpret_cast<char *>(dst + dstOffset * elemSize);

  // Handle edge case of rank-0 tensor.
  int64_t rank = srcShape.size();
  if (rank == 0) {
    memcpyFunc(dstPtr, srcPtr, elemSize);
    return;
  }

  std::vector<int64_t> indices(rank, 0);

  // Initialize index and scale strides.
  for (int64_t i = 0; i < rank; ++i) {
    srcStrides[i] *= elemSize;
    dstStrides[i] *= elemSize;
  }

  // The logic below comes from the original MLIR CRunner implementation of
  // strided memref copy. It could be further optimized if any of the
  // dimensions have unit stride.
  int64_t readIndex = 0, writeIndex = 0;
  while (true) {
    // Copy one element.
    memcpyFunc(dstPtr + writeIndex, srcPtr + readIndex, elemSize);
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      int64_t newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];
      if (srcShape[axis] != newIndex)
        break;
      if (axis == 0)
        return;
      indices[axis] = 0;
      readIndex -= srcShape[axis] * srcStrides[axis];
      writeIndex -= dstShape[axis] * dstStrides[axis];
    }
  }
}

void mrt::executeStridedByteCopy(
    uintptr_t src, int64_t srcOffsetBytes, const std::vector<int64_t> &srcShape,
    const std::vector<int64_t> &srcByteStrides, uintptr_t dst,
    int64_t dstOffsetBytes, const std::vector<int64_t> &dstShape,
    const std::vector<int64_t> &dstByteStrides, size_t elemSizeBytes,
    std::function<void(void *dst, void *src, size_t size)> memcpyFunc) {
  // Handle edge case of empty source tensor.
  if (std::all_of(srcShape.begin(), srcShape.end(),
                  [](int64_t dimSize) { return dimSize == 0; }))
    return;

  char *srcPtr = reinterpret_cast<char *>(src + srcOffsetBytes);
  char *dstPtr = reinterpret_cast<char *>(dst + dstOffsetBytes);

  // Handle edge case of rank-0 tensor.
  int64_t rank = srcShape.size();
  if (rank == 0) {
    memcpyFunc(dstPtr, srcPtr, elemSizeBytes);
    return;
  }

  std::vector<int64_t> indices(rank, 0);

  // The logic below comes from the original MLIR CRunner implementation of
  // strided memref copy. It could be further optimized if any of the
  // dimensions have unit stride.
  int64_t readIndex = 0, writeIndex = 0;
  while (true) {
    // Copy one element.
    memcpyFunc(dstPtr + writeIndex, srcPtr + readIndex, elemSizeBytes);
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      int64_t newIndex = ++indices[axis];
      readIndex += srcByteStrides[axis];
      writeIndex += dstByteStrides[axis];
      if (srcShape[axis] != newIndex)
        break;
      if (axis == 0)
        return;
      indices[axis] = 0;
      readIndex -= srcShape[axis] * srcByteStrides[axis];
      writeIndex -= dstShape[axis] * dstByteStrides[axis];
    }
  }
}
