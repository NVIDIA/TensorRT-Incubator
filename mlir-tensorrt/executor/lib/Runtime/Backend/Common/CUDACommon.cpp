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
#include "mlir-executor/Runtime/Backend/Common/CUDACommon.h"
#include "mlir-executor/Runtime/Backend/Common/NvPtxCompilerUtils.h"
#include "mlir-executor/Runtime/Support/Support.h"

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

StatusOr<CudaModulePtr> CudaModulePtr::create(ResourceTracker &tracker,
                                              llvm::StringRef ptxData,
                                              llvm::StringRef arch) {
  // JIT compile the PTX.
  assert(!ptxData.empty() && "expected ptx data with positive string length");
  StatusOr<std::unique_ptr<CuBinWrapper>> cubin =
      compilePtxToCuBin(ptxData.data(), ptxData.size(), arch);
  if (!cubin.isOk())
    return cubin.getStatus();
  if (*cubin == nullptr)
    return getInternalErrorStatus("failed to load PTX to cubin");

  CUmodule module{nullptr};
  CUresult result = cuModuleLoadDataEx(
      &module, reinterpret_cast<const void *>((*cubin)->data.data()), 0, 0, 0);
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
