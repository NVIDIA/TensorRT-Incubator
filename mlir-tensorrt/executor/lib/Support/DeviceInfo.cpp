//===- DeviceInfo.cpp -----------------------------------------------------===//
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
#include "mlir-executor/Support/DeviceInfo.h"
#include "mlir-executor/Support/CUDAWrappers.h"
#include "mlir-executor/Support/Status.h"

using namespace mlirtrt;

#ifdef MLIR_EXECUTOR_ENABLE_CUDA
static Status makeCudaStringError(cudaError_t errCode,
                                  llvm::StringRef context) {
  // Create a detailed error message using llvm::createStringError
  return getInternalErrorStatus("{0}: {1}", context,
                                cudaGetErrorString(errCode));
}
#endif

static StatusOr<DeviceInfo>
getDeviceInformationFromHostImpl(int cudaDeviceOridinal) {
#ifdef MLIR_EXECUTOR_ENABLE_CUDA
  cudaDeviceProp properties;
  cudaError_t err = cudaGetDeviceProperties(&properties, cudaDeviceOridinal);
  if (err != cudaSuccess)
    return makeCudaStringError(err, "failed to get cuda device properties");

  int ccMajor = 0;
  int ccMinor = 0;
  err = cudaDeviceGetAttribute(
      &ccMajor, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor,
      cudaDeviceOridinal);
  if (err != cudaSuccess)
    return makeCudaStringError(err,
                               "failed to get cuda device compute capability");
  err = cudaDeviceGetAttribute(
      &ccMinor, cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor,
      cudaDeviceOridinal);
  if (err != cudaSuccess)
    return makeCudaStringError(err,
                               "failed to get cuda device compute capability");

  // We want SM version as a single number.
  int64_t smVersion = ccMajor * 10 + ccMinor;
  DeviceInfo info;
  info.computeCapability = smVersion;
  info.maxSharedMemoryPerBlockKb = properties.sharedMemPerBlock / 1024;
  info.maxRegistersPerBlock = properties.regsPerBlock;
  return info;
#else
  return getInternalErrorStatus(
      "MLIR-Executor was not built with CUDA Runtime support");
#endif
}

StatusOr<llvm::SmallVector<DeviceInfo>>
mlirtrt::getAllDeviceInformationFromHost() {
#ifdef MLIR_EXECUTOR_ENABLE_CUDA
  int numDevices = 0;
  cudaError_t err = cudaGetDeviceCount(&numDevices);
  if (err != cudaSuccess)
    return makeCudaStringError(err, "failed to get cuda device count");

  llvm::SmallVector<DeviceInfo> deviceInfos;
  deviceInfos.reserve(numDevices);
  for (int i = 0; i < numDevices; ++i) {
    auto info = getDeviceInformationFromHostImpl(i);
    if (!info.isOk())
      return info.getStatus();
    deviceInfos.push_back(*info);
  }
  return deviceInfos;
#else
  return getInternalErrorStatus(
      "MLIR-Executor was not built with CUDA Runtime support");
#endif
}

StatusOr<DeviceInfo>
mlirtrt::getDeviceInformationFromHost(int32_t cudaDeviceOrdinal) {
#ifdef MLIR_EXECUTOR_ENABLE_CUDA
  return getDeviceInformationFromHostImpl(cudaDeviceOrdinal);
#else
  return getInternalErrorStatus(
      "MLIR-Executor was not built with CUDA support");
#endif
}
