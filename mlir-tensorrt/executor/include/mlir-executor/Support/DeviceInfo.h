//===- DeviceInfo.h ---------------------------------------------*- C++ -*-===//
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
/// Utilities for enumerating CUDA device information.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_SUPPORT_DEVICEINFO
#define MLIR_TENSORRT_SUPPORT_DEVICEINFO

#include "mlir-tensorrt-common/Support/Status.h"

namespace mlirtrt {

/// Encapsulates information about a CUDA device.
struct DeviceInfo {
  int64_t computeCapability;
  int64_t maxSharedMemoryPerBlockKb;
  // Maximum number of 4-byte registers per block.
  uint64_t maxRegistersPerBlock;
};

/// Infer target device information from the first visible CUDA device on the
/// host.
StatusOr<DeviceInfo> getDeviceInformationFromHost(int32_t cudaDeviceOrdinal);

/// Infer the device information from all visible CUDA devices.
StatusOr<llvm::SmallVector<DeviceInfo>> getAllDeviceInformationFromHost();

} // namespace mlirtrt

#endif // MLIR_TENSORRT_SUPPORT_DEVICEINFO
