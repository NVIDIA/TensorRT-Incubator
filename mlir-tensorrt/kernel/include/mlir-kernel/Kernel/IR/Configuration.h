//===- Configuration.h ----------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Utilities for GPU information (e.g. information about different
/// architectures).
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_KERNEL_UTILS_CONFIGURATION
#define MLIR_KERNEL_KERNEL_UTILS_CONFIGURATION

#include "mlir/Support/LLVM.h"
#include <utility>

namespace mlir {
class Location;

namespace NVVM {
class NVVMTargetAttr;
}

namespace kernel {

/// Convert a `gpu::TargetAttrInterface` to a compute capability integer (SM
/// version number) and a suffix letter (may be empty). Returns failure if the
/// target is not a NVVM target or if the compute capability is not set.
FailureOr<std::pair<int32_t, StringRef>>
targetInfoToChipInfo(NVVM::NVVMTargetAttr targetInfo);

// /// Queries the CUDA API for the specified device's compute capability (SM
// /// version, e.g. 8.0 for A100) and returns it as the pair (major verison
// /// number, minor version number).
FailureOr<std::pair<int32_t, int32_t>>
inferSMVersionFromCudaDevice(Location loc, int64_t deviceNumber);

} // namespace kernel
} // namespace mlir

#endif /* MLIR_KERNEL_KERNEL_UTILS_CONFIGURATION */
