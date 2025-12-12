//===- CUDAUtils.h --------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Utilities related to CUDA.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_UTILS_CUDAUTILS
#define MLIR_KERNEL_UTILS_CUDAUTILS

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include <cstdint>
#include <optional>

#define MLIR_TRT_CUDA_VERSION_GTE(major, minor, patch)                         \
  (MLIR_TRT_COMPILE_TIME_CUDA_VERSION_MAJOR > major ||                         \
   (MLIR_TRT_COMPILE_TIME_CUDA_VERSION_MAJOR == major &&                       \
    (MLIR_TRT_COMPILE_TIME_CUDA_VERSION_MINOR > minor ||                       \
     (MLIR_TRT_COMPILE_TIME_CUDA_VERSION_MINOR == minor &&                     \
      MLIR_TRT_COMPILE_TIME_CUDA_VERSION_PATCH >= patch))))

namespace mlir::kernel {

/// Returns the highest supported PTX version for the CUDA Toolkit used at
/// compile-time.
int32_t getHighestPTXVersion();

/// Returns unique target chip name from attributes attached to
/// `gpu::GPUModuleOp` by `set-gput-target` pass. Returns `std::nullopt` if
/// unique chip name couldn't be decided.
std::optional<StringRef> getUniqueTargetChip(mlir::gpu::GPUModuleOp module);

} // namespace mlir::kernel

#endif // MLIR_KERNEL_UTILS_CUDAUTILS
