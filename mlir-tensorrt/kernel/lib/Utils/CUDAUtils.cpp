//===- CUDAUtils.cpp ------------------------------------------------------===//
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

#include "mlir-kernel/Utils/CUDAUtils.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::kernel;

int32_t kernel::getHighestPTXVersion() {
#ifdef MLIR_TRT_ENABLE_CUDA
  if (MLIR_TRT_CUDA_VERSION_GTE(13, 0, 0))
    return 90;
  if (MLIR_TRT_CUDA_VERSION_GTE(12, 9, 0))
    return 88;
  if (MLIR_TRT_CUDA_VERSION_GTE(12, 8, 0))
    return 87;
  if (MLIR_TRT_CUDA_VERSION_GTE(12, 7, 0))
    return 86;
  if (MLIR_TRT_CUDA_VERSION_GTE(12, 5, 0))
    return 85;
  if (MLIR_TRT_CUDA_VERSION_GTE(12, 4, 0))
    return 84;
  if (MLIR_TRT_CUDA_VERSION_GTE(12, 3, 0))
    return 83;
  if (MLIR_TRT_CUDA_VERSION_GTE(12, 2, 0))
    return 82;
  if (MLIR_TRT_CUDA_VERSION_GTE(12, 1, 0))
    return 81;
  if (MLIR_TRT_CUDA_VERSION_GTE(12, 0, 0))
    return 80;
  if (MLIR_TRT_CUDA_VERSION_GTE(11, 8, 0))
    return 78;
  if (MLIR_TRT_CUDA_VERSION_GTE(11, 7, 0))
    return 77;
  if (MLIR_TRT_CUDA_VERSION_GTE(11, 6, 0))
    return 76;
  if (MLIR_TRT_CUDA_VERSION_GTE(11, 5, 0))
    return 75;
  if (MLIR_TRT_CUDA_VERSION_GTE(11, 4, 0))
    return 74;
  if (MLIR_TRT_CUDA_VERSION_GTE(11, 3, 0))
    return 73;
  if (MLIR_TRT_CUDA_VERSION_GTE(11, 2, 0))
    return 72;
  if (MLIR_TRT_CUDA_VERSION_GTE(11, 1, 0))
    return 71;
  if (MLIR_TRT_CUDA_VERSION_GTE(11, 0, 0))
    return 70;
  if (MLIR_TRT_CUDA_VERSION_GTE(10, 2, 0))
    return 65;
  llvm::report_fatal_error("Unsupported CUDA version");
#endif // MLIR_TRT_ENABLE_CUDA
  return 90;
}

std::optional<StringRef> kernel::getUniqueTargetChip(gpu::GPUModuleOp module) {
  ArrayAttr targets = module.getTargetsAttr();
  if (!targets)
    return std::nullopt;

  std::optional<StringRef> chip{};
  for (auto target : targets) {
    if (auto targetAttr = llvm::dyn_cast<NVVM::NVVMTargetAttr>(target)) {
      if (chip && *chip != targetAttr.getChip())
        return std::nullopt;
      if (!chip)
        chip = targetAttr.getChip();
    }
  }
  return chip;
}
