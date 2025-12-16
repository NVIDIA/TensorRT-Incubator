//===- Configuration.cpp --------------------------------------------------===//
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
#include "mlir-kernel/Kernel/IR/Configuration.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "llvm/Support/Regex.h"

#ifdef MLIR_TRT_ENABLE_CUDA
#include "cuda_runtime_api.h"
#endif // MLIR_TRT_ENABLE_CUDA

using namespace mlir;
using namespace mlir::kernel;

FailureOr<std::pair<int32_t, StringRef>>
kernel::targetInfoToChipInfo(NVVM::NVVMTargetAttr targetInfo) {
  llvm::Regex pattern(R"(sm_([0-9]+)([a-z]*))",
                      llvm::Regex::RegexFlags::IgnoreCase);
  llvm::StringRef arch = targetInfo.getChip();
  SmallVector<StringRef> matches;
  std::string error;
  if (pattern.match(arch, &matches, &error))
    return std::make_pair(std::stoi(matches[1].str()), matches[2]);

  return emitError(UnknownLoc::get(targetInfo.getContext()))
         << "failed to parse chip information from target attribute "
         << targetInfo << (error.empty() ? "" : ": ") << error;
}

FailureOr<std::pair<int32_t, int32_t>>
kernel::inferSMVersionFromCudaDevice(Location loc, int64_t deviceNumber) {
#ifdef MLIR_TRT_ENABLE_CUDA
  cudaDeviceProp props;
  cudaError_t err = cudaGetDeviceProperties(&props, deviceNumber);
  if (err != cudaSuccess)
    return emitError(loc) << "failed to get device properties: "
                          << cudaGetErrorString(err);
  return std::make_pair(props.major, props.minor);
#else
  return emitError(loc) << "not compiled with CUDA support";
#endif
}
