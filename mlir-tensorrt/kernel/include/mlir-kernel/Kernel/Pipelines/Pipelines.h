//===- Pipelines.h --------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Declarations for Kernel dialect pipelines.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_KERNEL_KERNEL_PIPELINES_PIPELINES
#define MLIR_KERNEL_KERNEL_PIPELINES_PIPELINES

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::kernel {

/// Builds a pipeline that performs initial Transform IR generation/application,
/// for the core transforms (i.e. tiling) that happen on tensor types.
/// The `generatorBenefit` list may be used to adjust the benefit of specific
/// schedule generators, which will influence which one is chosen. Each element
/// of the list must be a "name:benefit" pair.
void buildTransformIRPipeline(OpPassManager &pm, StringRef funcFilter,
                              int64_t computeCapability,
                              int64_t maxSharedMemoryPerBlockKb,
                              uint64_t maxRegistersPerBlock,
                              ArrayRef<std::string> generatorBenefit = {});

/// Builds a pipeline that performs post-bufferization optimizations.
void buildKernelMemRefOptimizationPipeline(OpPassManager &kernelModulePM);

/// Builds a pipeline that lowers bufferized and optimized GPU kernel IR to NVVM
/// and translates to PTX. This pass typically is run on the `gpu.module`
/// that contains a list of GPU kernel entrypoint functions. If `dumpPtxPath` is
/// provided, any generated PTX modules will be dumped as files to the specified
/// directory (and the directory is created if it does not exist).
void buildKernelLowerToPTXPipeline(OpPassManager &kernelModulePM,
                                   StringRef dumpPtxPath = "");

/// Register Kernel dialect pipelines.
void registerKernelPipelines();

/// Register external models for Kernel dialect GPUModuleLoweringAttrInterface
/// implementations.
void registerGPUModuleLoweringAttrExternalModels(DialectRegistry &registry);

} // namespace mlir::kernel

#endif // MLIR_KERNEL_KERNEL_PIPELINES_PIPELINES
