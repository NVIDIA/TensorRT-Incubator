//===- Passes.h -----------------------------------------------------------===//
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
/// Declarations for Kernel dialect transforms/passes.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMS_PASSES_H
#define MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMS_PASSES_H

#include "mlir-kernel/Kernel/IR/Enums.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "llvm/Support/CommandLine.h"
#include <memory>
#include <mlir/Pass/Pass.h>

//===----------------------------------------------------------------------===//
// Add Tablegen'd pass declarations and registration methods.
//===----------------------------------------------------------------------===//
namespace mlir::kernel {

namespace detail {
/// Shorthand adaptor for declaring LLVM CL options for bufferization
/// LayoutMapOption.
/// TODO: move this upstream.
inline llvm::cl::ValuesClass createBufferizationLayoutMapClOptions() {
  return ::llvm::cl::values(
      clEnumValN(::mlir::bufferization::LayoutMapOption::InferLayoutMap,
                 "infer-layout-map", "infer the layout"),
      clEnumValN(::mlir::bufferization::LayoutMapOption::IdentityLayoutMap,
                 "identity-layout-map", "use identity layout"),
      clEnumValN(::mlir::bufferization::LayoutMapOption::FullyDynamicLayoutMap,
                 "fully-dynamic-layout-map",
                 "use fully dynamic layout (most conservative)"));
}

/// Shorthand adaptor for declaring LLVM CL options for GPU module lowering
/// phases.
inline llvm::cl::ValuesClass createGpuModuleLoweringPhaseClOptions() {
  return ::llvm::cl::values(
      clEnumValN(::mlir::kernel::GPUModuleLoweringPhase::PreBufferization,
                 "pre-bufferization", "pre bufferization"),
      clEnumValN(::mlir::kernel::GPUModuleLoweringPhase::PostBufferization,
                 "post-bufferization", "post bufferization"),
      clEnumValN(::mlir::kernel::GPUModuleLoweringPhase::LowerToNVVM,
                 "lower-to-nvvm", "lower to NVVM"));
}
} // namespace detail

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"

} // namespace mlir::kernel

#endif // MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMS_PASSES_H
