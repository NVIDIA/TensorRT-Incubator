//===- Passes.h -----------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// CUDA dialect pass declarations.
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_CUDA_TRANSFORMS_PASSES_H
#define MLIR_TENSORRT_DIALECT_CUDA_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h" // IWYU pragma: keep

namespace mlir {
namespace cuda {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir-tensorrt/Dialect/CUDA/Transforms/Passes.h.inc"

} // namespace cuda
} // namespace mlir

#endif // MLIR_TENSORRT_DIALECT_CUDA_TRANSFORMS_PASSES_H
