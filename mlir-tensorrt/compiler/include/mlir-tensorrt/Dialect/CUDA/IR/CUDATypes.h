//===- CUDATypes.h --------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Declarations for CUDA dialect types.
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_CUDA_IR_CUDATYPES_H
#define MLIR_TENSORRT_DIALECT_CUDA_IR_CUDATYPES_H

#include "mlir/IR/Types.h" // IWYU pragma: keep

#define GET_TYPEDEF_CLASSES
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDAOpsTypes.h.inc"

#endif // MLIR_TENSORRT_DIALECT_CUDA_IR_CUDATYPES_H
