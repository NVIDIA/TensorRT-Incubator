//===- CUDAInterfaces.td --------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_CUDA_IR_CUDAINTERFACES_TD
#define MLIR_TENSORRT_DIALECT_CUDA_IR_CUDAINTERFACES_TD

include "mlir/IR/OpBase.td"

def CUDAOpInterface : OpInterface<"CUDAOpInterface"> {
  let description = [{
    All CUDA dialect ops has this interface.
  }];

  let cppNamespace = "::mlir::cuda";

  let methods = [];
}

#endif // MLIR_TENSORRT_DIALECT_CUDA_IR_CUDAINTERFACES_TD