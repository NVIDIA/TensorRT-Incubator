//===- Passes.td -------------------------------------------*- Tablegen -*-===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_CUDA_TRANSFORMS_PASSES_TD
#define MLIR_TENSORRT_DIALECT_CUDA_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// CUDAScheduleAsyncPass
//===----------------------------------------------------------------------===//

def CUDAScheduleAsyncPass : Pass<"cuda-schedule-async", "mlir::func::FuncOp"> {
  let summary = "Schedule CUDA commands across a stream pool and insert cross-stream event synchronization";

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::cuda::CUDADialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// CUDAInsertHostSyncPass
//===----------------------------------------------------------------------===//

def CUDAInsertHostSyncPass : Pass<"cuda-insert-host-sync", "mlir::ModuleOp"> {
  let summary = "Insert host-side synchronization only at correctness boundaries for async CUDA copies";

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::cuda::CUDADialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// CUDASimplifyStreamWaitPass
//===----------------------------------------------------------------------===//

def CUDASimplifyStreamWaitPass
    : Pass<"cuda-simplify-stream-wait", "mlir::func::FuncOp"> {
  let summary = "Remove redundant cuda.stream.wait_event operations";

  let dependentDialects = [
    "::mlir::cuda::CUDADialect",
    "::mlir::func::FuncDialect"
  ];
}

#endif // MLIR_TENSORRT_DIALECT_CUDA_TRANSFORMS_PASSES_TD
