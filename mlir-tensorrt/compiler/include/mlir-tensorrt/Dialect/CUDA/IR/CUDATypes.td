//===- CUDATypes.td -------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_CUDA_IR_CUDADTYPES_TD
#define MLIR_TENSORRT_DIALECT_CUDA_IR_CUDADTYPES_TD

include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.td"

include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// CUDA - Core Types
//===----------------------------------------------------------------------===//

def CUDA_Device : TypeDef<CUDA_Dialect, "Device", []> {
  let mnemonic = "device";
  let description = [{
    An opaque object which represents a CUDA device object (CUdevice).
  }];
}

def CUDA_Stream : TypeDef<CUDA_Dialect, "Stream", []> {
  let mnemonic = "stream";
  let description = [{
    An opaque object which represents a CUDA stream object (CUstream).
    A CUDA stream contains a sequence of operations that execute on GPU in the
    order in which they are issued by the host.
  }];
}

def CUDA_Event : TypeDef<CUDA_Dialect, "Event", []> {
  let mnemonic = "event";
  let description = [{
    An opaque object which represents a CUDA event object (CUevent).
    CUDA events provide markers for profiling and synchronization.
  }];
}

def CUDA_Module : TypeDef<CUDA_Dialect, "Module", []> {
  let mnemonic = "module";
  let description = [{
    An opaque object which represents a CUmodule. A module handle may point to
    PTX code, a CUBIN or a FATBIN from a file.
  }];
}

def CUDA_Function : TypeDef<CUDA_Dialect, "Function", []> {
  let mnemonic = "function";
  let description = [{
    An opaque object which represents a CUfunc. This is a function handle.
  }];
}

//===----------------------------------------------------------------------===//
// CUDA - cuBLAS Types
//===----------------------------------------------------------------------===//

def CUDA_BlasHandle : TypeDef<CUDA_Dialect, "BlasHandle", []> {
  let mnemonic = "blas.handle";
  let description = [{
    A handle to an opaque structure holding the cuBLAS library context.
  }];
}

def CUDA_BlasGemmAlgorithm : TypeDef<CUDA_Dialect, "BlasGemmAlgorithm", []> {
  let mnemonic = "blas.gemm_algorithm";
  let description = [{
    A handle to an opaque structure holding a gemm algorithm after selection based  
    on heuristic and problem specification.
  }];
}

#endif // MLIR_TENSORRT_DIALECT_CUDA_IR_CUDADTYPES_TD