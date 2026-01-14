//===- StreamSchedulableOpInterface.td ------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_COMMON_INTERFACES_STREAMSCHEDULABLEOPINTERFACE
#define MLIR_TENSORRT_COMMON_INTERFACES_STREAMSCHEDULABLEOPINTERFACE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// StreamSchedulableOp
//===----------------------------------------------------------------------===//

def StreamSchedulableOp : OpInterface<"StreamSchedulableOp"> {
  let description = [{
    Interface for ops that are schedulable on a CUDA stream.

    This is intentionally shared across dialects (e.g. CUDA, TensorRTRuntime,
    SPMD) so passes can reason about stream-bearing ops without ad-hoc scanning
    for `!cuda.stream` operand types.
  }];

  let cppNamespace = "::mtrt::compiler";

  let methods = [
    InterfaceMethod<[{
      Returns the single CUDA stream operand (`!cuda.stream`) for this op.
    }],
    /*retTy=*/"::mlir::OpOperand *",
    /*methodName=*/"getStreamOperand",
    /*args=*/(ins),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
      return &$_op.getStreamMutable();
    }]>
  ];
}

#endif // MLIR_TENSORRT_COMMON_INTERFACES_STREAMSCHEDULABLEOPINTERFACE
