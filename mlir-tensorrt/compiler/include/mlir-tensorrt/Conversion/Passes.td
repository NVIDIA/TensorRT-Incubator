//===- Passes.td -------------------------------------------*- Tablegen -*-===//
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
#ifndef MLIR_TENSORRT_CONVERSION_PASSES_TD
#define MLIR_TENSORRT_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"


#ifdef MLIR_TENSORRT_ENABLE_HLO

//===----------------------------------------------------------------------===//
// StablehloToPlan
//===----------------------------------------------------------------------===//

def ConvertStablehloToPlanPass : Pass<"convert-stablehlo-to-plan"> {
  let summary = "Convert certain StableHLO dialect ops to Plan dialect ops";
  let description = [{
    This pass converts supported Stable HLO dialect ops to Plan dialect ops.
    Currently the only conversions supported are:
    - `stablehlo.optimization_barrier` -> `plan.optimization_barrier`
  }];
  let dependentDialects = [
    "::mlir::plan::PlanDialect"
  ];
}

//===----------------------------------------------------------------------===//
// StablehloToTensorRT
//===----------------------------------------------------------------------===//

def ConvertStablehloToTensorRTPass : Pass<"convert-stablehlo-to-tensorrt"> {
  let summary = "Convert Stable HLO dialect to TensorRT dialect";
  let description = [{
    This pass converts supported Stable HLO ops to TensorRT dialect ops.
  }];
  let dependentDialects = [
    "::mlir::tensorrt::TensorRTDialect",
    "::mlir::quant::QuantDialect",
  ];
  let options = [
    Option<"convertLoops", "convert-loops", "bool", "false",
      "convert loops to TensorRT's loop layer">,
    Option<"convertConditionals", "convert-conditionals", "bool", "true",
      "convert conditionals to TensorRT's conditional layer">,
    Option<"trtMajorVersion", "trt-major-version", "int64_t", "10",
    "target TensorRT version for conversion">,
    Option<"preferEinsum", "prefer-einsum", "bool", "false",
      "prefer converting to 'tensorrt.einsum' over 'tensorrt.matrix_multiply'">
  ];
}

//===----------------------------------------------------------------------===//
// ChloToStableHloExt
//===----------------------------------------------------------------------===//

def ConvertChloToStableHloExtPass : Pass<"convert-chlo-to-stablehlo-ext"> {
  let summary = "Convert specific CHLO operations to stablehlo";
  let description = [{
    This pass converts a CHLO operations to StableHlo while also allowing
    for some CHLO operations to be preserved (see options).
  }];
  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect"
  ];

  let options = [
    Option<"preserveErf", "preserve-erf", "bool", "true",
      "do not convert chlo.erf ops">,
    Option<"preserveTopK", "preserve-topk", "bool", "true",
      "do not convert chlo.topk ops">,
  ];
}

#endif // MLIR_TENSORRT_ENABLE_HLO

//===----------------------------------------------------------------------===//
// HostToEmitC
//===----------------------------------------------------------------------===//

def ConvertHostToEmitCPass : Pass<"convert-host-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Convert host program IR to EmitC";
  let description = [{
    This pass aggregates conversion patterns for EmitC conversions.
  }];
  let dependentDialects = [
    "::mlir::emitc::EmitCDialect"
  ];
  let options = [
    Option<"artifactsDirectory", "artifacts-dir", "std::string",
      "\"\"", "">
  ];
}

//===----------------------------------------------------------------------===//
// TensorRTEmitCBuilderPass
//===----------------------------------------------------------------------===//
def ConvertTensorRTToEmitCPass : Pass<"convert-tensorrt-to-emitc",
      "::mlir::ModuleOp"> {
  let summary = "Generate an `EmitC` based builder for each `tensorrt` function";
  let description = [{
    The purpose of this conversion is enable the creation of a C++ source file
    that uses the TensorRT C++ API to build TRT engines representing each of
    the `func.func` in the module.

    Specifically, for each `func.func` in the module with symbol name `@[name]`
    that contains only `tensorrt` dialect operations (besides the func
    terminator), the pass creates two new functions (builder and tester) that
    do the following:

    1. `@[name]_builder`: Given a pointer to a `nvinfer1::INetworkDefinition`
      and a container for holding constant data, builds the equivalent
      TensorRT network using the TensorRT C++ API. This is represented in the
      IR using `emitc` dialect calls to call helper functions defined in
      `NvInferAdaptor.h`

    2.`@[name]_tester`: Creates the `nvinfer1::IBuilder`, optimization
      profiles, and sets other configuration, then calls `@[name]_builder` to
      construct the network, builds the TRT engine, and returns the serialized
      engine as a unique pointer.

    The original `func.func` and any other ops not compatible with translation
    to C++ (via conversion to EmitC) are erased.
  }];

  let dependentDialects = ["::mlir::emitc::EmitCDialect"];
}

//===----------------------------------------------------------------------===//
// LowerLinalgCopiesPass
//===----------------------------------------------------------------------===//

def LowerLinalgCopiesPass : Pass<"lower-linalg-copies"> {
  let summary = "Lower linalg.copy to memref.copy or other operations";
  let description = [{
    This pass lowers `linalg.copy` to `memref.copy`.
  }];
}

//===----------------------------------------------------------------------===//
// ConvertMemRefToCUDAPass
//===----------------------------------------------------------------------===//

def ConvertMemRefToCUDAPass : Pass<"convert-memref-to-cuda">{
  let summary = "Converts some memref ops to the CUDA dialect";
  let description = [{
    This pass converts a subset of memref operations to the CUDA dialect.
    Memref operations `memref.(alloc|copy|dealloc)` that have explicit
    memory space annotations will be converted into equivalent
    `cuda.(alloc|dealloc|copy(h2d|d2d|d2h))` operations. Since CUDA operations
    take stream operands, the streams are also materialized and synchronizations
    are inserted. Currently no analysis is performed and only a single "default"
    streams is used. Synchronizations are inserted after device-to-host copies.
  }];
  let dependentDialects = [
    "::mlir::cuda::CUDADialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConvertPlanToExecutorPass
//===----------------------------------------------------------------------===//

def ConvertPlanToExecutorPass : Pass<"convert-plan-to-executor",
                                     "::mlir::ModuleOp"> {
  let summary =
    "Converts plan dialect ops and attributes to the executor dialect";

  let description = [{
    This pass eliminates Plan dialect attributes from memref types
    (e.g. `#plan.memory_space` encodings) and ensures that no
    `plan` operations remain in the program.
  }];

  let dependentDialects = [
    "::mlir::executor::ExecutorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Executor Common Options
//===----------------------------------------------------------------------===//

defvar ConvertToExecutorOptions = [
  Option<"indexBitwidth", "index-bitwidth", "int64_t", "64",
  "all index types will be converted to signless integers of this bitwidth">,
  Option<"usePackedMemRefCConv", "use-packed-memref-cconv",
    "bool", "true",
    "convert memref arguments in functions to table/struct rather than to "
    "an unpacked list of scalars">
];

//===----------------------------------------------------------------------===//
// ConvertTensorRTToTensorRTRuntimePass
//===----------------------------------------------------------------------===//
def ConvertTensorRTToTensorRTRuntimePass : Pass<"convert-tensorrt-to-runtime",
                                            "::mlir::ModuleOp"> {
  let summary = "Converts `tensorrt` ops to Executor globals and"
    "TensorRTRuntime dialect operations";

  let description = [{
    This pass expects any functions representing TensorRT engines under a nested
    `tensorrt.module` have been translated to TensorRT engines and have their
    data attached to each function.

    Within the outer module, `executor` dialect global and constant resource
    operations are materialized to hold the binary data containing each serialized
    TensorRT engine and represent the loading of the serialized engine into
    a `!trtrt.execution_context`.

    Any `tensorrt.call` operation is lowered to a sequence of operations to
    retrieve the appropriate execution context and enqueue an execution of the
    operation on a stream.
  }];

  let dependentDialects = [
    "::mlir::trtrt::TensorRTRuntimeDialect",
    "::mlir::cuda::CUDADialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConvertTensorRTRuntimeToLLVMPass
//===----------------------------------------------------------------------===//
def ConvertTensorRTRuntimeToLLVMPass : Pass<"convert-tensorrt-runtime-to-llvm",
                                            "::mlir::ModuleOp"> {
  let summary = "Converts TensorRTRuntime dialect ops to LLVM dialect operations";

  let description = [{
    This pass contains patterns to convert `trtrt` dialect operations to `llvm`
    dialect operations.
  }];
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect",
  ];
  let options = [
    Option<"artifactsDirectory", "artifacts-dir", "std::string",
      "\"\"", "">
  ];
}

//===----------------------------------------------------------------------===//
// ConvertCUDAToExecutorPass
//===----------------------------------------------------------------------===//
def ConvertCUDAToExecutorPass : Pass<"convert-cuda-to-executor",
                                  "::mlir::ModuleOp"> {
  let summary = "Converts CUDA dialect ops to executor dialect operations";

  let description = [{
    This pass contains patterns to convert `cuda` dialect operations to `executor`
    dielact operations.
  }];
  let dependentDialects = [
    "::mlir::executor::ExecutorDialect",
    "::mlir::cuda::CUDADialect"
  ];
  let options = ConvertToExecutorOptions;
}

//===----------------------------------------------------------------------===//
// ConvertCUDAToLLVMPass
//===----------------------------------------------------------------------===//
def ConvertCUDAToLLVMPass : Pass<"convert-cuda-to-llvm",
                                  "::mlir::ModuleOp"> {
  let summary = "Converts CUDA dialect ops to LLVM dialect operations";

  let description = [{
    This pass contains patterns to convert `cuda` dialect operations to `llvm`
    dielact operations.
  }];
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect",
  ];
  let options = [
    Option<"artifactsDirectory", "artifacts-dir", "std::string",
      "\"\"", "">
  ];
}

//===----------------------------------------------------------------------===//
// HostToLLVM
//===----------------------------------------------------------------------===//
def ConvertHostToLLVMPass : Pass<"convert-host-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Convert host program IR to LLVM";
  let description = [{
    This pass aggregates conversion patterns for CUDA, TensorRTRuntime, and
    other upstream dialects in order to lower the post-bufferizatio host
    program IR to LLVM in a single pass.
  }];
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect"
  ];
  let options = [
  ];
}

//===----------------------------------------------------------------------===//
// ConvertTensorRTRuntimeToExecutorPass
//===----------------------------------------------------------------------===//
def ConvertTensorRTRuntimeToExecutorPass : Pass<"convert-tensorrt-runtime-to-executor",
                                            "::mlir::ModuleOp"> {
  let summary = "Converts TensorRTRuntime dialect ops to executor dialect operations";

  let description = [{
    This pass contains patterns for converting `trtrt` dialect operations to
    `executor` dialect operations.

    Currently the most important of these conversions is the  conversion of
    `trtrt.enqueue` (in bufferized memref form) to an opaque call to the
    variadic `executor.invoke`. At this point, we extract buffer pointers (and
    in the future, dynamic dimensions) from the `!executor.table` objects
    representing memref types and pass them to the invocation of
    `trtrt_enqueue`.

    Like other lowerings of operations to external function calls, this
    conversion effectively represents an ABI contract with the backend runtime.
  }];

  let dependentDialects = [
    "::mlir::executor::ExecutorDialect"
  ];
  let options = ConvertToExecutorOptions;
}

#ifdef MLIR_TENSORRT_ENABLE_HLO
//===----------------------------------------------------------------------===//
// ConvertStablehloToScfPass
//===----------------------------------------------------------------------===//
def ConvertStablehloToScfPass : Pass<"convert-stablehlo-to-scf"> {
  let summary = "Convert StableHLO control flows to SCF control flow.";
  let description = [{
    This pass converts StabeleHLO control flows to SCF control flow.
  }];
  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect"
  ];
}

//===----------------------------------------------------------------------===//
// StablehloScalarToArith
//===----------------------------------------------------------------------===//
def ConvertStablehloScalarToArithPass : Pass<"convert-stablehlo-scalar-to-arith"> {
  let summary = "Convert Stablehlo scalar ops to Arith dialect";
  let description = [{
    This pass tries to convert Stablehlo ops with single element tensor input/s to Arith
    or Math dialect op/s operating on scalars extracted from the input tensor/s.
    The purpose of this conversion is two fold
    1. Many Stablehlo ops with scalar inputs (e.g. bitcast_convert, shift_left, bitwise ops)
    are not supported by TensorRT.
    2. Even when an op with scalar operand/s is supported by TensorRT, operand/s
    needs to be expanded to 1D tensor in almost all cases.

  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::math::MathDialect"
  ];
}
#endif

#endif // MLIR_TENSORRT_CONVERSION_PASSES_TD
