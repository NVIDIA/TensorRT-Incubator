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


#ifdef MLIR_TRT_ENABLE_HLO

//===----------------------------------------------------------------------===//
// StablehloToPlan
//===----------------------------------------------------------------------===//

def ConvertStablehloToPlanPass : Pass<"convert-stablehlo-to-plan", "::mlir::ModuleOp"> {
  let summary = "Convert certain StableHLO dialect ops to Plan dialect ops";
  let description = [{
    This pass converts supported Stable HLO dialect ops to Plan dialect ops.
    Currently the only conversions supported are:
    - `stablehlo.optimization_barrier` -> `plan.optimization_barrier`
    - convert `tf.aliasing_output` argument attributes to `plan.aliasing_output`
    - convert `stablehlo.custom_call` to `executor.call_plugin` for
      `stablehlo.custom_call` that are targeted towards MTRT-compatible FFI
      calls.
  }];
  let dependentDialects = [
    "::mlir::cf::ControlFlowDialect",
    "::mlir::plan::PlanDialect",
    "::mlir::executor::ExecutorDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::cuda::CUDADialect"
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

#endif // MLIR_TRT_ENABLE_HLO

//===----------------------------------------------------------------------===//
// HostToEmitC
//===----------------------------------------------------------------------===//

def ConvertHostToEmitCPass : Pass<"convert-host-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Convert host program IR to EmitC";
  let description = [{
    This pass aggregates conversion patterns for EmitC conversions.
  }];
  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::executor::ExecutorDialect"
  ];
  let options = [
    Option<"emitAggregateInitDestroy", "emit-aggregate-init-destroy", "bool", "true",
      "Emit aggregate global `<module>_initialize_all` / `<module>_destroy_all` convenience helpers">,
  ];
}


def WrapModuleInEmitCClassPass : Pass<"wrap-module-in-emitc-class", "::mlir::ModuleOp"> {
  let summary = "Wrap an EmitC module in a single emitc.class (Program) state object";
  let description = [{
    This pass post-processes a module containing EmitC globals and functions by
    migrating module-scope `emitc.global` into `emitc.field` members of a single
    `emitc.class`, moving entrypoints into that class as methods, and
    consolidating initialization/teardown into `initialize()` and `destroy()`.

    It is intended to run after `convert-host-to-emitc`.
  }];
  let dependentDialects = [
    "::mlir::emitc::EmitCDialect"
  ];
}

def EmitCppSupportFilesPass : Pass<"emit-cpp-support-files", "::mlir::ModuleOp"> {
  let summary = "Emit EmitC C++ support files as executor.file_artifact ops";
  let description = [{
    This pass emits optional support files required to compile and test the C++
    produced by the EmitC host target. The files are emitted as
    `executor.file_artifact` operations, to be materialized by the
    `executor-serialize-artifacts` pass into the artifacts directory.

    The emitted files can include:
    - StandaloneCPP runtime source and header files (subset based on runtime usage)
    - An example CMake file
    - A simple test driver
  }];

  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::executor::ExecutorDialect"
  ];

  let options = [
    Option<"emitRuntimeFiles", "emit-runtime-files", "bool", "false",
      "Emit the required subset of StandaloneCPP runtime source/header files">,
    Option<"emitCMakeFile", "emit-cmake-file", "bool", "false",
      "Emit an example CMakeLists.txt file">,
    Option<"emitTestDriver", "emit-test-driver", "bool", "false",
      "Emit a simple test driver source file">,
    Option<"outputFile", "output-file", "std::string", "\"\"",
      "The path to the generated EmitC translation output file (for driver/CMake generation)">,
    Option<"entrypoint", "entrypoint", "std::string", "\"main\"",
      "Entrypoint function name (used by the generated test driver, if present)">,
    Option<"supportSubdir", "support-subdir", "std::string", "\"emitc_support\"",
      "Subdirectory under the artifacts directory to place emitted support files">,
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
// ConvertLinalgToCUDAPass
//===----------------------------------------------------------------------===//
def ConvertLinalgToCUDAPass : Pass<"convert-linalg-to-cuda"> {
  let summary = "Lower linalg.fill ops on device buffers to cuda.memset";
  let description = [{
    This pass converts `linalg.fill` (and `linalg.generic` fills) on
    device-visible buffers to `cuda.memset` using the default stream.
    Only contiguous buffers with element size of 1, 2, or 4 bytes are supported.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::cuda::CUDADialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::plan::PlanDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConvertKernelToCUDAPass
//===----------------------------------------------------------------------===//
def ConvertKernelToCUDAPass : Pass<"convert-kernel-to-cuda", "::mlir::ModuleOp"> {
  let summary = "Convert Kernel dialect to CUDA dialect";
  let description = [{
    This pass converts supported Kernel dialect ops to the CUDA dialect.

    In particular, any `kernel.call` operation is turned into a
    `cuda.launch` call and the operations to materialize e.g. CUDA
    streams and clean them up are automatically inserted.

    For each kernel that is invoked by `kernel.call` operations, a constant
    global for the PTX IR and globals for the CUmodule/CUfunc are materialized,
    forcing compilation of the kernel when the program globals are initialized.
  }];
  let dependentDialects = [
    "::mlir::cuda::CUDADialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::executor::ExecutorDialect"
  ];
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
  "all index types will be converted to signless integers of this bitwidth">
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
    "::mlir::executor::ExecutorDialect"
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
    "::mlir::executor::ExecutorDialect"
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
    "::mlir::LLVM::LLVMDialect",
    "::mlir::executor::ExecutorDialect"
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

#ifdef MLIR_TRT_ENABLE_HLO

//===----------------------------------------------------------------------===//
// StablehloToKernelPass
//===----------------------------------------------------------------------===//

def StablehloToKernelPass : Pass<"convert-stablehlo-to-kernel"> {
  let summary = "Convert Stablehlo dialect to Kernel dialect";
  let description = [{
    This pass converts supported Stablehlo dialect ops to the Kernel dialect.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::kernel::KernelDialect",
    "::mlir::math::MathDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConvertStablehloToArithPass
//===----------------------------------------------------------------------===//

def ConvertStablehloToArithPass :
      Pass<"convert-stablehlo-to-arith", "::mlir::func::FuncOp"> {
  let summary = "Convert any 'stablehlo.constant' into 'arith.constant'";

  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}

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
// StablehloToLinalg
//===----------------------------------------------------------------------===//

def StablehloToLinalgPass : Pass<"convert-stablehlo-to-linalg"> {
  let summary = "Convert Stablehlo dialect to Linalg dialect";
  let description = [{
    This pass converts supported Stablehlo dialect ops to the Linalg dialect.

    This pass just uses the patterns of the upstream
    `stablehlo-legalize-to-linalg` pass, but we correct an issue where
    operations inside of regions like `stablehlo.reduce` are converted.

    TODO: Remove this pass once the upstream pass is fixed.
  }];
  let dependentDialects = [
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::complex::ComplexDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::math::MathDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::shape::ShapeDialect"
  ];
}
#endif // MLIR_TRT_ENABLE_HLO

//===----------------------------------------------------------------------===//
// OutlineTensorRTOpsPass
//===----------------------------------------------------------------------===//

def OutlineTensorRTOpsPass : Pass<"outline-tensorrt-ops",
      "::mlir::ModuleOp"> {
  let summary = "Segment and outline all TensorRT Dialect ops into a 'tensorrt.module'";

  let dependentDialects = [
    "::mlir::plan::PlanDialect"
  ];
}

//===----------------------------------------------------------------------===//
// DropNestedModulesPass
//===----------------------------------------------------------------------===//
def DropNestedModulesPass : Pass<"drop-nested-modules", "::mlir::ModuleOp"> {
  let summary = "Drops any operation with the SymbolTable attribute directly"
    " nested within the top-level Module";
}

//===----------------------------------------------------------------------===//
// LinalgElementwiseFusionPass
//===----------------------------------------------------------------------===//

def LinalgElementwiseFusionPass : Pass<"mtrt-linalg-elementwise-fusion"> {
  let summary =
    "Performs elementwise fusion on linalg operations specific to MLIR-TensorRT";

  let description = [{
    Runs linalg elementwise fusion with a control function that is specific
    to MLIR-TensorRT.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::linalg::LinalgDialect"
  ];
}

//===----------------------------------------------------------------------===//
// MemRefCastEliminationPass
//===----------------------------------------------------------------------===//
def MemRefCastEliminationPass : Pass<"memref-cast-elimination"> {
  let summary = "Runs auxillary patterns to eliminate `memref.cast` operations";

  let description = [{
    Attempts to eliminate `memref.cast` operations where they can be statically
    determined to be superfluous but for whatever reason are not being
    removed by canonicalization.
  }];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SCFDetensorizePass
//===----------------------------------------------------------------------===//

def SCFDetensorizePass : Pass<"mtrt-scf-detensorize"> {
  let summary = "try to replace loop-carried scalar tensors with scalar-typed values";

  let description = [{

    Currently this pass only supports `scf.while`. It rewrites loop-carried SSA
    tensor values to scalar values if the below conditions are true:

    - The tensor type has one element.

    - In the 'before' region, the relevant block arguments are either forwarded
      to the successor region) or are only accessed via `tensor.extract`.

    - In the `after` region, the relevant block arguments are used by
      `tensor.extract` and the input to the same position in the yield
      terminator is defined by a `tensor.from_elements` op.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::scf::SCFDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SCFUnrollPass
//===----------------------------------------------------------------------===//

def SCFUnrollPass : Pass<"mtrt-scf-unroll"> {
   let summary = "Unroll for loops with static trip count";

   let description = [{
    Unroll `scf.for` loops with static trip count. Loops are fully unrolled
    if they have a cost below the `unroll-threshold`.

    Currently the cost is estimated by counting the number of operations in the
    loop body and multiplying it by the trip count.
   }];

   let dependentDialects = ["::mlir::scf::SCFDialect"];

   let options = [
    Option<"unrollThreshold", "unroll-threshold", "uint64_t",
    /*default=*/"100",
    "Unroll for loops if the cost is below this limit">
   ];
}

//===----------------------------------------------------------------------===//
// SCFForCyclicIterArgsPass
//===----------------------------------------------------------------------===//

def SCFForCyclicIterArgsPass : Pass<"mtrt-scf-for-cyclic-iter-args"> {
  let summary =
    "Eliminate scf.for iter args that only participate in cyclic shifts";

  let description = [{
    Identifies groups of `scf.for` iter args that are yielded as a cyclic
    permutation of the same block arguments. The loop-carried args are removed,
    and uses inside the loop are replaced with values selected from the init
    args based on the iteration index. The loop results are rewritten similarly
    based on the trip count.
  }];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SCFFloatStrengthReducePass
//===----------------------------------------------------------------------===//

def SCFFloatStrengthReducePass : Pass<"mtrt-scf-float-strength-reduce"> {
  let summary = "Convert float loop-carried variables to integer counters";

  let description = [{
    Identifies `scf.while` loops with floating-point loop-carried variables that
    are updated with constant additive steps (e.g., `arith.addf %iv, %const`).
    Transforms these loops to use integer counters with appropriate scale
    factors, and attempts to uplift to `scf.for` when possible.

    For example, a loop counting down from 1.0 to 0.05 with step -0.1 can be
    transformed to an integer loop counting from 10 to 1 with step -1, with
    a scale factor of 0.1 applied when the float value is needed.

    This transformation improves performance by:
    1. Using faster integer arithmetic for loop control
    2. Enabling further optimizations like loop unrolling
    3. Making loop bounds statically analyzable
  }];

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// FuncExtDuplicateFunctionEliminationPass
//===----------------------------------------------------------------------===//

def FuncExtDuplicateFunctionEliminationPass : Pass<"func-ext-duplicate-function-elimination",
  "::mlir::ModuleOp"> {
  let summary = "Deduplicates func.func operations";

  let description = [{
    This pass is based on the upstream `duplicate-function-elimination` pass
    and modified to handle more general cases and fix a couple bugs.

    An open TODO is to upstream these changes to the `duplicate-function-elimination` pass.

    This pass attempts to identify `func.func` that are equivalent representations
    of deduplicates them. It does this by first scanning all `func::FuncOp` operations
    (even those in nested symbol tables) and attempts to identify unique instances
    within each SymbolTable scope. Functions are only deduplicated within the scope
    of a single SymbolTable, but the pass will look at all scopes from inner to outer.

    It creates equivalence classes and rewrites the call operations to use only one
    func per class (which is selected to be the first one from each equivalence class).
    The other functions are discarded. A function is only selected for removal if
    we can be sure that all references to the symbol can be safely updated.

    Note that this pass does not try to reconcile argument or result attributes if
    two func.func are identical except for argument attributes, then they will be
    considered separate functions.
  }];
}

//===----------------------------------------------------------------------===//
// TensorExtPadToInsertSlicePass
//===----------------------------------------------------------------------===//

def TensorExtPadToInsertSlicePass : Pass<"tensor-ext-pad-to-insert-slice"> {
  let summary =
    "Convert tensor.pad to linalg.fill + tensor.insert_slice";

  let description = [{
    This pass converts `tensor.pad` operations to `linalg.fill` and
    `tensor.insert_slice` operations.
  }];

  let dependentDialects = [
    "::mlir::linalg::LinalgDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LinalgSimplifyExtractSlicePass
//===----------------------------------------------------------------------===//

def LinalgSimplifyExtractSlicePass : Pass<"linalg-simplify-extract-slice"> {
  let summary =
    "Runs auxillary patterns to eliminate `tensor.extract_slice` operations";

  let description = [{
    This pass tries to eliminate `tensor.extract_slice` operations in special
    cases not covered by the existing `tensor.extract_slice` transforms upstream
    (e.g. fusion and sequential tensor.extract_slice elimination). In
    particular, it removes `tensor.extract_slice` by fusing with a
    `TilingInterface` operation when both the slice op and the producer are in
    the same block and the producer has one use.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::linalg::LinalgDialect"
  ];
}

//===----------------------------------------------------------------------===//
// PlanExecuteConstantFoldableSubgraphs
//===----------------------------------------------------------------------===//

def PlanExecuteConstantFoldableSubgraphsPass : Pass<
      "plan-execute-constant-foldable-subgraphs",
      "::mlir::ModuleOp"> {
  let summary = "Compile and execute constant foldable subgraphs";

  let description = [{
    This pass compiles and executes e2e constant foldable subgraphs
    outlined by `PlanOutlineConstantFoldableSubgraphsPass`. In first step, outlined
    `func.func` is cloned into a separate `ModuleOp` and e2e execution happens.
    In second step, operands of `func.return` in the original `func.func`
    are replaced with the constants created from results of e2e execution.
    Inlining pass running after this pass will eventually replace
    `func.call` ops created in `PlanOutlineConstantFoldableSubgraphsPass`
    with result of constant folding entire subgraph.
  }];

  let dependentDialects = [
    "::mlir::plan::PlanDialect",
    "::mlir::cuda::CUDADialect",
    "::mlir::executor::ExecutorDialect",
    "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect",
  ];
}

#endif // MLIR_TENSORRT_CONVERSION_PASSES_TD
