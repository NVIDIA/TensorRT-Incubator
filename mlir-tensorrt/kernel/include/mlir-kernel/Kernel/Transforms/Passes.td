//===- Passes.td ----------------------------------------------------------===//
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
#ifndef MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMS_PASSES_TD
#define MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SharedAllocToGlobalPass
//===----------------------------------------------------------------------===//

def SharedAllocToGlobalPass : Pass<"kernel-shared-alloc-to-global",
                                          "gpu::GPUModuleOp"> {
  let summary = "converts shared-memory memref.alloc to memref.global";

  let description = [{
    Converts all `memref.alloc` representing shared memory (AKA workgroup memory)
    to be represented using `memref.global`.
  }];

  let options = [
    Option<"defaultAlignment", "default-alignment", "uint64_t", "16",
      "default address alignemnt used for the memref.global">
  ];
}

//===----------------------------------------------------------------------===//
// AnnotateKernelEntrypointsPass
//===----------------------------------------------------------------------===//

def AnnotateKernelEntrypointsPass : Pass<"kernel-annotate-entrypoints"> {
  let summary = "Annotates kernel entrypoints with the 'gpu.kernel' attribute";
  let description = [{
    This pass annotates kernel entrypoints with the 'gpu.kernel' attribute
    by checking to see which `gpu.module` functions are called from the host
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LowerToNVVMPass
//===----------------------------------------------------------------------===//
def LowerToNVVMPass : Pass<"kernel-lower-to-nvvm"> {
  let summary = "Applies final lowering to NVVM dialect IR";

  let description = [{
    If the target IR representing a CUDA kernel has been lowered into SPMD
    form, returns no results, has only `memref` or scalar-type operands, and
    obeys other restrictions described below, then this pass will lower that IR
    to NVVM (MLIR dialect) IR, which can further be translated to LLVM NVVM IR
    and then compiled to PTX using LLVM's PTX backend.

    #### Overview

    Assuming the op meets the preconditions described below, the lowering
    occurs in a two-step process:

      1. First, a set of rewrites are executed to prepare the IR for lowering to
         LLVM. Many dialects such as Affine, Math, and Arith define "expansion"
         passes that decompose more sophisticated or auxillary operations into
         simpler operations that can be lowered to LLVM. Therefore, we combine
         these rewrites together and execute them here to avoid having to put a
         separate pass into the pipeline.

      2. Second, the various conversion patterns from each of the allowed
         dialects' `X-to-llvm` conversion passes defined upstream are combined
         into a conversion rewrite set. This conversion is a "full" conversion
         and will fail if any ops or types in the input IR are not fully
         legalized.

    #### Preconditions

    The following dialects are allowed to be present in the input IR when the
    pass is called. If operations from any other dialect are present, the pass
    will fail.

    Fundamental low-level ops:

      Affine, Arith, Vector, ControlFlow, and Func dialect ops. See below for
      additional caveats for the Vector dialect.

    GPU and NVGPU dialect ops:

      GPU dialect `index` ops such as `blockDim` are allowed. These are ops that
      GPU dialect provides NVVM conversions for.

      All NVGPU dialect ops are allowed.

    Buffer access and management: some

      MemRef dialect ops are allowed, but see below regarding memref caveats.

    #### Caveats on the lowering of `vector` ops

      1. In the preparatory phase, `vector` ops that perform memory load/store
         (e.g. `vector.transfer_read|write`) will be converted to `vector.load`
         and `vector.store`. There should be no `vector.transfer_read|write`
         style ops. If the caller wants some special handling of higher-level
         Vector dialect operations, that should be handled prior to this pass.

    #### Caveats on the lowering of `memref` ops

      1. `memref.alloc` in the "global" address space (e.g. GPU DRAM) are illegal.

      2. `memref.alloc` in the "shared memory" address space should be converted
         to `memref.global` operations prior to this pass. This transform will
         be provided by a Kernel dialect Transform extension op.

    #### Caveats on the lowering of `func.func`

      1. `func.func` with bodies are treated as kernel entrypoints and will be
         annotated with NVVM attributes "nvvm.kernel".

  }];

  let dependentDialects = [
    "::mlir::NVVM::NVVMDialect",
    "::mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"preserveStructuredControlFlow",
           "preserve-structured-control-flow",
           "bool", "false",
           "if set, then SCF operations won't be lowered to CF (and then LLVM); "
           "this can be useful if planning to translate or raise LLVM to another "
           "representation besides LLVM IR">
  ];
}

//===----------------------------------------------------------------------===//
// PrepareLinalgForCodegen
//===----------------------------------------------------------------------===//
def PrepareLinalgForCodegenPass :Pass<"kernel-prepare-linalg-for-codegen"> {
  let summary = "performs rewrites on linalg operations"
                " to prepare for schedule generation";

  let description = [{
    This pass performs a number of rewrites on`linalg` and `tensor` operations.
    The purpose of the rewrites is to put the IR in a canonical before the
    codegen transform schedule is generated. Typically this pass would go
    immediately after a conversion pass (e.g. stablehlo-to-linalg).

    In particular, the pass will:
    - Try put all linalg operations into the form of `linalg.generic`. This means
      lowering named ops and lowering `linalg.map` operations.
    - Put ops into destination-passing style, including lowering `tensor.pad`
      to linalg.generic.
  }];

  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ApplyTransforms
//===----------------------------------------------------------------------===//
def ApplyTransformsPass : Pass<"kernel-apply-transforms"> {
  let summary = "Apply Transform schedule to each function in the module";
  let description = [{
    This pass applies a transform schedule to each function in the
    gpu.module. It requires each func.func in the module to have a
    corresponding transform.xxx whose attribute kernel.target_func matches
    the function name. For example, this pass can run after the
    `kernel-initial-transform-schedule` pass which generates the transform
    schedule.

    It transforms each func.func with the corresponding
    transform.xxx and deletes the transform.xxx after the transformation.

    It will emit Error and signal Pass Failure if a function does not have
    a corresponding transform schedule. It will leave a transform schedule
    untouched if it does not have a corresponding function.
  }];
  let dependentDialects = [
    "::mlir::kernel::KernelDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LowerKernelSortPass
//===----------------------------------------------------------------------===//

def LowerKernelSortPass : Pass<"kernel-lower-sort", "::mlir::ModuleOp"> {
  let summary = "Lower kernel.sort operations to generated GPU merge sort kernels";

  let description = [{
    This pass lowers `kernel.sort` operations by generating GPU merge sort
    kernels and replacing the operation with a call to a dispatch function.
    The pass:

    1. Generates the merge sort kernels (block sort, partition, merge) based
       on the configuration specified in the kernel.sort operation
    2. Inserts the generated kernels into the module's gpu.module
    3. Creates a dispatch function that uses `kernel.ext_call` to invoke the
       GPU kernels
    4. Replaces the kernel.sort operation with a `func.call` to the dispatch
       function

    This pass should be run before bufferization and before target assignment.
    The generated kernels follow the CUB merge sort algorithm and use a
    multi-pass approach with block-level sorting followed by iterative merging.
  }];

  let dependentDialects = [
    "::mlir::kernel::KernelDialect",
    "::mlir::func::FuncDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::gpu::GPUDialect"
  ];
}

//===----------------------------------------------------------------------===//
// KernelExpandOpsPass
//===----------------------------------------------------------------------===//

def KernelExpandOpsPass : Pass<"kernel-expand-ops"> {
  let summary = "Lower higher-level Kernel dialect operations to more primitive ops";

  let description = [{

    This pass is a collection of patterns for lowering higher-level utility
    operations. It should be run after lowering kernels to SIMT form.

    Currently it performs the following:

     - Inlines `kernel.combiner` operations

  }];
}

//===----------------------------------------------------------------------===//
// KernelRefineArgumentLayoutsPass
//===----------------------------------------------------------------------===//

def KernelRefineArgumentLayoutsPass : Pass<"kernel-refine-argument-layouts",
      "::mlir::ModuleOp"> {
  let summary = "Refine layouts on kernel function arguments based on callee usage";

  let description = [{
    Since the Kernel modules and any host module are bufferized independently,
    we ensure correctness by assuming kernel function memref arguments have fully
    dynamic layouts (unknown strides, unknown offset). This is usually pretty
    pessimistic, and often actual usage (e.g. via `kernel.call`) will
    provide arguments with identity layout maps (canonical generalized row-major
    layouts).

    This pass looks at all callers (`kernel.call` ops) and builds a map from
    kernel function callees to the `kernel.call` callees. If all callees use
    identity layout map for an argument, then the argument type of the kernel
    function is updated to use the identity layout instead of the fully dynamic
    layout.

    Note that currently refining the "fully dynamic" layout to something with
    partially unknown strides/offset is not supported. An argument is only updated
    if arguments with the identity layout are used for that argument at all call
    sites.
  }];

  let dependentDialects = [
    "::mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// KernelExpandMemRefArgsPass
//===----------------------------------------------------------------------===//

def KernelExpandMemRefArgsPass : Pass<"kernel-expand-memref-args",
                                      "::mlir::gpu::GPUModuleOp"> {
  let summary = "Expand memref arguments to use the optimized calling convention";

  let description = [{

    This pass expands memref arguments to use an optimized calling convention.
    Each memref argument is expanded into a list of types that represent the
    expanded form: an aligned pointer, and one index type for each dynamic
    offset/shape dim/stride.

    The following example illustrates the transformation:

     ```
     func.func @foo(
      %arg0: memref<4x?xf32, strided<offset: ?, strides: [?, 1]>>) {
       ...
     }
     ```

     is transformed into

     ```
     func.func @foo(%arg0_ptr: memref<f32>, %arg0_offset: i64,
                    %arg0_dim1: i64, %arg0_stride0: i64) {
       %arg0 = memref.reinterpret_cast %arg0_ptr to
                offset: [%arg0_offset],
                sizes: [4, %arg0_dim1],
                strides: [%arg0_stride0, 1]
                : memref<f32> to memref<4x?xf32, strided<offset: ?, strides:
                [?, 1]>>
       ...
     }
     ```

     We then use the "bare-pointer" calling convention to lower the function
     during the lowering to the lower-level target to ensure that the
     0-rank memref representing the aligned pointer is passed as a pointer.
  }];

  let dependentDialects = [
    "::mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SetGPUTargetPass
//===----------------------------------------------------------------------===//

def SetGPUTargetPass : Pass<"kernel-set-gpu-target", "::mlir::ModuleOp"> {
  let summary = "Set the GPU target for the module";

  let description = [{
    This pass sets the GPU target for the GPU container module.
  }];

  let options = [
    Option<"chip", "chip", "std::string", "\"\"",
      "Specifies the NVIDIA chip type string, e.g. 'sm_80'.">,
    Option<"features", "features", "std::string", "\"+ptx60\"",
      "Specifies the features to enable for the GPU target. "
      "Features are comma-separated.">,
    Option<"maxSharedMemoryPerBlockKb", "device-max-smem-per-block",
      "int64_t", "48",
      "Specifies the maximum amount of shared memory (in kilobytes) allowed per block">,
    Option<"maxRegistersPerBlock", "device-max-registers-per-block",
      "uint64_t", "65536",
      "Specifies the maximum number of 4-byte registers per thread block">,
    Option<"inferTargetFromHost", "infer-target-from-host",
      "bool", "false",
      "If true, the target will be inferred from the host system."
      " If this is set, the other options are ignored."
    >,
    Option<"populateHostSystemSpec", "populate-host-system-spec",
      "bool", "false",
      "If true, the top-level module will be populated with a DLTI system spec "
      "describing all GPUs in the system."
    >
  ];

  let dependentDialects = [
    "::mlir::DLTIDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::NVVM::NVVMDialect"
  ];
}

//===----------------------------------------------------------------------===//
// TranslateNVVMToPTX
//===----------------------------------------------------------------------===//
def TranslateNVVMToPTXPass : Pass<"translate-nvvm-to-ptx",
                                  "gpu::GPUModuleOp"> {
  let summary = "Translates LLVM/NVVM dialects into PTX";
  let description = [{
    Given an operation, this pass operates on the nested `llvm.func`
    operations, translating them to PTX. The PTX code will be stored as an
    attribute of the operation. The input operation must have been
    completely lowered to the LLVM/NVVM dialects.

    Target information is inferred from the GPU container module's
    target attribute(s).
  }];

  let dependentDialects = [
    "mlir::kernel::KernelDialect"
  ];
  let options = [
    Option<"dumpPtxPath", "dump-ptx", "std::string", "\"\"",
      "Specifies directory where PTX for compiled modules should be saved. Directories "
      " are created automatically. The default is empty (do not save the PTX).">
  ];
}

//===----------------------------------------------------------------------===//
// InferArgIndexingBoundsPass
//===----------------------------------------------------------------------===//

def InferArgIndexingBoundsPass : Pass<"kernel-infer-arg-indexing-bounds", "func::FuncOp"> {
  let summary =
    "Computes bounds on accessed indices of a function argument";

  let description = [{
    The pass takes a `func::FuncOp` as input, and it infers the lower and
    upper bounds of the memory access indices (via `memref.load` and
    `memref.store`) of the function arguments. The pass only runs on the
    function arguments which will be paged -- these arguments must be memref
    type and have an attribute `kernel.page_dim` that specifies which
    dimension of the memref will be paged.

    The inference computes the lower and upper bounds in the form of affine
    map. Because there can be multiple `memref.load` and `memref.store`s on
    one argument, and thus multiple lower and upper bounds, we store all these
    bounds as lists with the parameters `lower` and `upper` in the attribute
    `kernel.page_bounds`. In the end the pass attaches the attribute
    `kernel.page_bounds` to its corresponding function arguments that will
    be paged.

    The lower/upper bound affine maps can depend on function arguments
    as map operands. The map operands can be (1) a function argument of type
    `index`, or (2) loaded from a function argument of type
    `memref<...xindex>` with constant coordinates.
  }];
}

//===----------------------------------------------------------------------===//
// DispatchGPUModuleCompilationPass
//===----------------------------------------------------------------------===//

def DispatchGPUModuleCompilationPass : Pass<
      "kernel-dispatch-gpu-module-compilation",
      "::mlir::gpu::GPUModuleOp"> {
  let summary = "Dispatch the GPU module compilation to the appropriate strategy";

  let description = [{
    This pass dispatches the GPU module compilation to the appropriate strategy
    based on the target and the GPU module kind.
  }];

  let options = [
    Option<"phase", "phase",
           "GPUModuleLoweringPhase",
           "GPUModuleLoweringPhase::PreBufferization",
           "Specifies the phase of the compilation pipeline to run. ",
           "detail::createGpuModuleLoweringPhaseClOptions()">,
    Option<"debugDumpDir", "debug-dump-dir", "std::string", "\"\"",
           "Specifies the directory to dump the IR at each phase of the compilation pipeline.">
  ];
}

//===----------------------------------------------------------------------===//
// KernelSpecialFloatsTypeConversion
//===----------------------------------------------------------------------===//
def KernelSpecialFloatsTypeConversionPass : Pass<"kernel-special-floats-type-conversion",
                                                "::mlir::gpu::GPUModuleOp"> {
  let summary = "Applies type conversion to special floats like fp8, fp4 etc.";
  let description = [{
    This pass applies type conversions to special floating types such as fp8, fp4
    etc. The `LLVMTypeConverter` from the core LLVM (MLIR) repo treats special
    floats such as `f8E4M3FN` an invalid target type during `to LLVM dialect`
    conversion and performs the `special float -> integer of same bit width`
    conversion. However, the NV internal LLVM solid repo treats special floats
    such as `f8E4M3FN` a valid target type. This results in LLVM dialect IR that
    has `f8E4M3FN` data type operations just before conversion to LLVM IR, in
    case project is using LLVM solid.
    Core LLVM IR does not have special floating type support, so any special
    floating time in LLVM dialect IR with result in `LLVM Dialect IR -> LLVM IR`
    translation error. To achieve same behavior with both core LLVM and LLVM
    Solid, it becomes necessary to handle type conversion of special floats in
    a separate pass.

    For example, the following IR with `f8E4M3FN` type
    ```
    gpu.module @arith_emulations [#nvvm.target<chip = "sm_120">] {
      func.func @arith_extf_f8e4m3_f16(%arg0: f8E4M3FN) -> f16 {
        %0 = arith.extf %arg0 : f8E4M3FN to f16
        return %0 : f16
      }
    }
    ```
    becomes,
    ```
    gpu.module @arith_emulations [#nvvm.target<chip = "sm_120">] {
      func.func @arith_extf_f8e4m3_f16(%arg0: i8) -> f16 {
        %0 = arith.bitcast %arg0 : i8 to f8E4M3FN
        %1 = arith.extf %0 : f8E4M3FN to f16
        return %1 : f16
      }
    }
    ```
   In its current state, this pass handles `f8E4M3FN -> i8` and `f4E2M1FN -> i4` type
   conversions.
  }];
  let options = [
    ListOption<"specialTypes", "special-types", "std::string",
    "MLIR types that need special handling before conversion to LLVM IR."
    "If no value is provided, fp8 and fp4 types are added by default.">,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// KernelNormalizeQuantizedConversions
//===----------------------------------------------------------------------===//
def KernelNormalizeQuantizedConversionsPass : Pass<"kernel-normalize-quantized-conversions",
                                                "::mlir::gpu::GPUModuleOp"> {
  let summary = "Makes sure all conversions to and from quantized types happen via fp16.";
  let description = [{
    This pass applies the following two patterns to make sure all
    conversions to and from quantized types happen via fp16.

    - If `arith.truncf` is performing `x->f4/f8` truncation, where
    x ∈ {f16, f32, bf16, f64, ...}, it is decomposed into `x->f16->f4/f8`.
    - If `arith.extf` is performing `f4/f8 -> x` extension, where
    x ∈ {f16, f32, bf16, f64, ...}, it is decomposed into `f4/f8->f16->x`.

  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect"
  ];
}

#endif // MLIR_TENSORRT_DIALECT_KERNEL_TRANSFORMS_PASSES_TD
