//===- Passes.td -----------------------------------------------*- Tablegen -*-===//
//
// SPDX-FileCopyrightText: Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
#ifndef MLIR_EXECUTOR_PASSES_TD
#define MLIR_EXECUTOR_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Conversion pass options and passes
//===----------------------------------------------------------------------===//

defvar ConvertToExecutorOptions = [
  Option<"indexBitwidth", "index-bitwidth", "int64_t", "64",
  "all index types will be converted to signless integers of this bitwidth">
];

def ConvertLinalgToExecutorPass : Pass<"convert-linalg-to-executor"> {
  let summary = "Convert Linalg upstream dialect ops to 'executor' dialect";
  let description = [{
    The purpose of this conversion is to lower certain Linalg operations to
    the Executor dialect.

    Only certain bufferized Linalg operations are supported:

    - 'linalg.fill' filling a host buffer where the element type has size of
      1, 2, or 4 bytes.
  }];
  let dependentDialects = [
    "::mlir::executor::ExecutorDialect"
  ];
  let options = ConvertToExecutorOptions;
}

def ConvertMemRefToExecutorPass : Pass<"convert-memref-to-executor"> {
  let summary = "Convert MemRef dialect ops to Exectuor dialect ops";
  let description = [{
    The purpose of this conversion is to lower `memref` operations to more
    primitive operations of the Executor dialect. After this pass, the any
    `memref` operations should be converted and the `memref` type should be
    eliminated everywhere except at function boundaries. Remaining uses of the
    `memref` type as function arguments and at call sites are eliminated by the
    func-to-executor patterns applied in the `std-to-executor` pass.

  }];
  let dependentDialects = [
    "::mlir::executor::ExecutorDialect",
    ];
  let options = !listconcat(ConvertToExecutorOptions, [
    Option<"allowUncheckedMemrefCastConversion", "allow-unchecked-memref-cast-conversion",
        "bool", "true", "allow converting memref casts without runtime checks">
  ]);
}

def ConvertStdToExecutorPass : Pass<"convert-std-to-executor"> {
  let summary = "Convert standard upstream dialect ops to 'executor' dialect";
  let description = [{
    The purpose of this conversion is to lower or modify IR containing
    operations from three dialects (ControlFlow, Arith, Func) to prepare for
    translation to an interpreter or conversion to another lower-level dialect.

    This pass has the following prerequisuites:

    1. Any Executor op that should be lowered to a `func.call` must already have
       been lowered.

    2. No `memref` dialect operations should be present. Any usage of the
       `memref` type will be expanded to Executor dialect `table` types
       representing descriptors.

    3. There should be no `scf` dialect operations present. The
       `convert-scf-to-cf` should be used to lower `scf` operations.

    After this pass runs:

    1. No index types should be present. Index types are converted to signless
       integers of the specified bitwidth.

    2. No `memref` types should be present. All memref types, including at
       function boundaries, should be converted to `!executor.table<>` types
       that represent descriptors.
  }];
  let dependentDialects = [
    "::mlir::executor::ExecutorDialect",
    "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect",
    "::mlir::math::MathDialect"
  ];
  let options = ConvertToExecutorOptions;
}

def ConvertExecutorToExecutorPass : Pass<"convert-executor-to-executor"> {
  let summary = "Perform exector-to-executor lowerings";
  let description = [{
    The purpose of this conversion is to lower `executor` operations that
    operate on high-level types like `memref` or `index` into lower level
    types. These are mostly structural conversions, such as performing
    type conversions for the `executor.func` and `executor.call` operations.
  }];
  let dependentDialects = [
    "::mlir::executor::ExecutorDialect"
  ];
  let options = ConvertToExecutorOptions;
}

//===----------------------------------------------------------------------===//
// Transform passes (ModuleLikePass and executor transform passes)
//===----------------------------------------------------------------------===//

// This class represents an mlir::InterfacePass.
class ModuleLikePass<string passArg>
  : PassBase<passArg, "::mlir::ModuleLikePass">;

def ExecutorGenerateABIWrappersPass :
      ModuleLikePass<"executor-generate-abi-wrappers"> {
  let summary = "generates ABI wrappers for public functions";

  let description = [{
    This pass generates wrapper functions that are compatible with the
    Executor ABI for all public functions.

    The existing public functions are made private and only the new wrapper
    functions are public after the pass.

    If `force-undef-output-args` is specified, the pass will assign the
    `undef` ABI attribute to all output arguments. This means that the
    code in the ABI wrapper should never read the value of the object
    passed indirectly through the pointer argument. It can only write the
    value; the caller just provides spaces to hold the object but does
    not populate it. If the value type is a `tensor` or `memref` type,
    then this will force the downstream pipeline to pass a new allocation
    back to the caller. Otherwise, it is assumed that the caller provides
    the output buffer.

    It is legal for the compiler to automatically add the `undef` ABI
    attribute to any output argument. This can occur in situations where,
    for example, it is impossible for the caller to know the shape of a
    (statically ranked) memref. In this case, the compiler will automatically
    return an allocation and therefore `undef` is used to indicate that the
    caller only needs to allocate space to hold the descriptor.
  }];

  let dependentDialects = [
    "::mlir::executor::ExecutorDialect"
  ];

  let options = [
    Option<"forceUndefOutputArgs", "force-undef-output-args", "bool", "false",
      "force undefined output arguments to be specified as undef">
  ];
}

def ExecutorLowerABIOpsPass : ModuleLikePass<"executor-lower-abi-ops"> {
  let summary = "finalizes ABI wrapper returns after bufferization";
  let description = [{
    This pass is intended to run on exported Executor ABI wrapper functions
    (i.e. functions annotated with `executor.func_abi`) after module
    bufferization and after the ownership-based buffer deallocation pipeline.

    At this stage, `executor.abi.send` operations are generally not present.
    Module bufferization lowers ABI wrapper semantics as follows:

    - For outputs with `undef = false`, bufferization materializes any required
      `executor.abi.recv` + `memref.copy` into caller-provided storage; no
      live-out buffer is returned.
    - For outputs with `undef = true`, the callee returns the live-out memref
      value directly, and ownership is transferred to the caller.

    This pass finalizes ABI wrapper returns:

    - It rewrites return-like terminators to have an empty operand list and
      erases the function results (results are communicated exclusively via
      output arguments).
    - For scalar/complex return values, it ensures a corresponding
      `executor.abi.send` exists to write the value to the appropriate output
      pointer (unless already produced by `executor.abi.send`).
    - For memref return values whose corresponding output argument is marked
      `undef`, it inserts an `executor.abi.send` to model the required side
      effect of writing the memref descriptor to the output pointer. For
      non-`undef` memref outputs, no additional action is required because the
      data copy was materialized during bufferization.

    Additionally, this pass uses buffer origin/alias analysis to conservatively
    detect returned memref values that may alias each other, inserting
    alloc+copy as needed so that each returned `undef` buffer is a distinct
    allocation suitable for ownership transfer.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::executor::ExecutorDialect"
  ];
}

def ExecutorLowerAggregatesAtFunctionBoundariesPass :
      ModuleLikePass<"executor-lower-aggregates-at-function-boundaries"> {
  let summary = "lowers aggregates at function boundaries";

  let description = [{
    This pass lowers aggregates at function boundaries.

    There are three possible lowering modes:

    1. `direct`: Lower aggregates at function boundaries by passing
        aggregates by-value. If this option is specified, the pass
        has no effect; the IR is left unchanged.

    2. `indirect`: Lower aggregates at function boundaries by passing
        aggregates through indirect buffers. At call sites, we use
        stack-allocated buffers (alloca) to hold the aggregate.
        Aggregate results are promoted to destination pointer
        operands and the caller must allocate that memory.

    3. `unpacked`: Lower aggregates at function boundaries by passing
       them as a flat list of scalars. This expands all aggregates into
       many scalar arguments or results.

    If `indirect` or `unpacked` mode is specified, then the pass
    also inserts `executor.result` arguments to specify the original
    result index of the function prior to modification.
  }];

  let options = [
    Option<"mode", "mode", "std::string", "\"indirect\"",
      "the mode in which to lower aggregates at function boundaries">
  ];
}

def ExecutorPopulateFunctionMetadataPass
            : InterfacePass<"executor-populate-func-metadata", "FunctionOpInterface"> {
  let summary = "Populate function metadata on functions";

  let description = [{
    This pass create a FunctionMetadataAttr for each public function and attaches
    it to the function under the `executor.function_metadata` name.

    This pass must be run on bufferized IR, and it must be run prior to running
    any lowering or conversion of the bufferized IR to Executor or LLVM IR.
  }];
}

def ExecutorExpandOpsPass : Pass<"executor-expand-ops"> {
  let summary = "Decompose some Executor ops into more primitive operations";

  let description = [{
    This pass decomposes some Executor operations into more primitive ops.

    Specifically, it does the following depending on the pass options:

    - `lower-getoffset`: Lowers `executor.gep` operations into more primitive
      math operations. This pass should be invoked when translating to target
      languages or IRs that don't have a similar operation builtin. When
      converting Executor to LLVM IR, the GEP operation should be preserved.

    - `lower-alloca`: Lowers `executor.alloca` operations into pairs of
      `executor.(alloc|dealloc)` within the block scope. This should be used
      for targeting interpreters or languages that don't have an `alloca`
      equivalent.

  }];

  let options = [
    Option<"lowerAlloca", "lower-alloca", "bool", "true",
      "lower 'executor.alloca' into an alloc/dealloc pair within the block scope">,
    Option<"lowerGetOffset", "lower-getoffset", "bool", "true",
      "lower 'executor.getoffset' into more primitive arithmetic operations">,
    Option<"checkGetOffsetPrecisionLoss", "check-getoffset-precision-loss", "bool", "true",
      "insert runtime assertions to detect precision loss when getoffset result "
      "type differs from the DataLayout index type">
  ];
}

def ExecutorPackArgumentsPass : ModuleLikePass<"executor-pack-arguments"> {
  let summary = "changes calling convention of executor functions to packed";

  let description = [{

    This pass updates the calling convention of functions that contain a large
    number of arguments (the threshold is specified in the options). Some
    backends such as Lua cannot handle a number of arguments beyond some
    limit, so functions exceeding that limit must be rewritten to pack arguments
    into an aggregate structure.

    If the number of arguments exceeds the threshold, the function is
    rewritten to use the `packed` calling convention. This is accomplished
    by changing the function signature to accept a single executor Table
    object. The table object contains all the original argument types packed
    into the aggregate. The entry block then unpacks the individual elements
    as required.

    The metadata of the new function is the same as the old one (except for the
    calling convention). Therefore, runtime clients that inspect the metadata
    will see the original list of arguments. The runtime clients must check
    the calling convention to see whether or not the arguments are given as a
    flat list or if they are packed into an aggregate.
  }];

  let options = [
    Option<"maxArguments", "max-arguments", "int64_t", "100",
      "maximum number of arguments allowed for unpacked calling convention">
  ];
}

def ExecutorDecomposeAggregateLoadsAndStoresPass :
      ModuleLikePass<"executor-decompose-aggregate-loads-and-stores"> {
  let summary = "lowers load/store of aggregates into more primitive ops";

  let description = [{
    This pass replaces 'executor.load|store' operations on aggregates with
    more primitive operations.
  }];

  let options = [
    Option<"target", "target", "std::string", "\"lua\"",
      "a target which helps identify which ops must be lowered to calls">
  ];
}

def ExecutorLowerToRuntimeBuiltinsPass :
        ModuleLikePass<"executor-lower-to-runtime-builtins"> {
  let summary = "lowers some primitive operations to runtime builtin calls";

  let description = [{
    This pass replaces executor operations which must be converted to a call to
    a runtime-provieded function. The determination of which operations must
    be lowered to a call is based on the target.
  }];

  let options = [
    Option<"target", "target", "std::string", "\"lua\"",
      "a target which helps identify which ops must be lowered to calls">,
    Option<"indexBitwidth", "index-bitwidth", "int64_t", "64",
      "all index types will be converted to signless integers of this bitwidth">
  ];
}

def ExecutorLowerGlobalsPass : ModuleLikePass<"executor-lower-globals"> {
  let summary = "Lower globals to their final form before translation";

  let description = [{
    This pass lowers global declarations and use/set operations to their
    final pre-translation form.

    Adds a pass `executor-lower-globals` that lowers global declarations to
    their final pre-translation form. To lower `executor.global` operations with
    initialization regions, it creates a top-level function
    `executor_globals_init` that contains the concatenated IR to initialize
    globals for each `executor.global` operation that has an initialization
    region. Globals with an initial_value attribute are lowered to constant
    resource loads.

  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect"
  ];
}

def ExecutorAllocsToGlobalsPass : ModuleLikePass<"executor-allocs-to-globals"> {
  let summary = "Move long-lived allocations to global buffers";
  let description = [{
    This pass moves `memref`s that are allocated and deallocated in the same
    block and have a long live-range to globals. This moves the overhead of the
    allocation and deallocation of the temporary buffers out of the function
    execution codepath.
  }];

  let dependentDialects = [
    "::mlir::executor::ExecutorDialect"
  ];
}

def ExecutorEmulateUnsupportedFloatsPass :
        ModuleLikePass<"executor-emulate-unsupported-floats"> {
  let summary = "Emulate executor dialect operations for unsupported floats";
  let description = [{
    The Executor dialect doesn't provide runtime implementation of its operations
    for low-precision quantized float types such as `f4E2M1FN`. This pass emulates
    computations on the `f4E2M1FN` type using `f16`. Function boundaries maintain
    the `f4E2M1FN` type, while materialization between `f4E2M1FN` ↔ `f16` occurs
    via the executor's `extf` and `truncf` ops.

    Constant-like ops of type `f4E2M1FN` are converted to `f16` ops.

    The emulation target type is fixed as half-precision (f16). The option
    `unsupportedSourceTypes` can be used to specify unsupported executor runtime
    types. If no option is provided, `f4E2M1FN` is added by default, as it
    is not supported by the executor runtime.

  }];

  let options = [
    ListOption<"unsupportedSourceTypes", "unsupported-source-types", "std::string",
    "MLIR types without executor runtime support">,
  ];

  let dependentDialects = [
    "::mlir::executor::ExecutorDialect"
  ];
}

def ExecutorExpandMathOpsPass :
        ModuleLikePass<"executor-expand-math-ops"> {
  let summary = "Expand unsupported Math dialect operations";
  let description = [{
    This pass expands Math dialect operations that are not directly supported by
    the Executor dialect into compositions of supported operations.

    It uses two strategies:

    1. **Expansion patterns**: Operations like `math.sinh`, `math.cosh`,
       `math.powf`, `math.rsqrt`, `math.fma`, `math.clampf`, etc. are expanded
       into compositions of more primitive operations (e.g., `sinh(x)` becomes
       `(exp(x) - exp(-x)) / 2`).

    2. **Polynomial approximations**: Operations like `math.asin`, `math.acos`,
       `math.atan`, `math.erfc` that cannot be simply expanded are approximated
       using polynomial functions.

    This pass should be run before `convert-std-to-executor` to ensure all Math
    operations are lowered to forms that can be converted to Executor dialect.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect"
  ];
}

def ExecutorSerializeArtifactsPass : ModuleLikePass<"executor-serialize-artifacts"> {
  let summary = "Serializes artifacts to files";
  let description = [{
    This pass serializes artifacts to files.
  }];

  let options = [
    Option<"artifactsDirectory", "artifacts-directory", "std::string", "\"\"",
      "the directory to serialize artifacts to">,
    Option<"createManifest", "create-manifest", "bool", "true",
      "create a manifest file for the artifacts">,
  ];
}

#endif // MLIR_EXECUTOR_PASSES_TD
