#ifndef MLIR_TENSORRT_DIALECT_EXECUTOR_TRANSFORMS_PASSES_TD
#define MLIR_TENSORRT_DIALECT_EXECUTOR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

// This class represents an mlir::InterfacePass.
class ModuleLikePass<string passArg>
  : PassBase<passArg, "::mlir::ModuleLikePass">;

//===----------------------------------------------------------------------===//
// ExecutorPopulateFunctionMetadataPass
//===----------------------------------------------------------------------===//

def ExecutorPopulateFunctionMetadataPass
            : Pass<"executor-populate-func-metadata", "::mlir::func::FuncOp"> {
  let summary = "Populate function metadata on functions";

  let description = [{
    This pass create a FunctionMetadataAttr for each public function and attaches
    it to the fucntion under the `executor.function_metadata` name.

    This pass must be run on bufferized IR, and it must be run prior to running
    any lowering or conversion of the bufferized IR to Executor or LLVM IR.
  }];
}


//===----------------------------------------------------------------------===//
// ExecutorExpandOpsPass
//===----------------------------------------------------------------------===//

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
      "lower 'executor.getoffset' into more primitive arithmetic operations">
  ];
}

//===----------------------------------------------------------------------===//
// PackArguments
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ExecutorDecomposeAggregateLoadsAndStoresPass
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ExecutorLowerToRuntimeBuiltinsPass
//===----------------------------------------------------------------------===//

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
      "all index types will be converted to signless integers of this bitwidth">,
    Option<"usePackedMemRefCConv", "use-packed-memref-cconv",
      "bool", "true",
      "convert memref arguments in functions to table/struct rather than to "
      "an unpacked list of scalars">
  ];
}

//===----------------------------------------------------------------------===//
// ExecutorLowerGlobalsPass
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ExecutorAllocsToGlobalsPass
//===----------------------------------------------------------------------===//
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

#endif // MLIR_TENSORRT_DIALECT_EXECUTOR_TRANSFORMS_PASSES_TD
