//===- Ops.td -------------------------------------------------------------===//
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
#ifndef MLIR_KERNEL_KERNEL_IR_OPS
#define MLIR_KERNEL_KERNEL_IR_OPS

include "mlir-kernel/Kernel/IR/Dialect.td"
include "mlir-kernel/Kernel/IR/Enums.td"
include "mlir-kernel/Kernel/IR/Interfaces.td"
include "mlir-tensorrt-common/Interfaces/ToLoopsOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def Kernel_CallOp : Kernel_Op<"call", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  DestinationStyleOpInterface,
  CallOpInterface
]> {
  let summary = "calls a kernel defined in a `gpu.module`";

  let description = [{
    The `kernel.call` operation calls (launches/executes) a kernel that is
    defined within a `gpu.module`. The `kernel.call` operation cannot be
    nested within a `gpu.module` since it represents an operation executed on
    the host. The kernel accepts `grid_size` values (at least one and up to
    three values). The grid size indicates the size of the grid of threadblocks
    that should be launched. The total number of threadblocks launched is the
    product of all the values given to `grid_size`.

    A kernel may yield some number of results, in which case it is in
    destination-passing style and the results  are tied to the `outs` arguments.

    #### Example:

    ```mlir
    gpu.module @kernels {
      func.func @kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
        %c0 = arith.constant 0 : index
        %0 = memref.load %arg0[%c0] : memref<1xf32>
        memref.store %0, %arg1[%c0] : memref<1xf32>
        return
      }
    }

    func.func @caller(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
      kernel.call @kernels::@kernel(%arg0) outs(%arg1) : (memref<1xf32>)->(memref<1xf32>)
      return
    }
    ```
  }];

  let arguments = (ins Variadic<Index>:$grid_size,
                       Variadic<Index>:$block_size,
                       Variadic<AnyType>:$inputs,
                       Variadic<AnyType>:$outs,
                       SymbolRefAttr:$kernel_sym,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins "TypeRange":$types,
          "ValueRange":$gridSize,
          "ValueRange":$blockSize,
          "ValueRange":$inputs,
          "ValueRange":$outs,
          "SymbolRefAttr":$callee), [{
      build($_builder, $_state,
        types, gridSize, blockSize, inputs, outs,
        callee,
        ArrayAttr{}, ArrayAttr{});
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the function that is being called.
    FunctionOpInterface getKernelCallee(SymbolTableCollection &symbolTable);

    // Declare the outs as inits/outs to DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getOutsMutable(); }

    //===------------------------------------------------------------------===//
    // CallOpInterface
    //===------------------------------------------------------------------===//
    operand_range getArgOperands() {
      return getOperation()->getOperands().take_back(
        getInputs().size()+getOuts().size());
    }
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("kernel_sym");
    }

    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("kernel_sym", cast<SymbolRefAttr>(callee));
    }

    /// Return the number of leading arguments not passed to the callee.
    unsigned getNumNonForwardedArguments() {
      return getGridSize().size()+getBlockSize().size();
    }

    MutableOperandRange getArgOperandsMutable() {
      return MutableOperandRange(getOperation(),
        getNumNonForwardedArguments(),
        getInputs().size()+getOuts().size());
    }

    /// Check if the operand is in either `inputs` or `outs`.
    bool isForwardedOperand(OpOperand *operand) {
      return operand->getOperandNumber() >= getNumNonForwardedArguments();
    }

    /// If `operand` is forwarded to the callee, return which function
    /// argument it corresponds to (its position in the forwarded argument list).
    unsigned getForwardedArgumentIndex(OpOperand *operand) {
      assert(isForwardedOperand(operand) && "expected operand is forwarded");
      return operand->getOperandNumber() - getNumNonForwardedArguments();
    }
  }];

  let assemblyFormat = [{
    $kernel_sym `grid` `[` $grid_size `]`
    `block` `[` $block_size `]` `(` $inputs `)` `outs` `(` $outs `)`
    attr-dict `:` custom<KernelFunctionalType>(
      ref($inputs),
      ref($outs),
      type($inputs), type($outs), type($results))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExtCallOp
//===----------------------------------------------------------------------===//

def Kernel_ExtCallOp : Kernel_Op<"ext_call", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  CallOpInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "extended kernel call with tensor support and explicit aliasing";

  let description = [{
    The `kernel.ext_call` operation is an extended version of `kernel.call` that
    accepts tensors directly with explicit side-effect and aliasing annotations.

    Key features:
    - Single unified argument list (no ins/outs split)
    - Explicit side effects per argument: "r" (read), "rw" (read/write), "-" (none)
    - Explicit aliasing: maps arguments to results for in-place operations
    - Automatic bufferization integration

    The `result_aliases` attribute is an array of argument indices that specifies
    which arguments alias which results. For each `i` in `range(0, num_results)`,
    `result[i]` must alias `args[result_aliases[i]]`.

    The `effects` attribute specifies memory effects for each argument:
    - "r": read-only (for memref/tensor arguments)
    - "rw": read-write (for in-place modified arguments)
    - "-": no effect (for scalars)

    #### Example:

    ```mlir
    %results:2 = kernel.ext_call @kernels::@my_kernel
        grid[%gx] block[%bx]
        args(%keys, %temps, %ping : tensor<?xi32>, tensor<?xi32>, i1)
        result_aliases = [0, 1],  // result[0] aliases arg[0], result[1] aliases arg[1]
        effects = ["rw", "rw", "-"]  // read/write, read/write, no effect
        : tensor<?xi32>, tensor<?xi32>
    ```
  }];

  let arguments = (ins Variadic<Index>:$grid_size,
                       Variadic<Index>:$block_size,
                       Variadic<AnyType>:$args,
                       SymbolRefAttr:$kernel_sym,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       DenseI32ArrayAttr:$result_aliases,
                       StrArrayAttr:$effects);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins
          "::mlir::TypeRange":$resultTypes,
          "::mlir::ValueRange":$gridSize,
          "::mlir::ValueRange":$blockSize,
          "::mlir::ValueRange":$args,
          "::mlir::SymbolRefAttr":$callee,
          "::llvm::ArrayRef<int32_t>":$aliasingArgs,
          "::llvm::ArrayRef<::llvm::StringRef>":$effects), [{
      build($_builder, $_state,
        resultTypes, gridSize, blockSize, args,
        callee,
        ArrayAttr{}, ArrayAttr{},
        aliasingArgs,
        $_builder.getStrArrayAttr(effects));
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the function that is being called.
    FunctionOpInterface getKernelCallee(SymbolTableCollection &symbolTable);

    //===------------------------------------------------------------------===//
    // CallOpInterface
    //===------------------------------------------------------------------===//
    /// Get the argument operands passed to the callee.
    operand_range getArgOperands();

    /// Get the callable symbol reference.
    CallInterfaceCallable getCallableForCallee();

    /// Set the callable symbol reference.
    void setCalleeFromCallable(CallInterfaceCallable callee);

    /// Return the number of leading arguments not passed to the callee.
    unsigned getNumNonForwardedArguments();

    /// Get mutable range of argument operands.
    MutableOperandRange getArgOperandsMutable();

    /// Check if the operand is a kernel argument.
    bool isForwardedOperand(OpOperand *operand);

    /// If `operand` is forwarded to the callee, return which function
    /// argument it corresponds to (its position in the forwarded argument list).
    unsigned getForwardedArgumentIndex(OpOperand *operand);

    /// Check if argument at index has a write effect
    bool argHasWriteEffect(unsigned argIdx);

    /// Check if argument at index has a read effect
    bool argHasReadEffect(unsigned argIdx);
  }];

  let assemblyFormat = [{
    $kernel_sym `grid` `[` $grid_size `]`
    `block` `[` $block_size `]`
    `args` `(` $args `:` type($args) `)`
    `result_aliases` `=` $result_aliases
    `,` `effects` `=` $effects
    attr-dict (`:` type($results)^)?
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CombinerOp
//===----------------------------------------------------------------------===//

def Kernel_CombinerOp : Kernel_Op<"combiner", [
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
      ["getEntrySuccessorOperands"]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface,
      ["getAsmBlockArgumentNames"]>,
    SingleBlockImplicitTerminator<"::mlir::kernel::YieldOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let description = [{
    `kernel.combiner` represents a group of scalar operations which
    collectively represent an associative reduction operation.

    The purpose of this operation is to allow the programmer or a previous
    compiler pass to specify that a group of inline scalar operations
    has an associative property (and optionally one can also specify that the
    group is commutative).

    For example, the following 'kernel.combiner' represents the so-called
    "online softmax" reduction function, which is both associative and commutative:

    ```
    func.func @test_associative(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> (f32, f32) {
      %0, %1 = kernel.combiner (%arg0, %arg1, %arg2, %arg3) : f32, f32, f32, f32 {
      ^bb0(%in0: f32, %in1: f32, %out0: f32, %out1: f32):
        %max = arith.maximumf %in0, %out0 : f32
        %0 = arith.subf %in0, %max : f32
        %1 = arith.subf %out0, %max : f32
        %2 = math.exp %0 : f32
        %3 = math.exp %1 : f32
        %4 = arith.mulf %2, %in1 : f32
        %5 = arith.addf %4, %3 : f32
        kernel.yield %max, %5 : f32, f32
      }
      return %0, %1 : f32, f32
    }
    ```

    The assembly format is very basic and in the future could be improved. The
    semantic of the operation is simply that it forwards the operands to the
    block arguments, executes the operations in the body, and yields the results
    back to the parent. However, the operation also has certain requirements:

    1. It must have an even number of arguments
    2. The types of the first half of the arguments must match the last half.
    3. The number of results is half the number of arguments and the types
      are taken from the first half of the arguments.

    Conceptually, it indicates that the following represents an associative
    reduction on a pair of 2-tuples:

    ```
    f( [in0, in1], [out0, out1]) := [
      max(in0, out0),
      in1 * exp(in0 - max(in0, out0)) + in1 * exp(out0 - max(in0, out0))
    ]
    ````

    Note that no actual verification is performed to ensure that the operation
    is actually commutative since that would be very expensive. The analysis should
    come from the creator (programmer or compiler pass).
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[AnyFloat, AnyComplex, AnySignlessIntegerOrIndex]>>:$inputs,
    UnitAttr:$commutative);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` ( $inputs^ `)` ) : ( `)` ) ?
     attr-dict-with-keyword `:` type($inputs)
     $body
  }];

  let extraClassDeclaration = [{
    /// The "carried" args are the last half of the operands.
    ValueRange getReductionCarriedOperands() {
      return getInputs().drop_front(getInputs().size() / 2);
    }

    /// Return the block arguments corresponding to the
    /// carried values.
    Block::BlockArgListType getReductionCarriedRegionArgs() {
      return getBody().getArguments().drop_front(getInputs().size() / 2);
    }

    /// Return the yielded value corresponding to the specified
    /// carried value.
    Value getYieldedValueForCarriedArg(BlockArgument carriedArg) {
      return getBody().front().getTerminator()->getOperand(carriedArg.getArgNumber() -
        getResults().size());
    }

    /// The "reduction input" args are the first half of the operands.
    ValueRange getReductionInputOperands() {
      return getInputs().take_front(getInputs().size() / 2);
    }
  }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Kernel_YieldOp : Kernel_Op<"yield", [Pure, Terminator, ReturnLike]> {
  let arguments = (ins Variadic<AnyTypeOf<[AnyFloat, AnyComplex, AnySignlessIntegerOrIndex]>>:$results);

  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];

  let builders = [
    OpBuilder<(ins)>
  ];
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

def Kernel_SortOp : Kernel_Op<"sort", [
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Merge sort operator for keys-only or key-value pairs";

  let description = [{
    The `kernel.sort` operation performs a merge sort on the input tensor(s)
    and returns sorted result(s). It supports two modes:

    1. Keys-only: Sorts a single input tensor of keys
    2. Key-value: Sorts keys and rearranges corresponding values

    The sort is performed along the innermost (fastest-varying) dimension.
    The implementation uses a CUB-inspired two-stage GPU merge sort:
    - Stage 1: Block-level sort using bitonic sort within threads and merge across threads
    - Stage 2: Multi-pass global merge using merge path algorithm

    Configuration parameters:
    - `block_threads`: Number of threads per block (default: 128, recommended: 128)
    - `items_per_thread`: Nominal number of elements per thread (default: 4, recommended: 4)

    ## Type-Based Tuning (Automatic)

    The `items_per_thread` parameter is automatically scaled based on the key type's
    size to maintain constant register pressure. This follows CUB's
    `Nominal4BItemsToItems` strategy (see cub/util/util_math.cuh).

    Formula: `actual_items = min(nominal, max(1, nominal * 4 / sizeof(KeyType)))`

    Examples with `items_per_thread = 4`:
    - `i8`  (1 byte):  4 items/thread (no scaling needed)
    - `i32` (4 bytes): 4 items/thread (optimal baseline)
    - `i64` (8 bytes): 2 items/thread (scaled down to prevent register spills)
    - `f64` (8 bytes): 2 items/thread (scaled down to prevent register spills)

    This means you can use the same configuration for all types, and the compiler
    will automatically optimize for each type:

    ```mlir
    // Same config works optimally for all types!
    %sorted_i32 = kernel.sort ins(%keys_i32 : tensor<?xi32>)
                              outs(%out_i32 : tensor<?xi32>)
                              {block_threads = 128, items_per_thread = 4}
                              -> tensor<?xi32>

    %sorted_i64 = kernel.sort ins(%keys_i64 : tensor<?xi64>)
                              outs(%out_i64 : tensor<?xi64>)
                              {block_threads = 128, items_per_thread = 4}
                              -> tensor<?xi64>
    // Automatically uses 2 items/thread for i64 to prevent register spills!
    ```

    ## Recommended Configuration

    For best performance on Hopper/Blackwell GPUs with MLIR codegen:
    - `block_threads = 128`
    - `items_per_thread = 4`

    These defaults are empirically tuned for MLIR's codegen and provide 35% better
    performance than larger tile sizes. Smaller tiles maximize occupancy and enable
    better memory coalescing with our explicit loop unrolling optimizations.

    Note: These defaults differ from CUB's tuning (256x17) because:
    - MLIR codegen benefits from smaller tiles and better occupancy
    - Hopper architecture prefers more concurrent blocks with smaller working sets
    - Our loop unrolling optimizations work best with smaller tiles

    Users typically do not need to tune these parameters.

    ## Examples

    Example (keys-only with recommended config):
    ```mlir
    %sorted = kernel.sort ins(%keys : tensor<?xi32>)
                          outs(%out : tensor<?xi32>)
                          {block_threads = 128, items_per_thread = 4}
                          -> tensor<?xi32>
    ```

    Example (key-value):
    ```mlir
    %sorted_keys, %sorted_values = kernel.sort
        ins(%keys, %values : tensor<?xi32>, tensor<?xf32>)
        outs(%out_keys, %out_values : tensor<?xi32>, tensor<?xf32>)
        {block_threads = 128, items_per_thread = 4}
        -> tensor<?xi32>, tensor<?xf32>
    ```

    ## References

    - CUB DeviceMergeSort: https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceMergeSort.html
    - CUB Nominal4BItemsToItems: cub/util/util_math.cuh lines 86-91
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                       DefaultValuedAttr<I64Attr, "128">:$block_threads,
                       DefaultValuedAttr<I64Attr, "4">:$items_per_thread
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` ( $inputs^ `)` ) : ( `)` )?
    `<` `block_threads` `=` $block_threads
    `,` `items_per_thread` `=` $items_per_thread `>` attr-dict
    ( `:` type($inputs)^ )?
  }];
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

def Kernel_ScatterOp : Kernel_Op<"scatter", [
  DeclareOpInterfaceMethods<ToLoopsOpInterface, ["lowerToLoops"]>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
  DeclareOpInterfaceMethods<TilingInterface, [
    "getIterationDomain",
    "getLoopIteratorTypes",
    "getTiledImplementation",
    "getResultTilePosition",
    "generateResultTileValue"
  ]>,
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  SameVariadicOperandSize,
  SingleBlockImplicitTerminator<"::mlir::kernel::YieldOp">
]> {
  let description = [{
    The `kernel.scatter` operation performs scatters updates to the
    `init` tensor and returns a new updated tensor.

    The `init` tensor is the starting tensor, and the `updates` tensor
    contains the values to be scattered into the `init` tensor. The `indices`
    tensor specifies the indices of the elements in the `init` tensor
    that should be updated.

    The update is performed by combining the value from `init` with the
    scalar value from `updates` using the `update_computation` region.

    The semantics of this operation exactly match those of
    `stablehlo.scatter`. See the following spec for more details:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter

    This operation implements TilingInterface and ToLoopsOpInterface,
    and it is not currently bufferizable. The expected usage pattern
    is to lower to loops via ToLoopsOpInterface prior to
    bufferization.
  }];

  let arguments = (ins AnyShaped:$indices,
                       Variadic<AnyShaped>:$updates,
                       Variadic<AnyShaped>:$inits,
                       DenseI64ArrayAttr:$update_window_dims,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$inserted_window_dims,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$input_batching_dims,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$scatter_indices_batching_dims,
                       DenseI64ArrayAttr:$scatter_dims_to_operand_dims,
                       I64Attr:$index_vector_dim,
                       UnitAttr:$indices_are_sorted,
                       UnitAttr:$unique_indices);

  let results = (outs Variadic<AnyShaped>:$result);

  let regions = (region SizedRegion<1>:$update_computation /*scatter_i12*/);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `updates` `(` $updates `:` type($updates) `)`
    `into`    `(` $inits `:` type($inits) `)`
    `at`      `(` $indices `:` type($indices) `)`
    $update_computation
    attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    SmallVector<Value> generateScalarImplementationOnTensors(
        RewriterBase &b, Location loc,
        ValueRange updateIVs,
        ValueRange inputBbArgs);
  }];
}

#endif // MLIR_KERNEL_KERNEL_IR_OPS
