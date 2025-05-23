#ifndef MLIR_TENSORRT_DIALECT_CUDA_IR_CUDADOPS_TD
#define MLIR_TENSORRT_DIALECT_CUDA_IR_CUDADOPS_TD

include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.td"
include "mlir-tensorrt/Dialect/CUDA/IR/CUDATypes.td"
include "mlir-tensorrt/Dialect/CUDA/IR/CUDAInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Shorthand declarations
//===----------------------------------------------------------------------===//

class CUDA_Op<
  string mnemonic,
  list<Trait> traits = []
> : Op<
  CUDA_Dialect,
  mnemonic,
  !listconcat(traits, [CUDAOpInterface])
>;

def I4  : I<4>;
defvar CUDA_ArithFloatTypes = [F64, F32, BF16, F16, F8E4M3FN, BF16];
defvar CUDA_ArithIntTypes = [I64, I32, I16, I8, I4, Index];
def CUDA_Integer : AnyTypeOf<!listconcat(CUDA_ArithIntTypes, [I1])>;
def CUDA_Float : AnyTypeOf<CUDA_ArithFloatTypes>;
defvar CUDA_Float32Or64 = AnyTypeOf<[F64, F32]>;
def CUDA_Complex : Complex<CUDA_Float32Or64>;

//===----------------------------------------------------------------------===//
// CUDA - Device Management Ops
//===----------------------------------------------------------------------===//

def CUDA_DeviceCountOp : CUDA_Op<"num_devices", [Pure]> {
  let summary = "returns the number of CUDA devices on the host";
  let description = [{
    Returns the number of CUDA devices (e.g. GPUs) on the host. This is
    equivalent to the `cudaDeviceCount` CUDA runtime API call.
  }];
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def CUDA_GetDeviceOp : CUDA_Op<"get_device", [Pure]> {
  let summary = "returns the identifier of a particular device";
  let description = [{
    Returns the identifier for a particular CUDA device. Note that
    currently this is just the identity function since CUDA identifies
    devices by their ordinal.
  }];
  let arguments = (ins I32:$deviceNumber);
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict $deviceNumber `:` type($result)";
}

def CUDA_GetCurrentDeviceOp : CUDA_Op<"get_current_device", [Pure]> {
  let summary = "returns the ID of the device associated with the runtime execution context";
  let description = [{
    Returns the ID of the CUDA device that is active for the
    current runtime execution context. In our runtime model, each
    execution context is associated with a single CUDA device.
    Programs that represent the collective action of multiple devices
    must be shaded into either an SPMD program or different programs
    that are mapped onto the available devices.
  }];
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// CUDA - Module Management
//===----------------------------------------------------------------------===//

def CUDA_CompiledModuleOp : CUDA_Op<"compiled_module", [Symbol]> {
  let summary = "represents a compiled CUDA binary";

  let description = [{
    The `cuda.compiled_module` is a global operation that represents
    a compiled CUDA kernel. Its binary data is contained in `value`,
    which can be either PTX or compiled cuBin.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, ElementsAttr:$value);
  let assemblyFormat = "attr-dict $sym_name $value";

  let builders = [
    OpBuilder<(ins "StringRef":$sym_name, "TypedAttr":$attr), [{
      assert(isa<ElementsAttr>(attr) && "expected an ElementsAttr");
      build($_builder, $_state, sym_name, cast<ElementsAttr>(attr));
    }]>
  ];

  let hasVerifier = 1;
}

def CUDA_GetFunctionOp : CUDA_Op<"get_function", [Pure,
      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "retrieves a CUDA function handle from a compiled module";
  let description = [{
    The `cuda.get_function` operation references a `cuda.compiled_module`
    and returns an opaque `cuda.function` (representing a `CUfunction`
    from the CUDA driver API). The specific kernel entrypoint retrieved
    is given by the `kernel_name`.
  }];
  let arguments = (ins FlatSymbolRefAttr:$module,
                       StrAttr:$kernel_name);
  let results = (outs CUDA_Function:$result);
  let assemblyFormat = [{
    attr-dict $kernel_name `from` $module
  }];
}

//===----------------------------------------------------------------------===//
// CUDA - Execution Control
//===----------------------------------------------------------------------===//

def CUDA_LaunchOp : CUDA_Op<"launch", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface, ["getEffects"]>
    ]> {
  let arguments = (ins CUDA_Function:$func,
                       I32:$grid_x, I32:$grid_y, I32:$grid_z,
                       I32:$block_x, I32:$block_y, I32:$block_z,
                       I32:$dynamic_shared_mem,
                       CUDA_Stream:$stream,
                       Variadic<AnyTypeOf<[
                         CUDA_Integer, CUDA_Float, AnyMemRef, CUDA_Complex
                       ]>>:$args);

  let assemblyFormat = [{
    attr-dict
    $func `(` ($args^ `:` type($args))? `)` `with` `\n`
    ` ` `grid` `(` $grid_x `,` $grid_y `,` $grid_z `)` `\n`
    ` ` `block` `(` $block_x `,` $block_y `,` $block_z `)` `\n`
    ` ` `smem` `(` $dynamic_shared_mem `)` `stream` `(` $stream `)`
  }];
}


//===----------------------------------------------------------------------===//
// CUDA - Event Management Ops
//===----------------------------------------------------------------------===//

def CUDA_EventCreateOp : CUDA_Op<"event.create", [AlwaysSpeculatable]> {
  let summary = "Creates a CUDA event";
  let results = (outs CUDA_Event:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def CUDA_EventElapsedTimeOp : CUDA_Op<"event.elapsed"> {
  let summary = "Computes time elapsed between two CUDA events";
  let results = (outs F32:$result);
  let arguments = (ins CUDA_Event:$start,
                       CUDA_Event:$end);
  let assemblyFormat = [{
    attr-dict $start `,` $end `:` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// CUDA - Stream Management Ops
//===----------------------------------------------------------------------===//

def CUDA_StreamCreateOp : CUDA_Op<"stream.create", [AlwaysSpeculatable]> {
  let summary = "Creates an asynchronous CUDA stream";
  let results = (outs Res<CUDA_Stream, "",
                          [MemAlloc<DefaultResource, 0, FullEffect>]>:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def CUDA_GetGlobalStreamOp : CUDA_Op<"get_global_stream", [Pure]> {
  let summary = "gets handle to global CUDA stream";
  let arguments = (ins I64Attr:$index);
  let results = (outs CUDA_Stream:$result);
  let assemblyFormat = "attr-dict $index";
}

def CUDA_StreamWaitEventOp : CUDA_Op<"stream.wait_event",
    [MemoryEffectsOpInterface]> {
  let summary = "Makes stream wait on a CUDA event";
  let arguments = (ins CUDA_Stream:$stream, CUDA_Event:$event);
  let assemblyFormat = "attr-dict $stream `,` $event";

  let extraClassDeclaration = [{
    void getEffects(
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
      effects.emplace_back(MemoryEffects::Read::get());
      effects.emplace_back(MemoryEffects::Write::get());
    }
  }];
}

def CUDA_StreamRecordEventOp : CUDA_Op<"stream.record_event",
    [MemoryEffectsOpInterface]> {
  let summary = "Records an event in the given stream";
  let arguments = (ins CUDA_Stream:$stream, CUDA_Event:$event);
  let assemblyFormat = "attr-dict $stream `,` $event";

  let extraClassDeclaration = [{
    void getEffects(
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
      effects.emplace_back(MemoryEffects::Read::get());
      effects.emplace_back(MemoryEffects::Write::get());
    }
  }];
}

def CUDA_StreamSyncOp : CUDA_Op<"stream.sync",
    [MemoryEffectsOpInterface]> {
  let summary = "Wait for stream tasks to complete";
  let arguments = (ins CUDA_Stream:$stream);
  let assemblyFormat = "attr-dict $stream `:` type($stream)";

  let extraClassDeclaration = [{
    void getEffects(
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
      effects.emplace_back(MemoryEffects::Read::get());
      effects.emplace_back(MemoryEffects::Write::get());
    }
  }];
}

def CUDA_StreamDestroyOp : CUDA_Op<"stream.destroy"> {
  let summary = "Destroy CUDA stream";
  let arguments = (ins Arg<CUDA_Stream, "",
                        [MemFreeAt<0, FullEffect>]>:$stream);
  let assemblyFormat = "attr-dict $stream `:` type($stream)";
}

//===----------------------------------------------------------------------===//
// CUDA - Memory Management Ops
//===----------------------------------------------------------------------===//

def CUDA_AllocOp : CUDA_Op<"alloc", [AttrSizedOperandSegments]> {
  let description = [{
    `cuda.memory.alloc` performs an asynchronous memory allocation on the
    provided `stream`. The memory is associated with the provided device ID.

    The `dynamic_sizes` must be provided for each dynamic dimension of the
    result type.

    An optional alignment may be specified.
  }];
  let arguments = (ins
        Optional<CUDA_Stream>:$stream,
        Optional<I32>:$device,
        Variadic<Index>:$dynamic_sizes,
        OptionalAttr<I64Attr>:$alignment);
  let results = (outs Res<AnyMemRef, "",
    [MemAlloc<DefaultResource, 0, FullEffect>]>:$result);
  let assemblyFormat = [{
    `(` ($dynamic_sizes^ `)`) : (`)`)?
    (`stream` `(` $stream^ `)`)?
    (`device` `(`$device^ `)`)?
    (`align` $alignment^)?
    attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

class CUDA_CopyAsyncOp<string suffix> : CUDA_Op<"copy_"#suffix, []> {
  let arguments = (ins CUDA_Stream:$stream,
                       Arg<AnyMemRef, "memref to copy from",
                        [MemReadAt<0, FullEffect>]>:$source,
                       Arg<AnyMemRef, "memref to copy to",
                        [MemWriteAt<0, FullEffect>]>:$target);
  let assemblyFormat = [{
    attr-dict `stream` `(` $stream`)` $source `,` $target `:` type($source) `to` type($target)
  }];
  let hasVerifier = 1;
}

def CUDA_CopyH2DOp : CUDA_CopyAsyncOp<"h2d"> {
  let description = [{
    `cuda.memcpy_h2d` copies `num_bytes` from host buffer `source` to
    device buffer `target` at the specified offsets.
  }];
}

def CUDA_CopyD2DOp : CUDA_CopyAsyncOp<"d2d"> {
  let description = [{
    `cuda.memcpy_d2d` copies `num_bytes` from device buffer `source` to
    device buffer `target` at the specified offsets.
  }];
}

def CUDA_CopyD2HOp : CUDA_CopyAsyncOp<"d2h"> {
  let description = [{
    `cuda.memcpy_d2h` copies `num_bytes` from device buffer `source` to
    host buffer `target` at the specified offsets.
  }];
}

def CUDA_MemSetOp : CUDA_Op<"memset", []> {
  let arguments = (ins Arg<AnyMemRef, "", [MemWriteAt<0, FullEffect>]>:$memref,
                       AnyTypeOf<[I32, F32, I16, F16, BF16, I8, F8E4M3FN]>:$fill_value);
  let assemblyFormat = [{
    attr-dict $memref `with` $fill_value
    `:` type($memref) `,` type($fill_value)
  }];
  let hasVerifier = 1;
}

def CUDA_DeallocOp : CUDA_Op<"dealloc", []> {
  let arguments = (ins CUDA_Stream:$stream,
    Arg<AnyMemRef, "", [MemFreeAt<0, FullEffect>]>:$memref);
  let results = (outs);
  let assemblyFormat = [{
    attr-dict `stream` `(` $stream `)` $memref `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// CUDA - cuBLAS Ops
//===----------------------------------------------------------------------===//

def CUDA_BlasHandleCreateOp : CUDA_Op<"blas.handle.create",
    [AlwaysSpeculatable]> {
  let summary = "Creates cuBLAS handle";
  let description = [{
    Example,
    ```mlir
     %0 = cuda.blas.handle.create : !cuda.blas.handle
    ```
  }];
  let results = (outs Res<CUDA_BlasHandle, "",
                        [MemAlloc<DefaultResource, 0, FullEffect>]>:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def CUDA_BlasHeuristicAlgoSelectionOp : CUDA_Op<"blas.algo_select", [Pure]> {
  let summary = "Retrieve possible GEMM algorithm, given the problem size";
  let description = [{
    `cuda.blas.algo_select` tries to find the best possible cuBLAS GEMM algorithm
    for the given problem size using cuBLAS `cublasLtMatmulAlgoGetHeuristic` API. If
    no algorithm is found, it returns an error at the runtime.

    Problem size is defined with the following attributes,
    - `data_type`: Data type for input. For now, the input data type and cuBLAS
    matmul algorithm data type is same.
    - `size_a`: Size of input matrix A.

    NOTE: Unit of both the size and the stride is the number of elements of type
    `data_type`.

    - `stride_a`: Stride of input matrix A.

    NOTE: Even though the default cuBLAS format is `column-major`, default format
    used here and implemented at runtime is `row-major`.

    - `transpose_a`: If this unit attribute is set, matrix A is transposed before
    GEMM computation.
    - `size_b`: Size of input matrix B.
    - `stride_b`: Stride of input matrix B.
    - `transpose_b`: If this unit attribute is set, matrix B is transposed before
    GEMM computation.
    - `size_c`: Size of input matrix C.
    - `stride_c`: Stride of input matrix C.
    - `tile_sizes`: Desired CTA tile sizes for the M/N dimensions for the
    selected algorithm. 2 elements are expected in tile sizes. If any element
    is 0, then this argument will be considered as invalid.

    NOTE: All dimensions should be static.

    Example,
    ```mlir
    %r = cuda.blas.algo_select {
        data_type = f32,
        size_a = array<i64: 4, 4>,
        stride_a = array<i64: 4, 1>,
        size_b = array<i64: 4, 4>,
        stride_b = array<i64: 4, 1>,
        size_c = array<i64: 4, 4>,
        stride_c = array<i64: 4, 1>,
        tile_sizes = array<i64: 16, 16>
    }  %h : !cuda.blas.gemm_algorithm
    ```
    If matrix A and C needs to be transposed,

    ```mlir
    %r = cuda.blas.algo_select {
        data_type = f32,
        size_a = array<i64: 3, 2>,
        stride_a = array<i64: 2, 1>,
        transpose_a,
        size_b = array<i64: 3, 4>,
        stride_b = array<i64: 4, 1>,
        size_c = array<i64: 4, 2>,
        stride_c = array<i64: 2, 1>,
        tile_sizes = array<i64: 16, 16>
    }  %h : !cuda.blas.gemm_algorithm
    ```
  }];
  let arguments = (ins CUDA_BlasHandle:$handle,
                  TypeAttr:$data_type,
                  DenseI64ArrayAttr:$size_a,
                  DenseI64ArrayAttr:$stride_a,
                  UnitAttr:$transpose_a,
                  DenseI64ArrayAttr:$size_b,
                  DenseI64ArrayAttr:$stride_b,
                  UnitAttr:$transpose_b,
                  DenseI64ArrayAttr:$size_c,
                  DenseI64ArrayAttr:$stride_c,
                  DenseI64ArrayAttr:$tile_sizes);
  let results = (outs CUDA_BlasGemmAlgorithm:$result);
  let assemblyFormat = [{
    attr-dict $handle `:` type($result)
    }];
  let hasVerifier = 1;
}


def CUDA_BlasRunGemmOp : CUDA_Op<"blas.run_gemm", [
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  AttrSizedOperandSegments,
  DestinationStyleOpInterface,
  AllElementTypesMatch<["alpha", "mat_a", "mat_b", "beta", "mat_c"]>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Runs cuBLAS gemm algorithm";
  let description = [{
    `cuda.blas.run_gemm` runs GEMM or MatMul (Multiply-Accumulate) operation, based
    on input operands.
    GEMM is implemented as,
    `C = alpha*(A@B) + beta*C` (i.e. in-place GEMM).
    MatMul is implemented as,
    `C += A@B` (If `C` is non-zero, result of matmul `(A@B)` is accumulated into `C`.)

    NOTE: `@` represents matrix multiplication and `*` represents elementwise
    multiplication.

    - `handle`: cuBLAS Handle
    - `stream`: CUDA
    - `algo`: A gemm algorithm selected heuristically by `cuda.blas.algo_select` op
    for the given problem size.

    If the input has a `tensor` type, then `algo` should not be specified. The IR
    required to create `algo` is automatically added during bufferization.

    #### Implementing MatMul (C += A @ B)
    1. If input has `tensor` type, only operand `mat_a`, `mat_b` and `mat_c` are
    compulsory.
        Example,
        ```mlir
        %a = arith.constant dense<2.0> : tensor<2x2xf32>
        %b = arith.constant dense<2.0> : tensor<2x2xf32>
        %c = arith.constant dense<0.0> : tensor<2x2xf32>
        %h = cuda.blas.handle.create : !cuda.blas.handle
        %s = cuda.stream.create : !cuda.stream

        %r = cuda.blas.run_gemm %h stream (%s) inputs(%a, %b) out (%c) : !cuda.blas.handle,
        !cuda.stream, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>
        ```
    2. If input has `memref` form, all operands are compulsory. Runtime sets both
    `alpha` and `beta` to 1.
        Example,
        ```mlir
        %a = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32,
                #executor.memory_type<device>>
        %b = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32,
                #executor.memory_type<device>>
        %c = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32,
                #executor.memory_type<device>>

        cuda.blas.run_gemm %h stream (%s) algo (%r) inputs(%a, %b) out (%c) :
        !cuda.blas.handle, !cuda.stream, !cuda.blas.gemm_algorithm,
        memref<2x2xf32, #executor.memory_type<device>>, memref<2x2xf32, #executor.memory_type<device>>, memref<2x2xf32, #executor.memory_type<device>>
        ```

    #### Implementing GEMM (C = alpha*(A@B) + beta*C)
    1. If input has `tensor` form, all operands except `algo` are compulsory.
        Example,
        ```
        %a = arith.constant dense<2.0> : tensor<2x2xf32>
        %b = arith.constant dense<2.0> : tensor<2x2xf32>
        %c = arith.constant dense<2.0> : tensor<2x2xf32>

        // set alpha and beta to 1.
        %alpha = arith.constant dense<1.0> : tensor<1xf32>
        %beta = arith.constant dense<1.0> : tensor<1xf32>
        %h = cuda.blas.handle.create : !cuda.blas.handle
        %s = cuda.stream.create : !cuda.stream

        %r = cuda.blas.run_gemm %h stream (%s) inputs(alpha %alpha, %a, %b, beta %beta)
        out (%c) : !cuda.blas.handle, !cuda.stream, tensor<1xf32>, tensor<2x2xf32>,
        tensor<2x2xf32>, tensor<1xf32>, tensor<2x2xf32>
        ```
    2. If input has `memref` form, all operands are compulsory.
        Example,
        ```
        %a = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32,
                #executor.memory_type<device>>
        %b = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32,
                #executor.memory_type<device>>
        %c = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32,
                #executor.memory_type<device>>

        // set to intended values
        %alpha = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #executor.memory_type<device>>
        %beta = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #executor.memory_type<device>>

        cuda.blas.run_gemm %h stream (%s) algo (%r) inputs(alpha %alpha, %a, %b, beta %beta) out (%c) :
        !cuda.blas.handle, !cuda.stream, !cuda.blas.gemm_algorithm, memref<1xf32, #executor.memory_type<device>>,
        memref<2x2xf32, #executor.memory_type<device>>, memref<2x2xf32, #executor.memory_type<device>>,
        memref<1xf32, #executor.memory_type<device>>, memref<2x2xf32, #executor.memory_type<device>>
        ```

    NOTE: If input is `memref`, memory space is on device.
    NOTE: `transpose_<x>` attribute should be provided wherever necessary.
  }];
  let arguments = (ins UnitAttr:$transpose_a,
                  UnitAttr:$transpose_b,
                  CUDA_BlasHandle:$handle,
                  CUDA_Stream:$stream,
                  Optional<CUDA_BlasGemmAlgorithm>:$algo,
                  Optional<AnyShaped>:$alpha,
                  AnyShaped:$mat_a,
                  AnyShaped:$mat_b,
                  Optional<AnyShaped>:$beta,
                  AnyShaped:$mat_c);
  let results = (outs Variadic<AnyShaped>:$results);

  let assemblyFormat = [{
    $handle `stream` `(` $stream `)` (`algo` `(` $algo^ `)`)?
    `inputs` `(`(`alpha` $alpha^ `,`)? $mat_a  `,` $mat_b  (`,` `beta` $beta^)?`)`
    `out` `(` $mat_c `)` attr-dict `:` type(operands)
  }];
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getMatCMutable();
    }
  }];
  let hasVerifier = 1;
}


def CUDA_BlasHandleDestroyOp : CUDA_Op<"blas.handle.destroy"> {
  let summary = "Destroys cuBLAS handle";
  let description = [{
    Destroys cuBLAS handle.
    Example,
    ```mlir
    %0 = cuda.blas.handle.create : !cuda.blas.handle
    cuda.blas.handle.destroy %0 : !cuda.blas.handle
    ```
  }];
  let arguments = (ins Arg<CUDA_BlasHandle, "",
                          [MemFreeAt<0, FullEffect>]>:$handle);
  let results = (outs );
  let assemblyFormat = "attr-dict $handle `:` type($handle)";
}


#endif // MLIR_TENSORRT_DIALECT_CUDA_IR_CUDADOPS_TD
