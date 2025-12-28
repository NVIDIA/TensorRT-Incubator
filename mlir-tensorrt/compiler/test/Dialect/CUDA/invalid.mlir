// RUN: mlir-tensorrt-opt -split-input-file --verify-diagnostics %s

func.func @gemm_unsupported_data_type(%stream: !cuda.stream) {
  // set all elements of a, b, and c to 2.
  %a = arith.constant dense<2> : tensor<2x2xi8>
  %b = arith.constant dense<2> : tensor<2x2xi8>
  %c = arith.constant dense<2> : tensor<2x2xi8>

  // set alpha and beta to 1.
  %alpha = arith.constant dense<1> : tensor<1xi8>
  %beta = arith.constant dense<1> : tensor<1xi8>

  %h = cuda.blas.handle.create : !cuda.blas.handle

  // expected-error @below {{'cuda.blas.run_gemm' op Currently, only FP16, FP32, F64, and I32 types are supported.}}
  %r = cuda.blas.run_gemm %h stream (%stream) inputs(alpha %alpha, %a, %b, beta %beta) out (%c) : !cuda.blas.handle,
    !cuda.stream, tensor<1xi8>, tensor<2x2xi8>, tensor<2x2xi8>, tensor<1xi8>,
    tensor<2x2xi8>

  return
}

// -----

func.func @matmul_mixed_inputs(%stream: !cuda.stream) {
  %a = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32, strided<[6, 2]>>
  %b = memref.alloc() {alignment = 64 : i64} : memref<3x4xf32>
  %c = memref.alloc() {alignment = 64 : i64} : memref<2x4xf32>
  %h = cuda.blas.handle.create : !cuda.blas.handle
  %r = cuda.blas.algo_select {
    data_type = f32,
    size_a = array<i64: 2, 3>,
    stride_a = array<i64: 6, 2>,
    size_b = array<i64: 3, 4>,
    stride_b = array<i64: 4, 1>,
    size_c = array<i64: 2, 4>,
    stride_c = array<i64: 4, 1>,
    tile_sizes = array<i64: 0, 0>
  }  %h : !cuda.blas.gemm_algorithm
  // expected-error @below {{'cuda.blas.run_gemm' op cuBLAS needs last dimension to be contiguous i.e. stride 1}}
  cuda.blas.run_gemm %h stream (%stream) algo (%r) inputs(%a, %b) out (%c) :
    !cuda.blas.handle, !cuda.stream, !cuda.blas.gemm_algorithm, memref<2x3xf32, strided<[6, 2]>>,
    memref<3x4xf32>, memref<2x4xf32>
  return
}

// -----

func.func @matmul_non_contiguous_last_dim(%stream: !cuda.stream) {
  %a = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
  %b = memref.alloc() {alignment = 64 : i64} : memref<3x4xf32>
  %c = arith.constant dense<0.0> : tensor<2x4xf32>
  %h = cuda.blas.handle.create : !cuda.blas.handle
  %r = cuda.blas.algo_select {
    data_type = f32,
    size_a = array<i64: 2, 3>,
    stride_a = array<i64: 6, 2>,
    size_b = array<i64: 3, 4>,
    stride_b = array<i64: 4, 1>,
    size_c = array<i64: 2, 4>,
    stride_c = array<i64: 4, 1>,
    tile_sizes = array<i64: 16, 16>
  }  %h : !cuda.blas.gemm_algorithm
  // expected-error @below {{'cuda.blas.run_gemm' op If first input is `MemRefType`, all must be `MemRefType`.}}
  cuda.blas.run_gemm %h stream (%stream) algo (%r) inputs(%a, %b) out (%c) :
    !cuda.blas.handle, !cuda.stream, !cuda.blas.gemm_algorithm, memref<2x3xf32>,
    memref<3x4xf32>, tensor<2x4xf32>
  return
}

// -----

func.func @cuda_alloc(%arg0: index, %arg1: !cuda.stream) -> memref<10xf32, #executor.memory_type<device>> {
  // expected-error @below {{'cuda.alloc' op number of dynamic size operands (1) should be equal to the number of dynamic dimensions in the result type (0)}}
  %0 = cuda.alloc(%arg0) stream(%arg1) : memref<10xf32, #executor.memory_type<device>>
  return %0 : memref<10xf32, #executor.memory_type<device>>
}

// -----

func.func @cuda_alloc(%arg0: index, %arg1: !cuda.stream) -> memref<10xf32> {
  // expected-error @below {{'cuda.alloc' op number of dynamic size operands (1) should be equal to the number of dynamic dimensions in the result type (0)}}
  %0 = cuda.alloc(%arg0) stream(%arg1) : memref<10xf32>
  return %0 : memref<10xf32>
}

// -----

// expected-error @below {{'cuda.compiled_module' op expected exactly one of 'value' or 'file' to be specified}}
cuda.compiled_module @missing_payload

// -----

// expected-error @below {{'cuda.compiled_module' op expected exactly one of 'value' or 'file' to be specified}}
cuda.compiled_module @both_payloads dense<[0]> : vector<1xi8> file "kernels.ptx"

// -----

// expected-error @below {{'cuda.compiled_module' op expected kind=ptx when using a 'file' reference}}
cuda.compiled_module @llvm_from_file file "kernels.ll" {kind = #cuda.compiled_module_kind<LLVMIR>}
