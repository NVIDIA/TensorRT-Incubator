// RUN: mlir-tensorrt-opt %s -split-input-file -canonicalize | FileCheck %s

func.func @tensor_input(%stream: !cuda.stream){
  %h = cuda.blas.handle.create : !cuda.blas.handle

  %a_0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %b_0 = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %c_0 = arith.constant dense<0.0> : tensor<2x3xf32>

  %r_0 = cuda.blas.run_gemm %h stream (%stream) inputs(%a_0, %b_0) out (%c_0) : !cuda.blas.handle,
    !cuda.stream, tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  return
}

// CHECK-LABEL: tensor_input
// CHECK-NEXT: return

// -----

func.func @memref_input(%stream: !cuda.stream){
  %h = cuda.blas.handle.create : !cuda.blas.handle

  %a_0 = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
  %b_0 = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
  %c_0 = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>

  %r = cuda.blas.algo_select {
    data_type = f32,
    size_a = array<i64: 2, 2>,
    stride_a = array<i64: 2, 1>,
    size_b = array<i64: 2, 3>,
    stride_b = array<i64: 3, 1>,
    size_c = array<i64: 2, 3>,
    stride_c = array<i64: 3, 1>,
    tile_sizes = array<i64: 8, 16>
  }  %h : !cuda.blas.gemm_algorithm

  cuda.blas.run_gemm %h stream (%stream) algo (%r) inputs(%a_0, %b_0) out (%c_0) : !cuda.blas.handle,
    !cuda.stream, !cuda.blas.gemm_algorithm, memref<2x2xf32>, memref<2x3xf32>, memref<2x3xf32>
  return
}

// CHECK-LABEL: memref_input
// CHECK: %[[a_0:.+]] = memref.alloc()
// CHECK-NEXT: %[[b_0:.+]] = memref.alloc()
// CHECK-NEXT: %[[c_0:.+]] = memref.alloc()
// CHECK-NEXT: %[[algo:.+]] = cuda.blas.algo_select
// CHECK-NEXT: cuda.blas.run_gemm
// CHECK-NEXT: return