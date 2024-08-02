// RUN: mlir-tensorrt-opt -split-input-file -convert-cuda-to-executor -canonicalize %s | FileCheck %s

func.func @cuda_blas_gemm_algo_select_and_run() {
  %a = memref.alloc() {alignment = 64 : i64} : memref<100x200xf32>
  %b = memref.alloc() {alignment = 64 : i64} : memref<200x300xf32>
  %c = memref.alloc() {alignment = 64 : i64} : memref<100x300xf32>
  %beta = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
  %alpha = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
  %h = cuda.blas.handle.create : !cuda.blas.handle
  %s = cuda.stream.create : !cuda.stream
  %r = cuda.blas.algo_select {
    data_type = f32,
    size_a = array<i64: 100, 200>,
    stride_a = array<i64: 200, 1>,
    size_b = array<i64: 200, 300>,
    stride_b = array<i64: 300, 1>,
    size_c = array<i64: 100, 300>,
    stride_c = array<i64: 300, 1>, 
    tile_sizes = array<i64: 0, 0>
  }  %h : !cuda.blas.gemm_algorithm
  cuda.blas.run_gemm %h stream (%s) algo (%r) inputs(alpha %alpha, %a, %b, beta %beta) out (%c) :
    !cuda.blas.handle, !cuda.stream, !cuda.blas.gemm_algorithm, memref<1xf32>,
    memref<100x200xf32>, memref<200x300xf32>, memref<1xf32>, memref<100x300xf32>
  cuda.blas.handle.destroy %h : !cuda.blas.handle
  cuda.stream.destroy %s : !cuda.stream

  return
}

//       CHECK:   executor.func private @__cuda_stream_destroy(!executor.ptr<host>)
//       CHECK:   executor.func private @__cuda_blas_handle_destroy(!executor.opaque<"cuda_blas_handle">)
//       CHECK:   executor.func private @__cuda_blas_run_gemm(!executor.opaque<"cuda_blas_handle">, !executor.ptr<host>, !executor.opaque<"cuda_blas_gemm_algorithm">, !executor.ptr<host>, !executor.ptr<host>, !executor.ptr<host>, !executor.ptr<host>, !executor.ptr<host>)
//       CHECK:   executor.func private @__cuda_blas_algo_select(!executor.opaque<"cuda_blas_handle">, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !executor.opaque<"cuda_blas_gemm_algorithm">
//       CHECK:   executor.func private @__cuda_stream_create() -> !executor.ptr<host>
//       CHECK:   executor.func private @__cuda_blas_handle_create() -> !executor.opaque<"cuda_blas_handle">
// CHECK-LABEL: func.func @cuda_blas_gemm_algo_select_and_run
//       CHECK:     %[[c300_i64:.+]] = executor.constant 300 : i64
//       CHECK:     %[[c200_i64:.+]] = executor.constant 200 : i64
//       CHECK:     %[[c100_i64:.+]] = executor.constant 100 : i64
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[c1_i64:.+]] = executor.constant 1 : i64
//       CHECK:     %[[c3_i64:.+]] = executor.constant 3 : i64
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<100x200xf32>
//       CHECK:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[alloc]] : memref<100x200xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64>
//       CHECK:     %[[alloc_0:.+]] = memref.alloc() {alignment = 64 : i64} : memref<200x300xf32>
//       CHECK:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[alloc_0]] : memref<200x300xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64>
//       CHECK:     %[[alloc_1:.+]] = memref.alloc() {alignment = 64 : i64} : memref<100x300xf32>
//       CHECK:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[alloc_1]] : memref<100x300xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64>
//       CHECK:     %[[alloc_2:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
//       CHECK:     %[[v3:.+]] = builtin.unrealized_conversion_cast %[[alloc_2]] : memref<1xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     %[[alloc_3:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
//       CHECK:     %[[v4:.+]] = builtin.unrealized_conversion_cast %[[alloc_3]] : memref<1xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     %[[v5:.+]] = executor.call @__cuda_blas_handle_create() : () -> !executor.opaque<"cuda_blas_handle">
//       CHECK:     %[[v6:.+]] = executor.call @__cuda_stream_create() : () -> !executor.ptr<host>
//       CHECK:     %[[v7:.+]] = executor.call @__cuda_blas_algo_select(%[[v5]], %[[c3_i64]], %[[c1_i64]], %[[c100_i64]], %[[c200_i64]], %[[c200_i64]], %[[c1_i64]], %[[c0_i64]], %[[c200_i64]], %[[c300_i64]], %[[c300_i64]], %[[c1_i64]], %[[c0_i64]], %[[c100_i64]], %[[c300_i64]], %[[c300_i64]], %[[c1_i64]], %[[c0_i64]], %[[c0_i64]], %[[c0_i64]]) : (!executor.opaque<"cuda_blas_handle">, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !executor.opaque<"cuda_blas_gemm_algorithm">
//       CHECK:     %[[v8:.+]] = executor.table.get %[[v4]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     %[[v9:.+]] = executor.table.get %[[v0]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64>
//       CHECK:     %[[v10:.+]] = executor.table.get %[[v1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64>
//       CHECK:     %[[v11:.+]] = executor.table.get %[[v3]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     %[[v12:.+]] = executor.table.get %[[v2]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64>
//       CHECK:     executor.call @__cuda_blas_run_gemm(%[[v5]], %[[v6]], %[[v7]], %[[v8]], %[[v9]], %[[v10]], %[[v11]], %[[v12]]) : (!executor.opaque<"cuda_blas_handle">, !executor.ptr<host>, !executor.opaque<"cuda_blas_gemm_algorithm">, !executor.ptr<host>, !executor.ptr<host>, !executor.ptr<host>, !executor.ptr<host>, !executor.ptr<host>) -> ()
//       CHECK:     executor.call @__cuda_blas_handle_destroy(%[[v5]]) : (!executor.opaque<"cuda_blas_handle">) -> ()
//       CHECK:     executor.call @__cuda_stream_destroy(%[[v6]]) : (!executor.ptr<host>) -> ()

// -----

func.func @cuda_blas_gemm_algo_select_with_tile_size() {
  %h = cuda.blas.handle.create : !cuda.blas.handle
  %r = cuda.blas.algo_select {
    data_type = f32,
    size_a = array<i64: 100, 200>,
    stride_a = array<i64: 200, 1>,
    size_b = array<i64: 200, 300>,
    stride_b = array<i64: 300, 1>,
    size_c = array<i64: 100, 300>,
    stride_c = array<i64: 300, 1>,
    tile_sizes = array<i64: 16, 16>
  }  %h : !cuda.blas.gemm_algorithm
  return
}

//       CHECK:   executor.func private @__cuda_blas_algo_select(!executor.opaque<"cuda_blas_handle">, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !executor.opaque<"cuda_blas_gemm_algorithm">
//       CHECK:   executor.func private @__cuda_blas_handle_create() -> !executor.opaque<"cuda_blas_handle">
// CHECK-LABEL: func.func @cuda_blas_gemm_algo_select_with_tile_size
//       CHECK:     %[[c16_i64:.+]] = executor.constant 16 : i64
//       CHECK:     %[[c300_i64:.+]] = executor.constant 300 : i64
//       CHECK:     %[[c200_i64:.+]] = executor.constant 200 : i64
//       CHECK:     %[[c100_i64:.+]] = executor.constant 100 : i64
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[c1_i64:.+]] = executor.constant 1 : i64
//       CHECK:     %[[c3_i64:.+]] = executor.constant 3 : i64
//       CHECK:     %[[v5:.+]] = executor.call @__cuda_blas_handle_create() : () -> !executor.opaque<"cuda_blas_handle">
//       CHECK:     %[[v7:.+]] = executor.call @__cuda_blas_algo_select(%[[v5]], %[[c3_i64]], %[[c1_i64]], %[[c100_i64]], %[[c200_i64]], %[[c200_i64]], %[[c1_i64]], %[[c0_i64]], %[[c200_i64]], %[[c300_i64]], %[[c300_i64]], %[[c1_i64]], %[[c0_i64]], %[[c100_i64]], %[[c300_i64]], %[[c300_i64]], %[[c1_i64]], %[[c0_i64]], %[[c16_i64]], %[[c16_i64]]) : (!executor.opaque<"cuda_blas_handle">, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !executor.opaque<"cuda_blas_gemm_algorithm">