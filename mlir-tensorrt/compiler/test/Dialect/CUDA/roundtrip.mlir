// RUN: mlir-tensorrt-opt %s -split-input-file | mlir-tensorrt-opt -split-input-file | FileCheck %s

func.func @cuda_create_stream() -> !cuda.stream {
  %0 = cuda.stream.create : !cuda.stream
  return %0: !cuda.stream
}
// CHECK-LABEL: @cuda_create_stream
//       CHECK: %[[v0:.+]] = cuda.stream.create : !cuda.stream
//       CHECK: return %[[v0]] : !cuda.stream

// -----

func.func @cuda_create_event() -> !cuda.event {
  %0 = cuda.event.create : !cuda.event
  return %0: !cuda.event
}
// CHECK-LABEL: @cuda_create_event
//       CHECK: %[[v0:.+]] = cuda.event.create : !cuda.event
//       CHECK: return %[[v0]] : !cuda.event

// -----

func.func @cuda_stream_wait_event() {
  %0 = cuda.stream.create : !cuda.stream
  %1 = cuda.event.create : !cuda.event
  cuda.stream.wait_event %0, %1
  return
}
// CHECK-LABEL: @cuda_stream_wait_event
//       CHECK: %[[v0:.+]] = cuda.stream.create : !cuda.stream
//       CHECK: %[[v1:.+]] = cuda.event.create : !cuda.event
//       CHECK: cuda.stream.wait_event %[[v0]], %[[v1]]

// -----

func.func @cuda_stream_sync() {
  %0 = cuda.stream.create : !cuda.stream
  cuda.stream.sync %0 : !cuda.stream
  return
}
// CHECK-LABEL: @cuda_stream_sync
//       CHECK: %[[v0:.+]] = cuda.stream.create : !cuda.stream
//       CHECK: cuda.stream.sync %[[v0]] : !cuda.stream

// -----

func.func @cuda_stream_destroy() {
  %0 = cuda.stream.create : !cuda.stream
  cuda.stream.destroy %0 : !cuda.stream
  return
}
// CHECK-LABEL: @cuda_stream_destroy
//       CHECK: %[[v0:.+]] = cuda.stream.create : !cuda.stream
//       CHECK: cuda.stream.destroy %[[v0]] : !cuda.stream

// -----

func.func @cuda_blas_handle_create() -> !cuda.blas.handle {
  %0 = cuda.blas.handle.create : !cuda.blas.handle
  return %0 : !cuda.blas.handle
}
// CHECK-LABEL: @cuda_blas_handle_create
//       CHECK: %[[v0:.+]] = cuda.blas.handle.create : !cuda.blas.handle
//  CHECK-NEXT: return %[[v0]] : !cuda.blas.handle

// -----

func.func @cuda_blas_handle_destroy() {
  %0 = cuda.blas.handle.create : !cuda.blas.handle
  cuda.blas.handle.destroy %0 : !cuda.blas.handle
  return
}
// CHECK-LABEL: @cuda_blas_handle_destroy
//       CHECK: %[[v0:.+]] = cuda.blas.handle.create : !cuda.blas.handle
//  CHECK-NEXT: cuda.blas.handle.destroy %[[v0]] : !cuda.blas.handle
//  CHECK-NEXT: return

// -----

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
    tile_sizes = array<i64: 16, 16>
  }  %h : !cuda.blas.gemm_algorithm
  cuda.blas.run_gemm %h stream (%s) algo (%r) inputs(alpha %alpha, %a, %b, beta %beta) out (%c) : !cuda.blas.handle,
    !cuda.stream, !cuda.blas.gemm_algorithm, memref<1xf32>, memref<100x200xf32>,
    memref<200x300xf32>, memref<1xf32>, memref<100x300xf32>
  cuda.blas.handle.destroy %h : !cuda.blas.handle
  cuda.stream.destroy %s : !cuda.stream
  return
}

// CHECK-LABEL: @cuda_blas_gemm_algo_select_and_run
//       CHECK: %[[alloc:.+]] = memref.alloc() {{.*}} : memref<100x200xf32>
//  CHECK-NEXT: %[[alloc_0:.+]] = memref.alloc() {{.*}} : memref<200x300xf32>
//  CHECK-NEXT: %[[alloc_1:.+]] = memref.alloc() {{.*}} : memref<100x300xf32>
//  CHECK-NEXT: %[[alloc_2:.+]] = memref.alloc() {{.*}} : memref<1xf32>
//  CHECK-NEXT: %[[alloc_3:.+]] = memref.alloc() {{.*}} : memref<1xf32>
//  CHECK-NEXT: %[[v0:.+]] = cuda.blas.handle.create : !cuda.blas.handle
//  CHECK-NEXT: %[[v1:.+]] = cuda.stream.create : !cuda.stream
//  CHECK-NEXT: %[[v2:.+]] = cuda.blas.algo_select {data_type = f32, size_a = array<i64: 100, 200>, size_b = array<i64: 200, 300>, size_c = array<i64: 100, 300>, stride_a = array<i64: 200, 1>, stride_b = array<i64: 300, 1>, stride_c = array<i64: 300, 1>, tile_sizes = array<i64: 16, 16>} %[[v0]] : !cuda.blas.gemm_algorithm
//  CHECK-NEXT: cuda.blas.run_gemm %[[v0]] stream(%[[v1]]) algo(%[[v2]]) inputs(alpha %[[alloc_3]], %[[alloc]], %[[alloc_0]], beta %[[alloc_2]]) out(%[[alloc_1]]) : !cuda.blas.handle, !cuda.stream, !cuda.blas.gemm_algorithm, memref<1xf32>, memref<100x200xf32>, memref<200x300xf32>, memref<1xf32>, memref<100x300xf32>
//  CHECK-NEXT: cuda.blas.handle.destroy %[[v0]] : !cuda.blas.handle
//  CHECK-NEXT: cuda.stream.destroy %[[v1]] : !cuda.stream

// -----

func.func @cuda_blas_matmul_algo_select_and_run() {
  %a = memref.alloc() {alignment = 64 : i64} : memref<100x200xf32>
  %b = memref.alloc() {alignment = 64 : i64} : memref<200x300xf32>
  %c = memref.alloc() {alignment = 64 : i64} : memref<100x300xf32>
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
    tile_sizes = array<i64: 8, 8>
  }  %h : !cuda.blas.gemm_algorithm
  cuda.blas.run_gemm %h stream (%s) algo (%r) inputs(%a, %b) out (%c) : !cuda.blas.handle,
    !cuda.stream, !cuda.blas.gemm_algorithm, memref<100x200xf32>,
    memref<200x300xf32>, memref<100x300xf32>
  cuda.blas.handle.destroy %h : !cuda.blas.handle
  cuda.stream.destroy %s : !cuda.stream
  return
}

// CHECK-LABEL: @cuda_blas_matmul_algo_select_and_run
//       CHECK: %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<100x200xf32>
//  CHECK-NEXT: %[[alloc_0:.+]] = memref.alloc() {alignment = 64 : i64} : memref<200x300xf32>
//  CHECK-NEXT: %[[alloc_1:.+]] = memref.alloc() {alignment = 64 : i64} : memref<100x300xf32>
//  CHECK-NEXT: %[[v0:.+]] = cuda.blas.handle.create : !cuda.blas.handle
//  CHECK-NEXT: %[[v1:.+]] = cuda.stream.create : !cuda.stream
//  CHECK-NEXT: %[[v2:.+]] = cuda.blas.algo_select {data_type = f32, size_a = array<i64: 100, 200>, size_b = array<i64: 200, 300>, size_c = array<i64: 100, 300>, stride_a = array<i64: 200, 1>, stride_b = array<i64: 300, 1>, stride_c = array<i64: 300, 1>, tile_sizes = array<i64: 8, 8>} %0 : !cuda.blas.gemm_algorithm
//  CHECK-NEXT: cuda.blas.run_gemm %[[v0]] stream(%[[v1]]) algo(%[[v2]]) inputs( %[[alloc]], %[[alloc_0]]) out(%[[alloc_1]]) : !cuda.blas.handle, !cuda.stream, !cuda.blas.gemm_algorithm, memref<100x200xf32>, memref<200x300xf32>, memref<100x300xf32>
//  CHECK-NEXT: cuda.blas.handle.destroy %[[v0]] : !cuda.blas.handle
//  CHECK-NEXT: cuda.stream.destroy %[[v1]] : !cuda.stream

// -----


#device_space = #executor.memory_type<device>
!memref_type = memref<?x2x?xf32, #device_space>

func.func @device_alloc_dealloc(%arg0: index, %arg1: index,
  %stream: !cuda.stream, %device: i32) {
  %0 = cuda.alloc(%arg0, %arg1) stream(%stream) device(%device) : !memref_type
  cuda.dealloc stream(%stream) %0 : !memref_type
  return
}

// CHECK-LABEL: func.func @device_alloc_dealloc
//  CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index, %[[arg2:.+]]: !cuda.stream, %[[arg3:.+]]: i32)
//       CHECK:     %[[v0:.+]] = cuda.alloc(%[[arg0]], %[[arg1]]) stream(%[[arg2]]) device(%[[arg3]]) : memref<?x2x?xf32, #executor.memory_type<device>>
//       CHECK:     cuda.dealloc stream(%[[arg2]]) %[[v0]] : memref<?x2x?xf32, #executor.memory_type<device>>

// -----

#device_space = #executor.memory_type<device>
#host_space = #executor.memory_type<host>
!src_memref_type = memref<?x2x?xf32, #device_space>
!dst_memref_type = memref<?x2x?xf32, #device_space>

func.func @d2d_copy(%arg0: !src_memref_type, %arg1: !dst_memref_type, %stream: !cuda.stream) {
  cuda.copy_d2d stream(%stream) %arg0, %arg1 : !src_memref_type to !dst_memref_type
  return
}

// CHECK-LABEL: func.func @d2d_copy
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x2x?xf32, #executor.memory_type<device>>, %[[arg1:.+]]: memref<?x2x?xf32, #executor.memory_type<device>>, %[[arg2:.+]]: !cuda.stream) {
//       CHECK:     cuda.copy_d2d stream(%[[arg2]]) %[[arg0]], %[[arg1]] : memref<?x2x?xf32, #executor.memory_type<device>> to memref<?x2x?xf32, #executor.memory_type<device>>

// -----

#device_space = #executor.memory_type<device>
#host_space = #executor.memory_type<host>

!src_memref_type = memref<?x2x?xf32, #device_space>
!dst_memref_type = memref<?x2x?xf32, #host_space>

func.func @d2h_copy(%arg0: !src_memref_type, %arg1: !dst_memref_type, %stream: !cuda.stream) {
  cuda.copy_d2h stream(%stream) %arg0, %arg1 : !src_memref_type to !dst_memref_type
  return
}

// CHECK-LABEL: func.func @d2h_copy
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x2x?xf32, #executor.memory_type<device>>, %[[arg1:.+]]: memref<?x2x?xf32, #executor.memory_type<host>>, %[[arg2:.+]]: !cuda.stream) {
//       CHECK:     cuda.copy_d2h stream(%[[arg2]]) %[[arg0]], %[[arg1]] : memref<?x2x?xf32, #executor.memory_type<device>> to memref<?x2x?xf32, #executor.memory_type<host>>
// -----

#device_space = #executor.memory_type<host>
#host_space = #executor.memory_type<device>
!src_memref_type = memref<?x2x?xf32, #device_space>
!dst_memref_type = memref<?x2x?xf32, #device_space>

func.func @h2d_copy(%arg0: !src_memref_type, %arg1: !dst_memref_type, %stream: !cuda.stream) {
  cuda.copy_h2d stream(%stream) %arg0, %arg1 : !src_memref_type to !dst_memref_type
  return
}

// CHECK-LABEL: func.func @h2d_copy
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x2x?xf32, #executor.memory_type<host>>, %[[arg1:.+]]: memref<?x2x?xf32, #executor.memory_type<host>>, %[[arg2:.+]]: !cuda.stream) {
//       CHECK:     cuda.copy_h2d stream(%[[arg2]]) %[[arg0]], %[[arg1]] : memref<?x2x?xf32, #executor.memory_type<host>> to memref<?x2x?xf32, #executor.memory_type<host>>

// -----

#device_space = #executor.memory_type<unified>
#host_space = #executor.memory_type<host>
!src_memref_type = memref<?x2x?xf32, #host_space>
!dst_memref_type = memref<?x2x?xf32, #device_space>

func.func @h2u_copy(%arg0: !src_memref_type, %arg1: !dst_memref_type, %stream: !cuda.stream) {
  cuda.copy_h2d stream(%stream) %arg0, %arg1 : !src_memref_type to !dst_memref_type
  return
}

// CHECK-LABEL: func.func @h2u_copy
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x2x?xf32, #executor.memory_type<host>>, %[[arg1:.+]]: memref<?x2x?xf32, #executor.memory_type<unified>>, %[[arg2:.+]]: !cuda.stream)
//       CHECK:     cuda.copy_h2d stream(%[[arg2]]) %[[arg0]], %[[arg1]] : memref<?x2x?xf32, #executor.memory_type<host>> to memref<?x2x?xf32, #executor.memory_type<unified>>
