// RUN: mlir-tensorrt-opt -split-input-file -convert-cuda-to-executor %s | FileCheck %s

func.func @cuda_event(){
  %0 = cuda.event.create : !cuda.event
  %1 = cuda.stream.create : !cuda.stream
  cuda.stream.wait_event %1, %0
  return
}

// CHECK-LABEL: @cuda_event
//       CHECK: %[[v0:.+]] = executor.call @__cuda_event_create() : () -> !executor.ptr<host>
//       CHECK: %[[v1:.+]] = executor.call @__cuda_stream_create() : () -> !executor.ptr<host>
//       CHECK: executor.call @__cuda_stream_wait_event(%[[v1]], %[[v0]]) : (!executor.ptr<host>, !executor.ptr<host>) -> ()
//       CHECK: return

// -----

func.func @cuda_num_devices() -> i32 {
  %0 = cuda.num_devices : i32
  return %0 : i32
}

//       CHECK:   executor.func private @__cuda_num_devices() -> i32
// CHECK-LABEL: func.func @cuda_num_devices
//       CHECK:     %[[v0:.+]] = executor.call @__cuda_num_devices() : () -> i32
//       CHECK:     return %[[v0]] : i32

// -----

func.func @test_get_current_device() -> i32 {
  %0 = cuda.get_current_device
  return %0 : i32
}

// CHECK-LABEL: @test_get_current_device
//       CHECK:     %[[v0:.+]] = executor.call @__spmd_global_rank() : () -> i32
//       CHECK:     return %[[v0]] : i32

// -----

func.func @convert_cuda_get_device(%arg0: i32) -> i32 {
  %0 = cuda.get_device %arg0 : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @convert_cuda_get_device
//  CHECK-SAME: (%[[arg0:.+]]: i32) -> i32 {
//       CHECK:     %[[v0:.+]] = executor.call @__cuda_get_device(%[[arg0]]) : (i32) -> i32
//       CHECK:     return %[[v0]] : i32

// -----

!memref_4xi8 = memref<?x2x?xf32, #executor.memory_type<device>>

func.func @device_alloc(%arg0: index, %arg1: index, %stream: !cuda.stream, %device: i32) -> !memref_4xi8 {
  %0 = cuda.alloc(%arg0, %arg1) stream(%stream) device(%device) align 8 : !memref_4xi8
  return %0 : !memref_4xi8
}

// CHECK-LABEL: func.func @device_alloc
//  CHECK-SAME: (%[[arg0:.+]]: i64, %[[arg1:.+]]: i64, %[[arg2:.+]]: !executor.ptr<host>, %[[arg3:.+]]: i32) -> !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64> {
//       CHECK:     %[[c2_i64:.+]] = executor.constant 2 : i64
//       CHECK:     %[[c1_i64:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v0:.+]] = executor.muli %[[c1_i64]], %[[arg1]] : i64
//       CHECK:     %[[v1:.+]] = executor.muli %[[v0]], %[[c2_i64]] : i64
//       CHECK:     %[[v2:.+]] = executor.muli %[[v1]], %[[arg0]] : i64
//       CHECK:     %[[c4_i64:.+]] = executor.constant 4 : i64
//       CHECK:     %[[v3:.+]] = executor.muli %[[v2]], %[[c4_i64]] : i64
//       CHECK:     %[[v4:.+]] = executor.call @__cuda_alloc_device(%[[arg2]], %[[arg3]], %[[v3]], %{{.+}}) : (!executor.ptr<host>, i32, i64, i32) -> !executor.ptr<device>
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[v5:.+]] = executor.table.create(%[[v4]], %[[v4]], %[[c0_i64]], %[[arg0]], %[[c2_i64]], %[[arg1]], %[[v1]], %[[v0]], %[[c1_i64]] : !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64) : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     return %[[v5]]

// -----

func.func @memref_device_alloc_i1(%arg0: !cuda.stream, %device: i32) -> memref<1500x1500xi1, #executor.memory_type<device>> {
  %0 = cuda.alloc () stream(%arg0) device(%device) : memref<1500x1500xi1, #executor.memory_type<device>>
  return %0 : memref<1500x1500xi1, #executor.memory_type<device>>
}

// CHECK-LABEL: func.func @memref_device_alloc_i1
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: i32) -> !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64> {
//       CHECK:     %[[c1500_i64:.+]] = executor.constant 1500 : i64
//       CHECK:     %[[c1500_i64_0:.+]] = executor.constant 1500 : i64
//       CHECK:     %[[c1_i64:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v0:.+]] = executor.muli %[[c1_i64]], %[[c1500_i64_0]] : i64
//       CHECK:     %[[v1:.+]] = executor.muli %[[v0]], %[[c1500_i64]] : i64
//       CHECK:     %[[c1_i64_1:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v2:.+]] = executor.muli %[[v1]], %[[c1_i64_1]] : i64
//       CHECK:     %[[v3:.+]] = executor.call @__cuda_alloc_device(%[[arg0]], %[[arg1]], %[[v2]], %{{.+}}) : (!executor.ptr<host>, i32, i64, i32) -> !executor.ptr<device>
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[v4:.+]] = executor.table.create(%[[v3]], %[[v3]], %[[c0_i64]], %[[c1500_i64]], %[[c1500_i64_0]], %[[v0]], %[[c1_i64]] : !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64) : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>
//       CHECK:     return %[[v4]] : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>

// -----

!memref_4xi8 = memref<?x2x?xf32, #executor.memory_type<host_pinned>>

func.func @pinned_alloc(%arg0: index, %arg1: index, %stream: !cuda.stream, %device: i32) -> !memref_4xi8 {
  %0 = cuda.alloc(%arg0, %arg1) align 8 : !memref_4xi8
  return %0 : !memref_4xi8
}

// CHECK-LABEL: func.func @pinned_alloc
//  CHECK-SAME: (%[[arg0:.+]]: i64, %[[arg1:.+]]: i64, %[[arg2:.+]]: !executor.ptr<host>, %[[arg3:.+]]: i32) -> !executor.table<!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64, i64, i64, i64, i64> {
//   CHECK-DAG:     %[[c2_i64:.+]] = executor.constant 2 : i64
//   CHECK-DAG:     %[[c1_i64:.+]] = executor.constant 1 : i64
//   CHECK-DAG:     %[[v0:.+]] = executor.muli %[[c1_i64]], %[[arg1]] : i64
//   CHECK-DAG:     %[[v1:.+]] = executor.muli %[[v0]], %[[c2_i64]] : i64
//   CHECK-DAG:     %[[v2:.+]] = executor.muli %[[v1]], %[[arg0]] : i64
//   CHECK-DAG:     %[[c4_i64:.+]] = executor.constant 4 : i64
//   CHECK-DAG:     %[[v3:.+]] = executor.muli %[[v2]], %[[c4_i64]] : i64
//   CHECK-DAG:     %[[v4:.+]] = executor.call @__cuda_alloc_host_pinned(%[[v3]], %{{.+}}) : (i64, i32) -> !executor.ptr<host_pinned>
//   CHECK-DAG:     %[[c0_i64:.+]] = executor.constant 0 : i64
//   CHECK-DAG:     %[[v5:.+]] = executor.table.create(%[[v4]], %[[v4]], %[[c0_i64]], %[[arg0]], %[[c2_i64]], %[[arg1]], %[[v1]], %[[v0]], %[[c1_i64]] : !executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64, i64, i64, i64, i64) : <!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64, i64, i64, i64, i64>
//   CHECK-DAG:     return %[[v5]] : !executor.table<!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64, i64, i64, i64, i64>

// -----

!memref_4xi8 = memref<?x2x?xf32, #executor.memory_type<device>>
func.func @device_free(%arg0: !cuda.stream, %arg1: !memref_4xi8) {
  cuda.dealloc stream(%arg0) %arg1  : !memref_4xi8
  return
}

// CHECK-LABEL: func.func @device_free
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>) {
//       CHECK:     %[[v0:.+]] = executor.table.get %[[arg1]][0] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     executor.call @__cuda_free_device(%[[arg0]], %[[v0]]) : (!executor.ptr<host>, !executor.ptr<device>) -> ()

// -----

!memref_4xi8 = memref<?x2x?xf32, #executor.memory_type<host_pinned>>
func.func @pinned_free(%arg0: !cuda.stream, %arg1: !memref_4xi8) {
  cuda.dealloc stream(%arg0) %arg1  : !memref_4xi8
  return
}

// CHECK-LABEL: func.func @pinned_free
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: !executor.table<!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64, i64, i64, i64, i64>) {
//       CHECK:     %[[v0:.+]] = executor.table.get %[[arg1]][0] : <!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     executor.call @__cuda_free_host_pinned(%[[arg0]], %[[v0]])

// -----

#device_space = #executor.memory_type<device>
#host_space = #executor.memory_type<host>
!src_memref_type = memref<?x2x?xf32, #device_space>
!dst_memref_type = memref<?x2x?xf32, #device_space>

func.func @copy_d2d(%arg0: !src_memref_type, %arg1: !dst_memref_type, %stream: !cuda.stream) {
  cuda.copy_d2d stream(%stream) %arg0, %arg1 : !src_memref_type to !dst_memref_type
  return
}

// CHECK-LABEL: func.func @copy_d2d
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>, %[[arg1:.+]]: !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>, %[[arg2:.+]]: !executor.ptr<host>) {
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[v0:.+]] = executor.getoffset[%[[c0_i64]]] : (i64) -> i64, f32
//       CHECK:     %[[c0_i64_0:.+]] = executor.constant 0 : i64
//       CHECK:     %[[v1:.+]] = executor.getoffset[%[[c0_i64_0]]] : (i64) -> i64, f32
//       CHECK:     %[[c1_i64:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v2:.+]] = executor.table.get %[[arg0]][3] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v3:.+]] = executor.muli %[[c1_i64]], %[[v2]] : i64
//       CHECK:     %[[c2_i64:.+]] = executor.constant 2 : i64
//       CHECK:     %[[v4:.+]] = executor.muli %[[v3]], %[[c2_i64]] : i64
//       CHECK:     %[[v5:.+]] = executor.table.get %[[arg0]][5] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v6:.+]] = executor.muli %[[v4]], %[[v5]] : i64
//       CHECK:     %[[v7:.+]] = executor.getoffset[%[[v6]]] : (i64) -> i64, f32
//       CHECK:     %[[v8:.+]] = executor.table.get %[[arg0]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v9:.+]] = executor.table.get %[[arg1]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     executor.call @__cuda_memcpy_device2device(%[[arg2]], %[[v8]], %[[v0]], %[[v9]], %[[v1]], %[[v7]]) : (!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<device>, i64, i64) -> ()
// -----

func.func @copy_d2h_offset(%arg0: memref<128x16xf32, strided<[16, 1], offset: 16>, #executor.memory_type<device>>,
                           %arg1: memref<128x16xf32, strided<[16, 1], offset: 8>, #executor.memory_type<host>>,
                           %arg2: !cuda.stream) {
  cuda.copy_d2h stream(%arg2) %arg0, %arg1 :
    memref<128x16xf32, strided<[16, 1], offset: 16>, #executor.memory_type<device>>
    to memref<128x16xf32, strided<[16, 1], offset: 8>, #executor.memory_type<host>>
  return
}

// CHECK-LABEL: func.func @copy_d2h_offset
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>, %[[arg1:.+]]: !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64>, %[[arg2:.+]]: !executor.ptr<host>) {
//       CHECK:     %[[c16_i64:.+]] = executor.constant 16 : i64
//       CHECK:     %[[v0:.+]] = executor.getoffset[%[[c16_i64]]] : (i64) -> i64, f32
//       CHECK:     %[[c8_i64:.+]] = executor.constant 8 : i64
//       CHECK:     %[[v1:.+]] = executor.getoffset[%[[c8_i64]]] : (i64) -> i64, f32
//       CHECK:     %[[c2048_i64:.+]] = executor.constant 2048 : i64
//       CHECK:     %[[v2:.+]] = executor.getoffset[%[[c2048_i64]]] : (i64) -> i64, f32
//       CHECK:     %[[v3:.+]] = executor.table.get %[[arg0]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>
//       CHECK:     %[[v4:.+]] = executor.table.get %[[arg1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64>
//       CHECK:     executor.call @__cuda_memcpy_device2host(%[[arg2]], %[[v3]], %[[v0]], %[[v4]], %[[v1]], %[[v2]]) : (!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<host>, i64, i64)

// -----

!srcType = memref<6xf32, strided<[2], offset: 2>, #executor.memory_type<device>>
!dstType = memref<6xf32, strided<[2], offset: 4>, #executor.memory_type<host>>

func.func @copy_d2h_strided(%arg0: !srcType,
                           %arg1: !dstType, %arg2: !cuda.stream) {
  cuda.copy_d2h stream(%arg2) %arg0, %arg1 : !srcType to !dstType
  return
}

// CHECK-LABEL: func.func @copy_d2h_strided
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>, %[[arg1:.+]]: !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>, %[[arg2:.+]]: !executor.ptr<host>) {
//       CHECK:     %[[c1_i32:.+]] = executor.constant 1 : i32
//       CHECK:     %[[v0:.+]] = executor.alloca %[[c1_i32]] x !executor.table<i64, i64> : (i32) -> !executor.ptr<host>
//       CHECK:     %[[v1:.+]] = executor.getoffset[0, 0] : () -> i64, !executor.table<i64, i64>
//       CHECK:     %[[c6_i64:.+]] = executor.constant 6 : i64
//       CHECK:     executor.store %[[c6_i64]] to %[[v0]] + %[[v1]] : i64, !executor.ptr<host>, i64
//       CHECK:     %[[v2:.+]] = executor.getoffset[0, 1] : () -> i64, !executor.table<i64, i64>
//       CHECK:     %[[v3:.+]] = executor.table.get %[[arg0]][4] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//       CHECK:     executor.store %[[v3]] to %[[v0]] + %[[v2]] : i64, !executor.ptr<host>, i64
//       CHECK:     %[[c1_i32_0:.+]] = executor.constant 1 : i32
//       CHECK:     %[[v4:.+]] = executor.alloca %[[c1_i32_0]] x !executor.table<i64, i64> : (i32) -> !executor.ptr<host>
//       CHECK:     %[[v5:.+]] = executor.getoffset[0, 0] : () -> i64, !executor.table<i64, i64>
//       CHECK:     %[[c6_i64_1:.+]] = executor.constant 6 : i64
//       CHECK:     executor.store %[[c6_i64_1]] to %[[v4]] + %[[v5]] : i64, !executor.ptr<host>, i64
//       CHECK:     %[[v6:.+]] = executor.getoffset[0, 1] : () -> i64, !executor.table<i64, i64>
//       CHECK:     %[[v7:.+]] = executor.table.get %[[arg1]][4] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     executor.store %[[v7]] to %[[v4]] + %[[v6]] : i64, !executor.ptr<host>, i64
//       CHECK:     %[[c1_i64:.+]] = executor.constant 1 : i64
//       CHECK:     %[[c4_i64:.+]] = executor.constant 4 : i64
//       CHECK:     %[[v8:.+]] = executor.table.get %[[arg0]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//       CHECK:     %[[c2_i64:.+]] = executor.constant 2 : i64
//       CHECK:     %[[v9:.+]] = executor.table.get %[[arg1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     %[[c4_i64_2:.+]] = executor.constant 4 : i64
//       CHECK:     executor.call @__cuda_memcpy_strided_async_device2host(%[[arg2]], %[[c1_i64]], %[[c4_i64]], %[[v8]], %[[c2_i64]], %[[v0]], %[[v9]], %[[c4_i64_2]], %[[v4]]) : (!executor.ptr<host>, i64, i64, !executor.ptr<device>, i64, !executor.ptr<host>, !executor.ptr<host>, i64, !executor.ptr<host>)

// -----

!srcType = memref<1x8x4xf32, strided<[256, 4, 1], offset: ?>, #executor.memory_type<device>>
!dstType = memref<1x8x4xf32, strided<[128, 4, 1], offset: ?>, #executor.memory_type<host>>

func.func @memref_copy_contiguous_non_identity(%arg0: !srcType, %arg1: !dstType,
    %arg2: !cuda.stream) {
  cuda.copy_d2h stream(%arg2) %arg0, %arg1 : !srcType to !dstType
  return
}

// CHECK-LABEL: func.func @memref_copy_contiguous_non_identity
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>, %[[arg1:.+]]: !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64, i64, i64>, %[[arg2:.+]]: !executor.ptr<host>) {
//       CHECK:     %[[v0:.+]] = executor.table.get %[[arg0]][2] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v1:.+]] = executor.getoffset[%[[v0]]] : (i64) -> i64, f32
//       CHECK:     %[[v2:.+]] = executor.table.get %[[arg1]][2] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v3:.+]] = executor.getoffset[%[[v2]]] : (i64) -> i64, f32
//       CHECK:     %[[c32_i64:.+]] = executor.constant 32 : i64
//       CHECK:     %[[v4:.+]] = executor.getoffset[%[[c32_i64]]] : (i64) -> i64, f32
//       CHECK:     %[[v5:.+]] = executor.table.get %[[arg0]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v6:.+]] = executor.table.get %[[arg1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     executor.call @__cuda_memcpy_device2host(%[[arg2]], %[[v5]], %[[v1]], %[[v6]], %[[v3]], %[[v4]]) : (!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<host>, i64, i64) -> ()
//       CHECK:     return

// -----

#device_space = #executor.memory_type<device>
#host_space = #executor.memory_type<host>

!src_memref_type = memref<?x2x?xf32, #device_space>
!dst_memref_type = memref<?x2x?xf32, #host_space>

func.func @copy_d2h(%arg0: !src_memref_type, %arg1: !dst_memref_type,  %stream: !cuda.stream) {
  cuda.copy_d2h stream(%stream) %arg0, %arg1 : !src_memref_type to !dst_memref_type
  return
}


// -----

#device_space = #executor.memory_type<host>
#host_space = #executor.memory_type<device>
!src_memref_type = memref<?x2x?xf32, #device_space>
!dst_memref_type = memref<?x2x?xf32, #device_space>

func.func @copy_h2d(%arg0: !src_memref_type, %arg1: !dst_memref_type,  %stream: !cuda.stream) {
  cuda.copy_h2d stream(%stream) %arg0, %arg1 : !src_memref_type to !dst_memref_type
  return
}

// -----

!memref_4xi8 = memref<4xi8, #executor.memory_type<device>>

memref.global "private" @global2 : memref<4xi8, #executor.memory_type<host_pinned>> {alignment = 32 : i64}
memref.global "private" @globalDevice : !memref_4xi8 = dense<[5, 6, 7, 8]>

func.func @memref_global() -> (memref<4xi8, #executor.memory_type<host_pinned>>, !memref_4xi8) {
  %1 = memref.get_global @global2 : memref<4xi8, #executor.memory_type<host_pinned>>
  %2 = memref.get_global @globalDevice : !memref_4xi8
  return %1, %2 : memref<4xi8, #executor.memory_type<host_pinned>>, !memref_4xi8
}

// CHECK-LABEL:   executor.global @global2 : !executor.table<!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64> {
//       CHECK:     %[[c4_i64:.+]] = executor.constant 4 : i64
//       CHECK:     %[[c1_i64:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v0:.+]] = executor.muli %[[c1_i64]], %[[c4_i64]] : i64
//       CHECK:     %[[c1_i64_0:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v1:.+]] = executor.muli %[[v0]], %[[c1_i64_0]] : i64
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[v2:.+]] = executor.call @__cuda_stream_create() : () -> !executor.ptr<host>
//       CHECK:     %[[c32_i32:.+]] = executor.constant 32 : i32
//       CHECK:     %[[v5:.+]] = executor.call @__cuda_alloc_host_pinned(%[[v1]], %[[c32_i32]]) :
//       CHECK:     %[[v6:.+]] = executor.table.create(%[[v5]], %[[v5]], %[[c0_i64]], %[[c4_i64]], %[[c1_i64]] : !executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64) : <!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64>
//       CHECK:     %[[v7:.+]] = executor.table.get %[[v6]][1] : <!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64>
//       CHECK:     executor.call @__cuda_stream_sync(%[[v2]]) : (!executor.ptr<host>) -> ()
//       CHECK:     executor.call @__cuda_stream_destroy(%[[v2]]) : (!executor.ptr<host>) -> ()
//       CHECK:     executor.return %[[v6]] : !executor.table<!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64>
//       CHECK:   executor.data_segment @globalDevice_initializer constant dense<[5, 6, 7, 8]> : tensor<4xi8>
// CHECK-LABEL:   executor.global @globalDevice : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64> {
//       CHECK:     %[[c4_i64:.+]] = executor.constant 4 : i64
//       CHECK:     %[[c1_i64:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v0:.+]] = executor.muli %[[c1_i64]], %[[c4_i64]] : i64
//       CHECK:     %[[c1_i64_0:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v1:.+]] = executor.muli %[[v0]], %[[c1_i64_0]] : i64
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[v2:.+]] = executor.call @__cuda_stream_create() : () -> !executor.ptr<host>
//       CHECK:     %[[v3:.+]] = executor.call @__spmd_global_rank() : () -> i32
//       CHECK:     %[[v5:.+]] = executor.call @__cuda_alloc_device(%[[v2]], %[[v3]], %[[v1]], %{{.+}}) : (!executor.ptr<host>, i32, i64, i32) -> !executor.ptr<device>
//       CHECK:     %[[v6:.+]] = executor.table.create(%[[v5]], %[[v5]], %[[c0_i64]], %[[c4_i64]], %[[c1_i64]] : !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64) : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//       CHECK:     %[[v7:.+]] = executor.table.get %[[v6]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//       CHECK:     %[[v8:.+]] = executor.load_data_segment @globalDevice_initializer : !executor.ptr<host>
//       CHECK:     executor.call @__cuda_memcpy_host2device(%[[v2]], %[[v8]], %[[c0_i64]], %[[v7]], %[[c0_i64]], %[[v1]]) : (!executor.ptr<host>, !executor.ptr<host>, i64, !executor.ptr<device>, i64, i64) -> ()
//       CHECK:     executor.call @__cuda_stream_sync(%[[v2]]) : (!executor.ptr<host>) -> ()
//       CHECK:     executor.call @__cuda_stream_destroy(%[[v2]]) : (!executor.ptr<host>) -> ()
//       CHECK:     executor.return %[[v6]] : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
// CHECK-LABEL: func.func @memref_global
//       CHECK:     %[[v0:.+]] = executor.get_global @global2 : !executor.table<!executor.ptr<host_pinned>, !executor.ptr<host_pinned>, i64, i64, i64>
//       CHECK:     %[[v1:.+]] = executor.get_global @globalDevice : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//       CHECK:     return %[[v0]], %[[v1]] :


// -----

func.func @get_global_stream() {
  %0 = cuda.get_global_stream 0
  %1 = cuda.get_global_stream 0
  %2 = cuda.get_global_stream 1
  return
}

//       CHECK:   executor.global @stream1 constant : !executor.ptr<host>
//       CHECK:     %[[v0:.+]] = executor.call @__cuda_stream_create()
//       CHECK:     executor.return %[[v0]] : !executor.ptr<host>
//       CHECK:   executor.func private @__cuda_stream_create()
//       CHECK:   executor.global @stream0 constant : !executor.ptr<host>
//       CHECK:     %[[v0:.+]] = executor.call @__cuda_stream_create()
//       CHECK:     executor.return %[[v0]] : !executor.ptr<host>
// CHECK-LABEL: func.func @get_global_stream
//       CHECK:     %[[v0:.+]] = executor.get_global @stream0 : !executor.ptr<host>
//       CHECK:     %[[v1:.+]] = executor.get_global @stream0 : !executor.ptr<host>
//       CHECK:     %[[v2:.+]] = executor.get_global @stream1 : !executor.ptr<host>
//       CHECK:     return

// -----

cuda.compiled_module @kernels_cuModule_0 dense<0xFF> : vector<1xi8>

!memref_4xi80 = memref<4xf32, strided<[2], offset: 3>, #executor.memory_type<device>>
!memref_4xi81 = memref<4xf32, strided<[2], offset: 7>, #executor.memory_type<device>>

func.func @test_cuda_launch(
    %arg0: !memref_4xi80,
    %arg1: !memref_4xi81,
    %arg2: index, %arg3: index) {
  %0 = cuda.get_function "kernel" from @kernels_cuModule_0
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %1 = arith.index_cast %arg2 : index to i32
  %2 = arith.index_cast %arg3 : index to i32
  %3 = cuda.get_global_stream 0
  cuda.launch %0(%arg0, %arg1 : !memref_4xi80, !memref_4xi81) with
    grid(%1, %c1_i32, %c1_i32)
    block(%2, %c1_i32, %c1_i32)
    smem(%c0_i32) stream(%3)
  return
}

//   CHECK-LABEL:   executor.data_segment @kernels_cuModule_0_ptx_data constant dense<-1> : vector<1xi8>
//   CHECK-LABEL:   executor.global @kernels_cuModule_0_cuModule constant : !executor.ptr<host> {
//   CHECK-DAG:     %[[v0:.+]] = executor.call @__spmd_global_rank() : () -> i32
//   CHECK-DAG:     %[[v2:.+]] = executor.load_data_segment @kernels_cuModule_0_ptx_data : !executor.ptr<host>
//   CHECK-DAG:     %[[c1_i64:.+]] = executor.constant 1 : i64
//   CHECK-DAG:     %[[v3:.+]] = executor.call @__cuda_load_module(%[[v0]], %[[v2]], %[[c1_i64]]) : (i32, !executor.ptr<host>, i64) -> !executor.ptr<host>
//   CHECK-DAG:     executor.return %[[v3]] : !executor.ptr<host>
//   CHECK-LABEL:   executor.global @kernels_cuModule_0_cuModule_kernel_cuFunc constant : !executor.ptr<host> {
//   CHECK-DAG:     %[[v0:.+]] = executor.get_global @kernels_cuModule_0_cuModule : !executor.ptr<host>
//   CHECK-DAG:     %[[v1:.+]] = executor.str_literal "kernel"
//   CHECK-DAG:     %[[v2:.+]] = executor.call @__cuda_get_function(%[[v0]], %[[v1]]) : (!executor.ptr<host>, !executor.str_literal) -> !executor.ptr<host>
//   CHECK-DAG:     executor.return %[[v2]] : !executor.ptr<host>
// CHECK-LABEL: func.func @test_cuda_launch
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>, %[[arg1:.+]]: !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>, %[[arg2:.+]]: i64, %[[arg3:.+]]: i64) {
//   CHECK:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg3]] : i64 to index
//   CHECK:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : i64 to index
//   CHECK:     %[[v2:.+]] = executor.get_global @kernels_cuModule_0_cuModule_kernel_cuFunc : !executor.ptr<host>
//   CHECK:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK:     %[[v3:.+]] = arith.index_cast %[[v1]] : index to i32
//   CHECK:     %[[v4:.+]] = arith.index_cast %[[v0]] : index to i32
//   CHECK:     %[[v5:.+]] = executor.get_global @stream0 : !executor.ptr<host>
//   CHECK:     %[[v6:.+]] = executor.table.get %[[arg0]][0] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     %[[v7:.+]] = executor.table.get %[[arg0]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     %[[c3_i64:.+]] = executor.constant 3 : i64
//   CHECK:     %[[c4_i64:.+]] = executor.constant 4 : i64
//   CHECK:     %[[c2_i64:.+]] = executor.constant 2 : i64
//   CHECK:     %[[v8:.+]] = executor.table.get %[[arg1]][0] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     %[[v9:.+]] = executor.table.get %[[arg1]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     %[[c7_i64:.+]] = executor.constant 7 : i64
//   CHECK:     %[[c4_i64_0:.+]] = executor.constant 4 : i64
//   CHECK:     %[[c2_i64_1:.+]] = executor.constant 2 : i64
//   CHECK:     %[[c1_i32_2:.+]] = executor.constant 1 : i32
//   CHECK:     %[[v10:.+]] = executor.alloca %[[c1_i32_2]] x !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64> : (i32) -> !executor.ptr<host>
//   CHECK:     %[[c10_i32:.+]] = executor.constant 10 : i32
//   CHECK:     %[[v11:.+]] = executor.alloca %[[c10_i32]] x !executor.ptr<host> : (i32) -> !executor.ptr<host>
//   CHECK:     %[[v12:.+]] = executor.ptrtoint %[[v10]] : (!executor.ptr<host>) -> i64
//   CHECK:     %[[v13:.+]] = executor.getoffset[0, 0] : () -> i64, !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     executor.store %[[v6]] to %[[v10]] + %[[v13]] : !executor.ptr<device>, !executor.ptr<host>, i64
//   CHECK:     %[[v14:.+]] = executor.addi %[[v12]], %[[v13]] : i64
//   CHECK:     %[[v15:.+]] = executor.inttoptr %[[v14]] : (i64) -> !executor.ptr<host>
//   CHECK:     %[[v16:.+]] = executor.getoffset[0] : () -> i64, !executor.ptr<host>
//   CHECK:     executor.store %[[v15]] to %[[v11]] + %[[v16]] : !executor.ptr<host>, !executor.ptr<host>, i64
//   CHECK:     %[[v17:.+]] = executor.getoffset[0, 1] : () -> i64, !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     executor.store %[[v7]] to %[[v10]] + %[[v17]] : !executor.ptr<device>, !executor.ptr<host>, i64
//   CHECK:     %[[v18:.+]] = executor.addi %[[v12]], %[[v17]] : i64
//   CHECK:     %[[v19:.+]] = executor.inttoptr %[[v18]] : (i64) -> !executor.ptr<host>
//   CHECK:     %[[v20:.+]] = executor.getoffset[1] : () -> i64, !executor.ptr<host>
//   CHECK:     executor.store %[[v19]] to %[[v11]] + %[[v20]] : !executor.ptr<host>, !executor.ptr<host>, i64
//   CHECK:     %[[v21:.+]] = executor.getoffset[0, 2] : () -> i64, !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     executor.store %[[c3_i64]] to %[[v10]] + %[[v21]] : i64, !executor.ptr<host>, i64
//   CHECK:     %[[v22:.+]] = executor.addi %[[v12]], %[[v21]] : i64
//   CHECK:     %[[v23:.+]] = executor.inttoptr %[[v22]] : (i64) -> !executor.ptr<host>
//   CHECK:     %[[v24:.+]] = executor.getoffset[2] : () -> i64, !executor.ptr<host>
//   CHECK:     executor.store %[[v23]] to %[[v11]] + %[[v24]] : !executor.ptr<host>, !executor.ptr<host>, i64
//   CHECK:     %[[v25:.+]] = executor.getoffset[0, 3] : () -> i64, !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     executor.store %[[c4_i64]] to %[[v10]] + %[[v25]] : i64, !executor.ptr<host>, i64
//   CHECK:     %[[v26:.+]] = executor.addi %[[v12]], %[[v25]] : i64
//   CHECK:     %[[v27:.+]] = executor.inttoptr %[[v26]] : (i64) -> !executor.ptr<host>
//   CHECK:     %[[v28:.+]] = executor.getoffset[3] : () -> i64, !executor.ptr<host>
//   CHECK:     executor.store %[[v27]] to %[[v11]] + %[[v28]] : !executor.ptr<host>, !executor.ptr<host>, i64
//   CHECK:     %[[v29:.+]] = executor.getoffset[0, 4] : () -> i64, !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     executor.store %[[c2_i64]] to %[[v10]] + %[[v29]] : i64, !executor.ptr<host>, i64
//   CHECK:     %[[v30:.+]] = executor.addi %[[v12]], %[[v29]] : i64
//   CHECK:     %[[v31:.+]] = executor.inttoptr %[[v30]] : (i64) -> !executor.ptr<host>
//   CHECK:     %[[v32:.+]] = executor.getoffset[4] : () -> i64, !executor.ptr<host>
//   CHECK:     executor.store %[[v31]] to %[[v11]] + %[[v32]] : !executor.ptr<host>, !executor.ptr<host>, i64
//   CHECK:     %[[v33:.+]] = executor.getoffset[0, 5] : () -> i64, !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     executor.store %[[v8]] to %[[v10]] + %[[v33]] : !executor.ptr<device>, !executor.ptr<host>, i64
//   CHECK:     %[[v34:.+]] = executor.addi %[[v12]], %[[v33]] : i64
//   CHECK:     %[[v35:.+]] = executor.inttoptr %[[v34]] : (i64) -> !executor.ptr<host>
//   CHECK:     %[[v36:.+]] = executor.getoffset[5] : () -> i64, !executor.ptr<host>
//   CHECK:     executor.store %[[v35]] to %[[v11]] + %[[v36]] : !executor.ptr<host>, !executor.ptr<host>, i64
//   CHECK:     %[[v37:.+]] = executor.getoffset[0, 6] : () -> i64, !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     executor.store %[[v9]] to %[[v10]] + %[[v37]] : !executor.ptr<device>, !executor.ptr<host>, i64
//   CHECK:     %[[v38:.+]] = executor.addi %[[v12]], %[[v37]] : i64
//   CHECK:     %[[v39:.+]] = executor.inttoptr %[[v38]] : (i64) -> !executor.ptr<host>
//   CHECK:     %[[v40:.+]] = executor.getoffset[6] : () -> i64, !executor.ptr<host>
//   CHECK:     executor.store %[[v39]] to %[[v11]] + %[[v40]] : !executor.ptr<host>, !executor.ptr<host>, i64
//   CHECK:     %[[v41:.+]] = executor.getoffset[0, 7] : () -> i64, !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     executor.store %[[c7_i64]] to %[[v10]] + %[[v41]] : i64, !executor.ptr<host>, i64
//   CHECK:     %[[v42:.+]] = executor.addi %[[v12]], %[[v41]] : i64
//   CHECK:     %[[v43:.+]] = executor.inttoptr %[[v42]] : (i64) -> !executor.ptr<host>
//   CHECK:     %[[v44:.+]] = executor.getoffset[7] : () -> i64, !executor.ptr<host>
//   CHECK:     executor.store %[[v43]] to %[[v11]] + %[[v44]] : !executor.ptr<host>, !executor.ptr<host>, i64
//   CHECK:     %[[v45:.+]] = executor.getoffset[0, 8] : () -> i64, !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     executor.store %[[c4_i64_0]] to %[[v10]] + %[[v45]] : i64, !executor.ptr<host>, i64
//   CHECK:     %[[v46:.+]] = executor.addi %[[v12]], %[[v45]] : i64
//   CHECK:     %[[v47:.+]] = executor.inttoptr %[[v46]] : (i64) -> !executor.ptr<host>
//   CHECK:     %[[v48:.+]] = executor.getoffset[8] : () -> i64, !executor.ptr<host>
//   CHECK:     executor.store %[[v47]] to %[[v11]] + %[[v48]] : !executor.ptr<host>, !executor.ptr<host>, i64
//   CHECK:     %[[v49:.+]] = executor.getoffset[0, 9] : () -> i64, !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK:     executor.store %[[c2_i64_1]] to %[[v10]] + %[[v49]] : i64, !executor.ptr<host>, i64
//   CHECK:     %[[v50:.+]] = executor.addi %[[v12]], %[[v49]] : i64
//   CHECK:     %[[v51:.+]] = executor.inttoptr %[[v50]] : (i64) -> !executor.ptr<host>
//   CHECK:     %[[v52:.+]] = executor.getoffset[9] : () -> i64, !executor.ptr<host>
//   CHECK:     executor.store %[[v51]] to %[[v11]] + %[[v52]] : !executor.ptr<host>, !executor.ptr<host>, i64
//   CHECK:     %[[c10_i32_3:.+]] = executor.constant 10 : i32
//   CHECK:     executor.call @__cuda_launch(%[[v2]], %[[v3]], %[[c1_i32]], %[[c1_i32]], %[[v4]], %[[c1_i32]], %[[c1_i32]], %[[c0_i32]], %[[v5]], %[[v11]], %[[c10_i32_3]]) : (!executor.ptr<host>, i32, i32, i32, i32, i32, i32, i32, !executor.ptr<host>, !executor.ptr<host>, i32) -> ()
//   CHECK:     return
