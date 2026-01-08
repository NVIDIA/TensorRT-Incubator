// RUN: mlir-tensorrt-opt -split-input-file -convert-cuda-to-executor %s | FileCheck %s

func.func @cuda_event(){
  %0 = cuda.event.create : !cuda.event
  %device = cuda.get_active_device
  %1 = cuda.stream.create device(%device)
  cuda.stream.wait_event %1, %0
  return
}

// CHECK-LABEL: @cuda_event
//       CHECK: %[[v0:.+]] = executor.call @__cuda_event_create() : () -> !executor.ptr<host>
//       CHECK: %[[device:.+]] = executor.call @__cuda_get_active_device() : () -> i32
//       CHECK: %[[v1:.+]] = executor.call @__cuda_stream_create(%[[device]]) : (i32) -> !executor.ptr<host>
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

func.func @test_get_active_device() -> i32 {
  %0 = cuda.get_active_device
  return %0 : i32
}

// CHECK-LABEL: @test_get_active_device
//       CHECK:     %[[v0:.+]] = executor.call @__cuda_get_active_device() : () -> i32
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

func.func @test_get_program_device(%logical: i32) -> i32 {
  %0 = cuda.get_program_device %logical : i32
  return %0 : i32
}

// CHECK-LABEL: @test_get_program_device
//  CHECK-SAME: (%[[logical:.+]]: i32) -> i32 {
//   CHECK-DAG:   %[[v0:.+]] = executor.call @__cuda_get_program_device(%[[logical]]) : (i32) -> i32
//   CHECK-DAG:   return %[[v0]] : i32

// -----

!memref_4xi8 = memref<?x2x?xf32, #executor.memory_type<device>>

func.func @device_alloc(%arg0: index, %arg1: index, %stream: !cuda.stream, %device: i32) -> !memref_4xi8 {
  %0 = cuda.alloc(%arg0, %arg1) stream(%stream) align 8 : !memref_4xi8
  return %0 : !memref_4xi8
}

// CHECK-LABEL: func.func @device_alloc
//  CHECK-SAME: (%[[arg0_index:.+]]: index, %[[arg1_index:.+]]: index, %[[arg2_stream:.+]]: !cuda.stream, %[[arg3:.+]]: i32)
//   CHECK-DAG:     %[[arg0:.+]] = builtin.unrealized_conversion_cast %[[arg0_index]] : index to i64
//   CHECK-DAG:     %[[arg1:.+]] = builtin.unrealized_conversion_cast %[[arg1_index]] : index to i64
//   CHECK-DAG:     %[[arg2:.+]] = builtin.unrealized_conversion_cast %[[arg2_stream]] : !cuda.stream to !executor.ptr<host>
//       CHECK:     %[[c2_i64:.+]] = executor.constant 2 : i64
//       CHECK:     %[[c1_i64:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v0:.+]] = executor.muli %[[c1_i64]], %[[arg1]] : i64
//       CHECK:     %[[v1:.+]] = executor.muli %[[v0]], %[[c2_i64]] : i64
//       CHECK:     %[[v2:.+]] = executor.muli %[[v1]], %[[arg0]] : i64
//       CHECK:     %[[c4_i64:.+]] = executor.constant 4 : i64
//       CHECK:     %[[v3:.+]] = executor.muli %[[v2]], %[[c4_i64]] : i64
//       CHECK:     %[[v4:.+]] = executor.call @__cuda_alloc_device(%[[arg2]], %[[v3]], %{{.+}})
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[v5:.+]] = executor.table.create(%[[v4]], %[[v4]], %[[c0_i64]], %[[arg0]], %[[c2_i64]], %[[arg1]], %[[v1]], %[[v0]], %[[c1_i64]] : !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64) : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v6:.+]] = builtin.unrealized_conversion_cast %[[v5]]
//       CHECK:     return %[[v6]]

// -----

func.func @memref_device_alloc_i1(%arg0: !cuda.stream, %device: i32) -> memref<1500x1500xi1, #executor.memory_type<device>> {
  %0 = cuda.alloc () stream(%arg0) : memref<1500x1500xi1, #executor.memory_type<device>>
  return %0 : memref<1500x1500xi1, #executor.memory_type<device>>
}

// CHECK-LABEL: func.func @memref_device_alloc_i1
//  CHECK-SAME: (%[[arg0_stream:.+]]: !cuda.stream, %[[arg1:.+]]: i32) -> memref<1500x1500xi1, #executor.memory_type<device>> {
//       CHECK:     %[[arg0:.+]] = builtin.unrealized_conversion_cast %[[arg0_stream]] : !cuda.stream to !executor.ptr<host>
//       CHECK:     %[[c1500_i64:.+]] = executor.constant 1500 : i64
//       CHECK:     %[[c1500_i64_0:.+]] = executor.constant 1500 : i64
//       CHECK:     %[[c1_i64:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v0:.+]] = executor.muli %[[c1_i64]], %[[c1500_i64_0]] : i64
//       CHECK:     %[[v1:.+]] = executor.muli %[[v0]], %[[c1500_i64]] : i64
//       CHECK:     %[[c1_i64_1:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v2:.+]] = executor.muli %[[v1]], %[[c1_i64_1]] : i64
//       CHECK:     %[[v3:.+]] = executor.call @__cuda_alloc_device(%[[arg0]], %[[v2]], %{{.+}})
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[v4:.+]] = executor.table.create(%[[v3]], %[[v3]], %[[c0_i64]], %[[c1500_i64]], %[[c1500_i64_0]], %[[v0]], %[[c1_i64]] : !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64) : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>
//       CHECK:     %[[v5:.+]] = builtin.unrealized_conversion_cast %[[v4]] : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64> to memref<1500x1500xi1, #executor.memory_type<device>>
//       CHECK:     return %[[v5]] : memref<1500x1500xi1, #executor.memory_type<device>>

// -----

!memref_4xi8 = memref<?x2x?xf32, #executor.memory_type<host_pinned>>

func.func @pinned_alloc(%arg0: index, %arg1: index, %stream: !cuda.stream, %device: i32) -> !memref_4xi8 {
  %0 = cuda.alloc(%arg0, %arg1) align 8 : !memref_4xi8
  return %0 : !memref_4xi8
}

// CHECK-LABEL: func.func @pinned_alloc
//  CHECK-SAME: (%[[arg0_index:.+]]: index, %[[arg1_index:.+]]: index, %[[arg2_stream:.+]]: !cuda.stream, %[[arg3:.+]]: i32)
//   CHECK-DAG:     %[[arg0:.+]] = builtin.unrealized_conversion_cast %[[arg0_index]] : index to i64
//   CHECK-DAG:     %[[arg1:.+]] = builtin.unrealized_conversion_cast %[[arg1_index]] : index to i64
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
//   CHECK-DAG:     %[[v6:.+]] = builtin.unrealized_conversion_cast %[[v5]]
//   CHECK-DAG:     return %[[v6]]

// -----

!memref_4xi8 = memref<?x2x?xf32, #executor.memory_type<device>>
func.func @device_free(%arg0: !cuda.stream, %arg1: !memref_4xi8) {
  cuda.dealloc stream(%arg0) %arg1  : !memref_4xi8
  return
}

// CHECK-LABEL: func.func @device_free
//  CHECK-SAME: (%[[arg0_stream:.+]]: !cuda.stream, %[[arg1_memref:.+]]: memref<
//   CHECK-DAG:     %[[arg1:.+]] = builtin.unrealized_conversion_cast %[[arg1_memref]]
//   CHECK-DAG:     %[[arg0:.+]] = builtin.unrealized_conversion_cast %[[arg0_stream]]
//       CHECK:     %[[v0:.+]] = executor.table.get %[[arg1]][0] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     executor.call @__cuda_free_device(%[[arg0]], %[[v0]]) : (!executor.ptr<host>, !executor.ptr<device>) -> ()

// -----

!memref_4xi8 = memref<?x2x?xf32, #executor.memory_type<host_pinned>>
func.func @pinned_free(%arg0: !cuda.stream, %arg1: !memref_4xi8) {
  cuda.dealloc stream(%arg0) %arg1  : !memref_4xi8
  return
}

// CHECK-LABEL: func.func @pinned_free
//  CHECK-SAME: (%[[arg0_stream:.+]]: !cuda.stream, %[[arg1_memref:.+]]: memref<{{.*}}>
//   CHECK-DAG:     %[[arg1:.+]] = builtin.unrealized_conversion_cast %[[arg1_memref]]
//   CHECK-DAG:     %[[arg0:.+]] = builtin.unrealized_conversion_cast %[[arg0_stream]]
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
//  CHECK-SAME: (%[[arg0_src:.+]]: memref<{{.*}}>, %[[arg1_dst:.+]]: memref<{{.*}}>, %[[arg2_stream:.+]]: !cuda.stream) {
//   CHECK-DAG:     %[[arg1:.+]] = builtin.unrealized_conversion_cast %[[arg1_dst]] 
//   CHECK-DAG:     %[[arg0:.+]] = builtin.unrealized_conversion_cast %[[arg0_src]] 
//   CHECK-DAG:     %[[arg2:.+]] = builtin.unrealized_conversion_cast %[[arg2_stream]]
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
//  CHECK-SAME: (%[[arg0_src:.+]]: memref<128x16xf32, strided<[16, 1], offset: 16>, #executor.memory_type<device>>, %[[arg1_dst:.+]]: memref<128x16xf32, strided<[16, 1], offset: 8>, #executor.memory_type<host>>, %[[arg2_stream:.+]]: !cuda.stream) {
//   CHECK-DAG:     %[[arg1:.+]] = builtin.unrealized_conversion_cast %[[arg1_dst]] : memref<128x16xf32, strided<[16, 1], offset: 8>, #executor.memory_type<host>> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64>
//   CHECK-DAG:     %[[arg0:.+]] = builtin.unrealized_conversion_cast %[[arg0_src]] : memref<128x16xf32, strided<[16, 1], offset: 16>, #executor.memory_type<device>> to !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>
//   CHECK-DAG:     %[[arg2:.+]] = builtin.unrealized_conversion_cast %[[arg2_stream]] : !cuda.stream to !executor.ptr<host>
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
//  CHECK-SAME: (%[[arg0_src:.+]]: memref<{{.*}}>, %[[arg1_dst:.+]]: memref<{{.*}}>, %[[arg2_stream:.+]]: !cuda.stream) {
//   CHECK-DAG:     %[[arg1:.+]] = builtin.unrealized_conversion_cast %[[arg1_dst]] 
//   CHECK-DAG:     %[[arg0:.+]] = builtin.unrealized_conversion_cast %[[arg0_src]] 
//   CHECK-DAG:     %[[arg2:.+]] = builtin.unrealized_conversion_cast %[[arg2_stream]]
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
//  CHECK-SAME: (%[[arg0_src:.+]]: memref<{{.*}}>, %[[arg1_dst:.+]]: memref<{{.*}}>, %[[arg2_stream:.+]]: !cuda.stream) {
//   CHECK-DAG:     %[[arg1:.+]] = builtin.unrealized_conversion_cast %[[arg1_dst]] 
//   CHECK-DAG:     %[[arg0:.+]] = builtin.unrealized_conversion_cast %[[arg0_src]] 
//   CHECK-DAG:     %[[arg2:.+]] = builtin.unrealized_conversion_cast %[[arg2_stream]]
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

func.func @get_global_stream() {
  %c0 = arith.constant 0 : i32
  %device = cuda.get_program_device %c0 : i32
  %0 = cuda.get_global_stream device(%device) [0]
  %1 = cuda.get_global_stream device(%device) [0]
  %2 = cuda.get_global_stream device(%device) [1]
  return
}

//       CHECK:   executor.global @stream1 constant : !executor.ptr<host>
//       CHECK:     %[[device:.+]] = executor.call @__cuda_get_program_device(%{{.+}}) : (i32) -> i32
//       CHECK:     %[[v0:.+]] = executor.call @__cuda_stream_create(%[[device]])
//       CHECK:     executor.return %[[v0]] : !executor.ptr<host>
//       CHECK:   executor.global @stream0 constant : !executor.ptr<host>
//       CHECK:     %[[device:.+]] = executor.call @__cuda_get_program_device(%{{.+}}) : (i32) -> i32
//       CHECK:     %[[v0:.+]] = executor.call @__cuda_stream_create(%[[device]])
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
  %device = cuda.get_program_device %c0_i32 : i32
  %3 = cuda.get_global_stream device(%device) [0]
  cuda.launch %0(%arg0, %arg1 : !memref_4xi80, !memref_4xi81) with
    grid(%1, %c1_i32, %c1_i32)
    block(%2, %c1_i32, %c1_i32)
    smem(%c0_i32) stream(%3)
  return
}

// CHECK-LABEL: func.func @test_cuda_launch
//  CHECK-SAME: (%[[arg0_memref:.+]]: memref<{{.*}}>, %[[arg1_memref:.+]]: memref<{{.*}}>, %[[arg2:.+]]: index, %[[arg3:.+]]: index) {
//   CHECK-DAG:     %[[arg1:.+]] = builtin.unrealized_conversion_cast %[[arg1_memref]] 
//   CHECK-DAG:     %[[arg0:.+]] = builtin.unrealized_conversion_cast %[[arg0_memref]]
//   CHECK-DAG:     %[[v2:.+]] = executor.get_global @kernels_cuModule_0_cuModule_kernel_cuFunc : !executor.ptr<host>
//   CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[v3:.+]] = arith.index_cast %[[arg2]] : index to i32
//   CHECK-DAG:     %[[v4:.+]] = arith.index_cast %[[arg3]] : index to i32
//   CHECK-DAG:     %[[v5:.+]] = executor.get_global @stream0 : !executor.ptr<host>
//   CHECK-DAG:     %[[c1_i64:.+]] = executor.constant 1 : i64
//   CHECK-DAG:     %[[v6:.+]] = executor.table.get %[[arg0]][1]
//   CHECK-DAG:     %[[v7:.+]] = executor.table.get %[[arg1]][1]
//   CHECK-DAG:     %[[c0_i64:.+]] = executor.constant 0 : i64
//   CHECK-DAG:     %[[v8:.+]] = executor.alloca %[[c1_i64]] x !executor.ptr<device> : (i64) -> !executor.ptr<host>
//   CHECK-DAG:     executor.store %[[v6]] to %[[v8]] + %[[c0_i64]]
//   CHECK-DAG:     %[[v9:.+]] = executor.alloca %[[c1_i64]] x !executor.ptr<device> : (i64) -> !executor.ptr<host>
//   CHECK-DAG:     executor.store %[[v7]] to %[[v9]] + %[[c0_i64]]
//   CHECK-DAG:     %[[v10:.+]] = executor.alloca %[[c1_i64]] x !executor.table<!executor.ptr<host>, !executor.ptr<host>> : (i64) -> !executor.ptr<host>
//   CHECK-DAG:     %[[v11:.+]] = executor.getoffset[0, 0]
//   CHECK-DAG:     executor.store %[[v8]] to %[[v10]] + %[[v11]]
//   CHECK-DAG:     %[[v12:.+]] = executor.getoffset[0, 1]
//   CHECK-DAG:     executor.store %[[v9]] to %[[v10]] + %[[v12]]
//   CHECK-DAG:     executor.call @__cuda_launch(%[[v2]], %[[v3]], %[[c1_i32]], %[[c1_i32]], %[[v4]], %[[c1_i32]], %[[c1_i32]], %[[c0_i32]], %[[v5]], %[[v10]])
//       CHECK:     return