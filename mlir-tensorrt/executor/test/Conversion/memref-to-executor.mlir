// RUN: executor-opt %s -split-input-file -convert-memref-to-executor="index-bitwidth=32 use-packed-memref-cconv=false allow-unchecked-memref-cast-conversion=false" -canonicalize  -verify-diagnostics | FileCheck %s

!hostBuffer = memref<4xf32, #executor.memory_type<host>>

func.func @memref_host_alloc() -> !hostBuffer {
  %0 = memref.alloc () : !hostBuffer
  return %0 : !hostBuffer
}

// CHECK-LABEL: @memref_host_alloc
//  CHECK-SAME: () -> memref<4xf32, #executor.memory_type<host>> {
//   CHECK-DAG:     %[[c16:.+]] = executor.constant 16 : i32
//   CHECK-DAG:     %[[c0:.+]] = executor.constant 0 : i32
//   CHECK-DAG:     %[[c4:.+]] = executor.constant 4 : i32
//   CHECK-DAG:     %[[c1:.+]] = executor.constant 1 : i32
//       CHECK:     %[[v0:.+]] = executor.alloc %[[c16]] bytes align(%[[c16]]) : (i32, i32) -> !executor.ptr<host>
//       CHECK:     %[[v1:.+]] = executor.table.create(%[[v0]], %[[v0]], %[[c0]], %[[c4]], %[[c1]] : !executor.ptr<host>, !executor.ptr<host>, i32, i32, i32) : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[v1]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32> to memref<4xf32, #executor.memory_type<host>>
//       CHECK:     return %[[v2]] : memref<4xf32, #executor.memory_type<host>>

// -----

func.func @memref_default_alloc() -> memref<4xf32> {
  %0 = memref.alloc () : memref<4xf32>
  return %0 : memref<4xf32>
}

// CHECK-LABEL: @memref_default_alloc
//  CHECK-SAME: () -> memref<4xf32> {
//   CHECK-DAG:     %[[c16:.+]] = executor.constant 16 : i32
//   CHECK-DAG:     %[[c0:.+]] = executor.constant 0 : i32
//   CHECK-DAG:     %[[c4:.+]] = executor.constant 4 : i32
//   CHECK-DAG:     %[[c1:.+]] = executor.constant 1 : i32
//       CHECK:     %[[v0:.+]] = executor.alloc %[[c16]] bytes align(%[[c16]]) : (i32, i32) -> !executor.ptr<host>
//       CHECK:     %[[v1:.+]] = executor.table.create(%[[v0]], %[[v0]], %[[c0]], %[[c4]], %[[c1]] : !executor.ptr<host>, !executor.ptr<host>, i32, i32, i32) : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[v1]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32> to memref<4xf32>
//       CHECK:     return %[[v2]] : memref<4xf32>

// -----

func.func @memref_copy_offset(%arg0: memref<128x16xf32, strided<[16, 1], offset: 16>>,
                              %arg1: memref<128x16xf32, strided<[16, 1], offset: 8>>) {
  memref.copy %arg0, %arg1 : memref<128x16xf32, strided<[16, 1], offset: 16>> to memref<128x16xf32, strided<[16, 1], offset: 8>>
  return
}

// CHECK-LABEL: @memref_copy_offset
//  CHECK-SAME: (%[[arg0:.+]]: memref<128x16xf32, strided<[16, 1], offset: 16>>, %[[arg1:.+]]: memref<128x16xf32, strided<[16, 1], offset: 8>>) {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<128x16xf32, strided<[16, 1], offset: 16>> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<128x16xf32, strided<[16, 1], offset: 8>> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v2:.+]] = executor.getoffset[16] : () -> i32, f32
//       CHECK:     %[[v3:.+]] = executor.getoffset[8] : () -> i32, f32
//       CHECK:     %[[v4:.+]] = executor.getoffset[2048] : () -> i32, f32
//   CHECK-DAG:     %[[v5:.+]] = executor.table.get %[[v0]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v6:.+]] = executor.table.get %[[v1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     executor.memcpy %[[v5]] + %[[v2]] to %[[v6]] + %[[v3]] size %[[v4]] : !executor.ptr<host>, i32, !executor.ptr<host>, i32, i32

// -----

func.func @memref_copy_offset_f16(%arg0: memref<128x16xf16, strided<[16, 1], offset: 16>>,
                                  %arg1: memref<128x16xf16, strided<[16, 1], offset: 8>>) {
  memref.copy %arg0, %arg1 : memref<128x16xf16, strided<[16, 1], offset: 16>> to memref<128x16xf16, strided<[16, 1], offset: 8>>
  return
}

// CHECK-LABEL: @memref_copy_offset_f16
//  CHECK-SAME: (%[[arg0:.+]]: memref<128x16xf16, strided<[16, 1], offset: 16>>, %[[arg1:.+]]: memref<128x16xf16, strided<[16, 1], offset: 8>>) {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<128x16xf16, strided<[16, 1], offset: 16>> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<128x16xf16, strided<[16, 1], offset: 8>> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v2:.+]] = executor.getoffset[16] : () -> i32, f16
//       CHECK:     %[[v3:.+]] = executor.getoffset[8] : () -> i32, f16
//       CHECK:     %[[v4:.+]] = executor.getoffset[2048] : () -> i32, f16
//   CHECK-DAG:     %[[v5:.+]] = executor.table.get %[[v0]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//   CHECK-DAG:     %[[v6:.+]] = executor.table.get %[[v1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     executor.memcpy %[[v5]] + %[[v2]] to %[[v6]] + %[[v3]] size %[[v4]] : !executor.ptr<host>, i32, !executor.ptr<host>, i32, i32

// -----


func.func @scalar_edge_cases() -> i32 {
  %c0 = arith.constant 0 : i32
  %c99_i32 = arith.constant 99 : i32

  %0 = memref.alloc() : memref<i32>
  %1 = memref.alloc() : memref<i32>
  memref.store %c99_i32, %0[] : memref<i32>
  memref.store %c0, %1[] : memref<i32>
  memref.copy %0, %1 : memref<i32> to memref<i32>
  %load = memref.load %1[] : memref<i32>
  executor.print "i32 memcpy result = %d"(%load : i32)

  %c0_i1 = arith.constant 0 : i1
  %c1_i1 = arith.constant 1 : i1

  %2 = memref.alloc() : memref<i1>
  %3 = memref.alloc() : memref<i1>
  memref.store %c1_i1, %2[] : memref<i1>
  memref.store %c0_i1, %3[] : memref<i1>
  memref.copy %2, %3 : memref<i1> to memref<i1>
  %load_i1 = memref.load %3[] : memref<i1>
  executor.print "i1 memcpy result = %i1"(%load_i1 : i1)

  return %c0 : i32
}

// CHECK-LABEL: @scalar_edge_cases
//   CHECK-DAG:     %[[c0_i32:.+]] = executor.constant 0 : i32
//   CHECK-DAG:     %[[true:.+]] = arith.constant true
//   CHECK-DAG:     %[[false:.+]] = arith.constant false
//   CHECK-DAG:     %[[c16_i32:.+]] = executor.constant 16 : i32
//   CHECK-DAG:     %[[c0_i32_0:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[c99_i32:.+]] = arith.constant 99 : i32
//   CHECK-DAG:     %[[c1_i32:.+]] = executor.constant 1 : i32
//   CHECK-DAG:     %[[c4_i32:.+]] = executor.constant 4 : i32
//       CHECK:     %[[v0:.+]] = executor.alloc %[[c4_i32]] bytes align(%[[c16_i32]]) : (i32, i32) -> !executor.ptr<host>
//       CHECK:     %[[v1:.+]] = executor.alloc %[[c4_i32]] bytes align(%[[c16_i32]]) : (i32, i32) -> !executor.ptr<host>
//       CHECK:     executor.store %[[c99_i32]] to %[[v0]] + %[[c0_i32]] : i32, !executor.ptr<host>, i32
//       CHECK:     executor.store %[[c0_i32_0]] to %[[v1]] + %[[c0_i32]] : i32, !executor.ptr<host>, i32
//       CHECK:     %[[v2:.+]] = executor.getoffset[1] : () -> i32, i32
//       CHECK:     executor.memcpy %[[v0]] + %[[c0]] to %[[v1]] + %[[c0]] size %[[v2]] : !executor.ptr<host>, i32, !executor.ptr<host>, i32, i32
//       CHECK:     executor.load %[[v1]] + %[[c0]] : (!executor.ptr<host>, i32) -> i32
//       CHECK:     %[[v3:.+]] = executor.alloc %[[c1]] bytes align(%[[c16]]) : (i32, i32) -> !executor.ptr<host>
//       CHECK:     %[[v4:.+]] = executor.alloc %[[c1]] bytes align(%[[c16]]) : (i32, i32) -> !executor.ptr<host>
//       CHECK:     executor.store %[[true]] to %[[v3]] + %[[c0]] : i1, !executor.ptr<host>, i32
//       CHECK:     executor.store %[[false]] to %[[v4]] + %[[c0]] : i1, !executor.ptr<host>, i32
//       CHECK:     %[[v6:.+]] = executor.getoffset[1] : () -> i32, i1
//       CHECK:     executor.memcpy %[[v3]] + %[[c0]] to %[[v4]] + %[[c0]] size %[[v6]] : !executor.ptr<host>, i32, !executor.ptr<host>, i32, i32
//       CHECK:     executor.load %[[v4]] + %[[c0]] : (!executor.ptr<host>, i32) -> i1

// -----

func.func @alloc_i1() -> memref<1500x1500xi1, #executor.memory_type<host>> {
  %0 = memref.alloc () : memref<1500x1500xi1, #executor.memory_type<host>>
  return %0 : memref<1500x1500xi1, #executor.memory_type<host>>
}

// CHECK-LABEL: func.func @alloc_i1
//       CHECK:     %[[c0_i32:.+]] = executor.constant 0 : i32
//       CHECK:     %[[c16_i32:.+]] = executor.constant 16 : i32
//       CHECK:     %[[c1500_i32:.+]] = executor.constant 1500 : i32
//       CHECK:     %[[c1_i32:.+]] = executor.constant 1 : i32
//       CHECK:     %[[c2250000_i32:.+]] = executor.constant 2250000 : i32
//       CHECK:     %[[v0:.+]] = executor.alloc %[[c2250000_i32]] bytes align(%[[c16_i32]]) : (i32, i32) -> !executor.ptr<host>
//       CHECK:     %[[v1:.+]] = executor.table.create(%[[v0]], %[[v0]], %[[c0_i32]], %[[c1500_i32]], %[[c1500_i32]], %[[c1500_i32]], %[[c1_i32]] : !executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32) : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32>
//       CHECK:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[v1]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, i32, i32> to memref<1500x1500xi1, #executor.memory_type<host>>
//       CHECK:     return %[[v2]] : memref<1500x1500xi1, #executor.memory_type<host>>

// -----

!srcType = memref<6xf32, strided<[2], offset: 2>, #executor.memory_type<host>>
!dstType = memref<6xf32, strided<[2], offset: 4>, #executor.memory_type<host>>

func.func @memref_copy_h2h_strided(%arg0: !srcType,
                          %arg1: !dstType) {
  memref.copy %arg0, %arg1 : !srcType to !dstType
  return
}
// CHECK-LABEL: func.func @memref_copy_h2h_strided
//  CHECK-SAME: (%[[arg0:.+]]: memref<6xf32, strided<[2], offset: 2>, #executor.memory_type<host>>, %[[arg1:.+]]: memref<6xf32, strided<[2], offset: 4>, #executor.memory_type<host>>) {
//       CHECK:     %[[c6_i32:.+]] = executor.constant 6 : i32
//       CHECK:     %[[c2_i32:.+]] = executor.constant 2 : i32
//       CHECK:     %[[c4_i32:.+]] = executor.constant 4 : i32
//       CHECK:     %[[c1_i32:.+]] = executor.constant 1 : i32
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<6xf32, strided<[2], offset: 2>, #executor.memory_type<host>> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<6xf32, strided<[2], offset: 4>, #executor.memory_type<host>> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     %[[v2:.+]] = executor.table.get %[[v0]][0] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     %[[v3:.+]] = executor.table.get %[[v0]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     %[[v4:.+]] = executor.table.get %[[v1]][0] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     %[[v5:.+]] = executor.table.get %[[v1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     executor.strided_memref_copy(%[[c1_i32]], %[[c4_i32]], %[[v2]], %[[v3]], %[[c2_i32]], %[[c6_i32]], %[[c2_i32]], %[[v4]], %[[v5]], %[[c4_i32]], %[[c6_i32]], %[[c2_i32]]) : i32, i32, !executor.ptr<host>, !executor.ptr<host>, i32, i32, i32, !executor.ptr<host>, !executor.ptr<host>, i32, i32, i32
//       CHECK:     return

// -----

!srcType = memref<1x8x4xf32, strided<[32, 4, 1], offset: ?>, #executor.memory_type<host>>
!dstType = memref<1x8x4xf32, strided<[128, 4, 1], offset: ?>, #executor.memory_type<host>>

func.func @memref_copy_contiguous_non_identity(%arg0: !srcType, %arg1: !dstType) {
  memref.copy %arg0, %arg1 : !srcType to !dstType
  return
}

// CHECK-LABEL: @memref_copy_contiguous_non_identity
//       CHECK:   executor.memcpy

// -----

!srcType = memref<1x8x4xf32, strided<[256, 4, 1], offset: ?>, #executor.memory_type<host>>
!dstType = memref<1x8x4xf32, strided<[128, 4, 1], offset: ?>, #executor.memory_type<host>>

func.func @memref_copy_contiguous_non_identity2(%arg0: !srcType, %arg1: !dstType) {
  memref.copy %arg0, %arg1 : !srcType to !dstType
  return
}

// CHECK-LABEL: @memref_copy_contiguous_non_identity2
//       CHECK:   executor.memcpy


// -----

!srcType = memref<8x1x4xf32, strided<[4, 32, 1], offset: ?>, #executor.memory_type<host>>
!dstType = memref<8x1x4xf32, strided<[4, 32, 1], offset: ?>, #executor.memory_type<host>>

func.func @memref_copy_contiguous_non_identity3(%arg0: !srcType, %arg1: !dstType) {
  memref.copy %arg0, %arg1 : !srcType to !dstType
  return
}

// CHECK-LABEL: @memref_copy_contiguous_non_identity3
//       CHECK:   executor.memcpy

// -----

!srcType = memref<f32, strided<[], offset: ?>, #executor.memory_type<host>>
!dstType = memref<f32, strided<[], offset: ?>, #executor.memory_type<host>>

func.func @memref_copy_contiguous_non_identity4(%arg0: !srcType, %arg1: !dstType) {
  memref.copy %arg0, %arg1 : !srcType to !dstType
  return
}

// CHECK-LABEL: @memref_copy_contiguous_non_identity4
//       CHECK:   executor.memcpy

// -----

!srcType = memref<1xf32, strided<[32], offset: ?>, #executor.memory_type<host>>
!dstType = memref<1xf32, strided<[32], offset: ?>, #executor.memory_type<host>>

func.func @memref_copy_contiguous_non_identity5(%arg0: !srcType, %arg1: !dstType) {
  memref.copy %arg0, %arg1 : !srcType to !dstType
  return
}

// CHECK-LABEL: @memref_copy_contiguous_non_identity5
//       CHECK:   executor.memcpy

// -----

!srcType = memref<6xf32, strided<[2], offset: 2>>
!dstType = memref<6xf32, strided<[?], offset: 2>>
!dstType1 = memref<6xf32, strided<[2], offset: ?>>
!dstType2 = memref<6xf32, strided<[?], offset: ?>>
func.func @memref_cast(%arg0: !srcType) -> (!dstType, !dstType1, !dstType2) {
  %0 = memref.cast %arg0 : !srcType to !dstType
  %1 = memref.cast %arg0 : !srcType to !dstType1
  %2 = memref.cast %arg0 : !srcType to !dstType2
  return %0, %1, %2 : !dstType, !dstType1, !dstType2
}

// CHECK-LABEL: @memref_cast
//  CHECK-SAME: (%[[arg0:.+]]: memref<6xf32, strided<[2], offset: 2>>)
//   CHECK-DAG:  %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<6xf32, strided<[2], offset: 2>> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//   CHECK-DAG:  %[[v1:.+]] = builtin.unrealized_conversion_cast %[[v0]]
//   CHECK-DAG:  %[[v2:.+]] = builtin.unrealized_conversion_cast %[[v1]]
//   CHECK-DAG:  %[[v3:.+]] = builtin.unrealized_conversion_cast %[[v2]]
//   CHECK-DAG:  return %[[v1]], %[[v2]], %[[v3]] :


// -----

!srcType = memref<6xf32, strided<[?], offset: ?>>
!dstType = memref<6xf32, strided<[?], offset: 2>>
func.func @unsupported_memref_cast(%arg0: !srcType) -> !dstType {
  // expected-error @below {{failed to legalize operation 'memref.cast' that was explicitly marked illegal}}
  %0 = memref.cast %arg0 : !srcType to !dstType
  return %0 : !dstType
}

// -----

!srcType = memref<6xf32, strided<[?], offset: ?>>
!dstType = memref<6xf32, strided<[2], offset: ?>>
func.func @unsupported_memref_cast2(%arg0: !srcType) -> !dstType {
  // expected-error @below {{failed to legalize operation 'memref.cast' that was explicitly marked illegal}}
  %0 = memref.cast %arg0 : !srcType to !dstType
  return %0 : !dstType
}

// -----

!srcType = memref<1x20x12x64xf32, #executor.memory_type<host>>
!dstType = memref<1x20x12x64xf32, strided<[?, ?, ?, ?], offset: ?>, #executor.memory_type<host>>
func.func @memref_cast(%arg0: !srcType) -> !dstType {
  %cast = memref.cast %arg0 : !srcType to !dstType
  return %cast: !dstType
}

// -----

!memrefType = memref<?x2x?xf32, strided<[?, ?, 1], offset: ?>>

func.func @reinterpret_cast_memref_dynamic(
    %arg0: memref<f32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index)
  -> !memrefType {
  %reinterpret_cast = memref.reinterpret_cast %arg0
    to offset: [%arg1],
    sizes: [%arg2, 2, %arg3], strides: [%arg4, %arg5, 1] : memref<f32> to !memrefType
  return %reinterpret_cast : !memrefType
}

// CHECK-LABEL: @reinterpret_cast_memref_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: memref<f32>, %[[arg1:.+]]: index, %[[arg2:.+]]: index, %[[arg3:.+]]: index, %[[arg4:.+]]: index, %[[arg5:.+]]: index)
//   CHECK-DAG:     %[[c1_i32:.+]] = executor.constant 1 : i32
//   CHECK-DAG:     %[[c2_i32:.+]] = executor.constant 2 : i32
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<f32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : index to i32
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : index to i32
//   CHECK-DAG:     %[[v3:.+]] = builtin.unrealized_conversion_cast %[[arg3]] : index to i32
//   CHECK-DAG:     %[[v4:.+]] = builtin.unrealized_conversion_cast %[[arg4]] : index to i32
//   CHECK-DAG:     %[[v5:.+]] = builtin.unrealized_conversion_cast %[[arg5]] : index to i32
//       CHECK:     %[[v6:.+]] = executor.table.get %[[v0]][0] : <!executor.ptr<host>, !executor.ptr<host>, i32>
//       CHECK:     %[[v7:.+]] = executor.table.get %[[v0]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32>
//       CHECK:     %[[v8:.+]] = executor.table.create(%[[v6]], %[[v7]], %[[v1]], %[[v2]], %[[c2_i32]], %[[v3]], %[[v4]], %[[v5]], %[[c1_i32]] :
//       CHECK:     %[[v9:.+]] = builtin.unrealized_conversion_cast %[[v8]]
//       CHECK:     return %[[v9]] :

// -----

func.func @dynamic_copy(%arg0: memref<?x?xi32, #executor.memory_type<host>>, %arg1: memref<?x?xi32, #executor.memory_type<host>>) {
  memref.copy %arg0, %arg1 : memref<?x?xi32, #executor.memory_type<host>> to memref<?x?xi32, #executor.memory_type<host>>
  return
}

// CHECK-LABEL: @dynamic_copy
//  CHECK-SAME: (%[[arg0:.+]]: memref<{{.+}}>, %[[arg1:.+]]: memref<
//   CHECK-DAG:     %[[c0_i32:.+]] = executor.constant 0 : i32
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] :
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] :
//       CHECK:     %[[v2:.+]] = executor.table.get %[[v0]][3] :
//       CHECK:     %[[v3:.+]] = executor.table.get %[[v0]][4] :
//       CHECK:     %[[v4:.+]] = executor.muli %[[v2]], %[[v3]] : i32
//       CHECK:     %[[v5:.+]] = executor.getoffset[%[[v4]]] : (i32) -> i32, i32
//   CHECK-DAG:     %[[v7:.+]] = executor.table.get %[[v0]][1] :
//   CHECK-DAG:     %[[v8:.+]] = executor.table.get %[[v1]][1] :
//       CHECK:     executor.memcpy
//   CHECK-NOT:     cuda.stream.sync

// -----

// Memref.reshape operations should be eliminated by `memref-expand-ops` before running
// 'memref-to-executor'
func.func @memref_reshape_unsupported(%arg0: memref<?xi32>, %arg1: memref<2xindex>) -> memref<?x?xi32> {
  // expected-error @below {{failed to legalize operation 'memref.reshape' that was explicitly marked illegal}}
  %0 = memref.reshape %arg0 (%arg1) : (memref<?xi32>, memref<2xindex>) -> memref<?x?xi32>
  return %0 : memref<?x?xi32>
}

// -----

func.func @memref_reshape_unsupported(%input : memref<2x3xf32>, %shape : memref<?xindex>) {
  // expected-error @below {{failed to legalize operation 'memref.reshape' that was explicitly marked illegal}}
  %output = memref.reshape %input(%shape)
                : (memref<2x3xf32>, memref<?xindex>) -> memref<*xf32>
  return
}

// -----


!memref_4xi8 = memref<4xi8, #executor.memory_type<device>>

memref.global "private" @global1 : memref<4xi8, #executor.memory_type<host_pinned>> {alignment = 32 : i64}
memref.global "private" @global2 : !memref_4xi8 = dense<[5, 6, 7, 8]> {constant}
memref.global "private" @global3 : !memref_4xi8

func.func @memref_global() -> (memref<4xi8, #executor.memory_type<host_pinned>>, !memref_4xi8, !memref_4xi8) {
  %1 = memref.get_global @global1 : memref<4xi8, #executor.memory_type<host_pinned>>
  %2 = memref.get_global @global2 : !memref_4xi8
  %3 = memref.get_global @global3 : !memref_4xi8
  return %1, %2, %3 : memref<4xi8, #executor.memory_type<host_pinned>>, !memref_4xi8, !memref_4xi8
}

//   CHECK-DAG:   executor.data_segment @global1 uninitialized align 32 address_space <host_pinned> dense<0> : tensor<4xi8>
//   CHECK-DAG:   executor.data_segment @global2 constant address_space <device> dense<[5, 6, 7, 8]> : tensor<4xi8>
//   CHECK-DAG:   executor.data_segment @global3 uninitialized address_space <device> dense<0> : tensor<4xi8>
// CHECK-LABEL: func.func @memref_global
//    CHECK-DAG:     %[[c0_i32:.+]] = executor.constant 0 : i32
//    CHECK-DAG:     %[[c1_i32:.+]] = executor.constant 1 : i32
//    CHECK-DAG:     %[[c4_i32:.+]] = executor.constant 4 : i32
//    CHECK-DAG:     %[[v0:.+]] = executor.load_data_segment @global1 : !executor.ptr<host_pinned>
//    CHECK-DAG:     %[[v1:.+]] = executor.table.create(%[[v0]], %[[v0]], %[[c0_i32]], %[[c4_i32]], %[[c1_i32]] :
//    CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[v1]] :
//    CHECK-DAG:     %[[v3:.+]] = executor.load_data_segment @global2 : !executor.ptr<device>
//    CHECK-DAG:     %[[v4:.+]] = executor.table.create(%[[v3]], %[[v3]], %[[c0_i32]], %[[c4_i32]], %[[c1_i32]] :
//    CHECK-DAG:     %[[v5:.+]] = builtin.unrealized_conversion_cast %[[v4]] :
//    CHECK-DAG:     %[[v6:.+]] = executor.load_data_segment @global3 : !executor.ptr<device>
//    CHECK-DAG:     %[[v7:.+]] = executor.table.create(%[[v6]], %[[v6]], %[[c0_i32]], %[[c4_i32]], %[[c1_i32]] :
//    CHECK-DAG:     %[[v8:.+]] = builtin.unrealized_conversion_cast %[[v7]] :
//    CHECK-DAG:     return %[[v2]], %[[v5]], %[[v8]] :