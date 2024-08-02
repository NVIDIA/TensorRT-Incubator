// RUN: executor-opt %s -split-input-file -convert-memref-to-executor="allow-unchecked-memref-cast-conversion=false index-bitwidth=64" -cse -canonicalize | FileCheck %s


func.func @test_memref_of_memref_load_store_1d() {
  %c1 = arith.constant 1 : index
  %0 = memref.alloc () : memref<2xmemref<1024xf32>>

  %1 = memref.alloc () : memref<1024xf32>
  memref.store %1, %0[%c1] : memref<2xmemref<1024xf32>>

  %loaded = memref.load %0[%c1] : memref<2xmemref<1024xf32>>

  memref.dealloc %loaded : memref<1024xf32>
  memref.dealloc %0 : memref<2xmemref<1024xf32>>

  return
}

// CHECK-LABEL: @test_memref_of_memref_load_store_1d
//   CHECK-DAG:     %[[c4096_i64:.+]] = executor.constant 4096 : i64
//   CHECK-DAG:     %[[c80_i64:.+]] = executor.constant 80 : i64
//   CHECK-DAG:     %[[c16_i64:.+]] = executor.constant 16 : i64
//   CHECK-DAG:     %[[c1024_i64:.+]] = executor.constant 1024 : i64
//   CHECK-DAG:     %[[c0_i64:.+]] = executor.constant 0 : i64
//   CHECK-DAG:     %[[c64_i64:.+]] = executor.constant 64 : i64
//   CHECK-DAG:     %[[c1_i64:.+]] = executor.constant 1 : i64
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[c1]] : index to i64
//       CHECK:     %[[v1:.+]] = executor.alloc %[[c80_i64]] bytes align(%[[c64_i64]]) : (i64, i64) -> !executor.ptr<host>
//       CHECK:     %[[v2:.+]] = executor.alloc %[[c4096_i64]] bytes align(%[[c16_i64]]) : (i64, i64) -> !executor.ptr<host>
//       CHECK:     %[[v3:.+]] = executor.table.create(%[[v2]], %[[v2]], %[[c0_i64]], %[[c1024_i64]], %[[c1_i64]] : !executor.ptr<host>, !executor.ptr<host>, i64, i64, i64)
//       CHECK:     %[[v4:.+]] = executor.getoffset[%[[v0]]] : (i64) -> i64, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     executor.store %[[v3]] to %[[v1]] + %[[v4]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>, !executor.ptr<host>, i64
//       CHECK:     %[[v5:.+]] = executor.load %[[v1]] + %[[v4]] : (!executor.ptr<host>, i64) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     %[[v6:.+]] = executor.table.get %[[v5]][0] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     executor.dealloc %[[v6]] : <host>
//       CHECK:     executor.dealloc %[[v1]] : <host>

// -----

func.func @test_2d_memref_of_1d_memref_load_store() {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index

  %0 = memref.alloc () : memref<2x2xmemref<16xf32>>
  %1 = memref.alloc () : memref<16xf32>

  memref.store %1, %0[%c1, %c0] : memref<2x2xmemref<16xf32>>
  %loaded = memref.load %0[%c1, %c0] : memref<2x2xmemref<16xf32>>

  memref.dealloc %loaded : memref<16xf32>
  memref.dealloc %0 : memref<2x2xmemref<16xf32>>

  return
}

// CHECK-LABEL: @test_2d_memref_of_1d_memref_load_store
//   CHECK-DAG:     %[[c160_i64:.+]] = executor.constant 160 : i64
//   CHECK-DAG:     %[[c16_i64:.+]] = executor.constant 16 : i64
//   CHECK-DAG:     %[[c0_i64:.+]] = executor.constant 0 : i64
//   CHECK-DAG:     %[[c64_i64:.+]] = executor.constant 64 : i64
//   CHECK-DAG:     %[[c1_i64:.+]] = executor.constant 1 : i64
//   CHECK-DAG:     %[[c2_i64:.+]] = executor.constant 2 : i64
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[c1]] : index to i64
//       CHECK:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[c0]] : index to i64
//       CHECK:     %[[v2:.+]] = executor.alloc %[[c160_i64]] bytes align(%[[c64_i64]]) : (i64, i64) -> !executor.ptr<host>
//       CHECK:     %[[v3:.+]] = executor.alloc %[[c64_i64]] bytes align(%[[c16_i64]]) : (i64, i64) -> !executor.ptr<host>
//       CHECK:     %[[v4:.+]] = executor.table.create(%[[v3]], %[[v3]], %[[c0_i64]], %[[c16_i64]], %[[c1_i64]] : !executor.ptr<host>, !executor.ptr<host>, i64, i64, i64) : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     %[[v5:.+]] = executor.muli %[[v0]], %[[c2_i64]] : i64
//       CHECK:     %[[v6:.+]] = executor.addi %[[v5]], %[[v1]] : i64
//       CHECK:     %[[v7:.+]] = executor.getoffset[%[[v6]]] : (i64) -> i64, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     executor.store %[[v4]] to %[[v2]] + %[[v7]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>, !executor.ptr<host>, i64
//       CHECK:     %[[v8:.+]] = executor.load %[[v2]] + %[[v7]] : (!executor.ptr<host>, i64) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     %[[v9:.+]] = executor.table.get %[[v8]][0] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     executor.dealloc %[[v9]] : <host>
//       CHECK:     executor.dealloc %[[v2]] : <host>

// -----

func.func @test_dynamic_memref_of_0d_memref_load_store(%arg0: index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index

  %0 = memref.alloc (%arg0) : memref<?xmemref<f32>>
  %1 = memref.alloc () : memref<f32>

  memref.store %1, %0[%c1] : memref<?xmemref<f32>>
  %loaded = memref.load %0[%c1] : memref<?xmemref<f32>>

  memref.dealloc %loaded : memref<f32>
  memref.dealloc %0 : memref<?xmemref<f32>>

  return
}

// CHECK-LABEL: @test_dynamic_memref_of_0d_memref_load_store
//  CHECK-SAME: (%[[arg0:.+]]: index) {
//   CHECK-DAG:     %[[c16_i64:.+]] = executor.constant 16 : i64
//   CHECK-DAG:     %[[c4_i64:.+]] = executor.constant 4 : i64
//   CHECK-DAG:     %[[c0_i64:.+]] = executor.constant 0 : i64
//   CHECK-DAG:     %[[c32_i64:.+]] = executor.constant 32 : i64
//   CHECK-DAG:     %[[c24_i64:.+]] = executor.constant 24 : i64
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : index to i64
//       CHECK:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[c1]] : index to i64
//       CHECK:     %[[v2:.+]] = executor.muli %[[v0]], %[[c24_i64]] : i64
//       CHECK:     %[[v3:.+]] = executor.alloc %[[v2]] bytes align(%[[c32_i64]]) : (i64, i64) -> !executor.ptr<host>
//       CHECK:     %[[v4:.+]] = executor.alloc %[[c4_i64]] bytes align(%[[c16_i64]]) : (i64, i64) -> !executor.ptr<host>
//       CHECK:     %[[v5:.+]] = executor.table.create(%[[v4]], %[[v4]], %[[c0_i64]] : !executor.ptr<host>, !executor.ptr<host>, i64) : <!executor.ptr<host>, !executor.ptr<host>, i64>
//       CHECK:     %[[v6:.+]] = executor.getoffset[%[[v1]]] : (i64) -> i64, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64>
//       CHECK:     executor.store %[[v5]] to %[[v3]] + %[[v6]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64>, !executor.ptr<host>, i64
//       CHECK:     %[[v7:.+]] = executor.load %[[v3]] + %[[v6]] : (!executor.ptr<host>, i64) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64>
//       CHECK:     %[[v8:.+]] = executor.table.get %[[v7]][0] : <!executor.ptr<host>, !executor.ptr<host>, i64>
//       CHECK:     executor.dealloc %[[v8]] : <host>
//       CHECK:     executor.dealloc %[[v3]] : <host>
