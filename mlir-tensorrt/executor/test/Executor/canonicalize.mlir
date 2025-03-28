// RUN: executor-opt %s -split-input-file -canonicalize | FileCheck %s
// RUN: executor-opt %s -split-input-file -canonicalize --cse | FileCheck %s --check-prefix=CSE

func.func @add(%arg0: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %0 = executor.addi %arg0, %c0 : i32
  %1 = executor.addi %c0, %arg0 : i32
  %2 = executor.addi %0, %1 : i32
  %3 = executor.addi %c1, %c2 : i32
  return %2, %3 : i32, i32
}

// CHECK-LABEL: @add
//  CHECK-SAME: (%[[arg0:.+]]: i32) -> (i32, i32) {
//       CHECK:     %[[c3_i32:.+]] = executor.constant 3 : i32
//       CHECK:     %[[v0:.+]] = executor.addi %[[arg0]], %[[arg0]] : i32
//       CHECK:     return %[[v0]], %[[c3_i32]] : i32, i32

// -----

func.func @sub(%arg0: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %0 = executor.subi %arg0, %c0 : i32
  %1 = executor.subi %c0, %arg0 : i32
  %2 = executor.subi %0, %1 : i32
  %3 = executor.subi %c1, %c2 : i32
  return %2, %3 : i32, i32
}

// CHECK-LABEL: @sub
//  CHECK-SAME: (%[[arg0:.+]]: i32) -> (i32, i32) {
//       CHECK:     %[[cm1_i32:.+]] = executor.constant -1 : i32
//       CHECK:     %[[c0_i32:.+]] = arith.constant 0 : i32
//       CHECK:     %[[v0:.+]] = executor.subi %[[c0_i32]], %[[arg0]] : i32
//       CHECK:     %[[v1:.+]] = executor.subi %[[arg0]], %[[v0]] : i32
//       CHECK:     return %[[v1]], %[[cm1_i32]] : i32, i32

// -----

func.func @sdiv(%arg0: i32) -> (i32, i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %0 = executor.sdivi %arg0, %c1 : i32
  %1 = executor.sdivi %arg0, %c0 : i32
  %3 = executor.sdivi %c4, %c2 : i32
  return %0, %1, %3 : i32, i32, i32
}

// CHECK-LABEL: @sdiv
//  CHECK-SAME: (%[[arg0:.+]]: i32) -> (i32, i32, i32) {
//       CHECK:     %[[c2_i32:.+]] = executor.constant 2 : i32
//       CHECK:     %[[c0_i32:.+]] = arith.constant 0 : i32
//       CHECK:     %[[v0:.+]] = executor.sdivi %[[arg0]], %[[c0_i32]] : i32
//       CHECK:     return %[[arg0]], %[[v0]], %[[c2_i32]] : i32, i32, i32

// -----

func.func @srem(%arg0: i32) -> (i32, i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %0 = executor.sremi %arg0, %c1 : i32
  %1 = executor.sremi %arg0, %c0 : i32
  %3 = executor.sremi %c4, %c3 : i32
  return %0, %1, %3 : i32, i32, i32
}

// CHECK-LABEL: @srem
//  CHECK-SAME: (%[[arg0:.+]]: i32) -> (i32, i32, i32) {
//   CHECK-DAG:     %[[c1_i32:.+]] = executor.constant 1 : i32
//   CHECK-DAG:     %[[c0_i32:.+]] = executor.constant 0 : i32
//   CHECK-DAG:     %[[c0_i32_0:.+]] = arith.constant 0 : i32
//       CHECK:     %[[v0:.+]] = executor.sremi %[[arg0]], %[[c0_i32_0]] : i32
//       CHECK:     return %[[c0_i32]], %[[v0]], %[[c1_i32]] : i32, i32, i32

// -----

func.func @sfloordiv(%arg0: i32) -> (i32, i32) {
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %0 = executor.sfloor_divi %arg0, %c1 : i32
  %1 = executor.sfloor_divi %arg0, %c0 : i32
  return %0, %1 : i32, i32
}

// CHECK-LABEL: @sfloordiv
//  CHECK-SAME: (%[[arg0:.+]]: i32) -> (i32, i32) {
//       CHECK:     %[[c0_i32:.+]] = arith.constant 0 : i32
//       CHECK:     %[[v0:.+]] = executor.sfloor_divi %[[arg0]], %[[c0_i32]] : i32
//       CHECK:     return %[[arg0]], %[[v0]] : i32, i32

// -----

func.func @mul(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %0 = executor.muli %arg0, %c1 : i32
  %1 = executor.muli %c1, %arg1 : i32
  %2 = executor.muli %0, %1 : i32
  %3 = executor.muli %c2, %c2 : i32
  return %2, %3 : i32, i32
}

// CHECK-LABEL: @mul
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> (i32, i32) {
//       CHECK:     %[[c4_i32:.+]] = executor.constant 4 : i32
//       CHECK:     %[[v0:.+]] = executor.muli %[[arg0]], %[[arg1]] : i32
//       CHECK:     return %[[v0]], %[[c4_i32]] : i32, i32

// -----

func.func @mul_commutative(%arg0: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %0 = executor.muli %c2, %arg0 : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @mul_commutative
//  CHECK-SAME: (%[[arg0:.+]]: i32)
//  CHECK-NEXT:     %[[c2_i32:.+]] = arith.constant 2 : i32
//  CHECK-NEXT:     %[[v0:.+]] = executor.muli %[[arg0]], %[[c2_i32]]
//  CHECK-NEXT:     return %[[v0]]

// -----

!table = !executor.table<i32, f32, i64>

func.func @table_extract_create(%arg0: i32, %arg1: f32) -> (i32, i64) {
  %0 = executor.table.create (%arg0, %arg1 : i32, f32) : !table
  %1 = executor.table.get %0[0] : !table
  %2 = executor.table.get %0[2] : !table
  return %1, %2 : i32, i64
}

// CHECK-LABEL: @table_extract_create
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: f32) -> (i32, i64) {
//       CHECK:     %[[v0:.+]] = executor.table.create(%[[arg0]], %[[arg1]] : i32, f32) : <i32, f32, i64>
//       CHECK:     %[[v1:.+]] = executor.table.get %[[v0]][2] : <i32, f32, i64>
//       CHECK:     return %[[arg0]], %[[v1]] : i32, i64

// -----

!table = !executor.table<i32, f32, i64>

func.func @table_extract_insert(%arg0: !table) -> (f32, i32) {
  %c0 = executor.constant 0 : i32
  %0 = executor.table.set %c0 into %arg0[0] : i32, !table
  %1 = executor.table.get %0[1] : !table
  %2 = executor.table.get %0[0] : !table
  return %1, %2 : f32, i32
}

// CHECK-LABEL: @table_extract_insert
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<i32, f32, i64>) -> (f32, i32) {
//       CHECK:     %[[c0_i32:.+]] = executor.constant 0 : i32
//       CHECK:     %[[v0:.+]] = executor.table.get %[[arg0]][1] : <i32, f32, i64>
//       CHECK:     return %[[v0]], %[[c0_i32]] : f32, i32

// -----

!table = !executor.table<i32, f32, i64>

func.func @table_extract_insert_chain(%arg0: !table) -> (i32, i32) {
  %c0 = executor.constant 0 : i32
  %c1 = executor.constant 1.0 : f32
  %c2 = executor.constant 2 : i64
  %c3 = executor.constant 3 : i32
  %0 = executor.table.set %c0 into %arg0[0] : i32, !table
  %1 = executor.table.set %c1 into %0[1] : f32, !table
  %2 = executor.table.set %c2 into %1[2] : i64, !table
  %3 = executor.table.set %c3 into %2[0] : i32, !table
  %4 = executor.table.get %2[0] : !table
  %5 = executor.table.get %3[0] : !table
  return %4, %5 : i32, i32
}

// CHECK-LABEL: @table_extract_insert_chain
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<i32, f32, i64>) -> (i32, i32) {
//       CHECK:     %[[c0_i32:.+]] = executor.constant 0 : i32
//       CHECK:     %[[c3_i32:.+]] = executor.constant 3 : i32
//       CHECK:     return %[[c0_i32]], %[[c3_i32]] : i32, i32

// -----

func.func @alloc() -> (!executor.ptr<host>, !executor.ptr<host>) {
  %c16 = arith.constant 16 : i32
  %0 = executor.alloc %c16 bytes align(%c16) : (i32, i32) -> !executor.ptr<host>
  %1 = executor.alloc %c16 bytes align(%c16) : (i32, i32) -> !executor.ptr<host>
  executor.return %0, %1 : !executor.ptr<host>, !executor.ptr<host>
}

//   CSE-LABEL: @alloc
// CSE-COUNT-2:   executor.alloc

// -----

func.func private @magic_alloc() -> (!executor.ptr<device>)
executor.data_segment @__constant_1xi32_initializer dense<0> : tensor<1xi32>
executor.global @__constant_1xi32 constant : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32> {
  %c0_i32 = executor.constant 0 : i32
  %c4_i32 = executor.constant 4 : i32
  %c1_i32 = executor.constant 1 : i32
  %1 = func.call @magic_alloc() : () -> !executor.ptr<device>
  %2 = executor.load_data_segment @__constant_1xi32_initializer : !executor.ptr<host>
  %3 = executor.table.create(%1, %1, %c0_i32, %c4_i32, %c1_i32 : !executor.ptr<device>, !executor.ptr<device>, i32, i32, i32) : <!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
  executor.return %3 : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
}
func.func @canon_extract_from_const_global() -> !executor.table<!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32> {
  %0 = executor.get_global @__constant_1xi32 : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
  %1 = executor.table.get %0[0] : <!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
  %2 = executor.table.get %0[1] : <!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
  %3 = executor.table.get %0[2] : <!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
  %4 = executor.table.get %0[3] : <!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
  %5 = executor.table.get %0[4] : <!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
  %6 = executor.table.create(%1, %2, %3, %4, %5 : !executor.ptr<device>, !executor.ptr<device>, i32, i32, i32) : <!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
  return %6 : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
}

// CHECK-LABEL: func.func @canon_extract_from_const_global
//       CHECK:     %[[c1_i32:.+]] = executor.constant 1 : i32
//       CHECK:     %[[c4_i32:.+]] = executor.constant 4 : i32
//       CHECK:     %[[c0_i32:.+]] = executor.constant 0 : i32
//       CHECK:     %[[v0:.+]] = executor.get_global @__constant_1xi32 : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
//       CHECK:     %[[v1:.+]] = executor.table.get %[[v0]][0] : <!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
//       CHECK:     %[[v2:.+]] = executor.table.get %[[v0]][1] : <!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
//       CHECK:     %[[v3:.+]] = executor.table.create(%[[v1]], %[[v2]], %[[c0_i32]], %[[c4_i32]], %[[c1_i32]] : !executor.ptr<device>, !executor.ptr<device>, i32, i32, i32) : <!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
//       CHECK:     return %[[v3]] : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i32, i32, i32>
