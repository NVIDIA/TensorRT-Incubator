// RUN: executor-opt %s -split-input-file -executor-expand-ops -verify-diagnostics | FileCheck %s

func.func @lower_gep() -> i64 {
  %0 = executor.getoffset[1] : () -> i64, f32
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_gep
//   CHECK-DAG:     %[[c4_i64:.+]] = executor.constant 4 : i64
//   CHECK-DAG:     return %[[c4_i64]] : i64

// -----

!el_type = !executor.table<f32, f64>

func.func @lower_gep() -> i64 {
  %0 = executor.getoffset[1] : () -> i64, !el_type
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_gep
//   CHECK-DAG:     %[[c16_i64:.+]] = executor.constant 16 : i64
//   CHECK-DAG:     return %[[c16_i64]] : i64

// -----

!el_type = !executor.table<!executor.table<i32, f32>, !executor.table<i32, f32>>

func.func @lower_gep(%arg1: i64) -> i64 {
  %0 = executor.getoffset[%arg1, 1, 1] : (i64) -> i64, !el_type
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_gep
//  CHECK-SAME: (%[[arg1:.+]]: i64) -> i64 {
//   CHECK-DAG:     %[[c4_i64:.+]] = executor.constant 4 : i64
//   CHECK-DAG:     %[[c8_i64:.+]] = executor.constant 8 : i64
//   CHECK-DAG:     %[[c16_i64:.+]] = executor.constant 16 : i64
//   CHECK-DAG:     %[[v0:.+]] = executor.muli %[[arg1]], %[[c16_i64]] : i64
//   CHECK-DAG:     %[[v2:.+]] = executor.addi %[[v0]], %[[c8_i64]] : i64
//   CHECK-DAG:     %[[v3:.+]] = executor.addi %[[v2]], %[[c4_i64]] : i64
//   CHECK-DAG:     return %[[v3]] : i64

// -----

!el_type = !executor.table<!executor.table<i32, f64>, !executor.table<i64, f32>>

func.func @lower_gep(%arg1: i64) -> i64 {
  %0 = executor.getoffset[%arg1, 1, 1] : (i64) -> i64, !el_type
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_gep
//  CHECK-SAME: (%[[arg0:.+]]: i64) -> i64 {
//       CHECK:     %[[c8_i64:.+]] = executor.constant 8 : i64
//   CHECK-DAG:     %[[c16_i64:.+]] = executor.constant 16 : i64
//   CHECK-DAG:     %[[c32_i64:.+]] = executor.constant 32 : i64
//   CHECK-DAG:     %[[v0:.+]] = executor.muli %[[arg0]], %[[c32_i64]] : i64
//   CHECK-DAG:     %[[v1:.+]] = executor.addi %[[v0]], %[[c16_i64]] : i64
//   CHECK-DAG:     %[[v2:.+]] = executor.addi %[[v1]], %[[c8_i64]] : i64
//   CHECK-DAG:     return %[[v2]] : i64

// -----

!el_type = !executor.table<!executor.table<i32, f64>, !executor.table<i64, f32>>

func.func @lower_gep() -> i64 {
  %0 = executor.getoffset[0, 1, 1] : () -> i64, !el_type
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_gep
//   CHECK-DAG:     %[[c24_i64:.+]] = executor.constant 24 : i64
//   CHECK-DAG:     return %[[c24_i64]] : i64

// -----

// test 32-bit index width

!el_type = !executor.table<!executor.table<i32, f64>, !executor.table<i64, f32>>

builtin.module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<index, 32 : i64>
  >
} {
  func.func @lower_gep() -> i64 {
    // expected-error @below {{'executor.getoffset' op result type ('i64') does not match the width of the IndexType ('i32') specified by the DataLayout}}
    %0 = executor.getoffset[0, 1, 1] : () -> i64, !el_type
    return %0 : i64
  }
}

// -----

!el_type = !executor.table<!executor.table<i32, f64>, !executor.table<i64, f32>>

builtin.module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<index, 32 : i64>
  >
} {
  func.func @lower_gep() -> i32 {
    %0 = executor.getoffset[0, 1, 1] : () -> i32, !el_type
    return %0 : i32
  }
}

// CHECK-LABEL: func.func @lower_gep
//   CHECK-DAG:     %[[c24:.+]] = executor.constant 24 : i32
//   CHECK-DAG:     return %[[c24]] : i32

// -----

!el_type = !executor.table<!executor.table<i32, f64>, !executor.table<i64, f32>>

builtin.module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<index, 32 : i64>
  >
} {
  func.func @lower_gep() -> i32 {
    // expected-error @below {{'executor.getoffset' op index #0 (8589934591 : i64) cannot be converted losslessly to the width of the IndexType (i32) specified by the data layout}}
    %0 = executor.getoffset[0x1FFFFFFFF, 1, 1] : () -> i32, !el_type
    return %0 : i32
  }
}

