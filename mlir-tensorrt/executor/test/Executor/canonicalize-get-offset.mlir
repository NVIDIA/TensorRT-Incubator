// RUN: executor-opt %s -split-input-file -canonicalize | FileCheck %s

func.func @offset_simplify_const_indices_1(%arg1: i64) -> i64{
  %c1 = executor.constant 1 : i32
  %0 = executor.getoffset [%arg1, 1, %c1] : (i64, i32) -> i64, !executor.table<f32, vector<10xi1>>
  return %0 : i64
}

// CHECK-LABEL: func.func @offset_simplify_const_indices_1
//  CHECK-SAME: (%[[arg1:.+]]: i64) -> i64 {
//       CHECK:     %[[v0:.+]] = executor.getoffset[%[[arg1]], 1, 1] : (i64) -> i64, !executor.table<f32, vector<10xi1>>
//       CHECK:     return %[[v0]] : i64

// -----

func.func @offset_simplify_const_indices_2(%arg1: i64) -> i64{
  %c1 = executor.constant 1 : i32
  %0 = executor.getoffset [%c1] : (i32) -> i64, f32
  return %0 : i64
}

// CHECK-LABEL: func.func @offset_simplify_const_indices_2
//  CHECK-SAME: (%[[arg1:.+]]: i64) -> i64 {
//       CHECK:     %[[v0:.+]] = executor.getoffset[1] : () -> i64, f32
//       CHECK:     return %[[v0]] : i64

// -----

func.func @offset_simplify_zero_offset() -> i64 {
  %0 = executor.getoffset [0] : () -> i64, f32
  return %0 : i64
}

// CHECK-LABEL: func.func @offset_simplify_zero_offset
//       CHECK:     %[[c0:.+]] = executor.constant 0 : i64
//       CHECK:     return %[[c0]] : i64

// -----

func.func @offset_alignto_noop(%arg0: i64) -> i64 {
  %0 = executor.alignto %arg0, 1 : i64
  return %0 : i64
}

// CHECK-LABEL: func.func @offset_alignto_noop
//  CHECK-SAME: (%[[arg0:.+]]: i64) -> i64 {
//       CHECK:     return %[[arg0]] : i64

// -----

func.func @offset_alignto_redudndant(%arg0: i64) -> i64 {
  %0 = executor.alignto %arg0, 8 : i64
  %1 = executor.alignto %0, 4 : i64
  return %1 : i64
}

// CHECK-LABEL: func.func @offset_alignto_redudndant
//  CHECK-SAME: (%[[arg0:.+]]: i64)
//       CHECK:     %[[v0:.+]] = executor.alignto %[[arg0]], 8 : i64
//       CHECK:     return %[[v0]] : i64

// -----

func.func @offset_alignto_not_redudndant(%arg0: i64) -> i64 {
  %0 = executor.alignto %arg0, 4 : i64
  %1 = executor.alignto %0, 8 : i64
  return %1 : i64
}

// CHECK-LABEL: func.func @offset_alignto_not_redudndant
//  CHECK-SAME: (%[[arg0:.+]]: i64)
//       CHECK:     %[[v0:.+]] = executor.alignto %[[arg0]], 4 : i64
//       CHECK:     %[[v1:.+]] = executor.alignto %[[v0]], 8 : i64
//       CHECK:     return %[[v1]] : i64

// -----

func.func @offset_alignto_infer(%arg0: i64) -> i64 {
  %0 = executor.alignto %arg0, 8 : i64
  %c4 = executor.constant 4 : i64
  %1 = executor.addi %c4, %0 : i64
  %2 = executor.alignto %1, 4 : i64
  return %2 : i64
}

// CHECK-LABEL: func.func @offset_alignto_infer
//  CHECK-SAME: (%[[arg0:.+]]: i64) -> i64 {
//       CHECK:     %[[c4_i64:.+]] = executor.constant 4 : i64
//       CHECK:     %[[v0:.+]] = executor.alignto %[[arg0]], 8 : i64
//       CHECK:     %[[v1:.+]] = executor.addi %[[v0]], %[[c4_i64]] : i64
//       CHECK:     return %[[v1]] : i64

// -----

func.func @offset_alignto_infer2(%arg0: i64) -> i64 {
  %0 = executor.alignto %arg0, 8 : i64
  %c12 = executor.constant 12 : i64
  %1 = executor.addi %c12, %0 : i64
  %2 = executor.alignto %1, 4 : i64
  return %2 : i64
}

// CHECK-LABEL: func.func @offset_alignto_infer2
//  CHECK-SAME: (%[[arg0:.+]]: i64)
//       CHECK:     %[[c12_i64:.+]] = executor.constant 12 : i64
//       CHECK:     %[[v0:.+]] = executor.alignto %[[arg0]], 8 : i64
//       CHECK:     %[[v1:.+]] = executor.addi %[[v0]], %[[c12_i64]] : i64
//       CHECK:     return %[[v1]] : i64

// -----

func.func @offset_alignto_cant_simplify(%arg0: i64) -> i64 {
  %0 = executor.alignto %arg0, 8 : i64
  %c16 = executor.constant 16 : i64
  %1 = executor.addi %0, %c16 : i64
  %2 = executor.alignto %1, 16 : i64
  return %2 : i64
}

// CHECK-LABEL: func.func @offset_alignto_cant_simplify
//  CHECK-SAME: (%[[arg0:.+]]: i64)
//       CHECK:     %[[c16_i64:.+]] = executor.constant 16 : i64
//       CHECK:     %[[v0:.+]] = executor.alignto %[[arg0]], 8 : i64
//       CHECK:     %[[v1:.+]] = executor.addi %[[v0]], %[[c16_i64]] : i64
//       CHECK:     %[[v2:.+]] = executor.alignto %[[v1]], 16 : i64
//       CHECK:     return %[[v2]] : i64
