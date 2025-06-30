// RUN: executor-opt %s -split-input-file -executor-lower-to-runtime-builtins | FileCheck %s

func.func @alignto_lowering(%arg0: i32, %arg1: i64) -> (i32, i64) {
  %0 = executor.alignto %arg0, 2 : i32
  %1 = executor.alignto %arg1, 4 : i64
  return %0, %1 : i32, i64
}

//       CHECK:   executor.func private @_alignto_i64(i64, i32) -> i64
//       CHECK:   executor.func private @_alignto_i32(i32, i32) -> i32
// CHECK-LABEL: func.func @alignto_lowering
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: i64) -> (i32, i64) {
//       CHECK:     %[[c2_i32:.+]] = executor.constant 2 : i32
//       CHECK:     %[[v0:.+]] = executor.call @_alignto_i32(%[[arg0]], %[[c2_i32]]) : (i32, i32) -> i32
//       CHECK:     %[[c4_i32:.+]] = executor.constant 4 : i32
//       CHECK:     %[[v1:.+]] = executor.call @_alignto_i64(%[[arg1]], %[[c4_i32]]) : (i64, i32) -> i64
//       CHECK:     return %[[v0]], %[[v1]] : i32, i64
