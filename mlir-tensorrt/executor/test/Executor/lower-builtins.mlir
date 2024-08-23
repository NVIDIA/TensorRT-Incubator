// RUN: executor-opt %s -split-input-file -executor-lower-to-runtime-builtins | FileCheck %s

func.func @pointer_cast_lowering() -> (i32, i64) {
  %cst0 = executor.constant 123 : i32
  %ptr0 = executor.inttoptr %cst0 : (i32) -> !executor.ptr<host>
  %int0 = executor.ptrtoint %ptr0 : (!executor.ptr<host>) -> i32

  %cst1 = executor.constant 456 : i64
  %ptr1 = executor.inttoptr %cst1 : (i64) -> !executor.ptr<host>
  %int1 = executor.ptrtoint %ptr1 : (!executor.ptr<host>) -> i64
  return %int0, %int1 : i32, i64
}
//   CHECK-DAG:   executor.func private @_ptrtoint_i64_i64(!executor.ptr<host>) -> i64
//   CHECK-DAG:   executor.func private @_inttoptr_i64_i64(i64) -> !executor.ptr<host>
//   CHECK-DAG:   executor.func private @_ptrtoint_i64_i32(!executor.ptr<host>) -> i32
//   CHECK-DAG:   executor.func private @_inttoptr_i64_i32(i32) -> !executor.ptr<host>
// CHECK-LABEL: func.func @pointer_cast_lowering
//  CHECK-SAME: () -> (i32, i64) {
//       CHECK:     %[[c123_i32:.+]] = executor.constant 123 : i32
//       CHECK:     %[[v0:.+]] = executor.call @_inttoptr_i64_i32(%[[c123_i32]]) : (i32) -> !executor.ptr<host>
//       CHECK:     %[[v1:.+]] = executor.call @_ptrtoint_i64_i32(%[[v0]]) : (!executor.ptr<host>) -> i32
//       CHECK:     %[[c456_i64:.+]] = executor.constant 456 : i64
//       CHECK:     %[[v2:.+]] = executor.call @_inttoptr_i64_i64(%[[c456_i64]]) : (i64) -> !executor.ptr<host>
//       CHECK:     %[[v3:.+]] = executor.call @_ptrtoint_i64_i64(%[[v2]]) : (!executor.ptr<host>) -> i64
//       CHECK:     return %[[v1]], %[[v3]] : i32, i64

// -----

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
