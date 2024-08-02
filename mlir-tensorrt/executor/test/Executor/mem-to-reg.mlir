// RUN: executor-opt %s -split-input-file -mem2reg | FileCheck %s

func.func @mem2reg_test() -> f32 {
  %c1 = executor.constant 1 : i32
  %0 = executor.alloca %c1 x f32 : (i32) -> !executor.ptr<host>
  %offset = executor.constant 0 : i64
  %cst1 = executor.constant 1.0 : f32
  executor.store %cst1 to %0 + %offset : f32, !executor.ptr<host>, i64
  %1 = executor.load %0 + %offset : (!executor.ptr<host>, i64) -> f32
  return %1 : f32
}

// CHECK-LABEL: @mem2reg_test
//   CHECK-DAG:     %[[cst_f32:.+]] = executor.constant 1.{{[0e\+]+}} : f32
//   CHECK-DAG:     return %[[cst_f32]] : f32

// -----

func.func @mem2reg_test2() -> (f32, f32) {
  %c2 = executor.constant 2 : i32
  %0 = executor.alloca %c2 x f32 : (i32) -> !executor.ptr<host>
  %offset = executor.constant 0 : i64
  %offset4 = executor.constant 4 : i64
  %cst1 = executor.constant 1.0 : f32
  %cst2 = executor.constant 2.0 : f32
  executor.store %cst1 to %0 + %offset : f32, !executor.ptr<host>, i64
  executor.store %cst2 to %0 + %offset4 : f32, !executor.ptr<host>, i64
  %1 = executor.load %0 + %offset : (!executor.ptr<host>, i64) -> f32
  %2 = executor.load %0 + %offset4 : (!executor.ptr<host>, i64) -> f32
  return %1, %2 : f32, f32
}

// CHECK-LABEL: mem2reg_test2
// CHECK: executor.alloca
// CHECK-COUNT-2: executor.store
// CHECK-COUNT-2: executor.load

