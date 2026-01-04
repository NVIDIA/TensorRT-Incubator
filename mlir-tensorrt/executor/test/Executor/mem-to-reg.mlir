// RUN: executor-opt %s -split-input-file -mem2reg | FileCheck %s
// RUN: executor-opt %s -split-input-file -sroa -mem2reg | FileCheck %s --check-prefix=SROA

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

// -----

func.func @sroa_table_test() -> (i32, f32) {
  %c1 = executor.constant 1 : i32
  %0 = executor.alloca %c1 x !executor.table<i32, f32> : (i32) -> !executor.ptr<host>
  %offset0 = executor.constant 0 : i64
  %offset4 = executor.constant 4 : i64
  %val_i32 = executor.constant 42 : i32
  %val_f32 = executor.constant 3.14 : f32
  executor.store %val_i32 to %0 + %offset0 : i32, !executor.ptr<host>, i64
  executor.store %val_f32 to %0 + %offset4 : f32, !executor.ptr<host>, i64
  %1 = executor.load %0 + %offset0 : (!executor.ptr<host>, i64) -> i32
  %2 = executor.load %0 + %offset4 : (!executor.ptr<host>, i64) -> f32
  return %1, %2 : i32, f32
}

// SROA-LABEL: @sroa_table_test
// SROA-NOT: executor.alloca {{.*}} !executor.table
// SROA-NOT: executor.store
// SROA-NOT: executor.load
// SROA-DAG: %[[val_i32:.+]] = executor.constant 42 : i32
// SROA-DAG: %[[val_f32:.+]] = executor.constant {{.*}} : f32
// SROA: return %[[val_i32]], %[[val_f32]] : i32, f32

// -----

func.func @sroa_table_3fields() -> (i64, i32, f32) {
  %c1 = executor.constant 1 : i32
  %0 = executor.alloca %c1 x !executor.table<i64, i32, f32> : (i32) -> !executor.ptr<host>
  %offset0 = executor.constant 0 : i64
  %offset8 = executor.constant 8 : i64
  %offset12 = executor.constant 12 : i64
  %val0 = executor.constant 100 : i64
  %val1 = executor.constant 200 : i32
  %val2 = executor.constant 3.0 : f32
  executor.store %val0 to %0 + %offset0 : i64, !executor.ptr<host>, i64
  executor.store %val1 to %0 + %offset8 : i32, !executor.ptr<host>, i64
  executor.store %val2 to %0 + %offset12 : f32, !executor.ptr<host>, i64
  %1 = executor.load %0 + %offset0 : (!executor.ptr<host>, i64) -> i64
  %2 = executor.load %0 + %offset8 : (!executor.ptr<host>, i64) -> i32
  %3 = executor.load %0 + %offset12 : (!executor.ptr<host>, i64) -> f32
  return %1, %2, %3 : i64, i32, f32
}

// SROA-LABEL: @sroa_table_3fields
// SROA-NOT: executor.alloca {{.*}} !executor.table
// SROA-NOT: executor.store
// SROA-NOT: executor.load
// SROA-DAG: %[[val0:.+]] = executor.constant 100 : i64
// SROA-DAG: %[[val1:.+]] = executor.constant 200 : i32
// SROA-DAG: %[[val2:.+]] = executor.constant {{.*}} : f32
// SROA: return %[[val0]], %[[val1]], %[[val2]] : i64, i32, f32

