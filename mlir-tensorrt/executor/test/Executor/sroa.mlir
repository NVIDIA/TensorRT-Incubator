// RUN: executor-opt %s -split-input-file -sroa -cse | FileCheck %s

// CHECK-LABEL: @sroa_simple_table
func.func @sroa_simple_table() -> (i32, f32) {
  %c1 = executor.constant 1 : i32
  // CHECK-NOT: executor.alloca {{.*}} !executor.table
  // Verify two separate scalar allocas were created
  // CHECK-DAG: %[[alloca_f32:.+]] = executor.alloca {{.*}} x f32 : (i32) -> !executor.ptr<host>
  // CHECK-DAG: %[[alloca_i32:.+]] = executor.alloca {{.*}} x i32 : (i32) -> !executor.ptr<host>
  %0 = executor.alloca %c1 x !executor.table<i32, f32> : (i32) -> !executor.ptr<host>

  %offset0 = executor.constant 0 : i64
  %offset4 = executor.constant 4 : i64
  // CHECK-DAG: %[[val_i32:.+]] = executor.constant 42 : i32
  %val_i32 = executor.constant 42 : i32
  // CHECK-DAG: %[[val_f32:.+]] = executor.constant {{.*}} : f32
  %val_f32 = executor.constant 3.14 : f32

  // Verify stores are rewired to use the split allocas with zero offset
  // CHECK: executor.store %[[val_i32]] to %[[alloca_i32]] + {{.*}} : i32, !executor.ptr<host>, i64
  executor.store %val_i32 to %0 + %offset0 : i32, !executor.ptr<host>, i64

  // CHECK: executor.store %[[val_f32]] to %[[alloca_f32]] + {{.*}} : f32, !executor.ptr<host>, i64
  executor.store %val_f32 to %0 + %offset4 : f32, !executor.ptr<host>, i64

  // Verify loads are rewired to use the split allocas
  // CHECK: %[[load_i32:.+]] = executor.load %[[alloca_i32]] + {{.*}} : (!executor.ptr<host>, i64) -> i32
  %1 = executor.load %0 + %offset0 : (!executor.ptr<host>, i64) -> i32

  // CHECK: %[[load_f32:.+]] = executor.load %[[alloca_f32]] + {{.*}} : (!executor.ptr<host>, i64) -> f32
  %2 = executor.load %0 + %offset4 : (!executor.ptr<host>, i64) -> f32

  // CHECK: return %[[load_i32]], %[[load_f32]] : i32, f32
  return %1, %2 : i32, f32
}

// -----

// CHECK-LABEL: @sroa_3field_table
func.func @sroa_3field_table() -> (i64, i32, f32) {
  %c1 = executor.constant 1 : i32
  // CHECK-NOT: executor.alloca {{.*}} !executor.table
  // Verify three separate scalar allocas were created
  // CHECK-DAG: %[[alloca_f32:.+]] = executor.alloca {{.*}} x f32 : (i32) -> !executor.ptr<host>
  // CHECK-DAG: %[[alloca_i32:.+]] = executor.alloca {{.*}} x i32 : (i32) -> !executor.ptr<host>
  // CHECK-DAG: %[[alloca_i64:.+]] = executor.alloca {{.*}} x i64 : (i32) -> !executor.ptr<host>
  %0 = executor.alloca %c1 x !executor.table<i64, i32, f32> : (i32) -> !executor.ptr<host>

  %offset0 = executor.constant 0 : i64
  %offset8 = executor.constant 8 : i64
  %offset12 = executor.constant 12 : i64
  // CHECK-DAG: %[[val0:.+]] = executor.constant 100 : i64
  %val0 = executor.constant 100 : i64
  // CHECK-DAG: %[[val1:.+]] = executor.constant 200 : i32
  %val1 = executor.constant 200 : i32
  // CHECK-DAG: %[[val2:.+]] = executor.constant {{.*}} : f32
  %val2 = executor.constant 3.0 : f32

  // Verify stores are rewired to use the split allocas
  // CHECK: executor.store %[[val0]] to %[[alloca_i64]] + {{.*}} : i64, !executor.ptr<host>, i64
  executor.store %val0 to %0 + %offset0 : i64, !executor.ptr<host>, i64

  // CHECK: executor.store %[[val1]] to %[[alloca_i32]] + {{.*}} : i32, !executor.ptr<host>, i64
  executor.store %val1 to %0 + %offset8 : i32, !executor.ptr<host>, i64

  // CHECK: executor.store %[[val2]] to %[[alloca_f32]] + {{.*}} : f32, !executor.ptr<host>, i64
  executor.store %val2 to %0 + %offset12 : f32, !executor.ptr<host>, i64

  // Verify loads are rewired to use the split allocas
  // CHECK: %[[load_i64:.+]] = executor.load %[[alloca_i64]] + {{.*}} : (!executor.ptr<host>, i64) -> i64
  %1 = executor.load %0 + %offset0 : (!executor.ptr<host>, i64) -> i64

  // CHECK: %[[load_i32:.+]] = executor.load %[[alloca_i32]] + {{.*}} : (!executor.ptr<host>, i64) -> i32
  %2 = executor.load %0 + %offset8 : (!executor.ptr<host>, i64) -> i32

  // CHECK: %[[load_f32:.+]] = executor.load %[[alloca_f32]] + {{.*}} : (!executor.ptr<host>, i64) -> f32
  %3 = executor.load %0 + %offset12 : (!executor.ptr<host>, i64) -> f32

  // CHECK: return %[[load_i64]], %[[load_i32]], %[[load_f32]] : i64, i32, f32
  return %1, %2, %3 : i64, i32, f32
}

// -----

// CHECK-LABEL: @sroa_nested_table
func.func @sroa_nested_table() -> (i32, i32) {
  %c1 = executor.constant 1 : i32
  // CHECK-NOT: executor.alloca {{.*}} !executor.table
  // Both the outer table and nested table get fully destructured into two i32 allocas
  // CHECK-DAG: %[[alloca_i32_0:.+]] = executor.alloca {{.*}} x i32 : (i32) -> !executor.ptr<host>
  // CHECK-DAG: %[[alloca_i32_1:.+]] = executor.alloca {{.*}} x i32 : (i32) -> !executor.ptr<host>
  %0 = executor.alloca %c1 x !executor.table<i32, !executor.table<i32>> : (i32) -> !executor.ptr<host>

  %offset0 = executor.constant 0 : i64
  %offset4 = executor.constant 4 : i64
  // CHECK-DAG: %[[val0:.+]] = executor.constant 10 : i32
  %val0 = executor.constant 10 : i32
  // CHECK-DAG: %[[val1:.+]] = executor.constant 20 : i32
  %val1 = executor.constant 20 : i32

  // Verify stores are rewired - each store uses one of the split allocas
  // CHECK: executor.store %[[val0]] to %[[alloca_i32_1]] + {{.*}} : i32, !executor.ptr<host>, i64
  executor.store %val0 to %0 + %offset0 : i32, !executor.ptr<host>, i64

  // CHECK: executor.store %[[val1]] to %[[alloca_i32_0]] + {{.*}} : i32, !executor.ptr<host>, i64
  executor.store %val1 to %0 + %offset4 : i32, !executor.ptr<host>, i64

  // Verify loads are rewired
  // CHECK: %[[load0:.+]] = executor.load %[[alloca_i32_1]] + {{.*}} : (!executor.ptr<host>, i64) -> i32
  %1 = executor.load %0 + %offset0 : (!executor.ptr<host>, i64) -> i32

  // CHECK: %[[load1:.+]] = executor.load %[[alloca_i32_0]] + {{.*}} : (!executor.ptr<host>, i64) -> i32
  %2 = executor.load %0 + %offset4 : (!executor.ptr<host>, i64) -> i32

  // CHECK: return %[[load0]], %[[load1]] : i32, i32
  return %1, %2 : i32, i32
}

// -----

// Test that SROA does NOT trigger for scalar allocas (non-table types)
// CHECK-LABEL: @no_sroa_scalar
func.func @no_sroa_scalar() -> i32 {
  %c1 = executor.constant 1 : i32
  // CHECK: executor.alloca %{{.+}} x i32
  %0 = executor.alloca %c1 x i32 : (i32) -> !executor.ptr<host>

  %offset0 = executor.constant 0 : i64
  %val = executor.constant 42 : i32

  executor.store %val to %0 + %offset0 : i32, !executor.ptr<host>, i64
  %result = executor.load %0 + %offset0 : (!executor.ptr<host>, i64) -> i32

  return %result : i32
}

// -----

// Test that SROA does NOT trigger for multi-element table allocas
// CHECK-LABEL: @no_sroa_multi_element
func.func @no_sroa_multi_element() -> i32 {
  %c2 = executor.constant 2 : i32
  // CHECK: executor.alloca %{{.+}} x !executor.table<i32, f32>
  %0 = executor.alloca %c2 x !executor.table<i32, f32> : (i32) -> !executor.ptr<host>

  %offset0 = executor.constant 0 : i64
  %val = executor.constant 42 : i32

  executor.store %val to %0 + %offset0 : i32, !executor.ptr<host>, i64
  %result = executor.load %0 + %offset0 : (!executor.ptr<host>, i64) -> i32

  return %result : i32
}
