// RUN: executor-opt %s -split-input-file -executor-decompose-aggregate-loads-and-stores | FileCheck %s
// RUN: executor-opt %s -split-input-file -executor-decompose-aggregate-loads-and-stores -executor-expand-ops -cse -canonicalize | FileCheck %s --check-prefix=DECOMP


func.func @memref_load(%arg0: !executor.ptr<host>, %arg1: i64) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64> {
  %0 = executor.load %arg0 + %arg1 : (!executor.ptr<host>, i64) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  return %0 : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
}

// CHECK-LABEL: func.func @memref_load
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: i64) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64> {
//   CHECK-DAG:     %[[v0:.+]] = executor.getoffset[0, 0] : () -> i64, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//   CHECK-DAG:     %[[v1:.+]] = executor.addi %[[arg1]], %[[v0]] : i64
//   CHECK-DAG:     %[[v2:.+]] = executor.load %[[arg0]] + %[[v1]] : (!executor.ptr<host>, i64) -> !executor.ptr<host>
//   CHECK-DAG:     %[[v3:.+]] = executor.getoffset[0, 1] : () -> i64, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//   CHECK-DAG:     %[[v4:.+]] = executor.addi %[[arg1]], %[[v3]] : i64
//   CHECK-DAG:     %[[v5:.+]] = executor.load %[[arg0]] + %[[v4]] : (!executor.ptr<host>, i64) -> !executor.ptr<host>
//   CHECK-DAG:     %[[v6:.+]] = executor.getoffset[0, 2] : () -> i64, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//   CHECK-DAG:     %[[v7:.+]] = executor.addi %[[arg1]], %[[v6]] : i64
//   CHECK-DAG:     %[[v8:.+]] = executor.load %[[arg0]] + %[[v7]] : (!executor.ptr<host>, i64) -> i64
//   CHECK-DAG:     %[[v9:.+]] = executor.getoffset[0, 3] : () -> i64, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//   CHECK-DAG:     %[[v10:.+]] = executor.addi %[[arg1]], %[[v9]] : i64
//   CHECK-DAG:     %[[v11:.+]] = executor.load %[[arg0]] + %[[v10]] : (!executor.ptr<host>, i64) -> i64
//   CHECK-DAG:     %[[v12:.+]] = executor.getoffset[0, 4] : () -> i64, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//   CHECK-DAG:     %[[v13:.+]] = executor.addi %[[arg1]], %[[v12]] : i64
//   CHECK-DAG:     %[[v14:.+]] = executor.load %[[arg0]] + %[[v13]] : (!executor.ptr<host>, i64) -> i64
//   CHECK-DAG:     %[[v15:.+]] = executor.table.create(%[[v2]], %[[v5]], %[[v8]], %[[v11]], %[[v14]] : !executor.ptr<host>, !executor.ptr<host>, i64, i64, i64) : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//   CHECK-DAG:     return %[[v15]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>

// -----

!agg_type = !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>

func.func @agg_load_complex(%arg0: !executor.ptr<host>, %arg1: i64) -> !agg_type {
  %0 = executor.load %arg0 + %arg1 : (!executor.ptr<host>, i64) -> !agg_type
  return %0 : !agg_type
}

// CHECK-LABEL: func.func @agg_load_complex
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: i64) -> !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>> {
//   CHECK-DAG:     %[[v0:.+]] = executor.getoffset[0, 0] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//   CHECK-DAG:     %[[v1:.+]] = executor.addi %[[arg1]], %[[v0]] : i64
//   CHECK-DAG:     %[[v2:.+]] = executor.load %[[arg0]] + %[[v1]] : (!executor.ptr<host>, i64) -> !executor.ptr<host>
//   CHECK-DAG:     %[[v3:.+]] = executor.getoffset[0, 1] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//   CHECK-DAG:     %[[v4:.+]] = executor.addi %[[arg1]], %[[v3]] : i64
//   CHECK-DAG:     %[[v5:.+]] = executor.load %[[arg0]] + %[[v4]] : (!executor.ptr<host>, i64) -> i64
//   CHECK-DAG:     %[[v6:.+]] = executor.getoffset[0, 2] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//   CHECK-DAG:     %[[v7:.+]] = executor.addi %[[arg1]], %[[v6]] : i64
//   CHECK-DAG:     %[[v8:.+]] = executor.load %[[arg0]] + %[[v7]] : (!executor.ptr<host>, i64) -> i32
//   CHECK-DAG:     %[[v9:.+]] = executor.getoffset[0, 3] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//   CHECK-DAG:     %[[v10:.+]] = executor.addi %[[arg1]], %[[v9]] : i64
//   CHECK-DAG:     %[[v11:.+]] = executor.load %[[arg0]] + %[[v10]] : (!executor.ptr<host>, i64) -> f32
//   CHECK-DAG:     %[[v12:.+]] = executor.getoffset[0, 4] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//   CHECK-DAG:     %[[v13:.+]] = executor.addi %[[arg1]], %[[v12]] : i64
//   CHECK-DAG:     %[[v14:.+]] = executor.load %[[arg0]] + %[[v13]] : (!executor.ptr<host>, i64) -> f64
//   CHECK-DAG:     %[[v15:.+]] = executor.getoffset[0, 5] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//   CHECK-DAG:     %[[v16:.+]] = executor.addi %[[arg1]], %[[v15]] : i64
//   CHECK-DAG:     %[[v17:.+]] = executor.load %[[arg0]] + %[[v16]] : (!executor.ptr<host>, i64) -> i8
//   CHECK-DAG:     %[[v18:.+]] = executor.getoffset[0, 6] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//   CHECK-DAG:     %[[v19:.+]] = executor.addi %[[arg1]], %[[v18]] : i64
//   CHECK-DAG:     %[[v20:.+]] = executor.getoffset[0, 0] : () -> i64, !executor.table<f32, f32>
//   CHECK-DAG:     %[[v21:.+]] = executor.addi %[[v19]], %[[v20]] : i64
//   CHECK-DAG:     %[[v22:.+]] = executor.load %[[arg0]] + %[[v21]] : (!executor.ptr<host>, i64) -> f32
//   CHECK-DAG:     %[[v23:.+]] = executor.getoffset[0, 1] : () -> i64, !executor.table<f32, f32>
//   CHECK-DAG:     %[[v24:.+]] = executor.addi %[[v19]], %[[v23]] : i64
//   CHECK-DAG:     %[[v25:.+]] = executor.load %[[arg0]] + %[[v24]] : (!executor.ptr<host>, i64) -> f32
//   CHECK-DAG:     %[[v26:.+]] = executor.table.create(%[[v22]], %[[v25]] : f32, f32) : <f32, f32>
//   CHECK-DAG:     %[[v27:.+]] = executor.table.create(%[[v2]], %[[v5]], %[[v8]], %[[v11]], %[[v14]], %[[v17]], %[[v26]] : !executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>) : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//   CHECK-DAG:     return %[[v27]] : !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>

// -----

!agg_type = !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>

func.func @agg_store_complex(%arg0: !executor.ptr<host>, %arg1: i64, %arg2: !agg_type) {
  executor.store %arg2 to %arg0 + %arg1 : !agg_type, !executor.ptr<host>, i64
  return
}

// CHECK-LABEL: func.func @agg_store_complex
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: i64, %[[arg2:.+]]: !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>) {
//       CHECK-DAG:     %[[v0:.+]] = executor.table.get %[[arg2]][0] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v1:.+]] = executor.getoffset[0, 0] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v2:.+]] = executor.addi %[[arg1]], %[[v1]] : i64
//       CHECK-DAG:     executor.store %[[v0]] to %[[arg0]] + %[[v2]] : !executor.ptr<host>, !executor.ptr<host>, i64
//       CHECK-DAG:     %[[v3:.+]] = executor.table.get %[[arg2]][1] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v4:.+]] = executor.getoffset[0, 1] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v5:.+]] = executor.addi %[[arg1]], %[[v4]] : i64
//       CHECK-DAG:     executor.store %[[v3]] to %[[arg0]] + %[[v5]] : i64, !executor.ptr<host>, i64
//       CHECK-DAG:     %[[v6:.+]] = executor.table.get %[[arg2]][2] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v7:.+]] = executor.getoffset[0, 2] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v8:.+]] = executor.addi %[[arg1]], %[[v7]] : i64
//       CHECK-DAG:     executor.store %[[v6]] to %[[arg0]] + %[[v8]] : i32, !executor.ptr<host>, i64
//       CHECK-DAG:     %[[v9:.+]] = executor.table.get %[[arg2]][3] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v10:.+]] = executor.getoffset[0, 3] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v11:.+]] = executor.addi %[[arg1]], %[[v10]] : i64
//       CHECK-DAG:     executor.store %[[v9]] to %[[arg0]] + %[[v11]] : f32, !executor.ptr<host>, i64
//       CHECK-DAG:     %[[v12:.+]] = executor.table.get %[[arg2]][4] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v13:.+]] = executor.getoffset[0, 4] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v14:.+]] = executor.addi %[[arg1]], %[[v13]] : i64
//       CHECK-DAG:     executor.store %[[v12]] to %[[arg0]] + %[[v14]] : f64, !executor.ptr<host>, i64
//       CHECK-DAG:     %[[v15:.+]] = executor.table.get %[[arg2]][5] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v16:.+]] = executor.getoffset[0, 5] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v17:.+]] = executor.addi %[[arg1]], %[[v16]] : i64
//       CHECK-DAG:     executor.store %[[v15]] to %[[arg0]] + %[[v17]] : i8, !executor.ptr<host>, i64
//       CHECK-DAG:     %[[v18:.+]] = executor.table.get %[[arg2]][6] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v19:.+]] = executor.getoffset[0, 6] : () -> i64, !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//       CHECK-DAG:     %[[v20:.+]] = executor.addi %[[arg1]], %[[v19]] : i64
//       CHECK-DAG:     %[[v21:.+]] = executor.table.get %[[v18]][0] : <f32, f32>
//       CHECK-DAG:     %[[v22:.+]] = executor.getoffset[0, 0] : () -> i64, !executor.table<f32, f32>
//       CHECK-DAG:     %[[v23:.+]] = executor.addi %[[v20]], %[[v22]] : i64
//       CHECK-DAG:     executor.store %[[v21]] to %[[arg0]] + %[[v23]] : f32, !executor.ptr<host>, i64
//       CHECK-DAG:     %[[v24:.+]] = executor.table.get %[[v18]][1] : <f32, f32>
//       CHECK-DAG:     %[[v25:.+]] = executor.getoffset[0, 1] : () -> i64, !executor.table<f32, f32>
//       CHECK-DAG:     %[[v26:.+]] = executor.addi %[[v20]], %[[v25]] : i64
//       CHECK-DAG:     executor.store %[[v24]] to %[[arg0]] + %[[v26]] : f32, !executor.ptr<host>, i64
//       CHECK-DAG:     return

// DECOMP-LABEL: func.func @agg_store_complex
//  DECOMP-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: i64, %[[arg2:.+]]: !executor.table<!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>) {
//  DECOMP-DAG:     %[[c4_i64:.+]] = executor.constant 4 : i64
//  DECOMP-DAG:     %[[c36_i64:.+]] = executor.constant 36 : i64
//  DECOMP-DAG:     %[[c32_i64:.+]] = executor.constant 32 : i64
//  DECOMP-DAG:     %[[c24_i64:.+]] = executor.constant 24 : i64
//  DECOMP-DAG:     %[[c20_i64:.+]] = executor.constant 20 : i64
//  DECOMP-DAG:     %[[c16_i64:.+]] = executor.constant 16 : i64
//  DECOMP-DAG:     %[[c8_i64:.+]] = executor.constant 8 : i64
//  DECOMP-DAG:     %[[v0:.+]] = executor.table.get %[[arg2]][0] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//  DECOMP-DAG:     executor.store %[[v0]] to %[[arg0]] + %[[arg1]] : !executor.ptr<host>, !executor.ptr<host>, i64
//  DECOMP-DAG:     %[[v1:.+]] = executor.table.get %[[arg2]][1] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//  DECOMP-DAG:     %[[v2:.+]] = executor.addi %[[arg1]], %[[c8_i64]] : i64
//  DECOMP-DAG:     executor.store %[[v1]] to %[[arg0]] + %[[v2]] : i64, !executor.ptr<host>, i64
//  DECOMP-DAG:     %[[v3:.+]] = executor.table.get %[[arg2]][2] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//  DECOMP-DAG:     %[[v4:.+]] = executor.addi %[[arg1]], %[[c16_i64]] : i64
//  DECOMP-DAG:     executor.store %[[v3]] to %[[arg0]] + %[[v4]] : i32, !executor.ptr<host>, i64
//  DECOMP-DAG:     %[[v5:.+]] = executor.table.get %[[arg2]][3] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//  DECOMP-DAG:     %[[v6:.+]] = executor.addi %[[arg1]], %[[c20_i64]] : i64
//  DECOMP-DAG:     executor.store %[[v5]] to %[[arg0]] + %[[v6]] : f32, !executor.ptr<host>, i64
//  DECOMP-DAG:     %[[v7:.+]] = executor.table.get %[[arg2]][4] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//  DECOMP-DAG:     %[[v8:.+]] = executor.addi %[[arg1]], %[[c24_i64]] : i64
//  DECOMP-DAG:     executor.store %[[v7]] to %[[arg0]] + %[[v8]] : f64, !executor.ptr<host>, i64
//  DECOMP-DAG:     %[[v9:.+]] = executor.table.get %[[arg2]][5] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//  DECOMP-DAG:     %[[v10:.+]] = executor.addi %[[arg1]], %[[c32_i64]] : i64
//  DECOMP-DAG:     executor.store %[[v9]] to %[[arg0]] + %[[v10]] : i8, !executor.ptr<host>, i64
//  DECOMP-DAG:     %[[v11:.+]] = executor.table.get %[[arg2]][6] : <!executor.ptr<host>, i64, i32, f32, f64, i8, !executor.table<f32, f32>>
//  DECOMP-DAG:     %[[v12:.+]] = executor.addi %[[arg1]], %[[c36_i64]] : i64
//  DECOMP-DAG:     %[[v13:.+]] = executor.table.get %[[v11]][0] : <f32, f32>
//  DECOMP-DAG:     executor.store %[[v13]] to %[[arg0]] + %[[v12]] : f32, !executor.ptr<host>, i64
//  DECOMP-DAG:     %[[v14:.+]] = executor.table.get %[[v11]][1] : <f32, f32>
//  DECOMP-DAG:     %[[v15:.+]] = executor.addi %[[v12]], %[[c4_i64]] : i64
//  DECOMP-DAG:     executor.store %[[v14]] to %[[arg0]] + %[[v15]] : f32, !executor.ptr<host>, i64
//  DECOMP-DAG:     return
