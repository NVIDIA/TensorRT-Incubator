// RUN: kernel-opt %s -split-input-file -kernel-expand-ops | FileCheck %s

func.func @inline_combiner_op(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> (f32, f32) {
  %0, %1 = kernel.combiner (%arg0, %arg1, %arg2, %arg3) : f32, f32, f32, f32 {
  ^bb0(%a: f32, %c: f32, %b: f32, %d: f32):
    %max = arith.maximumf %a, %b : f32
    %0 = arith.subf %a, %max : f32
    %1 = arith.subf %b, %max : f32
    %2 = math.exp %0 : f32
    %3 = math.exp %1 : f32
    %4 = arith.mulf %2, %c : f32
    %5 = arith.mulf %3, %d : f32
    %6 = arith.addf %4, %5 : f32
    kernel.yield %max, %6 : f32, f32
  }
  return %0, %1 : f32, f32
}

// CHECK-LABEL: func.func @inline_combiner_op
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32, %[[arg2:.+]]: f32, %[[arg3:.+]]: f32)
//   CHECK-DAG:     %[[v0:.+]] = arith.maximumf %[[arg0]], %[[arg2]] : f32
//   CHECK-DAG:     %[[v1:.+]] = arith.subf %[[arg0]], %[[v0]] : f32
//   CHECK-DAG:     %[[v2:.+]] = arith.subf %[[arg2]], %[[v0]] : f32
//   CHECK-DAG:     %[[v3:.+]] = math.exp %[[v1]] : f32
//   CHECK-DAG:     %[[v4:.+]] = math.exp %[[v2]] : f32
//   CHECK-DAG:     %[[v5:.+]] = arith.mulf %[[v3]], %[[arg1]] : f32
//   CHECK-DAG:     %[[v6:.+]] = arith.mulf %[[v4]], %[[arg3]] : f32
//   CHECK-DAG:     %[[v7:.+]] = arith.addf %[[v5]], %[[v6]] : f32
//   CHECK-DAG:     return %[[v0]], %[[v7]] : f32, f32
