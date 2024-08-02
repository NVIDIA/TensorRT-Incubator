// RUN: mlir-tensorrt-opt %s -tensorrt-stablehlo-input-preprocessing -stablehlo-aggressive-simplification -split-input-file | FileCheck %s

func.func @trivial_right_shift(%arg0: tensor<i32>) -> tensor<i32> {
  %c32 = stablehlo.constant dense<32> : tensor<i32>
  %0 = stablehlo.shift_right_logical %arg0, %c32 : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: @trivial_right_shift
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>) -> tensor<i32> {
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<0> : tensor<i32>
//       CHECK:     return %[[v0]] : tensor<i32>

// -----

func.func @nontrivial_right_shift(%arg0: tensor<i32>) -> tensor<i32> {
  %c16 = stablehlo.constant dense<16> : tensor<i32>
  %0 = stablehlo.shift_right_logical %arg0, %c16 : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: @nontrivial_right_shift
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>) -> tensor<i32> {
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<16> : tensor<i32>
//       CHECK:     %[[v1:.+]] = stablehlo.shift_right_logical %[[arg0]], %[[v0]] : tensor<i32>
//       CHECK:     return %[[v1]] : tensor<i32>
