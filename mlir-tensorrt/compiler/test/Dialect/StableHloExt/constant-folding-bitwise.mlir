// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-ext-constant-folding | FileCheck %s

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

func.func @dynamic_trivial_right_shift(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %c32 = stablehlo.constant dense<32> : tensor<1xi32>
  %d = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %d.1 = stablehlo.reshape %d : (tensor<i32>) -> tensor<1xi32>
  %c32_1 = stablehlo.dynamic_broadcast_in_dim %c32, %d.1, dims = [0] : (tensor<1xi32>, tensor<1xi32>) -> (tensor<?xi32>)
  %0 = stablehlo.shift_right_logical %arg0, %c32_1 : tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: @dynamic_trivial_right_shift
//       CHECK: stablehlo.shift_right_logical

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

// -----

func.func @jax_random_seed(%arg0: tensor<i32>) -> (tensor<2xi32>) {
  %0 = stablehlo.constant dense<32> : tensor<i32>
  %1 = stablehlo.shift_right_logical %arg0, %0 : tensor<i32>
  %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i32>
  %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
  %4 = stablehlo.constant dense<4294967295> : tensor<i32>
  %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i32>
  %6 = stablehlo.and %arg0, %5 : tensor<i32>
  %7 = stablehlo.convert %6 : (tensor<i32>) -> tensor<i32>
  %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
  %9 = "stablehlo.concatenate"(%3, %8) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  return %9 : tensor<2xi32>
}

// CHECK-LABEL: @jax_random_seed
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>) -> tensor<2xi32> {
//   CHECK-DAG: %[[v1:.+]] = stablehlo.constant dense<0> : tensor<1xi32>
//   CHECK-DAG: %[[v3:.+]] = stablehlo.reshape %[[arg0]] : (tensor<i32>) -> tensor<1xi32>
//   CHECK-DAG: %[[v4:.+]] = stablehlo.concatenate %[[v1]], %[[v3]], dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
//   CHECK-DAG: return %[[v4]] : tensor<2xi32>
