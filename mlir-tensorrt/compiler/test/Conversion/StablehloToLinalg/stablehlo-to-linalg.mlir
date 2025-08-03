// RUN: mlir-tensorrt-opt %s -convert-stablehlo-to-linalg -split-input-file | FileCheck %s

func.func @reverse(%input: tensor<2048xf32>) -> tensor<2048xf32> {
  %result = "stablehlo.reverse"(%input) {
    dimensions = array<i64: 0>
  } : (tensor<2048xf32>) -> tensor<2048xf32>
  func.return %result : tensor<2048xf32>
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0) -> (d0)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<()[s0] -> (-s0 + 2047)>
// CHECK-LABEL: func.func @reverse
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2048xf32>) -> tensor<2048xf32>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<2048xf32>
//       CHECK:     %[[v1:.+]] = linalg.generic
// CHECK-SAME:        indexing_maps = [#[[$map]]],
// CHECK-SAME:        iterator_types = ["parallel"]
// CHECK-SAME:        outs(%[[v0]] : tensor<2048xf32>
//       CHECK:     ^bb0(%[[out:.+]]: f32):
//   CHECK-DAG:       %[[v2:.+]] = linalg.index 0 : index
//   CHECK-DAG:       %[[v3:.+]] = affine.apply #[[$map1]]()[%[[v2]]]
//   CHECK-DAG:       %[[extracted:.+]] = tensor.extract %[[arg0]][%[[v3]]]
//   CHECK-DAG:       linalg.yield %[[extracted]] : f32
//       CHECK:     return %[[v1]] : tensor<2048xf32>

// -----

// CHECK-LABEL: func.func @erf(
func.func @erf(%input: tensor<2048xf32>) -> tensor<2048xf32> {
  // CHECK: linalg.generic
  // CHECK:   math.erf %{{.*}} : f32
  %result = "chlo.erf"(%input) : (tensor<2048xf32>) -> tensor<2048xf32>
  return %result : tensor<2048xf32>
}

// -----

// CHECK-LABEL: func.func @erfc(
func.func @erfc(%input: tensor<2048xf32>) -> tensor<2048xf32> {
  // CHECK: linalg.generic
  // CHECK:   math.erfc %{{.*}} : f32
  %result = "chlo.erfc"(%input) : (tensor<2048xf32>) -> tensor<2048xf32>
  return %result : tensor<2048xf32>
}

// -----

// CHECK-LABEL: func.func @tanh(
func.func @tanh(%input: tensor<2048xf32>) -> tensor<2048xf32> {
  // CHECK: linalg.generic
  // CHECK:   math.tanh %{{.*}} : f32
  %result = "stablehlo.tanh"(%input) : (tensor<2048xf32>) -> tensor<2048xf32>
  return %result : tensor<2048xf32>
}

// -----

// CHECK-LABEL: func.func @tan(
func.func @tan(%input: tensor<2048xf32>) -> tensor<2048xf32> {
  // CHECK: linalg.generic
  // CHECK:   math.tan %{{.*}} : f32
  %result = "stablehlo.tan"(%input) : (tensor<2048xf32>) -> tensor<2048xf32>
  return %result : tensor<2048xf32>
}

// -----

// CHECK-LABEL: func.func @dot_general_algorithm_attr(
func.func @dot_general_algorithm_attr(%arg0: tensor<3x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<3x3xf32> {
  // CHECK: linalg.matmul
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<3x4xf32>, tensor<4x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
