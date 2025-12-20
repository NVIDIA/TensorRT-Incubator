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

// CHECK-LABEL: func.func @dot_general_algorithm_attr
func.func @dot_general_algorithm_attr(%arg0: tensor<3x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<3x3xf32> {
  // CHECK: linalg.matmul
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<3x4xf32>, tensor<4x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

// CHECK-LABEL: func.func @dot_algorithm_attr_reduce_precision
func.func @dot_algorithm_attr_reduce_precision(%arg0: tensor<3x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<3x3xf32> {
  // CHECK: linalg.matmul
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 3, allow_imprecise_accumulation = false> : (tensor<3x4xf32>, tensor<4x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

// CHECK-LABEL: func.func @dot_algorithm_attr_reduce_precision2
func.func @dot_algorithm_attr_reduce_precision2(%arg0: tensor<3x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<3x3xf32> {
  // CHECK: linalg.matmul
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], algorithm = <lhs_precision_type = bf16, rhs_precision_type = bf16, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 3, allow_imprecise_accumulation = false> : (tensor<3x4xf32>, tensor<4x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func.func @test_size1_reverse(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %0 = stablehlo.reverse %arg0, dims = [0] : tensor<1xf32>
  return %0 : tensor<1xf32>
}

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @test_size1_reverse(
// CHECK-DAG:   %[[v1:.*]] = tensor.empty() : tensor<1xf32>
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK:   %[[v2:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%[[v1]] : tensor<1xf32>) {
// CHECK:     ^bb0(%[[v3:.*]]: f32):
// CHECK:       %[[v6:.*]] = tensor.extract %arg0[%[[c0]]] : tensor<1xf32>
// CHECK:       linalg.yield %[[v6]] : f32
// CHECK:   } -> tensor<1xf32>
// CHECK:   return %[[v2]] : tensor<1xf32>

// -----

// CHECK-LABEL: func.func @get_dimension_size
func.func @get_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[dim:.+]] = tensor.dim %arg0, %[[c0]]
  // CHECK-DAG: %[[dim1:.+]] = tensor.dim %arg0, %[[c1]]
  // CHECK-DAG: %[[cast:.+]] = arith.index_castui %[[dim]]
  // CHECK-DAG: %[[cast1:.+]] = arith.index_castui %[[dim1]]
  // CHECK-DAG: %[[result:.+]] = tensor.from_elements %[[cast]]
  // CHECK-DAG: %[[result1:.+]] = tensor.from_elements %[[cast1]]
  // CHECK: return %[[result]], %[[result1]]
  %0 = "stablehlo.get_dimension_size"(%arg0) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %1 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  return %0, %1 : tensor<i32>, tensor<i32>
}
