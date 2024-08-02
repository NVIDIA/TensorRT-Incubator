// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-raise-qdq | FileCheck %s


func.func @dequantize_to_f32_static(%arg0: tensor<2x3x300x300xi8>) -> tensor<2x3x300x300xf32> {
  %cst = stablehlo.constant dense<8.000000e-01> : tensor<f32>
  %0 = stablehlo.convert %arg0 : (tensor<2x3x300x300xi8>) -> tensor<2x3x300x300xf32>
  %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3x300x300xf32>
  %2 = stablehlo.multiply %0, %1 : tensor<2x3x300x300xf32>
  return %2 : tensor<2x3x300x300xf32>
}

//  CHECK-LABEL: dequantize_to_f32_static
//   CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x300x300xi8>)
//   CHECK-NEXT: %[[v0:.+]] =  stablehlo.composite "tensorrt.pt_dq" %[[arg0]] {composite_attributes = {axis = -1 : i32, is_pointwise, scale = dense<8.000000e-01> : tensor<f32>}, decomposition = @pt_dq} : (tensor<2x3x300x300xi8>) -> tensor<2x3x300x300xf32>
//   CHECK-NEXT: return %[[v0]] : tensor<2x3x300x300xf32>
//  CHECK-LABEL: private @pt_dq
//   CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x300x300xi8>) -> tensor<2x3x300x300xf32>
//   CHECK-SAME: attributes {plan.decomposition}
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<8.000000e-01> : tensor<f32>
//   CHECK-NEXT: %[[v1:.+]] = stablehlo.broadcast_in_dim %[[v0]], dims = [] : (tensor<f32>) -> tensor<2x3x300x300xf32>
//   CHECK-NEXT: %[[v2:.+]] = stablehlo.convert %[[arg0]] : (tensor<2x3x300x300xi8>) -> tensor<2x3x300x300xf32>
//   CHECK-NEXT: %[[v3:.+]] = stablehlo.multiply %[[v2]], %[[v1]] : tensor<2x3x300x300xf32>
//   CHECK-NEXT: return %[[v3]] : tensor<2x3x300x300xf32>

// -----

func.func @dequantize_to_f32_dynamic(%arg0: tensor<?x?x?x?xi8>) -> tensor<?x?x?x?xf32> {
  %cst = stablehlo.constant dense<8.000000e-01> : tensor<f32>
  %0 = stablehlo.convert %arg0 : (tensor<?x?x?x?xi8>) -> tensor<?x?x?x?xf32>
  %1 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x?x?x?xi8>) -> tensor<i32>
  %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
  %3 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<?x?x?x?xi8>) -> tensor<i32>
  %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
  %5 = stablehlo.get_dimension_size %arg0, dim = 2 : (tensor<?x?x?x?xi8>) -> tensor<i32>
  %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
  %7 = stablehlo.get_dimension_size %arg0, dim = 3 : (tensor<?x?x?x?xi8>) -> tensor<i32>
  %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
  %9 = stablehlo.concatenate %2, %4, %6, %8, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %10 = stablehlo.dynamic_broadcast_in_dim %cst, %9, dims = [] : (tensor<f32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %11 = stablehlo.multiply %0, %10 : tensor<?x?x?x?xf32>
  return %11 : tensor<?x?x?x?xf32>
}

//  CHECK-LABEL: dequantize_to_f32_dynamic
//   CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?x?xi8>)
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.composite "tensorrt.pt_dq" %[[arg0]] {composite_attributes = {axis = -1 : i32, is_pointwise, scale = dense<8.000000e-01> : tensor<f32>}, decomposition = @pt_dq} : (tensor<?x?x?x?xi8>) -> tensor<?x?x?x?xf32>
//   CHECK-NEXT: return %[[v0]] : tensor<?x?x?x?xf32>
//  CHECK-LABEL: private @pt_dq
//   CHECK-SAME: attributes {plan.decomposition}

// -----

func.func @dequantize_to_f32_eager() -> tensor<258x256xf32> {
  %c = stablehlo.constant dense_resource<__elided__> : tensor<258x256xi8>
  %cst = stablehlo.constant dense<8.000000e-01> : tensor<f32>
  %0 = stablehlo.convert %c : (tensor<258x256xi8>) -> tensor<258x256xf32>
  %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<258x256xf32>
  %2 = stablehlo.multiply %0, %1 : tensor<258x256xf32>
  return %2 : tensor<258x256xf32>
}

// CHECK-LABEL: dequantize_to_f32_eager
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<258x256xi8>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.composite "tensorrt.pt_dq" %[[v0]] {composite_attributes = {axis = -1 : i32, is_pointwise, scale = dense<8.000000e-01> : tensor<f32>}, decomposition = @pt_dq} : (tensor<258x256xi8>) -> tensor<258x256xf32>
//  CHECK-NEXT: return %[[v1]] : tensor<258x256xf32>
// CHECK-LABEL: private @pt_dq
//  CHECK-SAME: attributes {plan.decomposition}

