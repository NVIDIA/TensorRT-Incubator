// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-raise-qdq | FileCheck %s

func.func @quantize_to_i8_static(%arg0: tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8> {
  %cst = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense_resource<__elided__> : tensor<3xf32>
  %cst_1 = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %cst_0, dims = [1] : (tensor<3xf32>) -> tensor<2x3x300x300xf32>
  %1 = stablehlo.divide %arg0, %0 : tensor<2x3x300x300xf32>
  %2 = stablehlo.round_nearest_even %1 : tensor<2x3x300x300xf32>
  %3 = stablehlo.clamp %cst_1, %2, %cst : (tensor<f32>, tensor<2x3x300x300xf32>, tensor<f32>) -> tensor<2x3x300x300xf32>
  %4 = stablehlo.convert %3 : (tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
  return %4 : tensor<2x3x300x300xi8>
}


//  CHECK-LABEL: quantize_to_i8_static
//   CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x300x300xf32>)
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.composite "tensorrt.pc_q" %[[arg0]] {composite_attributes = {axis = 1 : i32, is_pointwise, scale = dense_resource<__elided__> : tensor<3xf32>}, decomposition = @pc_q} : (tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
//   CHECK-NEXT: return %[[v0]] : tensor<2x3x300x300xi8>
//  CHECK-LABEL: private @pc_q
//   CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
//   CHECK-SAME: attributes {plan.decomposition}
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
//   CHECK-NEXT: %[[v1:.+]] = stablehlo.constant dense<1.270000e+02> : tensor<f32>
//   CHECK-NEXT: %[[v2:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<3xf32>
//   CHECK-NEXT: %[[v3:.+]] = stablehlo.broadcast_in_dim %[[v2]], dims = [1] : (tensor<3xf32>) -> tensor<2x3x300x300xf32>
//   CHECK-NEXT: %[[v4:.+]] = stablehlo.divide %[[arg0]], %[[v3]] : tensor<2x3x300x300xf32>
//   CHECK-NEXT: %[[v5:.+]] = stablehlo.round_nearest_even %[[v4]] : tensor<2x3x300x300xf32>
//   CHECK-NEXT: %[[v6:.+]] = stablehlo.clamp %[[v0]], %[[v5]], %[[v1]] : (tensor<f32>, tensor<2x3x300x300xf32>, tensor<f32>) -> tensor<2x3x300x300xf32>
//   CHECK-NEXT: %[[v7:.+]] = stablehlo.convert %[[v6]] : (tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
//   CHECK-NEXT: return %[[v7]] : tensor<2x3x300x300xi8>

// -----

func.func @quantize_to_i8_dynamic(%arg0: tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8> {
  %c = stablehlo.constant dense<3> : tensor<1xi32>
  %cst = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense_resource<__elided__> : tensor<3xf32>
  %cst_1 = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x3x?x?xf32>) -> tensor<i32>
  %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
  %2 = stablehlo.get_dimension_size %arg0, dim = 2 : (tensor<?x3x?x?xf32>) -> tensor<i32>
  %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
  %4 = stablehlo.get_dimension_size %arg0, dim = 3 : (tensor<?x3x?x?xf32>) -> tensor<i32>
  %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
  %6 = stablehlo.concatenate %1, %c, %3, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %7 = stablehlo.dynamic_broadcast_in_dim %cst_0, %6, dims = [1] : (tensor<3xf32>, tensor<4xi32>) -> tensor<?x3x?x?xf32>
  %8 = stablehlo.divide %arg0, %7 : tensor<?x3x?x?xf32>
  %9 = stablehlo.round_nearest_even %8 : tensor<?x3x?x?xf32>
  %10 = stablehlo.clamp %cst_1, %9, %cst : (tensor<f32>, tensor<?x3x?x?xf32>, tensor<f32>) -> tensor<?x3x?x?xf32>
  %11 = stablehlo.convert %10 : (tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8>
  return %11 : tensor<?x3x?x?xi8>
}

//  CHECK-LABEL: quantize_to_i8_dynamic
//   CHECK-SAME: (%[[arg0:.+]]: tensor<?x3x?x?xf32>)
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.composite "tensorrt.pc_q" %[[arg0]] {composite_attributes = {axis = 1 : i32, is_pointwise, scale = dense_resource<__elided__> : tensor<3xf32>}, decomposition = @pc_q} : (tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8>
//   CHECK-NEXT: return %[[v0]] : tensor<?x3x?x?xi8>
//  CHECK-LABEL: private @pc_q
//   CHECK-SAME: attributes {plan.decomposition}

// -----

func.func @quantize_to_i8_eager() -> tensor<258x256xi8> {
  %cst = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<258x256xf32>
  %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<256xf32>
  %cst_2 = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %cst_1, dims = [1] : (tensor<256xf32>) -> tensor<258x256xf32>
  %1 = stablehlo.divide %cst_0, %0 : tensor<258x256xf32>
  %2 = stablehlo.round_nearest_even %1 : tensor<258x256xf32>
  %3 = stablehlo.clamp %cst_2, %2, %cst : (tensor<f32>, tensor<258x256xf32>, tensor<f32>) -> tensor<258x256xf32>
  %4 = stablehlo.convert %3 : (tensor<258x256xf32>) -> tensor<258x256xi8>
  return %4 : tensor<258x256xi8>
}

//  CHECK-LABEL: quantize_to_i8_eager
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<258x256xf32>
//   CHECK-NEXT: %[[v1:.+]] = stablehlo.composite "tensorrt.pc_q" %[[v0]] {composite_attributes = {axis = 1 : i32, is_pointwise, scale = dense_resource<__elided__> : tensor<256xf32>}, decomposition = @pc_q} : (tensor<258x256xf32>) -> tensor<258x256xi8>
//   CHECK-NEXT: return %[[v1]] : tensor<258x256xi8>
//  CHECK-LABEL: private @pc_q
//   CHECK-SAME: attributes {plan.decomposition}

// -----

func.func @quantize_to_fp8() -> tensor<4x4xf8E4M3FN> {
  %cst = stablehlo.constant dense<4.480000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense<-4.480000e+02> : tensor<f32>
  %cst_1 = stablehlo.constant dense<[5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01]> : tensor<4xf32>
  %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<4x4xf32>
  %0 = stablehlo.broadcast_in_dim %cst_1, dims = [0] : (tensor<4xf32>) -> tensor<4x4xf32>
  %1 = stablehlo.divide %cst_2, %0 : tensor<4x4xf32>
  %2 = stablehlo.round_nearest_even %1 : tensor<4x4xf32>
  %3 = stablehlo.clamp %cst_0, %2, %cst : (tensor<f32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  %4 = stablehlo.convert %3 : (tensor<4x4xf32>) -> tensor<4x4xf8E4M3FN>
  return %4 : tensor<4x4xf8E4M3FN>
}

//  CHECK-LABEL: @quantize_to_fp8
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<4x4xf32>
//   CHECK-NEXT: %[[v1:.+]] = stablehlo.composite "tensorrt.pc_q" %[[v0]] {composite_attributes = {axis = 0 : i32, is_pointwise, scale = dense<[5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01]> : tensor<4xf32>}, decomposition = @pc_q} : (tensor<4x4xf32>) -> tensor<4x4xf8E4M3FN>
//   CHECK-NEXT: return %[[v1]] : tensor<4x4xf8E4M3FN>
//  CHECK-LABEL: private @pc_q
//   CHECK-SAME: attributes {plan.decomposition}

