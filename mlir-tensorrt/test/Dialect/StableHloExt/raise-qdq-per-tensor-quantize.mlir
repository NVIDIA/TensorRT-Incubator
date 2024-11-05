// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-raise-qdq | FileCheck %s

func.func @quantize_to_i8_static(%arg0: tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8> {
  %cst = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense<8.000000e-01> : tensor<f32>
  %cst_1 = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x3x300x300xf32>
  %1 = stablehlo.divide %arg0, %0 : tensor<2x3x300x300xf32>
  %2 = stablehlo.round_nearest_even %1 : tensor<2x3x300x300xf32>
  %3 = stablehlo.clamp %cst_1, %2, %cst : (tensor<f32>, tensor<2x3x300x300xf32>, tensor<f32>) -> tensor<2x3x300x300xf32>
  %4 = stablehlo.convert %3 : (tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
  return %4 : tensor<2x3x300x300xi8>
}

//  CHECK-LABEL: quantize_to_i8_static
//   CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x300x300xf32>)
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.composite "tensorrt.pt_q" %[[arg0]] {composite_attributes = {axis = -1 : i32, scale = dense<8.000000e-01> : tensor<f32>}, decomposition = @pt_q} : (tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
//   CHECK-NEXT: return %[[v0]] : tensor<2x3x300x300xi8>
//  CHECK-LABEL: private @pt_q
//   CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
//   CHECK-SAME: attributes {plan.decomposition}
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
//   CHECK-NEXT: %[[v1:.+]] = stablehlo.constant dense<1.270000e+02> : tensor<f32>
//   CHECK-NEXT: %[[v2:.+]] = stablehlo.constant dense<8.000000e-01> : tensor<f32>
//   CHECK-NEXT: %[[v3:.+]] = stablehlo.broadcast_in_dim %[[v2]], dims = [] : (tensor<f32>) -> tensor<2x3x300x300xf32>
//   CHECK-NEXT: %[[v4:.+]] = stablehlo.divide %[[arg0]], %[[v3]] : tensor<2x3x300x300xf32>
//   CHECK-NEXT: %[[v5:.+]] = stablehlo.round_nearest_even %[[v4]] : tensor<2x3x300x300xf32>
//   CHECK-NEXT: %[[v6:.+]] = stablehlo.clamp %[[v0]], %[[v5]], %[[v1]] : (tensor<f32>, tensor<2x3x300x300xf32>, tensor<f32>) -> tensor<2x3x300x300xf32>
//   CHECK-NEXT: %[[v7:.+]] = stablehlo.convert %[[v6]] : (tensor<2x3x300x300xf32>) -> tensor<2x3x300x300xi8>
//   CHECK-NEXT: return %[[v7]] : tensor<2x3x300x300xi8>

// -----

func.func @quantize_to_i8_dynamic(%arg0: tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8> {
  %c = stablehlo.constant dense<3> : tensor<1xi32>
  %cst = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense<8.000000e-01> : tensor<f32>
  %cst_1 = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x3x?x?xf32>) -> tensor<i32>
  %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
  %2 = stablehlo.get_dimension_size %arg0, dim = 2 : (tensor<?x3x?x?xf32>) -> tensor<i32>
  %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
  %4 = stablehlo.get_dimension_size %arg0, dim = 3 : (tensor<?x3x?x?xf32>) -> tensor<i32>
  %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
  %6 = stablehlo.concatenate %1, %c, %3, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %7 = stablehlo.dynamic_broadcast_in_dim %cst_0, %6, dims = [] : (tensor<f32>, tensor<4xi32>) -> tensor<?x3x?x?xf32>
  %8 = stablehlo.divide %arg0, %7 : tensor<?x3x?x?xf32>
  %9 = stablehlo.round_nearest_even %8 : tensor<?x3x?x?xf32>
  %10 = stablehlo.clamp %cst_1, %9, %cst : (tensor<f32>, tensor<?x3x?x?xf32>, tensor<f32>) -> tensor<?x3x?x?xf32>
  %11 = stablehlo.convert %10 : (tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8>
  return %11 : tensor<?x3x?x?xi8>
}

//  CHECK-LABEL: quantize_to_i8_dynamic
//   CHECK-SAME: (%[[arg0:.+]]: tensor<?x3x?x?xf32>)
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.composite "tensorrt.pt_q" %[[arg0]] {composite_attributes = {axis = -1 : i32, scale = dense<8.000000e-01> : tensor<f32>}, decomposition = @pt_q} : (tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8>
//   CHECK-NEXT: return %[[v0]] : tensor<?x3x?x?xi8>
//  CHECK-LABEL: private @pt_q
//   CHECK-SAME: attributes {plan.decomposition}

// -----

func.func @quantize_to_i8_eager() -> tensor<4x4xi8> {
  %cst = stablehlo.constant dense<1.270000e+02> : tensor<f32>
  %cst_0 = stablehlo.constant dense<-1.280000e+02> : tensor<f32>
  %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<4x4xf32>
  %cst_2 = stablehlo.constant dense<8.000000e-01> : tensor<4x4xf32>
  %0 = stablehlo.divide %cst_1, %cst_2 : tensor<4x4xf32>
  %1 = stablehlo.round_nearest_even %0 : tensor<4x4xf32>
  %2 = stablehlo.clamp %cst_0, %1, %cst : (tensor<f32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  %3 = stablehlo.convert %2 : (tensor<4x4xf32>) -> tensor<4x4xi8>
  return %3 : tensor<4x4xi8>
}

// CHECK-LABEL: quantize_to_i8_eager
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<4x4xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.composite "tensorrt.pt_q" %[[v0]] {composite_attributes = {axis = -1 : i32, scale = dense<8.000000e-01> : tensor<4x4xf32>}, decomposition = @pt_q} : (tensor<4x4xf32>) -> tensor<4x4xi8>
//  CHECK-NEXT: return %[[v1]] : tensor<4x4xi8>
//  CHECK-LABEL: private @pt_q
//   CHECK-SAME: attributes {plan.decomposition}

// -----

func.func @quantize_bf16_to_fp8() -> tensor<2xf8E4M3FN> {
    %cst = stablehlo.constant dense<4.480000e+02> : tensor<f32>
    %cst_0 = stablehlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xbf16>
    %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %cst_2 = stablehlo.constant dense<-4.480000e+02> : tensor<f32>
    %0 = stablehlo.convert %cst_2 : (tensor<f32>) -> tensor<bf16>
    %1 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<bf16>) -> tensor<2xbf16>
    %2 = stablehlo.divide %cst_0, %1 : tensor<2xbf16>
    %3 = stablehlo.round_nearest_even %2 : tensor<2xbf16>
    %4 = stablehlo.convert %cst : (tensor<f32>) -> tensor<bf16>
    %5 = stablehlo.clamp %0, %3, %4 : (tensor<bf16>, tensor<2xbf16>, tensor<bf16>) -> tensor<2xbf16>
    %6 = stablehlo.convert %5 : (tensor<2xbf16>) -> tensor<2xf8E4M3FN>
    return %6 : tensor<2xf8E4M3FN>
}

// CHECK-LABEL: quantize_bf16_to_fp8
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xbf16>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.composite "tensorrt.pt_q" %[[v0]] {composite_attributes = {axis = -1 : i32, scale = dense<5.000000e-01> : tensor<bf16>}, decomposition = @pt_q} : (tensor<2xbf16>) -> tensor<2xf8E4M3FN>
//  CHECK-NEXT: return %[[v1]] : tensor<2xf8E4M3FN>
//  CHECK-LABEL: private @pt_q
//   CHECK-SAME: attributes {plan.decomposition}
