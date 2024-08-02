// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-raise-qdq | FileCheck %s

// Block quantization is weight only quantization thus input is always constant.

func.func @small_weights() -> tensor<4x4xi4> {
  %cst = stablehlo.constant dense<7.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<-8.000000e+00> : tensor<f32>
  %cst_1 = stablehlo.constant dense<[[5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01], [5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01]]> : tensor<2x4xf32>
  %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<4x4xf32>
  %0 = stablehlo.broadcast_in_dim %cst_1, dims = [1, 2] : (tensor<2x4xf32>) -> tensor<2x2x4xf32>
  %1 = stablehlo.reshape %0 : (tensor<2x2x4xf32>) -> tensor<4x4xf32>
  %2 = stablehlo.divide %cst_2, %1 : tensor<4x4xf32>
  %3 = stablehlo.round_nearest_even %2 : tensor<4x4xf32>
  %4 = stablehlo.clamp %cst_0, %3, %cst : (tensor<f32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  %5 = stablehlo.convert %4 : (tensor<4x4xf32>) -> tensor<4x4xi4>
  return %5 : tensor<4x4xi4>
}


//  CHECK-LABEL: small_weights
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<4x4xf32>
//   CHECK-NEXT: %[[v1:.+]] = stablehlo.composite "tensorrt.block_q" %[[v0]] {composite_attributes = {axis = -1 : i32, is_pointwise, scale = dense<{{\[}}[5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01], [5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01]{{\]}}> : tensor<2x4xf32>}, decomposition = @block_q} : (tensor<4x4xf32>) -> tensor<4x4xi4>
//   CHECK-NEXT: return %[[v1]] : tensor<4x4xi4>
//  CHECK-LABEL: private @block_q
//   CHECK-SAME: (%[[arg0:.+]]: tensor<4x4xf32>)
//   CHECK-SAME: attributes {plan.decomposition}
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<-8.000000e+00> : tensor<f32>
//   CHECK-NEXT: %[[v1:.+]] = stablehlo.constant dense<7.000000e+00> : tensor<f32>
//   CHECK-NEXT: %[[v2:.+]] = stablehlo.constant dense<{{\[}}[5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01], [5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01]{{\]}}> : tensor<2x4xf32>
//   CHECK-NEXT: %[[v3:.+]] = stablehlo.broadcast_in_dim %[[v2]], dims = [1, 2] : (tensor<2x4xf32>) -> tensor<2x2x4xf32>
//   CHECK-NEXT: %[[v4:.+]] = stablehlo.reshape %[[v3]] : (tensor<2x2x4xf32>) -> tensor<4x4xf32>
//   CHECK-NEXT: %[[v5:.+]] = stablehlo.divide %[[arg0]], %[[v4]] : tensor<4x4xf32>
//   CHECK-NEXT: %[[v6:.+]] = stablehlo.round_nearest_even %[[v5]] : tensor<4x4xf32>
//   CHECK-NEXT: %[[v7:.+]] = stablehlo.clamp %[[v0]], %[[v6]], %[[v1]] : (tensor<f32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
//   CHECK-NEXT: %[[v8:.+]] = stablehlo.convert %[[v7]] : (tensor<4x4xf32>) -> tensor<4x4xi4>
//   CHECK-NEXT: return %[[v8]] : tensor<4x4xi4>

// -----

func.func @large_weights() -> tensor<258x256xi4> {
  %cst = stablehlo.constant dense<7.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<258x256xf32>
  %cst_1 = stablehlo.constant dense_resource<__elided__> : tensor<2x256xf32>
  %cst_2 = stablehlo.constant dense<-8.000000e+00> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %cst_1, dims = [1, 2] : (tensor<2x256xf32>) -> tensor<129x2x256xf32>
  %1 = stablehlo.reshape %0 : (tensor<129x2x256xf32>) -> tensor<258x256xf32>
  %2 = stablehlo.divide %cst_0, %1 : tensor<258x256xf32>
  %3 = stablehlo.round_nearest_even %2 : tensor<258x256xf32>
  %4 = stablehlo.clamp %cst_2, %3, %cst : (tensor<f32>, tensor<258x256xf32>, tensor<f32>) -> tensor<258x256xf32>
  %5 = stablehlo.convert %4 : (tensor<258x256xf32>) -> tensor<258x256xi4>
  return %5 : tensor<258x256xi4>
}

//  CHECK-LABEL: large_weights
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<258x256xf32>
//   CHECK-NEXT: %[[v1:.+]] = stablehlo.composite "tensorrt.block_q" %[[v0]] {composite_attributes = {axis = -1 : i32, is_pointwise, scale = dense_resource<__elided__> : tensor<2x256xf32>}, decomposition = @block_q} : (tensor<258x256xf32>) -> tensor<258x256xi4>
//   CHECK-NEXT: return %[[v1]] : tensor<258x256xi4>
//  CHECK-LABEL: private @block_q
//   CHECK-SAME: attributes {plan.decomposition}