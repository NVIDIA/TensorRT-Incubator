// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-raise-qdq | FileCheck %s

// Block dequantization is weight only dequantization thus input is always constant.

func.func @small_weights() -> tensor<4x4xf32> {
  %c = stablehlo.constant dense<2> : tensor<4x4xi4>
  %cst = stablehlo.constant dense<[[5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01], [5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01]]> : tensor<2x4xf32>
  %0 = stablehlo.convert %c : (tensor<4x4xi4>) -> tensor<4x4xf32>
  %1 = stablehlo.broadcast_in_dim %cst, dims = [1, 2] : (tensor<2x4xf32>) -> tensor<2x2x4xf32>
  %2 = stablehlo.reshape %1 : (tensor<2x2x4xf32>) -> tensor<4x4xf32>
  %3 = stablehlo.multiply %0, %2 : tensor<4x4xf32>
  return %3 : tensor<4x4xf32>
}



//  CHECK-LABEL: small_weights
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<2> : tensor<4x4xi4>
//   CHECK-NEXT: %[[v1:.+]] = stablehlo.composite "tensorrt.block_dq" %[[v0]] {composite_attributes = {axis = -1 : i32, scale = dense<{{\[}}[5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01], [5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01]{{\]}}> : tensor<2x4xf32>}, decomposition = @block_dq} : (tensor<4x4xi4>) -> tensor<4x4xf32>
//   CHECK-NEXT: return %[[v1]] : tensor<4x4xf32>
//  CHECK-LABEL: private @block_dq
//   CHECK-SAME: (%[[arg0:.+]]: tensor<4x4xi4>)
//   CHECK-SAME: attributes {plan.decomposition}
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense<{{\[}}[5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01], [5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01]{{\]}}> : tensor<2x4xf32>
//   CHECK-NEXT: %[[v1:.+]] = stablehlo.broadcast_in_dim %[[v0]], dims = [1, 2] : (tensor<2x4xf32>) -> tensor<2x2x4xf32>
//   CHECK-NEXT: %[[v2:.+]] = stablehlo.reshape %[[v1]] : (tensor<2x2x4xf32>) -> tensor<4x4xf32>
//   CHECK-NEXT: %[[v3:.+]] = stablehlo.convert %[[arg0]] : (tensor<4x4xi4>) -> tensor<4x4xf32>
//   CHECK-NEXT: %[[v4:.+]] = stablehlo.multiply %[[v3]], %[[v2]] : tensor<4x4xf32>
//   CHECK-NEXT: return %[[v4]] : tensor<4x4xf32>


// -----

func.func @large_weights() -> tensor<258x256xf32> {
  %c = stablehlo.constant dense_resource<__elided__> : tensor<258x256xi4>
  %cst = stablehlo.constant dense_resource<__elided__> : tensor<2x256xf32>
  %0 = stablehlo.convert %c : (tensor<258x256xi4>) -> tensor<258x256xf32>
  %1 = stablehlo.broadcast_in_dim %cst, dims = [1, 2] : (tensor<2x256xf32>) -> tensor<129x2x256xf32>
  %2 = stablehlo.reshape %1 : (tensor<129x2x256xf32>) -> tensor<258x256xf32>
  %3 = stablehlo.multiply %0, %2 : tensor<258x256xf32>
  return %3 : tensor<258x256xf32>
}


//  CHECK-LABEL: large_weights
//   CHECK-NEXT: %[[v0:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<258x256xi4>
//   CHECK-NEXT: %[[v1:.+]] = stablehlo.composite "tensorrt.block_dq" %[[v0]] {composite_attributes = {axis = -1 : i32, scale = dense_resource<__elided__> : tensor<2x256xf32>}, decomposition = @block_dq} : (tensor<258x256xi4>) -> tensor<258x256xf32>
//   CHECK-NEXT: return %[[v1]] : tensor<258x256xf32>
//  CHECK-LABEL: private @block_dq
//   CHECK-SAME: attributes {plan.decomposition}