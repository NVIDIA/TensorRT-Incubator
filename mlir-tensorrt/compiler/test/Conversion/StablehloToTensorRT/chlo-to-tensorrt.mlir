// RUN: mlir-tensorrt-opt %s -split-input-file -convert-stablehlo-to-tensorrt | FileCheck %s

func.func @erf(%arg0: tensor<10xf32>, %arg1: tensor<10xf16>) -> (tensor<10xf32>, tensor<10xf16>) {
  %0 = chlo.erf %arg0 : tensor<10xf32> -> tensor<10xf32>
  %1 = chlo.erf %arg1 : tensor<10xf16> -> tensor<10xf16>
  return %0, %1 : tensor<10xf32>, tensor<10xf16>
}

// CHECK-LABEL: @erf
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf16>) -> (tensor<10xf32>, tensor<10xf16>) {
//       CHECK:     %[[v0:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kERF>} %[[arg0]] : tensor<10xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kERF>} %[[arg1]] : tensor<10xf16>
//       CHECK:     return %[[v0]], %[[v1]] : tensor<10xf32>, tensor<10xf16>

// -----

func.func @erf_scalar(%arg0: tensor<f32>) -> (tensor<f32>) {
  %0 = chlo.erf %arg0 : tensor<f32> -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @erf_scalar
//       CHECK: %[[exp:.+]] = tensorrt.expand_rank %[[arg0:.+]] : tensor<f32> to tensor<1xf32>
//       CHECK: %[[un:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kERF>} %[[exp:.+]] : tensor<1xf32>
//       CHECK: %[[coll:.+]] = tensorrt.collapse_rank %[[un]] : tensor<1xf32> to tensor<f32>

// -----

func.func @top_k(%arg0: tensor<1x50257xf32>) -> (tensor<1x50xf32>, tensor<1x50xi32>) {
  %values, %indices = chlo.top_k(%arg0, k = 50) : tensor<1x50257xf32> -> (tensor<1x50xf32>, tensor<1x50xi32>)
  return %values, %indices : tensor<1x50xf32>, tensor<1x50xi32>
}

// CHECK-LABEL: @top_k
//  CHECK:      %{{.+}}, %{{.+}} = tensorrt.top_k <kMAX> {axis = 1 : i64, k = 50 : i64}
//  CHECK-SAME: %{{.+}} : tensor<1x50257xf32> -> tensor<1x50xf32>, tensor<1x50xi32>

// -----

func.func @topk_1d(%arg0: tensor<4xf16>) -> (tensor<3xf16> {jax.result_info = "[0]"}, tensor<3xi32> {jax.result_info = "[1]"}) {
    %values, %indices = chlo.top_k(%arg0, k = 3) {largest = true} : tensor<4xf16> -> (tensor<3xf16>, tensor<3xi32>)
    return %values, %indices : tensor<3xf16>, tensor<3xi32>
}

// CHECK-LABEL: topk_1d
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf16>)
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.expand_rank %[[arg0]] : tensor<4xf16> to tensor<4x1xf16>
//  CHECK-NEXT: %[[values:.+]], %[[indices:.+]] = tensorrt.top_k <kMAX> {{.*}} %[[v0]] : tensor<4x1xf16> -> tensor<3x1xf16>, tensor<3x1xi32>
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.collapse_rank %[[values]]
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.collapse_rank %[[indices]]
//  CHECK-NEXT: return %[[v1]], %[[v2]] : tensor<3xf16>, tensor<3xi32>