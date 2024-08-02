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