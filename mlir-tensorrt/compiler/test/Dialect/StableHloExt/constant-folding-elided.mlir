// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-ext-constant-folding | FileCheck %s

func.func @reshape_folder_elided() -> (tensor<1024x1024xf32>, tensor<1024x1024x1xf32>) {
  %0 = stablehlo.constant dense_resource<__elided__> : tensor<1024x1024xf32>
  %1 = stablehlo.reshape %0 : (tensor<1024x1024xf32>) -> tensor<1024x1024x1xf32>
  return %0, %1 : tensor<1024x1024xf32>, tensor<1024x1024x1xf32>
}

// CHECK-LABEL: @reshape_folder_elided
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<1024x1024x1xf32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<1024x1024xf32>
//       CHECK:     return %[[v1]], %[[v0]]

// -----

func.func @convert_folder_elided() -> (tensor<1024xi32>, tensor<1024xf32>, tensor<1024xf16>) {
  %0 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
  %1 = stablehlo.constant dense_resource<__elided__> : tensor<1024xf16>
  %2 = stablehlo.convert %0 : (tensor<1024xf32>) -> tensor<1024xf16>
  %3 = stablehlo.convert %1 : (tensor<1024xf16>) -> tensor<1024xf32>
  %4 = stablehlo.convert %1 : (tensor<1024xf16>) -> tensor<1024xi32>
  return %4, %3, %2 : tensor<1024xi32>, tensor<1024xf32>, tensor<1024xf16>
}

// CHECK-LABEL: @convert_folder_elided
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<1024xf16>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<1024xf32>
//   CHECK-DAG:     %[[v2:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<1024xi32>
//       CHECK:     return %[[v2]], %[[v1]], %[[v0]] : tensor<1024xi32>, tensor<1024xf32>, tensor<1024xf16>

// -----

func.func @transpose_fold_elided() -> tensor<2x4xf32> {
  %0 = stablehlo.constant dense_resource<__elided__> : tensor<4x2xf32>
  %1 = "stablehlo.transpose"(%0) {permutation = array<i64: 1, 0>} :
    (tensor<4x2xf32>) -> tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// CHECK-LABEL: @transpose_fold_elided
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense_resource<__elided__> : tensor<2x4xf32>
//       CHECK:     return %[[v0]] : tensor<2x4xf32>
