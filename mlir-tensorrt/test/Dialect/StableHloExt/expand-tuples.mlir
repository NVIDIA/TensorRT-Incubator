// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-ext-expand-tuples --stablehlo-aggressive-simplification | FileCheck %s

func.func @main(%arg0: tensor<1x1xf32>, %arg1: tensor<1x8x8x16xf32>) -> tuple<tensor<1024xf32>, tensor<1xf32>> {
  %1 = stablehlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<1xf32>
  %2 = stablehlo.reshape %arg1 : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
  %3 = stablehlo.tuple %2, %1 : tuple<tensor<1024xf32>, tensor<1xf32>>
  func.return %3 : tuple<tensor<1024xf32>, tensor<1xf32>>
}

// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x1xf32>, %[[arg1:.+]]: tensor<1x8x8x16xf32>)
//  CHECK-NEXT: %[[v0:.*]] = stablehlo.reshape %[[arg0]] : (tensor<1x1xf32>) -> tensor<1xf32>
//  CHECK-NEXT: %[[v1:.*]] = stablehlo.reshape %[[arg1]] : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
//  CHECK-NEXT: return %[[v1]], %[[v0]] : tensor<1024xf32>, tensor<1xf32>

// -----

// Test nesting of tuples
func.func @main(%arg0: tensor<1x1xf32>, %arg1: tensor<1x8x8x16xf32>) -> tuple<tensor<1024xf32>, tuple<tuple<tensor<1024xf32>, tensor<1xf32>>, tensor<1xf32>>> {
  %1 = stablehlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<1xf32>
  %2 = stablehlo.reshape %arg1 : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
  %3 = stablehlo.tuple %2, %1 : tuple<tensor<1024xf32>, tensor<1xf32>>
  %4 = stablehlo.tuple %3, %1 : tuple<tuple<tensor<1024xf32>, tensor<1xf32>>, tensor<1xf32>>
  %5 = stablehlo.tuple %2, %4 : tuple<tensor<1024xf32>, tuple<tuple<tensor<1024xf32>, tensor<1xf32>>, tensor<1xf32>>>
  func.return %5 : tuple<tensor<1024xf32>, tuple<tuple<tensor<1024xf32>, tensor<1xf32>>, tensor<1xf32>>>
}

// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x1xf32>, %[[arg1:.+]]: tensor<1x8x8x16xf32>)
//  CHECK-NEXT: %[[v0:.*]] = stablehlo.reshape %[[arg0]] : (tensor<1x1xf32>) -> tensor<1xf32>
//  CHECK-NEXT: %[[v1:.*]] = stablehlo.reshape %[[arg1]] : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
//  CHECK-NEXT:  return %[[v1]], %[[v1]], %[[v0]], %[[v0]] : tensor<1024xf32>, tensor<1024xf32>, tensor<1xf32>, tensor<1xf32>

// -----

func.func @main(%arg0: tensor<1x224x224x3xf16>, %arg1: tensor<f32>) -> tensor<1x224x224x3xf16> {
  func.return %arg0 : tensor<1x224x224x3xf16>
}

// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x224x224x3xf16>, %[[arg1:.+]]: tensor<f32>)
//  CHECK-NEXT: return %[[arg0]] : tensor<1x224x224x3xf16>

// -----

func.func @main(%arg0: tuple<tensor<1024xf32>, tensor<1xf32>>) -> tuple<tensor<1024xf32>, tensor<1xf32>> {
  func.return %arg0 : tuple<tensor<1024xf32>, tensor<1xf32>>
}

// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1024xf32>, %[[arg1:.+]]: tensor<1xf32>)
//  CHECK-NEXT: return %[[arg0]], %[[arg1]] : tensor<1024xf32>, tensor<1xf32>

// -----

func.func @main() -> tuple<> {
  %0 = stablehlo.tuple  : tuple<>
  func.return %0 : tuple<>
}

// CHECK-LABEL: @main()
//  CHECK-NEXT: return