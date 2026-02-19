// RUN: mlir-tensorrt-opt -split-input-file -pass-pipeline="builtin.module(linalg-simplify-extract-slice)" %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @non_rank_reducing_slice(%arg0: tensor<2x4x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4x1xf32> {
  %cst = arith.constant 5.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x4x4xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x4x4xf32>) outs(%0 : tensor<2x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x4x4xf32>
  %extracted_slice = tensor.extract_slice %1[0, 0, 0] [2, 4, 1] [1, 1, 1] : tensor<2x4x4xf32> to tensor<2x4x1xf32>
  return %extracted_slice : tensor<2x4x1xf32>
}

// CHECK-LABEL: @non_rank_reducing_slice
// CHECK: %[[v0:.+]] = linalg.generic
// CHECK: return %[[v0]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
func.func @multiple(%arg0: tensor<2x4x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x2x1xf32> {
  %cst = arith.constant 5.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x4x4xf32>
  %1 = tensor.empty() : tensor<4x2x4xf32>
  %2:2 = linalg.generic {indexing_maps = [#map, #map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg0 : tensor<2x4x4xf32>, tensor<2x4x4xf32>)
      outs(%0, %1 : tensor<2x4x4xf32>, tensor<4x2x4xf32>) {
  ^bb0(%in: f32, %in1: f32, %out1: f32, %out2: f32):
    linalg.yield %in, %in1 : f32, f32
  } -> (tensor<2x4x4xf32>, tensor<4x2x4xf32>)
  %extracted_slice = tensor.extract_slice %2#1[0, 0, 0] [2, 2, 1] [1, 1, 1] : tensor<4x2x4xf32> to tensor<2x2x1xf32>
  return %extracted_slice : tensor<2x2x1xf32>
}

// CHECK-LABEL: @multiple
// CHECK: %[[v0:.+]]:2 = linalg.generic
// CHECK: return %[[v0]]#1

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @rank_reducing_slice(%arg0: tensor<2x4x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %cst = arith.constant 5.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x4x4xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x4x4xf32>) outs(%0 : tensor<2x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x4x4xf32>
  %extracted_slice = tensor.extract_slice %1[0, 0, 0] [2, 4, 1] [1, 1, 1] : tensor<2x4x4xf32> to tensor<2x4xf32>
  return %extracted_slice : tensor<2x4xf32>
}

//     CHECK: %[[v0:.+]] = linalg.generic
// CHECK-NOT: return %[[v0]]
