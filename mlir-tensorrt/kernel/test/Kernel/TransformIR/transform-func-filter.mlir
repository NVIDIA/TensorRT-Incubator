// RUN: kernel-opt -split-input-file %s -kernel-initial-transform-schedule="func-filter=kernel.func" | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (0, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @trt_matrix_multiply(%arg0: tensor<2048x1024xf32>, %arg1: tensor<1024x2048xf32>) -> tensor<2048x2048xf32> attributes {kernel.func} {
  %0 = tensor.empty() : tensor<2048x2048xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<2048x1024xf32>, tensor<1024x2048xf32>) outs(%1 : tensor<2048x2048xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %3, %out : f32
    linalg.yield %4 : f32
  } -> tensor<2048x2048xf32>
  return %2 : tensor<2048x2048xf32>
}
func.func @cos_matmul(%arg0: tensor<?x1024x1024xf32>, %arg1: tensor<1x1024x1024xf32>) -> tensor<?x1024x1024xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x1024x1024xf32>
  %0 = tensor.empty(%dim) : tensor<?x1024x1024xf32>
  %1 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x1024x1024xf32>) outs(%0 : tensor<?x1024x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = math.cos %in : f32
    linalg.yield %5 : f32
  } -> tensor<?x1024x1024xf32>
  %c0_0 = arith.constant 0 : index
  %dim_1 = tensor.dim %1, %c0_0 : tensor<?x1024x1024xf32>
  %2 = tensor.empty(%dim_1) : tensor<?x1024x1024xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x1024x1024xf32>) -> tensor<?x1024x1024xf32>
  %4 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1, %arg1 : tensor<?x1024x1024xf32>, tensor<1x1024x1024xf32>) outs(%3 : tensor<?x1024x1024xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %5 = arith.mulf %in, %in_2 : f32
    %6 = arith.addf %5, %out : f32
    linalg.yield %6 : f32
  } -> tensor<?x1024x1024xf32>
  return %4 : tensor<?x1024x1024xf32>
}

// CHECK:  transform.sequence  failures(propagate) attributes {kernel.target_func = @trt_matrix_multiply}
// CHECK-NOT:  transform.sequence  failures(propagate) attributes {kernel.target_func = @cos_matmul}