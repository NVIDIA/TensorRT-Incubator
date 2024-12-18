// RUN: mlir-tensorrt-opt %s -stablehlo-gather-to-slice | FileCheck %s

// Generated from JAX expression: 'input[:, 1::2, ...]'
func.func @slice_1d(%arg0: tensor<12x128x4x12x1xf32>) -> tensor<12x64x4x12x1xf32> {
  %0 = stablehlo.constant dense<1> : tensor<64xi32>
  %1 = stablehlo.constant dense<2> : tensor<64xi32>
  %2 = stablehlo.iota dim = 0 : tensor<64xi32>
  %3 = stablehlo.multiply %1, %2 : tensor<64xi32>
  %4 = stablehlo.add %0, %3 : tensor<64xi32>
  %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<64xi32>) -> tensor<64x1xi32>
  %6 = "stablehlo.gather"(%arg0, %5) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3, 4], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 12, 1, 4, 12, 1>} : (tensor<12x128x4x12x1xf32>, tensor<64x1xi32>) -> tensor<12x64x4x12x1xf32>
  return %6 : tensor<12x64x4x12x1xf32>
}

// CHECK-LABEL: @slice_1d
//  CHECK-SAME: (%[[arg0:.+]]: tensor<12x128x4x12x1xf32>)
//       CHECK:     %[[v0:.+]] = stablehlo.slice %[[arg0]] [0:12, 1:128:2, 0:4, 0:12, 0:1] : (tensor<12x128x4x12x1xf32>) -> tensor<12x64x4x12x1xf32>
//       CHECK:     return %[[v0]]

// -----

// Generated from JAX expression: 'input[:, ::2, ...]'
func.func @slice_1d_2(%arg0: tensor<12x128x4x12x1xf32>) -> tensor<12x64x4x12x1xf32> {
  %0 = stablehlo.constant dense<0> : tensor<64xi32>
  %1 = stablehlo.constant dense<2> : tensor<64xi32>
  %2 = stablehlo.iota dim = 0 : tensor<64xi32>
  %3 = stablehlo.multiply %1, %2 : tensor<64xi32>
  %4 = stablehlo.add %0, %3 : tensor<64xi32>
  %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<64xi32>) -> tensor<64x1xi32>
  %6 = "stablehlo.gather"(%arg0, %5) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3, 4], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 12, 1, 4, 12, 1>} : (tensor<12x128x4x12x1xf32>, tensor<64x1xi32>) -> tensor<12x64x4x12x1xf32>
  return %6 : tensor<12x64x4x12x1xf32>
}

// CHECK-LABEL: @slice_1d_2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<12x128x4x12x1xf32>)
//       CHECK:     %[[v0:.+]] = stablehlo.slice %[[arg0]] [0:12, 0:127:2, 0:4, 0:12, 0:1] : (tensor<12x128x4x12x1xf32>) -> tensor<12x64x4x12x1xf32>
//       CHECK:     return %[[v0]]

// -----

// Generated from JAX expression: 'input[:, 1::2, :, 3::4, :]'
func.func @slice_2d(%arg0: tensor<12x128x4x12x1xf32>) -> tensor<12x64x4x3x1xf32> {
  %0 = stablehlo.constant dense<4> : tensor<3xi32>
  %1 = stablehlo.constant dense<3> : tensor<3xi32>
  %2 = stablehlo.constant dense<1> : tensor<64xi32>
  %3 = stablehlo.constant dense<2> : tensor<64xi32>
  %4 = stablehlo.iota dim = 0 : tensor<64xi32>
  %5 = stablehlo.multiply %3, %4 : tensor<64xi32>
  %6 = stablehlo.add %2, %5 : tensor<64xi32>
  %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<64xi32>) -> tensor<64x3x1xi32>
  %8 = stablehlo.iota dim = 0 : tensor<3xi32>
  %9 = stablehlo.multiply %0, %8 : tensor<3xi32>
  %10 = stablehlo.add %1, %9 : tensor<3xi32>
  %11 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<3xi32>) -> tensor<64x3x1xi32>
  %12 = stablehlo.concatenate %7, %11, dim = 2 : (tensor<64x3x1xi32>, tensor<64x3x1xi32>) -> tensor<64x3x2xi32>
  %13 = "stablehlo.gather"(%arg0, %12) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 4], collapsed_slice_dims = [1, 3], start_index_map = [1, 3], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 12, 1, 4, 1, 1>} : (tensor<12x128x4x12x1xf32>, tensor<64x3x2xi32>) -> tensor<12x64x4x3x1xf32>
  return %13 : tensor<12x64x4x3x1xf32>
}

// CHECK-LABEL: @slice_2d
//  CHECK-SAME: (%[[arg0:.+]]: tensor<12x128x4x12x1xf32>) -> tensor<12x64x4x3x1xf32>
//       CHECK:     %[[v0:.+]] = stablehlo.slice %[[arg0]] [0:12, 1:128:2, 0:4, 3:12:4, 0:1] : (tensor<12x128x4x12x1xf32>) -> tensor<12x64x4x3x1xf32>
//       CHECK:     return %[[v0]]
