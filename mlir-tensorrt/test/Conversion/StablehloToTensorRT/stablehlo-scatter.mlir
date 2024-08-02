// RUN: mlir-tensorrt-opt -split-input-file %s --convert-stablehlo-to-tensorrt | FileCheck %s

func.func @whisper_jax_scatter(%arg0: tensor<1x51865xf32>) -> tensor<1x51865xf32> {
  %0 = stablehlo.constant dense<50257> : tensor<1x1xi32>
  %1 = stablehlo.constant dense<0xFF800000> : tensor<1x1xf32>
  %2 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<1x51865xf32>) -> tensor<51865x1xf32>
  %3 = "stablehlo.scatter"(%2, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    stablehlo.return %arg2 : tensor<f32>
  }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, tensorrt.canonicalized_scatter, unique_indices = false} : (tensor<51865x1xf32>, tensor<1x1xi32>, tensor<1x1xf32>) -> tensor<51865x1xf32>
  %4 = "stablehlo.transpose"(%3) {permutation = array<i64: 1, 0>} : (tensor<51865x1xf32>) -> tensor<1x51865xf32>
  return %4 : tensor<1x51865xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @whisper_jax_scatter
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x51865xf32>) -> tensor<1x51865xf32> {
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<50257> : tensor<1x1xi32>
//       CHECK:     %[[cst_f32:.+]] = tensorrt.constant dense<0xFF800000> : tensor<1x1xf32>
//       CHECK:     %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] : tensor<1x51865xf32> to tensor<51865x1xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.scatter_nd data(%[[v0]] : tensor<51865x1xf32>) indices(%[[cst_i32]] : tensor<1x1xi32>) updates(%[[cst_f32]] : tensor<1x1xf32>)
//       CHECK:     %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v1]] : tensor<51865x1xf32> to tensor<1x51865xf32>
//       CHECK:     return %[[v2]] : tensor<1x51865xf32>

// -----

func.func @whisper_jax_scatter2(%arg0: tensor<1x51865xf32>, %arg1: tensor<88x1xi32>) -> tensor<1x51865xf32> {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<88x1xf32>
  %1 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<1x51865xf32>) -> tensor<51865x1xf32>
  %2 = "stablehlo.scatter"(%1, %arg1, %0) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    stablehlo.return %arg3 : tensor<f32>
  }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, tensorrt.canonicalized_scatter, unique_indices = false} : (tensor<51865x1xf32>, tensor<88x1xi32>, tensor<88x1xf32>) -> tensor<51865x1xf32>
  %3 = "stablehlo.transpose"(%2) {permutation = array<i64: 1, 0>} : (tensor<51865x1xf32>) -> tensor<1x51865xf32>
  return %3 : tensor<1x51865xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: @whisper_jax_scatter2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x51865xf32>, %[[arg1:.+]]: tensor<88x1xi32>) -> tensor<1x51865xf32> {
//       CHECK:   %[[cst_f32:.+]] = tensorrt.constant dense<0xFF800000> : tensor<88x1xf32>
//       CHECK:   %[[v0:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[arg0]] : tensor<1x51865xf32> to tensor<51865x1xf32>
//       CHECK:   %[[v1:.+]] = tensorrt.scatter_nd data(%[[v0]] : tensor<51865x1xf32>) indices(%[[arg1]] : tensor<88x1xi32>) updates(%[[cst_f32]] : tensor<88x1xf32>)
//       CHECK:   %[[v2:.+]] = tensorrt.transpose {permutation = #[[$map]]} %[[v1]] : tensor<51865x1xf32> to tensor<1x51865xf32>
//       CHECK:   return %[[v2]] : tensor<1x51865xf32>

// -----

func.func @stablehlo_scatter_canonicalize(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<2xi32>, %arg3: tensor<2x3xf32>, %arg4: tensor<2x3xf32>) -> tensor<3x3xf32> {
  %0 = stablehlo.reshape %arg2 : (tensor<2xi32>) -> tensor<2x1xi32>
  %1:2 = "stablehlo.scatter"(%arg0, %arg1, %0, %arg3, %arg4) ({
  ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>):
    stablehlo.return %arg5, %arg7 : tensor<f32>, tensor<f32>
  }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, tensorrt.canonicalized_scatter, unique_indices = false} : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<2x1xi32>, tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>)
  return %1#0 : tensor<3x3xf32>
}

// CHECK-LABEL: @stablehlo_scatter_canonicalize
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x3xf32>, %[[arg1:.+]]: tensor<3x3xf32>, %[[arg2:.+]]: tensor<2xi32>, %[[arg3:.+]]: tensor<2x3xf32>, %[[arg4:.+]]: tensor<2x3xf32>) -> tensor<3x3xf32> {
//       CHECK:   %[[v0:.+]] = tensorrt.expand_rank %[[arg2]] : tensor<2xi32> to tensor<2x1xi32>
//       CHECK:   %[[v1:.+]] = tensorrt.scatter_nd data(%[[arg0]] : tensor<3x3xf32>) indices(%[[v0]] : tensor<2x1xi32>) updates(%[[arg3]] : tensor<2x3xf32>)
//       CHECK:   %[[v2:.+]] = tensorrt.scatter_nd data(%[[arg1]] : tensor<3x3xf32>) indices(%[[v0]] : tensor<2x1xi32>) updates(%[[arg4]] : tensor<2x3xf32>)
//       CHECK:   return %[[v1]] : tensor<3x3xf32>
