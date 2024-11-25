// RUN: mlir-tensorrt-opt %s --stablehlo-ext-canonicalize-scatter --stablehlo-aggressive-simplification -split-input-file | FileCheck %s


func.func @whisper_jax_scatter(%arg0: tensor<1x51865xf32>) -> tensor<1x51865xf32> {
  %cst = arith.constant dense<50257> : tensor<1x1xi32>
  %0 = stablehlo.constant dense<0xFF800000> : tensor<1x1x1xf32>
  %1 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<1x51865xf32>) -> tensor<51865x1xf32>
  %2 = "stablehlo.scatter"(%1, %cst, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    stablehlo.return %arg2 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 2],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    unique_indices = false
  } : (tensor<51865x1xf32>, tensor<1x1xi32>, tensor<1x1x1xf32>) -> tensor<51865x1xf32>
  %3 = "stablehlo.transpose"(%2) {permutation = array<i64: 1, 0>} : (tensor<51865x1xf32>) -> tensor<1x51865xf32>
  return %3 : tensor<1x51865xf32>
}


// CHECK-LABEL: @whisper_jax_scatter
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x51865xf32>)
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.constant dense<0xFF800000> : tensor<1x1xf32>
//   CHECK-DAG:     %[[cst:.+]] = arith.constant dense<50257> : tensor<1x1xi32>
//       CHECK:     %[[v1:.+]] = stablehlo.reshape %[[arg0]]
//       CHECK:     %[[v2:.+]] = "stablehlo.scatter"(%[[v1]], %[[cst]], %[[v0]])
//  CHECK-SAME:       indices_are_sorted = false
//  CHECK-SAME:       #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>

//  CHECK-SAME:       unique_indices = false
//  CHECK-NEXT:     ^bb0(%[[arg1:.+]]: tensor<f32>, %[[arg2:.+]]: tensor<f32>):
//  CHECK-NEXT:       stablehlo.return %[[arg2]] : tensor<f32>
//  CHECK-NEXT:     }) {tensorrt.canonicalized_scatter}
//  CHECK-SAME:        : (tensor<51865x1xf32>, tensor<1x1xi32>, tensor<1x1xf32>) -> tensor<51865x1xf32>

//       CHECK:     %[[v3:.+]] = stablehlo.reshape
//       CHECK:     return %[[v3]] : tensor<1x51865xf32>

// -----

func.func @whisper_jax_scatter2(%arg0: tensor<1x51865xf32>, %arg1: tensor<88x1xi32>) -> tensor<1x51865xf32> {
  %0 = stablehlo.constant dense<0xFF800000> : tensor<88x1x1xf32>
  %1 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<1x51865xf32>) -> tensor<51865x1xf32>
  %2 = "stablehlo.scatter"(%1, %arg1, %0) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    stablehlo.return %arg3 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 2],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >, unique_indices = false
  } : (tensor<51865x1xf32>, tensor<88x1xi32>, tensor<88x1x1xf32>) -> tensor<51865x1xf32>
  %3 = "stablehlo.transpose"(%2) {permutation = array<i64: 1, 0>} : (tensor<51865x1xf32>) -> tensor<1x51865xf32>
  return %3 : tensor<1x51865xf32>
}

// CHECK-LABEL: @whisper_jax_scatter2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x51865xf32>, %[[arg1:.+]]: tensor<88x1xi32>) -> tensor<1x51865xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<0xFF800000> : tensor<88x1xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.reshape
//       CHECK:     %[[v2:.+]] = "stablehlo.scatter"(%[[v1]], %[[arg1]], %[[v0]])
//  CHECK-SAME:      indices_are_sorted = false
//  CHECK-SAME:      scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>
//  CHECK-SAME:      unique_indices = false
//  CHECK-NEXT:     ^bb0(%[[arg2:.+]]: tensor<f32>, %[[arg3:.+]]: tensor<f32>):
//  CHECK-NEXT:       stablehlo.return %[[arg3]] : tensor<f32>
//  CHECK-NEXT:     }) {tensorrt.canonicalized_scatter}
//  CHECK-SAME:      (tensor<51865x1xf32>, tensor<88x1xi32>, tensor<88x1xf32>) -> tensor<51865x1xf32>
//       CHECK:     %[[v3:.+]] = stablehlo.reshape
//       CHECK:     return %[[v3]] : tensor<1x51865xf32>
// -----

func.func @stablehlo_scatter_canonicalize(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<2xi32>, %arg3: tensor<2x3xf32>, %arg4: tensor<2x3xf32>) -> tensor<3x3xf32> {
  %expanded   = stablehlo.reshape %arg2 : (tensor<2xi32>) -> tensor<2x1xi32>
  %expanded_0 = stablehlo.reshape %arg3 : (tensor<2x3xf32>) -> tensor<2x1x3xf32>
  %expanded_1 = stablehlo.reshape %arg4 : (tensor<2x3xf32>) -> tensor<2x1x3xf32>
  %0:2 = "stablehlo.scatter"(%arg0, %arg1, %expanded, %expanded_0, %expanded_1) ({
  ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>):
    stablehlo.return %arg5, %arg7 : tensor<f32>, tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 2],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    unique_indices = false
  } : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<2x1xi32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>)
  return %0#0 : tensor<3x3xf32>
}

// CHECK-LABEL: @stablehlo_scatter_canonicalize
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x3xf32>, %[[arg1:.+]]: tensor<3x3xf32>, %[[arg2:.+]]: tensor<2xi32>, %[[arg3:.+]]: tensor<2x3xf32>, %[[arg4:.+]]: tensor<2x3xf32>) -> tensor<3x3xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.reshape %[[arg2]] : (tensor<2xi32>) -> tensor<2x1xi32>
//       CHECK:     %[[v2:.+]]:2 = "stablehlo.scatter"(%[[arg0]], %[[arg1]], %[[v0]], %[[arg3]], %[[arg4]])
//  CHECK-SAME:       indices_are_sorted = false
//  CHECK-SAME:       scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>
//  CHECK-SAME:       unique_indices = false
//  CHECK-NEXT:     ^bb0(%[[arg5:.+]]: tensor<f32>, %[[arg6:.+]]: tensor<f32>, %[[arg7:.+]]: tensor<f32>, %[[arg8:.+]]: tensor<f32>):
//  CHECK-NEXT:       stablehlo.return %[[arg5]], %[[arg7]] : tensor<f32>, tensor<f32>
//  CHECK-NEXT:     }) {tensorrt.canonicalized_scatter}
//  CHECK-SAME:       : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<2x1xi32>, tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>)
//       CHECK:     return %[[v2]]#0 : tensor<3x3xf32>

// -----

func.func @stablehlo_scatter_canonicalize_negative(%arg0: tensor<3x3xf32>, %arg1: tensor<2x1x2xi32>, %arg2: tensor<2x1x1x3xf32>) -> tensor<3x3xf32> {
  %collapsed = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<2x1x2xi32> into tensor<2x2xi32>
  %collapsed_0 = tensor.collapse_shape %arg2 [[0, 1], [2], [3]] : tensor<2x1x1x3xf32> into tensor<2x1x3xf32>
  %0 = "stablehlo.scatter"(%arg0, %collapsed, %collapsed_0) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg3 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 2],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1>,
    unique_indices = false
  } : (tensor<3x3xf32>, tensor<2x2xi32>, tensor<2x1x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// We don't do anything for this type of scatter. It can be converted into a series of
// slices and inserts (using scatter_elements) during stablehlo-to-tensorrt conversion.

// CHECK-LABEL: @stablehlo_scatter_canonicalize_negative
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x3xf32>, %[[arg1:.+]]: tensor<2x1x2xi32>, %[[arg2:.+]]: tensor<2x1x1x3xf32>) -> tensor<3x3xf32> {
//   CHECK-NOT: tensorrt.canonicalized_scatter
