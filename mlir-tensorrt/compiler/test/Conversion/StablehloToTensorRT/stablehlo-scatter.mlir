// RUN: mlir-tensorrt-opt -split-input-file %s --convert-stablehlo-to-tensorrt | FileCheck %s

func.func @whisper_jax_scatter(%arg0: tensor<1x51865xf32>) -> tensor<1x51865xf32> {
  %0 = stablehlo.constant dense<50257> : tensor<1x1xi32>
  %1 = stablehlo.constant dense<0xFF800000> : tensor<1x1xf32>
  %2 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<1x51865xf32>) -> tensor<51865x1xf32>
  %3 = "stablehlo.scatter"(%2, %0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    stablehlo.return %arg2 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    tensorrt.canonicalized_scatter,
    unique_indices = false
  } : (tensor<51865x1xf32>, tensor<1x1xi32>, tensor<1x1xf32>) -> tensor<51865x1xf32>
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
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    tensorrt.canonicalized_scatter,
    unique_indices = false
  } : (tensor<51865x1xf32>, tensor<88x1xi32>, tensor<88x1xf32>) -> tensor<51865x1xf32>
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

func.func @scatter_multiple_inputs(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<2xi32>, %arg3: tensor<2x3xf32>, %arg4: tensor<2x3xf32>) -> tensor<3x3xf32> {
  %0 = stablehlo.reshape %arg2 : (tensor<2xi32>) -> tensor<2x1xi32>
  %1:2 = "stablehlo.scatter"(%arg0, %arg1, %0, %arg3, %arg4) ({
  ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>):
    stablehlo.return %arg6, %arg8 : tensor<f32>, tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    tensorrt.canonicalized_scatter,
    unique_indices = false
  } : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<2x1xi32>, tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>)
  return %1#0 : tensor<3x3xf32>
}

// CHECK-LABEL: @scatter_multiple_inputs
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x3xf32>, %[[arg1:.+]]: tensor<3x3xf32>, %[[arg2:.+]]: tensor<2xi32>, %[[arg3:.+]]: tensor<2x3xf32>, %[[arg4:.+]]: tensor<2x3xf32>)
//       CHECK:   %[[v0:.+]] = tensorrt.expand_rank %[[arg2]] : tensor<2xi32> to tensor<2x1xi32>
//       CHECK:   %[[v1:.+]] = tensorrt.scatter_nd data(%[[arg0]] : tensor<3x3xf32>) indices(%[[v0]] : tensor<2x1xi32>) updates(%[[arg3]] : tensor<2x3xf32>)
//       CHECK:   %[[v2:.+]] = tensorrt.scatter_nd data(%[[arg1]] : tensor<3x3xf32>) indices(%[[v0]] : tensor<2x1xi32>) updates(%[[arg4]] : tensor<2x3xf32>)
//       CHECK:   return %[[v1]] : tensor<3x3xf32>

// -----

!input_type = tensor<8xf32>
!index_type = tensor<4x1xi32>
!updates_type = tensor<4xf32>

func.func @scattern_nd_1d_input(%arg0: !input_type, %arg1: !index_type, %arg2: !updates_type) -> !input_type {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    tensorrt.canonicalized_scatter,
    unique_indices = false
  } : (!input_type, !index_type, !updates_type) -> !input_type
  return %0 : !input_type
}

// CHECK-LABEL: func.func @scattern_nd_1d_input
//  CHECK-SAME: (%[[arg0:.+]]: tensor<8xf32>, %[[arg1:.+]]: tensor<4x1xi32>, %[[arg2:.+]]: tensor<4xf32>)
//   CHECK-DAG:     %[[v0:.+]] = tensorrt.scatter_nd data(%[[arg0]] : tensor<8xf32>) indices(%[[arg1]] : tensor<4x1xi32>) updates(%[[arg2]] : tensor<4xf32>)
//   CHECK-DAG:     return %[[v0]]

// -----

!input_type = tensor<4x4x4xf32>
!index_type = tensor<2x1xi32>
!updates_type = tensor<2x4x4xf32>

func.func @scattern_nd_2d_input(%arg0: !input_type, %arg1: !index_type, %arg2: !updates_type) -> !input_type {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 2],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    tensorrt.canonicalized_scatter,
    unique_indices = false
  } : (!input_type, !index_type, !updates_type) -> !input_type
  return %0 : !input_type
}

// CHECK-LABEL: func.func @scattern_nd_2d_input
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x4x4xf32>, %[[arg1:.+]]: tensor<2x1xi32>, %[[arg2:.+]]: tensor<2x4x4xf32>)
//   CHECK-DAG:     %[[v0:.+]] = tensorrt.scatter_nd data(%[[arg0]] : tensor<4x4x4xf32>) indices(%[[arg1]] : tensor<2x1xi32>) updates(%[[arg2]] : tensor<2x4x4xf32>)
//   CHECK-DAG:     return %[[v0]]

// -----

!input_type = tensor<1x4xf32>
!index_type = tensor<1xi32>
!updates_type = tensor<4xf32>

func.func @scattern_nd_2d_2d_input_1d_index(%arg0: !input_type, %arg1: !index_type, %arg2: !updates_type) -> !input_type {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [0],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 0
    >,
    tensorrt.canonicalized_scatter,
    unique_indices = false
  } : (!input_type, !index_type, !updates_type) -> !input_type
  return %0 : !input_type
}

// CHECK-LABEL: func.func @scattern_nd_2d_2d_input_1d_index
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x4xf32>, %[[arg1:.+]]: tensor<1xi32>, %[[arg2:.+]]: tensor<4xf32>) -> tensor<1x4xf32> {
//   CHECK-DAG:     %[[v0:.+]] = tensorrt.scatter_nd data(%[[arg0]] : tensor<1x4xf32>) indices(%[[arg1]] : tensor<1xi32>) updates(%[[arg2]] : tensor<4xf32>)
//   CHECK-DAG:     return %[[v0]] : tensor<1x4xf32>

// -----

!input_type = tensor<4x4x4xf32>
!index_type = tensor<2x2x1xi32>
!updates_type = tensor<2x2x4x4xf32>

func.func @scattern_nd_3d_updates(%arg0: !input_type, %arg1: !index_type, %arg2: !updates_type) -> !input_type {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 2
    >,
    tensorrt.canonicalized_scatter,
    unique_indices = false
  } : (!input_type, !index_type, !updates_type) -> !input_type
  return %0 : !input_type
}

// CHECK-LABEL: func.func @scattern_nd_3d_updates
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x4x4xf32>, %[[arg1:.+]]: tensor<2x2x1xi32>, %[[arg2:.+]]: tensor<2x2x4x4xf32>)
//   CHECK-DAG:     %[[v0:.+]] = tensorrt.scatter_nd data(%[[arg0]] : tensor<4x4x4xf32>) indices(%[[arg1]] : tensor<2x2x1xi32>) updates(%[[arg2]] : tensor<2x2x4x4xf32>)
//   CHECK-DAG:     return %[[v0]] : tensor<4x4x4xf32>

// -----

!input_type = tensor<4x4x4xf32>
!index_type = tensor<2x1x2xi32>
!updates_type = tensor<2x2x4x4xf32>

func.func @not_scattern_nd_wrong_index_dim(%arg0: !input_type, %arg1: !index_type, %arg2: !updates_type) -> !input_type {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    tensorrt.canonicalized_scatter,
    unique_indices = false
  } : (!input_type, !index_type, !updates_type) -> !input_type
  return %0 : !input_type
}

// CHECK-LABEL: @not_scattern_nd_wrong_index_dim
//   CHECK-NOT: tensorrt.scatter_nd

// -----

!input_type = tensor<4x4x4xf32>
!index_type = tensor<2x2x1xi32>
!updates_type = tensor<2x2x4x4xf32>

func.func @not_scatter_nd_wrong_scatter_to_operand_dims(
    %arg0: !input_type, %arg1: !index_type, %arg2: !updates_type)
    -> !input_type {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1],
      index_vector_dim = 2
    >,
    tensorrt.canonicalized_scatter,
    unique_indices = false
  } : (!input_type, !index_type, !updates_type) -> !input_type
  return %0 : !input_type
}

// CHECK-LABEL: @not_scatter_nd_wrong_scatter_to_operand_dims
//   CHECK-NOT: tensorrt.scatter_nd

// -----

// This test is a regression test for a bug where one of the patterns
// was crashing due to empty `scatter_dims_to_operand_dims` array.

// CHECK-LABEL: func.func @scatter_zero_ext_regression
func.func @scatter_zero_ext_regression(%arg0 : tensor<f32>, %arg1 : tensor<1x0xi32>, %arg2 : tensor<1xf32>) -> tensor<f32> {
  // CHECK: %[[v0:.+]] = tensorrt.scatter_nd data(%[[arg0]] : tensor<f32>) indices(%[[arg1]] : tensor<1x0xi32>) updates(%[[arg2]] 
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      "stablehlo.return"(%arg4) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<f32>, tensor<1x0xi32>, tensor<1xf32>) -> tensor<f32>  
  // CHECK: return %[[v0]] : tensor<f32>
  func.return %0 : tensor<f32>
}
