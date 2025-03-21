// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-ext-constant-folding | FileCheck %s

!input_type = tensor<1x8xf32>
!index_type = tensor<1x1xi32>
!updates_type = tensor<1x8xf32>

func.func @trivial_scatter_nd_2d_input(%arg0: !input_type, %arg1: !updates_type) -> !input_type {
%c0 = stablehlo.constant dense<0> : !index_type
  %0 = "stablehlo.scatter"(%arg0, %c0, %arg1) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
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
  } : (!input_type, !index_type, !updates_type) -> !input_type
  return %0 : !input_type
}

// CHECK-LABEL: func.func @trivial_scatter_nd_2d_input
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x8xf32>, %[[arg1:.+]]: tensor<1x8xf32>)
//       CHECK:     return %[[arg1]] : tensor<1x8xf32>

// -----

!input_type = tensor<1x4x4xf32>
!index_type = tensor<1x1xi32>
!updates_type = tensor<1x4x4xf32>

func.func @trivial_scatter_nd_3d_input(%arg0: !input_type, %arg1: !updates_type) -> !input_type {
%c0 = stablehlo.constant dense<0> : !index_type
  %0 = "stablehlo.scatter"(%arg0, %c0, %arg1) ({
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

// CHECK-LABEL: func.func @trivial_scatter_nd_3d_input
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x4x4xf32>, %[[arg1:.+]]: tensor<1x4x4xf32>)
//   CHECK-DAG:     return %[[arg1]] : tensor<1x4x4xf32>

// -----

!input_type = tensor<1x1018xi1>
!index_type = tensor<1xi32>
!updates_type = tensor<1018xi1>

func.func @trivial_scatter_nd_requires_reshape(%arg0: !input_type, %arg1: !updates_type) -> !input_type {
%c0 = stablehlo.constant dense<0> : !index_type
  %0 = "stablehlo.scatter"(%arg0, %c0, %arg1) ({
  ^bb0(%arg3: tensor<i1>, %arg4: tensor<i1>):
    stablehlo.return %arg4 : tensor<i1>
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

// CHECK-LABEL: func.func @trivial_scatter_nd_requires_reshape
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
//   CHECK-DAG:   %[[v0:.+]] = stablehlo.reshape %[[arg1]] : (tensor<1018xi1>) -> tensor<1x1018xi1>
//   CHECK-DAG:   return %[[v0]] : tensor<1x1018xi1>

// -----

!input_type = tensor<8xf32>
!index_type = tensor<4x1xi32>
!updates_type = tensor<4xf32>

func.func @not_trivial_scatter_nd_multi_slice(%arg0: !input_type, %arg1: !updates_type) -> !input_type {
  %c0 = stablehlo.constant dense<0> : !index_type
  %0 = "stablehlo.scatter"(%arg0, %c0, %arg1) ({
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

// CHECK-LABEL: @not_trivial_scatter_nd_multi_slice
//       CHECK:  stablehlo.scatter

// -----

!input_type = tensor<8xf32>
!index_type = tensor<1xi32>
!updates_type = tensor<8xf32>

func.func @limitation1(
    %arg0: !input_type, %arg2: !updates_type) -> !input_type {
  %c0 = stablehlo.constant dense<0> : !index_type
  %0 = "stablehlo.scatter"(%arg0, %c0, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [0],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 0
    >,
    tensorrt.canonicalized_scatter,
    unique_indices = false
  } : (!input_type, !index_type, !updates_type) -> !input_type
  return %0 : !input_type
}

// We don't recognize this case yet.

// CHECK-LABEL: func.func @limitation1
//       CHECK: stablehlo.scatter
