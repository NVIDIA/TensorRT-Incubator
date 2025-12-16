// RUN: mlir-tensorrt-opt %s -split-input-file -convert-stablehlo-to-kernel | FileCheck %s

func.func @scatter_0(%arg0: tensor<2x3x10xf16>, %arg1: tensor<3x2x1xi32>, %arg2: tensor<3x2x3xf16>)
      -> tensor<2x3x10xf16> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2)
    <{scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [2],
        input_batching_dims = [0, 1],
        scatter_indices_batching_dims = [1, 0],
        scatter_dims_to_operand_dims = [2],
        index_vector_dim = 2>}> ({
  ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):
    stablehlo.return %arg4 : tensor<f16>
  }) : (tensor<2x3x10xf16>, tensor<3x2x1xi32>, tensor<3x2x3xf16>) -> tensor<2x3x10xf16>
  return %0 : tensor<2x3x10xf16>
}

// CHECK-LABEL: func.func @scatter_0
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x10xf16>, %[[arg1:.+]]: tensor<3x2x1xi32>, %[[arg2:.+]]: tensor<
//       CHECK:     %[[v0:.+]] = kernel.scatter updates(%[[arg2]] : tensor<3x2x3xf16>)
//  CHECK-SAME:       into(%[[arg0]] : tensor<2x3x10xf16>)
//  CHECK-SAME:       at(%[[arg1]] : tensor<3x2x1xi32>)
//       CHECK:     ^bb0(%[[arg3:.+]]: f16, %[[arg4:.+]]: f16):
//       CHECK:       kernel.yield %[[arg4]] : f16
//       CHECK:      index_vector_dim = 2 : i64,
//  CHECK-SAME:      input_batching_dims = array<i64: 0, 1>,
//  CHECK-SAME:      scatter_dims_to_operand_dims = array<i64: 2>,
//  CHECK-SAME:      scatter_indices_batching_dims = array<i64: 1, 0>,
//  CHECK-SAME:      update_window_dims = array<i64: 2>
//       CHECK:     return %[[v0]]

// -----


func.func @scatter_2(%arg0: tensor<10xf32>, %arg1: tensor<3x1xi32>, %arg2: tensor<3x2xf32>) -> (tensor<10xf32>) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2)
    <{scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [1],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1>}> ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    stablehlo.return %arg4 : tensor<f32>
  }) : (tensor<10xf32>, tensor<3x1xi32>, tensor<3x2xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: func.func @scatter_2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<3x1xi32>, %[[arg2:.+]]: tensor<3x2xf32>)
//       CHECK:     %[[v0:.+]] = kernel.scatter updates(%[[arg2]] : tensor<3x2xf32>)
//  CHECK-SAME:       into(%[[arg0]] : tensor<10xf32>)
//  CHECK-SAME:       at(%[[arg1]] : tensor<3x1xi32>)
//       CHECK:     ^bb0(%[[arg3:.+]]: f32, %[[arg4:.+]]: f32):
//       CHECK:       kernel.yield %[[arg4]] : f32
//       CHECK:      index_vector_dim = 1 : i64,
//  CHECK-SAME:      scatter_dims_to_operand_dims = array<i64: 0>,
//  CHECK-SAME:      update_window_dims = array<i64: 1>
//       CHECK:     return %[[v0]] : tensor<10xf32>

// -----


func.func @scatter_4(%arg0: tensor<5xbf16>, %arg1: tensor<2x1xi32>, %arg2: tensor<2xbf16>)
      -> (tensor<5xbf16>) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2)
    <{scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1>}> ({
  ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
    stablehlo.return %arg4 : tensor<bf16>
  }) : (tensor<5xbf16>, tensor<2x1xi32>, tensor<2xbf16>) -> tensor<5xbf16>
  return %0 : tensor<5xbf16>
}

// CHECK-LABEL: func.func @scatter_4
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor<{{.+}}>, %[[arg2:.+]]:
//       CHECK:     %[[v0:.+]] = kernel.scatter updates(%[[arg2]] : tensor<2xbf16>)
//   CHECK-SAME:       into(%[[arg0]] : tensor<5xbf16>)
//   CHECK-SAME:       at(%[[arg1]] : tensor<2x1xi32>)
//       CHECK:     ^bb0(%[[arg3:.+]]: bf16, %[[arg4:.+]]: bf16):
//       CHECK:       kernel.yield %[[arg4]] : bf16
//       CHECK:      index_vector_dim = 1 : i64,
//   CHECK-SAME:      inserted_window_dims = array<i64: 0>,
//   CHECK-SAME:      scatter_dims_to_operand_dims = array<i64: 0>,
//   CHECK-SAME:      update_window_dims = array<i64>
//       CHECK:     return %[[v0]]


// -----


func.func @scatter_11(%arg0: tensor<2x5xf16>, %arg1: tensor<2x2x1xi32>, %arg2: tensor<2x2xf16>)
    -> (tensor<2x5xf16>) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2)
    <{scatter_dimension_numbers = #stablehlo.scatter<
        inserted_window_dims = [1],
        input_batching_dims = [0],
        scatter_indices_batching_dims = [0],
        scatter_dims_to_operand_dims = [1],
        index_vector_dim = 2>}> ({
  ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):
    stablehlo.return %arg4 : tensor<f16>
  }) : (tensor<2x5xf16>, tensor<2x2x1xi32>, tensor<2x2xf16>) -> tensor<2x5xf16>
  return %0 : tensor<2x5xf16>
}

// CHECK-LABEL: func.func @scatter_11
//  CHECK-SAME: (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor<{{.+}}>, %[[arg2:.+]]:
//       CHECK:     %[[v0:.+]] = kernel.scatter updates(%[[arg2]] : tensor<2x2xf16>)
//  CHECK-SAME:       into(%[[arg0]] : tensor<2x5xf16>)
//  CHECK-SAME:       at(%[[arg1]] : tensor<2x2x1xi32>)
//       CHECK:     ^bb0(%[[arg3:.+]]: f16, %[[arg4:.+]]: f16):
//       CHECK:       kernel.yield %[[arg4]] : f16
//       CHECK:      index_vector_dim = 2 : i64,
//  CHECK-SAME:      input_batching_dims = array<i64: 0>,
//  CHECK-SAME:      inserted_window_dims = array<i64: 1>,
//  CHECK-SAME:      scatter_dims_to_operand_dims = array<i64: 1>,
//  CHECK-SAME:      scatter_indices_batching_dims = array<i64: 0>,
//  CHECK-SAME:      update_window_dims = array<i64>
//       CHECK:     return %[[v0]]

// -----


func.func @scatter_add_0(%arg0: tensor<10x5xf16>, %arg1: tensor<3x1xi32>,
      %arg2: tensor<3x3xf16>)
      -> (tensor<10x5xf16>) {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2)
    <{scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [1],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1>}> ({
  ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<f16>
    stablehlo.return %1 : tensor<f16>
  }) : (tensor<10x5xf16>, tensor<3x1xi32>, tensor<3x3xf16>) -> tensor<10x5xf16>
  return %0 : tensor<10x5xf16>
}

// CHECK-LABEL: func.func @scatter_add_0
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x5xf16>, %[[arg1:.+]]: tensor<3x1xi32>, %[[arg2:.+]]:
//       CHECK:     %[[v0:.+]] = kernel.scatter updates(%[[arg2]] : tensor<3x3xf16>)
//  CHECK-SAME:       into(%[[arg0]] : tensor<10x5xf16>)
//  CHECK-SAME:       at(%[[arg1]] : tensor<3x1xi32>)
//       CHECK:     ^bb0(%[[arg3:.+]]: f16, %[[arg4:.+]]: f16):
//       CHECK:       %[[v1:.+]] = arith.addf %[[arg3]], %[[arg4]] : f16
//       CHECK:       kernel.yield %[[v1]] : f16
//       CHECK:      index_vector_dim = 1 : i64,
//  CHECK-SAME:      inserted_window_dims = array<i64: 0>,
//  CHECK-SAME:      scatter_dims_to_operand_dims = array<i64: 0>,
//  CHECK-SAME:      update_window_dims = array<i64: 1>
//       CHECK:     return %[[v0]] : tensor<10x5xf16>

// -----

func.func @scatter_apply_0(%arg0: tensor<2x3x10xf16>, %arg1: tensor<3x2x1xi32>, %arg2: tensor<3x2x3xf16>)
        -> tensor<2x3x10xf16> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2)
    <{scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [2],
        input_batching_dims = [0, 1],
        scatter_indices_batching_dims = [1, 0],
        scatter_dims_to_operand_dims = [2],
        index_vector_dim = 2>}>
  ({
  ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):
    %1 = stablehlo.sine %arg3 : tensor<f16>
    stablehlo.return %1 : tensor<f16>
  }) : (tensor<2x3x10xf16>, tensor<3x2x1xi32>, tensor<3x2x3xf16>) -> tensor<2x3x10xf16>
  return %0 : tensor<2x3x10xf16>
}

// CHECK-LABEL: func.func @scatter_apply_0
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x10xf16>, %[[arg1:.+]]: tensor<3x2x1xi32>, %[[arg2:.+]]: tensor<
//       CHECK:     %[[v0:.+]] = kernel.scatter updates(%[[arg2]] : tensor<3x2x3xf16>)
//  CHECK-SAME:         into(%[[arg0]] : tensor<2x3x10xf16>)
//  CHECK-SAME:         at(%[[arg1]] : tensor<3x2x1xi32>)
//       CHECK:     ^bb0(%[[arg3:.+]]: f16, %[[arg4:.+]]: f16):
//       CHECK:       %[[v1:.+]] = math.sin %[[arg3]] : f16
//       CHECK:       kernel.yield %[[v1]] : f16
//       CHECK:      index_vector_dim = 2 : i64,
//  CHECK-SAME:      input_batching_dims = array<i64: 0, 1>,
//  CHECK-SAME:      scatter_dims_to_operand_dims = array<i64: 2>,
//  CHECK-SAME:      scatter_indices_batching_dims = array<i64: 1, 0>,
//  CHECK-SAME:      update_window_dims = array<i64: 2>
//       CHECK:     return %[[v0]] : tensor<2x3x10xf16>

// -----

func.func @scatter_unique_indices(%arg0: tensor<3x2xi32>, %arg1: tensor<2xi32>,
      %arg2: tensor<1xi32>)
    -> (tensor<3x2xi32>) {
  %0 = "stablehlo.scatter"(%arg0, %arg2, %arg1) <{
    indices_are_sorted = true,
    scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [0],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0]>,
        unique_indices = true}> ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    stablehlo.return %arg4 : tensor<i32>
  }) : (tensor<3x2xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  return %0 : tensor<3x2xi32>
}

// CHECK-LABEL: func.func @scatter_unique_indices
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x2xi32>, %[[arg1:.+]]: tensor<2xi32>, %[[arg2:.+]]:
//       CHECK:     %[[v0:.+]] = kernel.scatter updates(%[[arg1]] : tensor<2xi32>)
//  CHECK-SAME:       into(%[[arg0]] : tensor<3x2xi32>)
//  CHECK-SAME:       at(%[[arg2]] : tensor<1xi32>)
//       CHECK:     ^bb0(%[[arg3:.+]]: i32, %[[arg4:.+]]: i32):
//       CHECK:       kernel.yield %[[arg4]] : i32
//       CHECK:      index_vector_dim = 0 : i64,
//  CHECK-SAME:      indices_are_sorted,
//  CHECK-SAME:      inserted_window_dims = array<i64: 0>,
//  CHECK-SAME:      scatter_dims_to_operand_dims = array<i64: 0>,
//  CHECK-SAME:      unique_indices,
//  CHECK-SAME:      update_window_dims = array<i64: 0>
//       CHECK:     return %[[v0]] : tensor<3x2xi32>

// -----

func.func @unsupported_type(%input_tensor: tensor<200x100x300x!quant.uniform<i8:f32, 2.000000e+00:15>>,
    %scatter_indices: tensor<10x2xi16>, %updates: tensor<10x300x!quant.uniform<i8:f32, 2.000000e+00:15>>) ->
      tensor<200x100x300x!quant.uniform<i16:f32, 2.000000e+00:15>> {
  %0 = "stablehlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>, %rhs: tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>):
    %add = stablehlo.add %lhs, %rhs : tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>
    "stablehlo.return"(%add) : (tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300x!quant.uniform<i8:f32, 2.000000e+00:15>>, tensor<10x2xi16>,
      tensor<10x300x!quant.uniform<i8:f32, 2.000000e+00:15>>) ->
      tensor<200x100x300x!quant.uniform<i16:f32, 2.000000e+00:15>>
  func.return %0 : tensor<200x100x300x!quant.uniform<i16:f32, 2.000000e+00:15>>
}

// CHECK-LABEL: func.func @unsupported_type
//       CHECK:   stablehlo.scatter

// -----

func.func @unsupported_promotion(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf64> {
  %0 = "stablehlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f64>, %rhs: tensor<f64>):
    %add = stablehlo.add %lhs, %rhs : tensor<f64>
    "stablehlo.return"(%add) : (tensor<f64>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf64>
  func.return %0 : tensor<200x100x300xf64>
}

// CHECK-LABEL: func.func @unsupported_promotion
//       CHECK:   stablehlo.scatter

// -----

func.func @result_requires_shape_cast(%arg0: tensor<3x2xi32>,
      %arg1: tensor<2xi32>,
      %arg2: tensor<1xi32>)
    -> (tensor<?x?xi32>) {
  %0 = "stablehlo.scatter"(%arg0, %arg2, %arg1) <{
    indices_are_sorted = true,
    scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [0],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0]>,
        unique_indices = true}> ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    stablehlo.return %arg4 : tensor<i32>
  }) : (tensor<3x2xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// CHECK-LABEL: func.func @result_requires_shape_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x2xi32>, %[[arg1:.+]]: tensor<2xi32>, %[[arg2:.+]]:
//       CHECK:     %[[v0:.+]] = kernel.scatter updates(%[[arg1]] : tensor<2xi32>)
//  CHECK-SAME:       into(%[[arg0]] : tensor<3x2xi32>)
//  CHECK-SAME:       at(%[[arg2]] : tensor<1xi32>)
//       CHECK:     ^bb0(%[[arg3:.+]]: i32, %[[arg4:.+]]: i32):
//       CHECK:       kernel.yield %[[arg4]] : i32
//       CHECK:      index_vector_dim = 0 : i64,
//  CHECK-SAME:      indices_are_sorted,
//  CHECK-SAME:      inserted_window_dims = array<i64: 0>,
//  CHECK-SAME:      scatter_dims_to_operand_dims = array<i64: 0>,
//  CHECK-SAME:      unique_indices,
//  CHECK-SAME:      update_window_dims = array<i64: 0>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[v0]] : tensor<3x2xi32> to tensor<?x?xi32>
//       CHECK:     return %[[cast]]
