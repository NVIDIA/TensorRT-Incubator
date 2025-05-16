// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-canonicalize-dot-general | FileCheck %s

func.func @dot_general_flatten_outer_dims(%arg0: tensor<1x1x32x32x256xf32>,
                                                %arg1: tensor<1x8x256xf32>) -> tensor<1x1x32x32x8xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
    batching_dims = [0] x [0],
    contracting_dims = [4] x [2],
    precision = [DEFAULT, DEFAULT]
      : (tensor<1x1x32x32x256xf32>, tensor<1x8x256xf32>) -> tensor<1x1x32x32x8xf32>
  return %0 : tensor<1x1x32x32x8xf32>
}

// CHECK-LABEL: func.func @dot_general
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x1x32x32x256xf32>, %[[arg1:.+]]: tensor<1x8x256xf32>) -> tensor<1x1x32x32x8xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 1, 2, 3, 4] : (tensor<1x1x32x32x256xf32>) -> tensor<1x1x32x32x256xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<1x1x32x32x256xf32>) -> tensor<1x1024x256xf32>
//       CHECK:     %[[v2:.+]] = stablehlo.transpose %[[arg1]], dims = [0, 1, 2] : (tensor<1x8x256xf32>) -> tensor<1x8x256xf32>
//       CHECK:     %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<1x8x256xf32>) -> tensor<1x8x256xf32>
//       CHECK:     %[[v4:.+]] = stablehlo.dot_general %[[v1]], %[[v3]], batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x1024x256xf32>, tensor<1x8x256xf32>) -> tensor<1x1024x8xf32>
//       CHECK:     %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<1x1024x8xf32>) -> tensor<1x1x32x32x8xf32>
//       CHECK:     return %[[v5]] : tensor<1x1x32x32x8xf32>

// -----

// TODO: handle this case

func.func @dot_general_flatten_outer_dims_input_erased(%arg0: tensor<1x?x?x?x256xf32>,
                                          %arg1: tensor<1x8x256xf32>) -> tensor<1x1x32x32x8xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
    batching_dims = [0] x [0],
    contracting_dims = [4] x [2],
    precision = [DEFAULT, DEFAULT]
      : (tensor<1x?x?x?x256xf32>, tensor<1x8x256xf32>) -> tensor<1x1x32x32x8xf32>
  return %0 : tensor<1x1x32x32x8xf32>
}

// CHECK-LABEL: func.func @dot_general_flatten_outer_dims_input_erased
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.dot_general {{.*}} -> tensor<1x1x32x32x8xf32>
//  CHECK-NEXT:     return %[[v0]] :

// -----

func.func @dot_general_flatten_outer_dims_type_erased(%arg0: tensor<1x1x32x32x256xf32>,
                                                      %arg1: tensor<1x8x256xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
    batching_dims = [0] x [0],
    contracting_dims = [4] x [2],
    precision = [DEFAULT, DEFAULT]
      : (tensor<1x1x32x32x256xf32>, tensor<1x8x256xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}

// CHECK-LABEL: func.func @dot_general_flatten_outer_dims_type_erased
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.dot_general {{.*}} -> tensor<?x?x?x?x?xf32>
//  CHECK-NEXT:     return %[[v0]] :

// -----

func.func @dot_general_to_mul(%arg0: tensor<12x1x1x1xf32>, %arg1: tensor<12xf32>) -> tensor<12x1x1x1xf32> {
  %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<12xf32>, tensor<12x1x1x1xf32>) -> tensor<12x1x1x1xf32>
  return %0 : tensor<12x1x1x1xf32>
}

// CHECK-LABEL: func.func @dot_general_to_mul
//  CHECK-SAME: (%[[arg0:.+]]: tensor<12x1x1x1xf32>, %[[arg1:.+]]: tensor<12xf32>) -> tensor<12x1x1x1xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.broadcast_in_dim %[[arg1]], dims = [0] : (tensor<12xf32>) -> tensor<12x1x1x1xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.multiply %[[v0]], %[[arg0]] : tensor<12x1x1x1xf32>
//       CHECK:     return %[[v1]] : tensor<12x1x1x1xf32>

// -----

!lhs_type = tensor<12x1x1x1xcomplex<f32>>
!rhs_type = tensor<12xcomplex<f32>>
!result_type = tensor<12x1x1x1xcomplex<f64>>

func.func @complex_dot_general_to_mul(%arg0: !lhs_type,
                                      %arg1: !rhs_type) -> !result_type {
  %0 = stablehlo.dot_general %arg1, %arg0,
    batching_dims = [0] x [0],
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (!rhs_type, !lhs_type) -> !result_type
  return %0 : !result_type
}

// CHECK-LABEL: func.func @complex_dot_general_to_mul
//  CHECK-SAME: (%[[arg0:.+]]: tensor<12x1x1x1xcomplex<f32>>, %[[arg1:.+]]: tensor<12xcomplex<f32>>)
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.broadcast_in_dim %[[arg1]], {{.*}} -> tensor<12x1x1x1xcomplex<f32>>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.convert %[[v0]] : (tensor<12x1x1x1xcomplex<f32>>) -> tensor<12x1x1x1xcomplex<f64>>
//   CHECK-DAG:     %[[v2:.+]] = stablehlo.convert %[[arg0]] : (tensor<12x1x1x1xcomplex<f32>>) -> tensor<12x1x1x1xcomplex<f64>>
//   CHECK-DAG:     %[[v3:.+]] = stablehlo.multiply %[[v1]], %[[v2]]
//       CHECK:     return %[[v3]]

// -----

// TODO: this can be converted to mul without broadcast, then we can handle the dynamic shapes.
// We only can't handle this right now since broadcast requires dynamic shape.

func.func @dot_general_to_mul_batch_dynamic(%arg0: tensor<?x1x1x1xf32>, %arg1: tensor<?xf32>) -> tensor<?x1x1x1xf32> {
  %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<?xf32>, tensor<?x1x1x1xf32>) -> tensor<?x1x1x1xf32>
  return %0 : tensor<?x1x1x1xf32>
}

// CHECK-LABEL: func.func @dot_general_to_mul_batch_dynamic
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.dot_general {{.*}} -> tensor<?x1x1x1xf32>
//  CHECK-NEXT:     return %[[v0]] : tensor<?x1x1x1xf32>

// -----

// TODO: this can be converted to mul without broadcast, then we can handle the dynamic shapes.
// We only can't handle this right now since broadcast requires dynamic shape.

func.func @dot_general_to_mul_inputs_only_dynamic(%arg0: tensor<?x1x1x1xf32>, %arg1: tensor<?xf32>) -> tensor<12x1x1x1xf32> {
  %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<?xf32>, tensor<?x1x1x1xf32>) -> tensor<12x1x1x1xf32>
  return %0 : tensor<12x1x1x1xf32>
}

// CHECK-LABEL: func.func @dot_general_to_mul_inputs_only_dynamic
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.dot_general {{.*}} -> tensor<12x1x1x1xf32>
//  CHECK-NEXT:     return %[[v0]] : tensor<12x1x1x1xf32>

// -----

func.func @dot_general_to_dims_erased(%arg0: tensor<12x1x1x1xf32>, %arg1: tensor<12xf32>) -> tensor<?x?x?x?xf32> {
  %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<12xf32>, tensor<12x1x1x1xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func.func @dot_general_to_dims_erased
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.dot_general {{.*}} -> tensor<?x?x?x?xf32>
//  CHECK-NEXT:     return %[[v0]] : tensor<?x?x?x?xf32>

// -----

func.func @dot_general_to_mul2(%arg0: tensor<10x12x10x10xf32>, %arg1: tensor<12x10xf32>) -> tensor<12x10x10x10xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [1, 2] x [0, 1], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<10x12x10x10xf32>, tensor<12x10xf32>) -> tensor<12x10x10x10xf32>
  return %0 : tensor<12x10x10x10xf32>
}

// CHECK-LABEL: func.func @dot_general_to_mul2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x12x10x10xf32>, %[[arg1:.+]]: tensor<12x10xf32>) -> tensor<12x10x10x10xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [1, 2, 0, 3] : (tensor<10x12x10x10xf32>) -> tensor<12x10x10x10xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.broadcast_in_dim %[[arg1]], dims = [0, 1] : (tensor<12x10xf32>) -> tensor<12x10x10x10xf32>
//       CHECK:     %[[v2:.+]] = stablehlo.multiply %[[v0]], %[[v1]] : tensor<12x10x10x10xf32>
//       CHECK:     return %[[v2]] : tensor<12x10x10x10xf32>

// -----

func.func @dot_general_to_mul3(%arg0: tensor<12x10x10x10xf32>, %arg1: tensor<12x10x1x1xf32>) -> tensor<12x10x10x10x1x1xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<12x10x10x10xf32>, tensor<12x10x1x1xf32>) -> tensor<12x10x10x10x1x1xf32>
  return %0 : tensor<12x10x10x10x1x1xf32>
}

// CHECK-LABEL: func.func @dot_general_to_mul3
//  CHECK-SAME: (%[[arg0:.+]]: tensor<12x10x10x10xf32>, %[[arg1:.+]]: tensor<12x10x1x1xf32>) -> tensor<12x10x10x10x1x1xf32> {
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.broadcast_in_dim %[[arg0]], dims = [0, 1, 2, 3] : (tensor<12x10x10x10xf32>) -> tensor<12x10x10x10x1x1xf32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.broadcast_in_dim %[[arg1]], dims = [0, 1, 4, 5] : (tensor<12x10x1x1xf32>) -> tensor<12x10x10x10x1x1xf32>
//   CHECK-DAG:     %[[v2:.+]] = stablehlo.multiply %[[v0]], %[[v1]] : tensor<12x10x10x10x1x1xf32>
//       CHECK:     return %[[v2]] : tensor<12x10x10x10x1x1xf32>

// -----

func.func @dot_general_to_mul4(%arg0: tensor<12x10x10x10xf32>, %arg1: tensor<10x12x1x1xf32>) -> tensor<12x10x10x10x1x1xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [1, 0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<12x10x10x10xf32>, tensor<10x12x1x1xf32>) -> tensor<12x10x10x10x1x1xf32>
  return %0 : tensor<12x10x10x10x1x1xf32>
}

// CHECK-LABEL: func.func @dot_general_to_mul4
//  CHECK-SAME: (%[[arg0:.+]]: tensor<12x10x10x10xf32>, %[[arg1:.+]]: tensor<10x12x1x1xf32>) -> tensor<12x10x10x10x1x1xf32> {
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.transpose %[[arg1]], dims = [1, 0, 2, 3] : (tensor<10x12x1x1xf32>) -> tensor<12x10x1x1xf32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.broadcast_in_dim %[[arg0]], dims = [0, 1, 2, 3] : (tensor<12x10x10x10xf32>) -> tensor<12x10x10x10x1x1xf32>
//   CHECK-DAG:     %[[v2:.+]] = stablehlo.broadcast_in_dim %[[v0]], dims = [0, 1, 4, 5] : (tensor<12x10x1x1xf32>) -> tensor<12x10x10x10x1x1xf32>
//   CHECK-DAG:     %[[v3:.+]] = stablehlo.multiply %[[v1]], %[[v2]] : tensor<12x10x10x10x1x1xf32>
//       CHECK:     return %[[v3]] : tensor<12x10x10x10x1x1xf32>

// -----

func.func @dot_general_batch_dims_front(%arg0: tensor<2x64x1024x12xf32>, %arg1: tensor<2x12x1x1024xf32>)
     -> tensor<2x12x64x1xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
    lhs_batching_dimensions = [0, 3],
    rhs_batching_dimensions = [0, 1],
    lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [3]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}
    : (tensor<2x64x1024x12xf32>, tensor<2x12x1x1024xf32>) -> tensor<2x12x64x1xf32>
  return %0 : tensor<2x12x64x1xf32>
}

// CHECK-LABEL: func.func @dot_general_batch_dims_front
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x64x1024x12xf32>, %[[arg1:.+]]: tensor<2x12x1x1024xf32>) -> tensor<2x12x64x1xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 3, 1, 2] : (tensor<2x64x1024x12xf32>) -> tensor<2x12x64x1024xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.dot_general %[[v0]], %[[arg1]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x12x64x1024xf32>, tensor<2x12x1x1024xf32>) -> tensor<2x12x64x1xf32>
//       CHECK:     return %[[v1]] : tensor<2x12x64x1xf32>

// -----

func.func @dot_general_batch_dims_front_input_dims_erased(%arg0: tensor<?x?x1024x12xf32>, %arg1: tensor<2x12x1x1024xf32>)
     -> tensor<2x12x64x1xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
    lhs_batching_dimensions = [0, 3],
    rhs_batching_dimensions = [0, 1],
    lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [3]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}
    : (tensor<?x?x1024x12xf32>, tensor<2x12x1x1024xf32>) -> tensor<2x12x64x1xf32>
  return %0 : tensor<2x12x64x1xf32>
}

// CHECK-LABEL: func.func @dot_general_batch_dims_front_input_dims_erased
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x1024x12xf32>, %[[arg1:.+]]: tensor<2x12x1x1024xf32>) -> tensor<2x12x64x1xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 3, 1, 2] : (tensor<?x?x1024x12xf32>) -> tensor<?x12x?x1024xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.dot_general %[[v0]], %[[arg1]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<?x12x?x1024xf32>, tensor<2x12x1x1024xf32>) -> tensor<2x12x64x1xf32>
//       CHECK:     return %[[v1]] : tensor<2x12x64x1xf32>

// -----

func.func @dot_general_batch_dims_front_dims_erased(%arg0: tensor<2x64x1024x12xf32>, %arg1: tensor<2x12x1x1024xf32>)
     -> tensor<?x?x?x?xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
    lhs_batching_dimensions = [0, 3],
    rhs_batching_dimensions = [0, 1],
    lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [3]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}
    : (tensor<2x64x1024x12xf32>, tensor<2x12x1x1024xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func.func @dot_general_batch_dims_front_dims_erased
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.dot_general {{.*}} -> tensor<?x?x?x?xf32>
//  CHECK-NEXT:     return %[[v0]] : tensor<?x?x?x?xf32>

// -----

func.func @contracting_collapse_1(%arg0: tensor<2x3x5x4xf32>, %arg1: tensor<2x4x5x3xf32>) -> tensor<2x3x3xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
    batching_dims = [0] x [0],
    contracting_dims = [3, 2] x [1, 2],
    precision = [DEFAULT, DEFAULT] : (tensor<2x3x5x4xf32>, tensor<2x4x5x3xf32>) -> tensor<2x3x3xf32>
  return %0: tensor<2x3x3xf32>
}

// CHECK-LABEL: contracting_collapse_1
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x5x4xf32>, %[[arg1:.+]]: tensor<2x4x5x3xf32>
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 1, 3, 2] : (tensor<2x3x5x4xf32>) -> tensor<2x3x4x5xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<2x3x4x5xf32>) -> tensor<2x3x20xf32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.transpose %[[arg1]], dims = [0, 3, 1, 2] : (tensor<2x4x5x3xf32>) -> tensor<2x3x4x5xf32>
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<2x3x4x5xf32>) -> tensor<2x3x20xf32>
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.dot_general %[[v1]], %[[v3]], batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x3x20xf32>, tensor<2x3x20xf32>) -> tensor<2x3x3xf32>
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
//  CHECK-NEXT: return %[[v5]] : tensor<2x3x3xf32>

// -----

func.func @contracting_collapse_2(%arg0: tensor<2x3x5xf32>, %arg1: tensor<3x2x4xf32>) -> tensor<5x4xf32>{
  %0 = stablehlo.dot_general %arg0, %arg1,
    contracting_dims = [0, 1] x [1, 0],
    precision = [DEFAULT, DEFAULT] : (tensor<2x3x5xf32>, tensor<3x2x4xf32>) -> tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: contracting_collapse_2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x5xf32>, %[[arg1:.+]]: tensor<3x2x4xf32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [2, 0, 1] : (tensor<2x3x5xf32>) -> tensor<5x2x3xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<5x2x3xf32>) -> tensor<5x6xf32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.transpose %[[arg1]], dims = [2, 1, 0] : (tensor<3x2x4xf32>) -> tensor<4x2x3xf32>
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<4x2x3xf32>) -> tensor<4x6xf32>
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.dot_general %[[v1]], %[[v3]], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x6xf32>, tensor<4x6xf32>) -> tensor<5x4xf32>
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<5x4xf32>) -> tensor<5x4xf32>
//  CHECK-NEXT: return %[[v5]] : tensor<5x4xf32>

// -----

func.func @contracting_collapse_3(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<4x5x6x2xf32>) -> tensor<3x6xf32>{
  %0 = stablehlo.dot_general %arg0, %arg1,
    contracting_dims = [0, 2, 3] x [3, 0, 1],
    precision = [DEFAULT, DEFAULT] : (tensor<2x3x4x5xf32>, tensor<4x5x6x2xf32>) -> tensor<3x6xf32>
  return %0 : tensor<3x6xf32>
}

// CHECK-LABEL: contracting_collapse_3
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x4x5xf32>, %[[arg1:.+]]: tensor<4x5x6x2xf32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [1, 0, 2, 3] : (tensor<2x3x4x5xf32>) -> tensor<3x2x4x5xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<3x2x4x5xf32>) -> tensor<3x40xf32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.transpose %[[arg1]], dims = [2, 3, 0, 1] : (tensor<4x5x6x2xf32>) -> tensor<6x2x4x5xf32>
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<6x2x4x5xf32>) -> tensor<6x40xf32>
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.dot_general %[[v1]], %[[v3]], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x40xf32>, tensor<6x40xf32>) -> tensor<3x6xf32>
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<3x6xf32>) -> tensor<3x6xf32>
//  CHECK-NEXT: return %[[v5]] : tensor<3x6xf32>

// -----

func.func @contracting_collapse_4(%arg0: tensor<2x2x3x4x2xf32>, %arg1: tensor<2x2x4x3x5xf32>) -> tensor<2x2x2x5xf32>{
  %0 = stablehlo.dot_general %arg0, %arg1,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [2, 3] x [3, 2],
    precision = [DEFAULT, DEFAULT] : (tensor<2x2x3x4x2xf32>, tensor<2x2x4x3x5xf32>) -> tensor<2x2x2x5xf32>
  return %0 : tensor<2x2x2x5xf32>
}

// CHECK-LABEL: contracting_collapse_4
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x2x3x4x2xf32>, %[[arg1:.+]]: tensor<2x2x4x3x5xf32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 1, 4, 2, 3] : (tensor<2x2x3x4x2xf32>) -> tensor<2x2x2x3x4xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<2x2x2x3x4xf32>) -> tensor<2x2x2x12xf32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.transpose %[[arg1]], dims = [0, 1, 4, 3, 2] : (tensor<2x2x4x3x5xf32>) -> tensor<2x2x5x3x4xf32>
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<2x2x5x3x4xf32>) -> tensor<2x2x5x12xf32>
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.dot_general %[[v1]], %[[v3]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x2x2x12xf32>, tensor<2x2x5x12xf32>) -> tensor<2x2x2x5xf32>
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<2x2x2x5xf32>) -> tensor<2x2x2x5xf32>
//  CHECK-NEXT: return %[[v5]] : tensor<2x2x2x5xf32>

// -----

// TODO: Add support for this case.
func.func @contracting_collapse_5(%arg0: tensor<2x?x?x4x2xf32>, %arg1: tensor<2x?x4x3x5xf32>) -> tensor<2x2x2x5xf32>{
  %0 = stablehlo.dot_general %arg0, %arg1,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [2, 3] x [3, 2],
    precision = [DEFAULT, DEFAULT] : (tensor<2x?x?x4x2xf32>, tensor<2x?x4x3x5xf32>) -> tensor<2x2x2x5xf32>
  return %0 : tensor<2x2x2x5xf32>
}

// CHECK-LABEL: contracting_collapse_5
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x?x?x4x2xf32>, %[[arg1:.+]]: tensor<2x?x4x3x5xf32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.dot_general %[[arg0]], %[[arg1]], batching_dims = [0, 1] x [0, 1], contracting_dims = [2, 3] x [3, 2], precision = [DEFAULT, DEFAULT] : (tensor<2x?x?x4x2xf32>, tensor<2x?x4x3x5xf32>) -> tensor<2x2x2x5xf32>
//  CHECK-NEXT: return %[[v0]] : tensor<2x2x2x5xf32>

// -----

// TODO: Add support for this case.
func.func @contracting_collapse_6(%arg0: tensor<2x2x3x4x2xf32>, %arg1: tensor<2x2x4x3x5xf32>) -> tensor<?x?x?x?xf32>{
  %0 = stablehlo.dot_general %arg0, %arg1,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [2, 3] x [3, 2],
    precision = [DEFAULT, DEFAULT] : (tensor<2x2x3x4x2xf32>, tensor<2x2x4x3x5xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: contracting_collapse_6
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x2x3x4x2xf32>, %[[arg1:.+]]: tensor<2x2x4x3x5xf32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.dot_general %[[arg0]], %[[arg1]], batching_dims = [0, 1] x [0, 1], contracting_dims = [2, 3] x [3, 2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x3x4x2xf32>, tensor<2x2x4x3x5xf32>) -> tensor<?x?x?x?xf32>
//  CHECK-NEXT: return %[[v0]] : tensor<?x?x?x?xf32>

// -----

func.func public @contracting_collapse_to_vec_mat(%arg0: tensor<3x4xf32>, %arg1: tensor<1x4x3xf32>) -> (tensor<1xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg1,
    contracting_dims = [0, 1] x [2, 1],
    precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<1x4x3xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
}

// CHECK-LABEL: contracting_collapse_to_vec_mat
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x4xf32>, %[[arg1:.+]]: tensor<1x4x3xf32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 1] : (tensor<3x4xf32>) -> tensor<3x4xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<3x4xf32>) -> tensor<1x12xf32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.transpose %[[arg1]], dims = [0, 2, 1] : (tensor<1x4x3xf32>) -> tensor<1x3x4xf32>
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<1x3x4xf32>) -> tensor<1x12xf32>
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.dot_general %[[v1]], %[[v3]], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x12xf32>, tensor<1x12xf32>) -> tensor<1x1xf32>
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<1x1xf32>) -> tensor<1xf32>
//  CHECK-NEXT: return %[[v5]] : tensor<1xf32>

// -----

func.func public @contracting_collapse_to_mat_vec(%arg0: tensor<1x4x3xf32>, %arg1: tensor<3x4xf32>) -> (tensor<1xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg1,
    contracting_dims = [2, 1] x [0, 1],
    precision = [DEFAULT, DEFAULT] : (tensor<1x4x3xf32>, tensor<3x4xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
}

// CHECK-LABEL: contracting_collapse_to_mat_vec
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x4x3xf32>, %[[arg1:.+]]: tensor<3x4xf32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 2, 1] : (tensor<1x4x3xf32>) -> tensor<1x3x4xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<1x3x4xf32>) -> tensor<1x12xf32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.transpose %[[arg1]], dims = [0, 1] : (tensor<3x4xf32>) -> tensor<3x4xf32>
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<3x4xf32>) -> tensor<1x12xf32>
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.dot_general %[[v1]], %[[v3]], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x12xf32>, tensor<1x12xf32>) -> tensor<1x1xf32>
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<1x1xf32>) -> tensor<1xf32>
//  CHECK-NEXT: return %[[v5]] : tensor<1xf32>

// -----

func.func public @contracting_collapse_to_vec_vec(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x4x3xf32>) -> (tensor<f32>) {
    %0 = stablehlo.dot_general %arg0, %arg1,
    contracting_dims = [0, 1, 2] x [0, 2, 1],
    precision = [DEFAULT, DEFAULT] : (tensor<2x3x4xf32>, tensor<2x4x3xf32>) -> tensor<f32>
    return %0 : tensor<f32>
}

// CHECK-LABEL: contracting_collapse_to_vec_vec
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x3x4xf32>, %[[arg1:.+]]: tensor<2x4x3xf32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.transpose %[[arg0]],  dims = [0, 1, 2] : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<2x3x4xf32>) -> tensor<1x24xf32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.transpose %[[arg1]], dims = [0, 2, 1] : (tensor<2x4x3xf32>) -> tensor<2x3x4xf32>
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<2x3x4xf32>) -> tensor<1x24xf32>
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.dot_general %[[v1]], %[[v3]], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x24xf32>, tensor<1x24xf32>) -> tensor<1x1xf32>
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<1x1xf32>) -> tensor<f32>
//  CHECK-NEXT: return %[[v5]] : tensor<f32>

// -----

func.func public @contracting_collapse_to_batch_vec_vec(%arg0: tensor<2x2x3x4x2xf32>, %arg1: tensor<2x2x2x4x3xf32>) -> (tensor<2x2xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg1,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [2, 4, 3] x [4, 2, 3],
    precision = [DEFAULT, DEFAULT] : (tensor<2x2x3x4x2xf32>, tensor<2x2x2x4x3xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: contracting_collapse_to_batch_vec_vec
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x2x3x4x2xf32>, %[[arg1:.+]]: tensor<2x2x2x4x3xf32>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 1, 2, 4, 3] : (tensor<2x2x3x4x2xf32>) -> tensor<2x2x3x2x4xf32>
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<2x2x3x2x4xf32>) -> tensor<2x2x1x24xf32>
//  CHECK-NEXT: %[[v2:.+]] = stablehlo.transpose %[[arg1]], dims = [0, 1, 4, 2, 3] : (tensor<2x2x2x4x3xf32>) -> tensor<2x2x3x2x4xf32>
//  CHECK-NEXT: %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<2x2x3x2x4xf32>) -> tensor<2x2x1x24xf32>
//  CHECK-NEXT: %[[v4:.+]] = stablehlo.dot_general %[[v1]], %[[v3]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x2x1x24xf32>, tensor<2x2x1x24xf32>) -> tensor<2x2x1x1xf32>
//  CHECK-NEXT: %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<2x2x1x1xf32>) -> tensor<2x2xf32>
//  CHECK-NEXT: return %[[v5]] : tensor<2x2xf32>

// -----

func.func @hlo_dot_general3(%arg0: tensor<32x49x32xf32>, %arg1: tensor<32x1x32x49xf32>) -> tensor<32x49x1x49xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<32x49x32xf32>, tensor<32x1x32x49xf32>) -> tensor<32x49x1x49xf32>
  return %0 : tensor<32x49x1x49xf32>
}

// CHECK-LABEL: func.func @hlo_dot_general3
//  CHECK-SAME: (%[[arg0:.+]]: tensor<32x49x32xf32>, %[[arg1:.+]]: tensor<32x1x32x49xf32>)
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 1, 2] : (tensor<32x49x32xf32>) -> tensor<32x49x32xf32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<32x49x32xf32>) -> tensor<32x49x32xf32>
//   CHECK-DAG:     %[[v2:.+]] = stablehlo.transpose %[[arg1]], dims = [0, 1, 3, 2] : (tensor<32x1x32x49xf32>) -> tensor<32x1x49x32xf32>
//   CHECK-DAG:     %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<32x1x49x32xf32>) -> tensor<32x49x32xf32>
//   CHECK-DAG:     %[[v4:.+]] = stablehlo.dot_general %[[v1]], %[[v3]], batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<32x49x32xf32>, tensor<32x49x32xf32>) -> tensor<32x49x49xf32>
//   CHECK-DAG:     %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<32x49x49xf32>) -> tensor<32x49x1x49xf32>
//   CHECK-DAG:     return %[[v5]]

// -----

func.func @hlo_dot_general4(%arg0: tensor<32x5x49x32xf32>, %arg1: tensor<32x5x1x32x49xf32>) -> tensor<32x5x49x1x49xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [3]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<32x5x49x32xf32>, tensor<32x5x1x32x49xf32>) -> tensor<32x5x49x1x49xf32>
  return %0 : tensor<32x5x49x1x49xf32>
}

// CHECK-LABEL: func.func @hlo_dot_general4
//  CHECK-SAME: (%[[arg0:.+]]: tensor<32x5x49x32xf32>, %[[arg1:.+]]: tensor<32x5x1x32x49xf32>)
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.transpose %[[arg0]], dims = [0, 1, 2, 3] : (tensor<32x5x49x32xf32>) -> tensor<32x5x49x32xf32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.reshape %[[v0]] : (tensor<32x5x49x32xf32>) -> tensor<32x5x49x32xf32>
//   CHECK-DAG:     %[[v2:.+]] = stablehlo.transpose %[[arg1]], dims = [0, 1, 2, 4, 3] : (tensor<32x5x1x32x49xf32>) -> tensor<32x5x1x49x32xf32>
//   CHECK-DAG:     %[[v3:.+]] = stablehlo.reshape %[[v2]] : (tensor<32x5x1x49x32xf32>) -> tensor<32x5x49x32xf32>
//   CHECK-DAG:     %[[v4:.+]] = stablehlo.dot_general %[[v1]], %[[v3]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<32x5x49x32xf32>, tensor<32x5x49x32xf32>) -> tensor<32x5x49x49xf32>
//   CHECK-DAG:     %[[v5:.+]] = stablehlo.reshape %[[v4]] : (tensor<32x5x49x49xf32>) -> tensor<32x5x49x1x49xf32>
//   CHECK-DAG:     return %[[v5]]
