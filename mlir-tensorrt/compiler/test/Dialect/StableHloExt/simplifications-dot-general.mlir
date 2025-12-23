
// RUN: mlir-tensorrt-opt %s --stablehlo-ext-simplifications -split-input-file | FileCheck %s

func.func @simplify_dot_general_no_contraction_to_mul_needs_transpose_and_broadcast(
  %arg0: tensor<10x12x10x10xf32>,
  %arg1: tensor<12x10xf32>
) -> tensor<12x10x10x10xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
    batching_dims = [1, 2] x [0, 1],
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (tensor<10x12x10x10xf32>, tensor<12x10xf32>) -> tensor<12x10x10x10xf32>
  return %0 : tensor<12x10x10x10xf32>
}

// CHECK-LABEL: func.func @simplify_dot_general_no_contraction_to_mul_needs_transpose_and_broadcast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x12x10x10xf32>, %[[arg1:.+]]: tensor<12x10xf32>) -> tensor<12x10x10x10xf32> {
//       CHECK:     %[[t0:.+]] = stablehlo.transpose %[[arg0]], dims = [1, 2, 0, 3] : (tensor<10x12x10x10xf32>) -> tensor<12x10x10x10xf32>
//       CHECK:     %[[b1:.+]] = stablehlo.broadcast_in_dim %[[arg1]], dims = [0, 1] : (tensor<12x10xf32>) -> tensor<12x10x10x10xf32>
//       CHECK:     %[[m0:.+]] = stablehlo.multiply %[[t0]], %[[b1]] : tensor<12x10x10x10xf32>
//       CHECK:     return %[[m0]] : tensor<12x10x10x10xf32>

// -----

func.func @simplify_dot_general_no_contraction_to_mul_outer_product(
  %lhs: tensor<2x3xf32>,
  %rhs: tensor<5xf32>
) -> tensor<2x3x5xf32> {
  %0 = stablehlo.dot_general %lhs, %rhs,
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (tensor<2x3xf32>, tensor<5xf32>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

// CHECK-LABEL: func.func @simplify_dot_general_no_contraction_to_mul_outer_product
//  CHECK-SAME: (%[[lhs:.+]]: tensor<2x3xf32>, %[[rhs:.+]]: tensor<5xf32>) -> tensor<2x3x5xf32> {
//   CHECK-DAG:     %[[b0:.+]] = stablehlo.broadcast_in_dim %[[lhs]], dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x5xf32>
//   CHECK-DAG:     %[[b1:.+]] = stablehlo.broadcast_in_dim %[[rhs]], dims = [2] : (tensor<5xf32>) -> tensor<2x3x5xf32>
//       CHECK:     %[[m0:.+]] = stablehlo.multiply %[[b0]], %[[b1]] : tensor<2x3x5xf32>
//       CHECK:     return %[[m0]] : tensor<2x3x5xf32>

// -----

// Both sides broadcast into a full outer-product result.
func.func @simplify_dot_general_no_contraction_to_mul_outer_product_2d_2d(
  %lhs: tensor<2x3xf32>,
  %rhs: tensor<4x5xf32>
) -> tensor<2x3x4x5xf32> {
  %0 = stablehlo.dot_general %lhs, %rhs,
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (tensor<2x3xf32>, tensor<4x5xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>
}

// CHECK-LABEL: func.func @simplify_dot_general_no_contraction_to_mul_outer_product_2d_2d
//  CHECK-SAME: (%[[lhs:.+]]: tensor<2x3xf32>, %[[rhs:.+]]: tensor<4x5xf32>) -> tensor<2x3x4x5xf32> {
//   CHECK-DAG:     %[[b0:.+]] = stablehlo.broadcast_in_dim %[[lhs]], dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x4x5xf32>
//   CHECK-DAG:     %[[b1:.+]] = stablehlo.broadcast_in_dim %[[rhs]], dims = [2, 3] : (tensor<4x5xf32>) -> tensor<2x3x4x5xf32>
//       CHECK:     %[[m0:.+]] = stablehlo.multiply %[[b0]], %[[b1]] : tensor<2x3x4x5xf32>
//       CHECK:     return %[[m0]] : tensor<2x3x4x5xf32>

// -----

// No broadcast or transpose required: just a direct multiply.
func.func @simplify_dot_general_no_contraction_to_mul_no_broadcast_no_transpose(
  %lhs: tensor<12x10xf32>,
  %rhs: tensor<12x10xf32>
) -> tensor<12x10xf32> {
  %0 = stablehlo.dot_general %lhs, %rhs,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (tensor<12x10xf32>, tensor<12x10xf32>) -> tensor<12x10xf32>
  return %0 : tensor<12x10xf32>
}

// CHECK-LABEL: func.func @simplify_dot_general_no_contraction_to_mul_no_broadcast_no_transpose
//  CHECK-SAME: (%[[lhs:.+]]: tensor<12x10xf32>, %[[rhs:.+]]: tensor<12x10xf32>) -> tensor<12x10xf32> {
//       CHECK:     %[[m0:.+]] = stablehlo.multiply %[[lhs]], %[[rhs]] : tensor<12x10xf32>
//       CHECK:     return %[[m0]] : tensor<12x10xf32>

// -----

// Broadcast only one side (lhs expands into rhs' trailing non-batch dims).
func.func @simplify_dot_general_no_contraction_to_mul_broadcast_lhs_only(
  %lhs: tensor<12x10xf32>,
  %rhs: tensor<12x10x5xf32>
) -> tensor<12x10x5xf32> {
  %0 = stablehlo.dot_general %lhs, %rhs,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (tensor<12x10xf32>, tensor<12x10x5xf32>) -> tensor<12x10x5xf32>
  return %0 : tensor<12x10x5xf32>
}

// CHECK-LABEL: func.func @simplify_dot_general_no_contraction_to_mul_broadcast_lhs_only
//  CHECK-SAME: (%[[lhs:.+]]: tensor<12x10xf32>, %[[rhs:.+]]: tensor<12x10x5xf32>) -> tensor<12x10x5xf32> {
//       CHECK:     %[[b0:.+]] = stablehlo.broadcast_in_dim %[[lhs]], dims = [0, 1] : (tensor<12x10xf32>) -> tensor<12x10x5xf32>
//       CHECK:     %[[m0:.+]] = stablehlo.multiply %[[b0]], %[[rhs]] : tensor<12x10x5xf32>
//       CHECK:     return %[[m0]] : tensor<12x10x5xf32>

// -----

// Promotes operands to the result element type before multiplying.
func.func @simplify_dot_general_no_contraction_to_mul_promote_to_result_type(
  %lhs: tensor<12x10xf16>,
  %rhs: tensor<12x10xf16>
) -> tensor<12x10xf32> {
  %0 = stablehlo.dot_general %lhs, %rhs,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (tensor<12x10xf16>, tensor<12x10xf16>) -> tensor<12x10xf32>
  return %0 : tensor<12x10xf32>
}

// CHECK-LABEL: func.func @simplify_dot_general_no_contraction_to_mul_promote_to_result_type
//  CHECK-SAME: (%[[lhs:.+]]: tensor<12x10xf16>, %[[rhs:.+]]: tensor<12x10xf16>) -> tensor<12x10xf32> {
//   CHECK-DAG:     %[[c0:.+]] = stablehlo.convert %[[lhs]] : (tensor<12x10xf16>) -> tensor<12x10xf32>
//   CHECK-DAG:     %[[c1:.+]] = stablehlo.convert %[[rhs]] : (tensor<12x10xf16>) -> tensor<12x10xf32>
//       CHECK:     %[[m0:.+]] = stablehlo.multiply %[[c0]], %[[c1]] : tensor<12x10xf32>
//       CHECK:     return %[[m0]] : tensor<12x10xf32>

// -----

// Dynamic batch dim, outer-product in non-batch dims.
func.func @simplify_dot_general_no_contraction_to_mul_dynamic_batch_outer_product(
  %lhs: tensor<?x4xf32>,
  %rhs: tensor<?x5xf32>
) -> tensor<?x4x5xf32> {
  %0 = stablehlo.dot_general %lhs, %rhs,
    batching_dims = [0] x [0],
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (tensor<?x4xf32>, tensor<?x5xf32>) -> tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>
}

// CHECK-LABEL: func.func @simplify_dot_general_no_contraction_to_mul_dynamic_batch_outer_product
//  CHECK-SAME: (%[[lhs:.+]]: tensor<?x4xf32>, %[[rhs:.+]]: tensor<?x5xf32>) -> tensor<?x4x5xf32> {
//   CHECK-DAG:     %[[b0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[lhs]], %{{.*}}, dims = [0, 1] {known_nonexpanding_dimensions = array<i64: 0, 1>} : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<?x4x5xf32>
//   CHECK-DAG:     %[[b1:.+]] = stablehlo.dynamic_broadcast_in_dim %[[rhs]], %{{.*}}, dims = [0, 2] {known_nonexpanding_dimensions = array<i64: 0, 1>} : (tensor<?x5xf32>, tensor<3xi32>) -> tensor<?x4x5xf32>
//       CHECK:     %[[m0:.+]] = stablehlo.multiply %[[b0]], %[[b1]] : tensor<?x4x5xf32>
//       CHECK:     return %[[m0]] : tensor<?x4x5xf32>

// -----

// Dynamic batch dims, no broadcast/transpose required.
func.func @simplify_dot_general_no_contraction_to_mul_dynamic_batch_direct_mul(
  %lhs: tensor<?x?xf32>,
  %rhs: tensor<?x?xf32>
) -> tensor<?x?xf32> {
  %0 = stablehlo.dot_general %lhs, %rhs,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @simplify_dot_general_no_contraction_to_mul_dynamic_batch_direct_mul
//  CHECK-SAME: (%[[lhs:.+]]: tensor<?x?xf32>, %[[rhs:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
//       CHECK:     %[[m0:.+]] = stablehlo.multiply %[[lhs]], %[[rhs]] : tensor<?x?xf32>
//       CHECK:     return %[[m0]] : tensor<?x?xf32>

// -----

// Dynamic batch dim where batching dim is not leading on LHS: requires transpose.
func.func @simplify_dot_general_no_contraction_to_mul_dynamic_batch_needs_transpose(
  %lhs: tensor<10x?x7xf32>,
  %rhs: tensor<?xf32>
) -> tensor<?x10x7xf32> {
  %0 = stablehlo.dot_general %lhs, %rhs,
    batching_dims = [1] x [0],
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (tensor<10x?x7xf32>, tensor<?xf32>) -> tensor<?x10x7xf32>
  return %0 : tensor<?x10x7xf32>
}

// CHECK-LABEL: func.func @simplify_dot_general_no_contraction_to_mul_dynamic_batch_needs_transpose
//  CHECK-SAME: (%[[lhs:.+]]: tensor<10x?x7xf32>, %[[rhs:.+]]: tensor<?xf32>) -> tensor<?x10x7xf32> {
//       CHECK:     %[[t0:.+]] = stablehlo.transpose %[[lhs]], dims = [1, 0, 2] : (tensor<10x?x7xf32>) -> tensor<?x10x7xf32>
//       CHECK:     %[[b0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[rhs]], %{{.*}}, dims = [0] {known_nonexpanding_dimensions = array<i64: 0>} : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x10x7xf32>
//       CHECK:     %[[m0:.+]] = stablehlo.multiply %[[t0]], %[[b0]] : tensor<?x10x7xf32>
//       CHECK:     return %[[m0]] : tensor<?x10x7xf32>

// -----

!lhs_type = tensor<12x1x1x1xcomplex<f32>>
!rhs_type = tensor<12xcomplex<f32>>
!result_type = tensor<12x1x1x1xcomplex<f64>>

// Promotes operands to the result element type before multiplying.
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
//  CHECK-SAME: (%[[arg0:.+]]: tensor<12x1x1x1xcomplex<f32>>, %[[arg1:.+]]: tensor<12xcomplex<f32>>) -> tensor<12x1x1x1xcomplex<f64>> {
//   CHECK-DAG:     %[[c0:.+]] = stablehlo.convert %[[arg1]] : (tensor<12xcomplex<f32>>) -> tensor<12xcomplex<f64>>
//   CHECK-DAG:     %[[c1:.+]] = stablehlo.convert %[[arg0]] : (tensor<12x1x1x1xcomplex<f32>>) -> tensor<12x1x1x1xcomplex<f64>>
//   CHECK-DAG:     %[[r0:.+]] = stablehlo.reshape %[[c0]] : (tensor<12xcomplex<f64>>) -> tensor<12x1x1x1xcomplex<f64>>
//       CHECK:     %[[m0:.+]] = stablehlo.multiply %[[r0]], %[[c1]] : tensor<12x1x1x1xcomplex<f64>>
//       CHECK:     return %[[m0]] : tensor<12x1x1x1xcomplex<f64>>

// -----

// Negative: result element type is *narrower* than operands, so we do not rewrite.
func.func @dot_general_no_contraction_narrowing_result_negative(
  %lhs: tensor<12x10xf32>,
  %rhs: tensor<12x10xf32>
) -> tensor<12x10xf16> {
  %0 = stablehlo.dot_general %lhs, %rhs,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (tensor<12x10xf32>, tensor<12x10xf32>) -> tensor<12x10xf16>
  return %0 : tensor<12x10xf16>
}

// CHECK-LABEL: func.func @dot_general_no_contraction_narrowing_result_negative
//  CHECK-SAME: (%[[lhs:.+]]: tensor<12x10xf32>, %[[rhs:.+]]: tensor<12x10xf32>) -> tensor<12x10xf16> {
//       CHECK:     %[[v0:.+]] = stablehlo.dot_general %[[lhs]], %[[rhs]]
//  CHECK-SAME:       contracting_dims = [] x []
//       CHECK:     return %[[v0]] : tensor<12x10xf16>
