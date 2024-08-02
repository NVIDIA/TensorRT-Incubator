// RUN: tensorrt-opt %s -split-input-file -tensorrt-apply-bug-wars=tensorrt-strongly-typed=true | FileCheck %s

func.func @main(%arg0: tensor<6xi8>) -> tensor<6xi8> {
  %cst_f32 = tensorrt.constant dense<1.000000e+00> : tensor<1xf32>
  %2 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kABS>} %arg0 : tensor<6xi8>
  return %2 : tensor<6xi8>
}

// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<6xi8>) -> tensor<6xi8> {
//       CHECK:     %[[v0:.+]] = tensorrt.expand_rank %[[arg0]] : tensor<6xi8> to tensor<1x6xi8>
//       CHECK:     %[[v1:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kABS>} %[[v0]] : tensor<1x6xi8>
//       CHECK:     %[[v2:.+]] = tensorrt.collapse_rank %[[v1]] : tensor<1x6xi8> to tensor<6xi8>
//       CHECK:     return %[[v2]] : tensor<6xi8>

// -----

func.func @tensorrt_matmul_to_vec_vec_mul(%lhs: tensor<1x3xf32>, %rhs: tensor<1x3xf32>) -> (tensor<f32>) {
  %0 = tensorrt.matrix_multiply
  {op0 = #tensorrt.matrix_operation<kNONE>,
  op1 = #tensorrt.matrix_operation<kTRANSPOSE>}
  ins(%lhs, %rhs : tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x1xf32>
  %1 = tensorrt.collapse_rank %0 : tensor<1x1xf32> to tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: func.func @tensorrt_matmul_to_vec_vec_mul
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.collapse_rank
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.collapse_rank
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kVECTOR>, op1 = #tensorrt.matrix_operation<kVECTOR>} ins(%[[v0]], %[[v1]] : tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
//  CHECK-NEXT: return %[[v2]] : tensor<f32>

// -----

func.func @tensorrt_matmul_to_vec_vec_mul_2(%lhs: tensor<1x3xf32>, %rhs: tensor<3x1xf32>) -> (tensor<f32>) {
  %0 = tensorrt.matrix_multiply
  {op0 = #tensorrt.matrix_operation<kNONE>,
  op1 = #tensorrt.matrix_operation<kNONE>}
  ins(%lhs, %rhs : tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
  %1 = tensorrt.collapse_rank %0 : tensor<1x1xf32> to tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: func.func @tensorrt_matmul_to_vec_vec_mul_2
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.collapse_rank
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.collapse_rank
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kVECTOR>, op1 = #tensorrt.matrix_operation<kVECTOR>} ins(%[[v0]], %[[v1]] : tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
//  CHECK-NEXT: return %[[v2]] : tensor<f32>