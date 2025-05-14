// RUN: tensorrt-opt %s -split-input-file -tensorrt-apply-wars="tensorrt-strongly-typed=true force-default-slice-in-bounds" | FileCheck %s

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

// -----

func.func @tensorrt_default_slice_force_in_bounds(
    %arg0: tensor<10x10xf32>,
    %arg1: tensor<2xi32>,
    %arg2: tensor<?x10xf32>,
    %arg3: tensor<2xi32>) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = tensorrt.slice %arg0[%arg1 : tensor<2xi32>][10, 10][1, 1] : tensor<10x10xf32> to tensor<?x?xf32>
  %1 = tensorrt.slice %arg0[%arg1 : tensor<2xi32>][%arg3: tensor<2xi32>][1, 1] : tensor<10x10xf32> to tensor<?x?xf32>
  %2 = tensorrt.slice %arg2[%arg1 : tensor<2xi32>][%arg3: tensor<2xi32>][1, 1] : tensor<?x10xf32> to tensor<?x?xf32>
  return %0, %1, %2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

// CHECK-LABEL: func.func @tensorrt_default_slice_force_in_bounds
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x10xf32>, %[[arg1:.+]]: tensor<2xi32>, %[[arg2:.+]]: tensor<?x10xf32>, %[[arg3:.+]]: tensor<2xi32>)
//   CHECK-DAG:     %[[cst_i32:.+]] = tensorrt.constant dense<0> : tensor<2xi32>
//   CHECK-DAG:     %[[cst_i32_0:.+]] = tensorrt.constant dense<10> : tensor<2xi32>
//   CHECK-DAG:     %[[cst_i32_1:.+]] = tensorrt.constant dense<1> : tensor<2xi32>
//   CHECK-DAG:     %[[cst_i32_2:.+]] = tensorrt.constant dense<9> : tensor<2xi32>
//   CHECK-DAG:     %[[v0:.+]] = tensorrt.element_wise <kMIN>(%[[arg1]], %[[cst_i32_2]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v1:.+]] = tensorrt.element_wise <kMAX>(%[[v0]], %[[cst_i32]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v2:.+]] = tensorrt.slice %[[arg0]][%[[v1]]: tensor<2xi32>][10, 10][1, 1] : tensor<10x10xf32> to tensor<?x?xf32>
//   CHECK-DAG:     %[[v3:.+]] = tensorrt.element_wise <kMIN>(%[[arg1]], %[[cst_i32_2]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v4:.+]] = tensorrt.element_wise <kMAX>(%[[v3]], %[[cst_i32]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v5:.+]] = tensorrt.element_wise <kMIN>(%[[arg3]], %[[cst_i32_0]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v6:.+]] = tensorrt.element_wise <kMAX>(%[[v5]], %[[cst_i32]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v7:.+]] = tensorrt.slice %[[arg0]][%[[v4]]: tensor<2xi32>][%[[v6]]: tensor<2xi32>][1, 1] : tensor<10x10xf32> to tensor<?x?xf32>
//   CHECK-DAG:     %[[v8:.+]] = tensorrt.shape %[[arg2]] : tensor<?x10xf32> -> tensor<2xi32>
//   CHECK-DAG:     %[[v9:.+]] = tensorrt.element_wise <kSUB>(%[[v8]], %[[cst_i32_1]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v10:.+]] = tensorrt.element_wise <kMIN>(%[[arg1]], %[[v9]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v11:.+]] = tensorrt.element_wise <kMAX>(%[[v10]], %[[cst_i32]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v12:.+]] = tensorrt.element_wise <kMIN>(%[[arg3]], %[[v8]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v13:.+]] = tensorrt.element_wise <kMAX>(%[[v12]], %[[cst_i32]] : tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v14:.+]] = tensorrt.slice %[[arg2]][%[[v11]]: tensor<2xi32>][%[[v13]]: tensor<2xi32>][1, 1] : tensor<?x10xf32> to tensor<?x?xf32>
//   CHECK-DAG:     return %[[v2]], %[[v7]], %[[v14]] : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>

// -----

func.func @trt_identity_cast_ui8_i32(%arg0: tensor<10xui8>) -> tensor<10xi32> {
  %0 = tensorrt.identity %arg0 : tensor<10xui8> to tensor<10xi32>
  return %0 : tensor<10xi32>
}

//  CHECK-LABEL: @trt_identity_cast_ui8_i32
//   CHECK-SAME: (%[[arg0:.+]]: tensor<10xui8>) -> tensor<10xi32>
//   CHECK-NEXT: %[[v0:.+]] = tensorrt.identity %[[arg0]] : tensor<10xui8> to tensor<10xf32>
//   CHECK-NEXT: %[[v1:.+]] = tensorrt.identity %[[v0]] : tensor<10xf32> to tensor<10xi32>
//   CHECK-NEXT: return %[[v1]] : tensor<10xi32>

// -----

func.func @trt_identity_cast_i32_ui8(%arg0: tensor<10xi32>) -> tensor<10xui8> {
  %0 = tensorrt.identity %arg0 : tensor<10xi32> to tensor<10xui8>
  return %0 : tensor<10xui8>
}

//  CHECK-LABEL: @trt_identity_cast_i32_ui8
//   CHECK-SAME: (%[[arg0:.+]]: tensor<10xi32>) -> tensor<10xui8>
//   CHECK-NEXT: %[[v0:.+]] = tensorrt.identity %[[arg0]] : tensor<10xi32> to tensor<10xf32>
//   CHECK-NEXT: %[[v1:.+]] = tensorrt.identity %[[v0]] : tensor<10xf32> to tensor<10xui8>
//   CHECK-NEXT: return %[[v1]] : tensor<10xui8>