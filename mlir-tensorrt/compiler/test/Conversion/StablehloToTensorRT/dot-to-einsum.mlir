// RUN: mlir-tensorrt-opt %s -split-input-file -convert-stablehlo-to-tensorrt=prefer-einsum=true | FileCheck %s

!lhs = tensor<2x10x20x30x40xf32>
!rhs = tensor<2x10x20x30x40xf32>
!result = tensor<2x10x20x10x20xf32>

// CHECK-LABEL: @dot_general_multiple_contraction_dims1
// CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
func.func @dot_general_multiple_contraction_dims1(%arg0: !lhs, %arg1: !rhs) -> !result {
  // CHECK: %[[v0:.+]] = tensorrt.einsum
  // CHECK-SAME: "adebc,afgbc->adefg"
  // CHECK-SAME: ins(%[[arg0]], %[[arg1]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [3,4],
      rhs_contracting_dimensions = [3,4]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (!lhs, !rhs) -> !result
  // CHECK: return %[[v0]]
  return %0 : !result
}

// -----

!lhs = tensor<2x10x30x20x40xf32>
!rhs = tensor<2x30x10x20x40xf32>
!result = tensor<2x10x20x10x20xf32>

// CHECK-LABEL: @dot_general_multiple_contraction_dims2
// CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
func.func @dot_general_multiple_contraction_dims2(%arg0: !lhs, %arg1: !rhs) -> !result {
  // CHECK: %[[v0:.+]] = tensorrt.einsum
  // CHECK-SAME: "adbec,abfgc->adefg"
  // CHECK-SAME: ins(%[[arg0]], %[[arg1]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2,4],
      rhs_contracting_dimensions = [1,4]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (!lhs, !rhs) -> !result
  // CHECK: return %[[v0]]
  return %0 : !result
}

// -----

// CHECK-LABEL: @dot_general_multiple_outer_product_dims
// CHECK-SAME: (%[[arg0:.+]]: tensor<32x49x32xf32>, %[[arg1:.+]]: tensor<32x1x32x49xf32>)
func.func @dot_general_multiple_outer_product_dims(%arg0: tensor<32x49x32xf32>,
                                                   %arg1: tensor<32x1x32x49xf32>) -> tensor<32x49x1x49xf32> {
  // CHECK: %[[v0:.+]] = tensorrt.einsum
  // CHECK-SAME: "acb,adbe->acde"
  // CHECK-SAME: ins(%[[arg0]], %[[arg1]] : tensor<32x49x32xf32>, tensor<32x1x32x49xf32>) -> tensor<32x49x1x49xf32>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<32x49x32xf32>, tensor<32x1x32x49xf32>) -> tensor<32x49x1x49xf32>
  // CHECK: return %[[v0]]
  return %0 : tensor<32x49x1x49xf32>
}

// -----

!lhs = tensor<?x?x32x64xf32>
!rhs = tensor<?x?x64x100xf32>
!result = tensor<?x?x32x100xf32>

// CHECK-LABEL: @simple_dot_general1
// CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
func.func @simple_dot_general1(%arg0: !lhs, %arg1: !rhs) -> !result {
  // CHECK: %[[v0:.+]] = tensorrt.einsum
  // CHECK-SAME: "abdc,abce->abde"
  // CHECK-SAME: ins(%[[arg0]], %[[arg1]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (!lhs, !rhs) -> !result
  // CHECK: return %[[v0]]
  return %0 : !result
}


// -----

!lhs = tensor<?x?x64x32xf32>
!rhs = tensor<?x?x64x100xf32>
!result = tensor<?x?x32x100xf32>

// CHECK-LABEL: @simple_dot_general2
// CHECK-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
func.func @simple_dot_general2(%arg0: !lhs, %arg1: !rhs) -> !result {
  // CHECK: %[[v0:.+]] = tensorrt.einsum
  // CHECK-SAME: "abcd,abce->abde"
  // CHECK-SAME: ins(%[[arg0]], %[[arg1]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (!lhs, !rhs) -> !result
  // CHECK: return %[[v0]]
  return %0 : !result
}


// -----

// CHECK-LABEL: func.func @dot_general_promoted_result_type
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x64xf16>, %[[arg1:.+]]: tensor<?x?x64x100xf16>)
func.func @dot_general_promoted_result_type(%arg0: tensor<?x?x64xf16>, %arg1: tensor<?x?x64x100xf16>) -> tensor<?x?x100xf32> {
//       CHECK: %[[v0:.+]] = tensorrt.identity %[[arg0]] : tensor<?x?x64xf16> to tensor<?x?x64xf32>
//       CHECK: %[[v1:.+]] = tensorrt.identity %[[arg1]] : tensor<?x?x64x100xf16> to tensor<?x?x64x100xf32>
//       CHECK: %[[v2:.+]] = tensorrt.einsum
//  CHECK-SAME:   "abc,abcd->abd"
//  CHECK-SAME:   ins(%[[v0]], %[[v1]] : tensor<?x?x64xf32>, tensor<?x?x64x100xf32>) -> tensor<?x?x100xf32>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<?x?x64xf16>, tensor<?x?x64x100xf16>) -> tensor<?x?x100xf32>
  // CHECK: return %[[v2]]
  return %0 : tensor<?x?x100xf32>
}

