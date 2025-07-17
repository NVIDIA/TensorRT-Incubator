// RUN: mlir-tensorrt-opt %s -split-input-file -convert-stablehlo-to-tensorrt=prefer-einsum=true | FileCheck %s --check-prefixes=EINSUM,BOTH
// RUN: mlir-tensorrt-opt %s -split-input-file -convert-stablehlo-to-tensorrt=prefer-einsum=false | FileCheck %s --check-prefixes=MATMUL,BOTH

!lhs = tensor<2x10x20x30x40xf32>
!rhs = tensor<2x10x20x30x40xf32>
!result = tensor<2x10x20x10x20xf32>

// BOTH-LABEL: @dot_general_multiple_contraction_dims1
// BOTH-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
func.func @dot_general_multiple_contraction_dims1(%arg0: !lhs, %arg1: !rhs) -> !result {
  // BOTH: %[[v0:.+]] = tensorrt.einsum
  // BOTH-SAME: "adebc,afgbc->adefg"
  // BOTH-SAME: ins(%[[arg0]], %[[arg1]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [3,4],
      rhs_contracting_dimensions = [3,4]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (!lhs, !rhs) -> !result
  // BOTH: return %[[v0]]
  return %0 : !result
}

// -----

!lhs = tensor<2x10x30x20x40xf32>
!rhs = tensor<2x30x10x20x40xf32>
!result = tensor<2x10x20x10x20xf32>

// BOTH-LABEL: @dot_general_multiple_contraction_dims2
// BOTH-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
func.func @dot_general_multiple_contraction_dims2(%arg0: !lhs, %arg1: !rhs) -> !result {
  // BOTH: %[[v0:.+]] = tensorrt.einsum
  // BOTH-SAME: "adbec,abfgc->adefg"
  // BOTH-SAME: ins(%[[arg0]], %[[arg1]]
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2,4],
      rhs_contracting_dimensions = [1,4]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (!lhs, !rhs) -> !result
  // BOTH: return %[[v0]]
  return %0 : !result
}

// -----

// BOTH-LABEL: @dot_general_multiple_outer_product_dims
// BOTH-SAME: (%[[arg0:.+]]: tensor<32x49x32xf32>, %[[arg1:.+]]: tensor<32x1x32x49xf32>)
func.func @dot_general_multiple_outer_product_dims(%arg0: tensor<32x49x32xf32>,
                                                   %arg1: tensor<32x1x32x49xf32>) -> tensor<32x49x1x49xf32> {
  // BOTH: %[[v0:.+]] = tensorrt.einsum
  // BOTH-SAME: "acb,adbe->acde"
  // BOTH-SAME: ins(%[[arg0]], %[[arg1]] : tensor<32x49x32xf32>, tensor<32x1x32x49xf32>) -> tensor<32x49x1x49xf32>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<32x49x32xf32>, tensor<32x1x32x49xf32>) -> tensor<32x49x1x49xf32>
  // BOTH: return %[[v0]]
  return %0 : tensor<32x49x1x49xf32>
}

// -----

!lhs = tensor<?x?x32x64xf32>
!rhs = tensor<?x?x64x100xf32>
!result = tensor<?x?x32x100xf32>

// EINSUM-LABEL: @simple_dot_general1
// EINSUM-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
// MATMUL-LABEL: func.func @simple_dot_general1
func.func @simple_dot_general1(%arg0: !lhs, %arg1: !rhs) -> !result {
  // EINSUM: %[[v0:.+]] = tensorrt.einsum
  // EINSUM-SAME: "abdc,abce->abde"
  // EINSUM-SAME: ins(%[[arg0]], %[[arg1]]
  //       MATMUL: %[[v0:.+]] = tensorrt.matrix_multiply
  //  MATMUL-SAME:  op0 = #tensorrt.matrix_operation<kNONE>
  //  MATMUL-SAME:  op1 = #tensorrt.matrix_operation<kNONE>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (!lhs, !rhs) -> !result
  // EINSUM: return %[[v0]]
  return %0 : !result
}


// -----

!lhs = tensor<?x?x64x32xf32>
!rhs = tensor<?x?x64x100xf32>
!result = tensor<?x?x32x100xf32>

// EINSUM-LABEL: @simple_dot_general2
// EINSUM-SAME: (%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: tensor<{{.*}}>)
// MATMUL-LABEL: func.func @simple_dot_general2
func.func @simple_dot_general2(%arg0: !lhs, %arg1: !rhs) -> !result {
  // EINSUM: %[[v0:.+]] = tensorrt.einsum
  // EINSUM-SAME: "abcd,abce->abde"
  // EINSUM-SAME: ins(%[[arg0]], %[[arg1]]
  //       MATMUL: %[[v0:.+]] = tensorrt.matrix_multiply
  //  MATMUL-SAME:  op0 = #tensorrt.matrix_operation<kTRANSPOSE>
  //  MATMUL-SAME:  op1 = #tensorrt.matrix_operation<kNONE>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (!lhs, !rhs) -> !result
  // EINSUM: return %[[v0]]
  return %0 : !result
}


// -----

// MATMUL-LABEL: func.func @dot_general_promoted_result_type
// EINSUM-LABEL: func.func @dot_general_promoted_result_type
//  EINSUM-SAME: (%[[arg0:.+]]: tensor<?x?x64xf16>, %[[arg1:.+]]: tensor<?x?x64x100xf16>)
func.func @dot_general_promoted_result_type(%arg0: tensor<?x?x64xf16>, %arg1: tensor<?x?x64x100xf16>) -> tensor<?x?x100xf32> {
//       EINSUM: %[[v0:.+]] = tensorrt.identity %[[arg0]] : tensor<?x?x64xf16> to tensor<?x?x64xf32>
//       EINSUM: %[[v1:.+]] = tensorrt.identity %[[arg1]] : tensor<?x?x64x100xf16> to tensor<?x?x64x100xf32>
//       EINSUM: %[[v2:.+]] = tensorrt.einsum
//  EINSUM-SAME:   "abc,abcd->abd"
//  EINSUM-SAME:   ins(%[[v0]], %[[v1]] : tensor<?x?x64xf32>, tensor<?x?x64x100xf32>) -> tensor<?x?x100xf32>
//       MATMUL: %[[v0:.+]] = tensorrt.matrix_multiply
//  MATMUL-SAME:  op0 = #tensorrt.matrix_operation<kVECTOR>
//  MATMUL-SAME:  op1 = #tensorrt.matrix_operation<kNONE>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<?x?x64xf16>, tensor<?x?x64x100xf16>) -> tensor<?x?x100xf32>
  // EINSUM: return %[[v2]]
  return %0 : tensor<?x?x100xf32>
}

// -----

// BOTH-LABEL: @dot_general_non_contiguous_batching_dims
// BOTH-SAME: (%[[arg0:.+]]: tensor<1x256x16x72xbf16>, %[[arg1:.+]]: tensor<1x256x16x72xbf16>)
func.func @dot_general_non_contiguous_batching_dims(
    %arg0: tensor<1x256x16x72xbf16>, %arg1: tensor<1x256x16x72xbf16>)
    -> tensor<1x16x256x256xbf16> {
  // BOTH: %[[v0:.+]] = tensorrt.einsum
  // BOTH-SAME: "adbc,aebc->abde"
  // BOTH-SAME: ins(%[[arg0]], %[[arg1]]
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x256x16x72xbf16>, tensor<1x256x16x72xbf16>) -> tensor<1x16x256x256xbf16>
  // BOTH: return %[[v0]]
  return %0 : tensor<1x16x256x256xbf16>
}

