// RUN: mlir-tensorrt-opt -split-input-file %s --convert-stablehlo-to-tensorrt -mlir-print-ir-after-failure | FileCheck %s

func.func @gather(%arg0: tensor<1xi32>, %arg1: tensor<1x1xi32>) -> tensor<1x1xi32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false, slice_sizes = array<i64: 1>
  } : (tensor<1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
  return %0 : tensor<1x1xi32>
}

// CHECK-LABEL: @gather(
//       CHECK:     %[[v0:.+]] = tensorrt.gather {axis = 0 : i64} ins(%{{.+}}, %{{.+}} : tensor<1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
//       CHECK:     return %[[v0]] : tensor<1x1xi32

// -----

func.func @gather1(%arg0: tensor<6xi32>, %arg1: tensor<3x1xi32>) -> tensor<3x1xi32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false, slice_sizes = array<i64: 1>
  } : (tensor<6xi32>, tensor<3x1xi32>) -> tensor<3x1xi32>
  return %0 : tensor<3x1xi32>
}

// CHECK-LABEL: @gather1(
//       CHECK:     %[[v0:.+]] = tensorrt.gather {axis = 0 : i64} ins(%{{.+}}, %{{.+}} : tensor<6xi32>, tensor<3x1xi32>) -> tensor<3x1xi32>
//       CHECK:     return %[[v0]] : tensor<3x1xi32>


// -----

func.func @gather2(%arg0: tensor<4x4x4xi32>, %arg1: tensor<6x1xi32>) -> tensor<6x1x4x4xi32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3], start_index_map = [0], index_vector_dim = 1>,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 4, 4>
  } : (tensor<4x4x4xi32>, tensor<6x1xi32>) -> tensor<6x1x4x4xi32>
  return %0 : tensor<6x1x4x4xi32>
}

// CHECK-LABEL:  @gather2(
//       CHECK:     %[[v0:.+]] = tensorrt.gather {axis = 0 : i64} ins(%{{.+}}, %{{.+}} : tensor<4x4x4xi32>, tensor<6x1xi32>) -> tensor<6x1x4x4xi32>
//       CHECK:     return %[[v0]] : tensor<6x1x4x4xi32>

// -----

func.func @gather_unsupported(%arg0: tensor<3x4x2xi32>, %arg1: tensor<2x3x2xi32>) -> tensor<2x3x2x2xi32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
      slice_sizes = array<i64: 1, 2, 2>,
      indices_are_sorted = false
  } : (tensor<3x4x2xi32>, tensor<2x3x2xi32>) -> tensor<2x3x2x2xi32>
  return %0 : tensor<2x3x2x2xi32>
}

// CHECK-LABEL: @gather_unsupported(
//   CHECK-NOT:   tensorrt.gather

// -----

func.func @gather_using_concat_slices(%arg0: tensor<64x1x1x12xf16> , %arg1: tensor<1x1xi32>) -> tensor<1x32x1x1x12xf16> {
  %2 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2, 3, 4],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    slice_sizes = array<i64: 32, 1, 1, 12>
  } : (tensor<64x1x1x12xf16>, tensor<1x1xi32>) -> tensor<1x32x1x1x12xf16>
  return %2 : tensor<1x32x1x1x12xf16>
}

// CHECK-LABEL: @gather_using_concat_slices
//  CHECK-SAME: (%[[arg0:.+]]: tensor<64x1x1x12xf16>, %[[arg1:.+]]: tensor<1x1xi32>) -> tensor<1x32x1x1x12xf16> {
//       CHECK:     %[[v0:.+]] = tensorrt.slice %[[arg1]][0, 0][1, 1][1, 1] : tensor<1x1xi32> to tensor<1x1xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.collapse_rank %[[v0]] : tensor<1x1xi32> to tensor<1xi32>
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<0> : tensor<3xi32>
//       CHECK:     %[[v2:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[v1]], %[[cst_i32]] : tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
//       CHECK:     %[[v3:.+]] = tensorrt.slice %[[arg0]][%[[v2]]: tensor<4xi32>][32, 1, 1, 12][1, 1, 1, 1] : tensor<64x1x1x12xf16> to tensor<32x1x1x12xf16>
//       CHECK:     %[[v4:.+]] = tensorrt.expand_rank %[[v3]] : tensor<32x1x1x12xf16> to tensor<1x32x1x1x12xf16>
//       CHECK:     %[[v5:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[v4]] : tensor<1x32x1x1x12xf16>) -> tensor<1x32x1x1x12xf16>
//       CHECK:     return %[[v5]] : tensor<1x32x1x1x12xf16>

// -----

func.func @gather_using_concat_slices(%arg0: tensor<64x1x1x12xf16> , %arg1: tensor<4x1xi32>) -> tensor<4x32x1x1x12xf16> {
  %2 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2, 3, 4],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    slice_sizes = array<i64: 32, 1, 1, 12>
  } : (tensor<64x1x1x12xf16>, tensor<4x1xi32>) -> tensor<4x32x1x1x12xf16>
  return %2 : tensor<4x32x1x1x12xf16>
}

// CHECK-LABEL: @gather_using_concat_slices
//  CHECK-SAME: (%[[arg0:.+]]: tensor<64x1x1x12xf16>, %[[arg1:.+]]: tensor<4x1xi32>) -> tensor<4x32x1x1x12xf16> {
//       CHECK:     %[[v0:.+]] = tensorrt.slice %[[arg1]][0, 0][1, 1][1, 1] : tensor<4x1xi32> to tensor<1x1xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.collapse_rank %[[v0]] : tensor<1x1xi32> to tensor<1xi32>
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<0> : tensor<3xi32>
//       CHECK:     %[[v2:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[v1]], %[[cst_i32]] : tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
//       CHECK:     %[[v3:.+]] = tensorrt.slice %[[arg0]][%[[v2]]: tensor<4xi32>][32, 1, 1, 12][1, 1, 1, 1] : tensor<64x1x1x12xf16> to tensor<32x1x1x12xf16>
//       CHECK:     %[[v4:.+]] = tensorrt.expand_rank %[[v3]] : tensor<32x1x1x12xf16> to tensor<1x32x1x1x12xf16>
//       CHECK:     %[[v5:.+]] = tensorrt.slice %[[arg1]][1, 0][1, 1][1, 1] : tensor<4x1xi32> to tensor<1x1xi32>
//       CHECK:     %[[v6:.+]] = tensorrt.collapse_rank %[[v5]] : tensor<1x1xi32> to tensor<1xi32>
//       CHECK:     %[[cst_i32_0:.+]] = tensorrt.constant dense<0> : tensor<3xi32>
//       CHECK:     %[[v7:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[v6]], %[[cst_i32_0]] : tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
//       CHECK:     %[[v8:.+]] = tensorrt.slice %[[arg0]][%[[v7]]: tensor<4xi32>][32, 1, 1, 12][1, 1, 1, 1] : tensor<64x1x1x12xf16> to tensor<32x1x1x12xf16>
//       CHECK:     %[[v9:.+]] = tensorrt.expand_rank %[[v8]] : tensor<32x1x1x12xf16> to tensor<1x32x1x1x12xf16>
//       CHECK:     %[[v10:.+]] = tensorrt.slice %[[arg1]][2, 0][1, 1][1, 1] : tensor<4x1xi32> to tensor<1x1xi32>
//       CHECK:     %[[v11:.+]] = tensorrt.collapse_rank %[[v10]] : tensor<1x1xi32> to tensor<1xi32>
//       CHECK:     %[[cst_i32_1:.+]] = tensorrt.constant dense<0> : tensor<3xi32>
//       CHECK:     %[[v12:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[v11]], %[[cst_i32_1]] : tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
//       CHECK:     %[[v13:.+]] = tensorrt.slice %[[arg0]][%[[v12]]: tensor<4xi32>][32, 1, 1, 12][1, 1, 1, 1] : tensor<64x1x1x12xf16> to tensor<32x1x1x12xf16>
//       CHECK:     %[[v14:.+]] = tensorrt.expand_rank %[[v13]] : tensor<32x1x1x12xf16> to tensor<1x32x1x1x12xf16>
//       CHECK:     %[[v15:.+]] = tensorrt.slice %[[arg1]][3, 0][1, 1][1, 1] : tensor<4x1xi32> to tensor<1x1xi32>
//       CHECK:     %[[v16:.+]] = tensorrt.collapse_rank %[[v15]] : tensor<1x1xi32> to tensor<1xi32>
//       CHECK:     %[[cst_i32_2:.+]] = tensorrt.constant dense<0> : tensor<3xi32>
//       CHECK:     %[[v17:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[v16]], %[[cst_i32_2]] : tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
//       CHECK:     %[[v18:.+]] = tensorrt.slice %[[arg0]][%[[v17]]: tensor<4xi32>][32, 1, 1, 12][1, 1, 1, 1] : tensor<64x1x1x12xf16> to tensor<32x1x1x12xf16>
//       CHECK:     %[[v19:.+]] = tensorrt.expand_rank %[[v18]] : tensor<32x1x1x12xf16> to tensor<1x32x1x1x12xf16>
//       CHECK:     %[[v20:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[v4]], %[[v9]], %[[v14]], %[[v19]] : tensor<1x32x1x1x12xf16>, tensor<1x32x1x1x12xf16>, tensor<1x32x1x1x12xf16>, tensor<1x32x1x1x12xf16>) -> tensor<4x32x1x1x12xf16>

// -----

// checks that the limit on batch size is obeyed
func.func @gather_using_concat_slices_unsupported(%arg0: tensor<64x1x1x12xf16> , %arg1: tensor<5x1xi32>) -> tensor<5x32x1x1x12xf16> {
  %2 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2, 3, 4],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    slice_sizes = array<i64: 32, 1, 1, 12>
  } : (tensor<64x1x1x12xf16>, tensor<5x1xi32>) -> tensor<5x32x1x1x12xf16>
  return %2 : tensor<5x32x1x1x12xf16>
}

// CHECK-LABEL: @gather_using_concat_slices_unsupported
//   CHECK-NOT:  tensorrt.slice

// -----

func.func @gather_canonical_4d(%arg0: tensor<1x1x32x32x256xf32>, %arg1: tensor<392x4xi32>) -> tensor<392x1x1x1x1x256xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1, 2, 3, 4, 5],
        start_index_map = [0, 1, 2, 3],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1, 1, 1, 256>
    } : (tensor<1x1x32x32x256xf32>, tensor<392x4xi32>) -> tensor<392x1x1x1x1x256xf32>
  return %0 : tensor<392x1x1x1x1x256xf32>
}

// CHECK-LABEL: func.func @gather_canonical_4d
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x1x32x32x256xf32>, %[[arg1:.+]]: tensor<392x4xi32>) -> tensor<392x1x1x1x1x256xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.gather_nd data(%[[arg0]]) indices(%[[arg1]]) : (tensor<1x1x32x32x256xf32>, tensor<392x4xi32>) -> tensor<392x256xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.reshape %[[v0]] : tensor<392x256xf32> to tensor<392x1x1x1x1x256xf32>
//       CHECK:     return %[[v1]]

// -----

func.func @gather_canonical_3d(%arg0: tensor<4x5x6x7x8xf32>, %arg1: tensor<16x17x3xi32>) -> tensor<16x17x1x1x1x7x8xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2, 3, 4, 5, 6],
        start_index_map = [0, 1, 2],
        index_vector_dim = 2
      >,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1, 1, 7, 8>
    } : (tensor<4x5x6x7x8xf32>, tensor<16x17x3xi32>) -> tensor<16x17x1x1x1x7x8xf32>
  return %0 : tensor<16x17x1x1x1x7x8xf32>
}

// CHECK-LABEL: func.func @gather_canonical_3d
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x5x6x7x8xf32>, %[[arg1:.+]]: tensor<16x17x3xi32>) -> tensor<16x17x1x1x1x7x8xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.gather_nd data(%[[arg0]]) indices(%[[arg1]]) : (tensor<4x5x6x7x8xf32>, tensor<16x17x3xi32>) -> tensor<16x17x7x8xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.reshape %[[v0]] : tensor<16x17x7x8xf32> to tensor<16x17x1x1x1x7x8xf32>
//       CHECK:     return %[[v1]]

// -----

func.func @gather_canonical_scalar(%arg0: tensor<4x5x6x7x8xf32>, %arg1: tensor<16x17x5xi32>) -> tensor<16x17x1x1x1x1x1xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2, 3, 4, 5, 6],
        start_index_map = [0, 1, 2, 3, 4],
        index_vector_dim = 2
      >,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 1, 1, 1, 1>
    } : (tensor<4x5x6x7x8xf32>, tensor<16x17x5xi32>) -> tensor<16x17x1x1x1x1x1xf32>
  return %0 : tensor<16x17x1x1x1x1x1xf32>
}

// CHECK-LABEL: func.func @gather_canonical_scalar
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x5x6x7x8xf32>, %[[arg1:.+]]: tensor<16x17x5xi32>) -> tensor<16x17x1x1x1x1x1xf32>
//  CHECK-NEXT:     %[[v0:.+]] = tensorrt.gather_nd data(%[[arg0]]) indices(%[[arg1]]) : (tensor<4x5x6x7x8xf32>, tensor<16x17x5xi32>) -> tensor<16x17xf32>
//  CHECK-NEXT:     %[[v1:.+]] = tensorrt.reshape %[[v0]] : tensor<16x17xf32> to tensor<16x17x1x1x1x1x1xf32>
//  CHECK-NEXT:     return %[[v1]]

// -----

// CHECK-LABEL: func.func @gather_negative
//  CHECK-NEXT:  stablehlo.gather
//  CHECK-NEXT:  return
func.func @gather_negative(%arg0: tensor<1x1x64x12xf16> , %arg1: tensor<1xi32>) -> tensor<1x1x64x4xf16> {
  %2 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0, 1, 2, 3],
      start_index_map = [3],
      index_vector_dim = 0
    >,
    indices_are_sorted = true,
    slice_sizes = array<i64: 1, 1, 64, 4>
  } : (tensor<1x1x64x12xf16>, tensor<1xi32>) -> tensor<1x1x64x4xf16>
  return %2 : tensor<1x1x64x4xf16>
}
