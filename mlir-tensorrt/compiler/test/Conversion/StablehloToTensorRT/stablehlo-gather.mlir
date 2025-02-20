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

// -----

func.func @simple_gather_dynamic(%arg0: tensor<?x?x256x256xi32>, %arg1: tensor<?xi32>) -> tensor<?x?x256x256xi32> {
  %c1 = stablehlo.constant dense<1> : tensor<1xi32>
  %c256 = stablehlo.constant dense<256> : tensor<1xi32>
  %dim = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<?x?x256x256xi32>) -> tensor<i32>
  %dim.1 = stablehlo.reshape %dim : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %c1, %dim.1, %c256, %c256, dim = 0 :
    (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %shape) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    indices_are_sorted = false, slice_sizes = array<i64: 1>
  } : (tensor<?x?x256x256xi32>, tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x256x256xi32>
  return %0 : tensor<?x?x256x256xi32>
}

// CHECK-LABEL: func.func @simple_gather_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x256x256xi32>, %[[arg1:.+]]: tensor<?xi32>)
//   CHECK-DAG:     %[[v5:.+]] = tensorrt.gather {axis = 0 : i64} ins(%[[arg0]], %[[arg1]] : tensor<?x?x256x256xi32>, tensor<?xi32>) -> tensor<?x?x256x256xi32>
//   CHECK-DAG:     return %[[v5]] : tensor<?x?x256x256xi32>

// -----

func.func @negative_gather_dynamic(%arg0: tensor<?x?x256x256xi32>, %arg1: tensor<?xi32>) -> tensor<?x?x256x256xi32> {
  %c1 = stablehlo.constant dense<1> : tensor<1xi32>
  %c256 = stablehlo.constant dense<256> : tensor<1xi32>
  // Wrong dimension index, should be dim = 1.
  %dim = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x?x256x256xi32>) -> tensor<i32>
  %dim.1 = stablehlo.reshape %dim : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %c1, %dim.1, %c256, %c256, dim = 0 :
    (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %shape) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    indices_are_sorted = false, slice_sizes = array<i64: 1>
  } : (tensor<?x?x256x256xi32>, tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x256x256xi32>
  return %0 : tensor<?x?x256x256xi32>
}

// CHECK-LABEL: func.func @negative_gather_dynamic(
//   CHECK-NOT: tensorrt.gather

// -----

func.func @negative_gather_dynamic2(%arg0: tensor<?x?x256x256xi32>, %arg1: tensor<?xi32>) -> tensor<?x?x256x256xi32> {
  %c1 = stablehlo.constant dense<1> : tensor<1xi32>
  %c256 = stablehlo.constant dense<256> : tensor<1xi32>
  // Dimension size should be arg0, not arg1
  %dim = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  %dim.1 = stablehlo.reshape %dim : (tensor<i32>) -> tensor<1xi32>
  %shape = stablehlo.concatenate %c1, %dim.1, %c256, %c256, dim = 0 :
    (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %shape) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    indices_are_sorted = false, slice_sizes = array<i64: 1>
  } : (tensor<?x?x256x256xi32>, tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x256x256xi32>
  return %0 : tensor<?x?x256x256xi32>
}

// CHECK-LABEL: func.func @negative_gather_dynamic2(
//   CHECK-NOT: tensorrt.gather
