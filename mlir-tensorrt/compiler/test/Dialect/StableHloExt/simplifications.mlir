// RUN: mlir-tensorrt-opt %s --stablehlo-ext-simplifications -split-input-file | FileCheck %s

func.func @simplify_trivial_min(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = stablehlo.minimum %arg0, %arg0 :  tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_min
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>) -> tensor<1xi32> {
//       CHECK:     return %[[arg0]] : tensor<1xi32>

// -----
func.func @simplify_trivial_max(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = stablehlo.maximum %arg0, %arg0 :  tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_max
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>) -> tensor<1xi32> {
//       CHECK:     return %[[arg0]] : tensor<1xi32>

// -----

func.func @simplify_trivial_min_requires_cast(%arg0: tensor<1xi32>) -> tensor<?xi32> {
  %0 = stablehlo.minimum %arg0, %arg0 :  (tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_min_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>) -> tensor<?xi32> {
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg0]] : tensor<1xi32> to tensor<?xi32>
//       CHECK:     return %[[cast]] : tensor<?xi32>

// -----

// Verifies that redundant `tensor.cast` are eliminated.

func.func @simplify_cast_cancel(%arg0: tensor<1xi32>, %arg1: tensor<?xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<?xi32> {
  %0 = stablehlo.minimum %arg0, %arg0 :  (tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1 = tensor.cast %0 : tensor<?xi32> to tensor<1xi32>
  %2 = stablehlo.real_dynamic_slice %arg1, %arg2, %1, %arg3 : (tensor<?xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %2 : tensor<?xi32>
}

// CHECK-LABEL: func.func @simplify_cast_cancel
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>, %[[arg1:.+]]: tensor<?xi32>, %[[arg2:.+]]: tensor<1xi32>, %[[arg3:.+]]: tensor<1xi32>) -> tensor<?xi32> {
//       CHECK:     %[[v0:.+]] = stablehlo.real_dynamic_slice %[[arg1]], %[[arg2]], %[[arg0]], %[[arg3]] : (tensor<?xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
//       CHECK:     return %[[v0]] : tensor<?xi32>

// -----

func.func @simplify_trivial_slice(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.slice %arg0 [0:2] : (tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xi32>)
//  CHECK-NEXT:     return %[[arg0]] : tensor<2xi32>

// -----

func.func @simplify_trivial_slice_nd(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
  %0 = stablehlo.slice %arg0 [0:2, 0:4:1] : (tensor<2x4xi32>) -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice_nd
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x4xi32>)
//  CHECK-NEXT:     return %[[arg0]] : tensor<2x4xi32>

// -----

func.func @simplify_trivial_slice_requires_cast(%arg0: tensor<2xi32>) -> tensor<?xi32> {
  %0 = stablehlo.slice %arg0 [0:2] : (tensor<2xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xi32>) -> tensor<?xi32> {
//  CHECK-NEXT:     %[[cast:.+]] = tensor.cast %[[arg0]] : tensor<2xi32> to tensor<?xi32>
//  CHECK-NEXT:     return %[[cast]] : tensor<?xi32>

// -----

func.func @simplify_trivial_slice_negative(%arg0: tensor<2xi32>) -> tensor<?xi32> {
  %0 = stablehlo.slice %arg0 [0:2:2] : (tensor<2xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice_negative
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xi32>)
//  CHECK-NEXT:   %[[v0:.+]] = stablehlo.slice %[[arg0]] [0:2:2]
//  CHECK-NEXT:   return %[[v0]]

// -----

func.func @simplify_trivial_slice_0d(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.slice %arg0 [] : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice_0d
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>) -> tensor<i32> {
//  CHECK-NEXT:     return %[[arg0]] : tensor<i32>

// -----

func.func @simplify_trivial_slice_empty(%arg0: tensor<0xi32>) -> tensor<0xi32> {
  %0 = stablehlo.slice %arg0 [0:0] : (tensor<0xi32>) -> tensor<0xi32>
  return %0 : tensor<0xi32>
}

// CHECK-LABEL: func.func @simplify_trivial_slice_empty
//  CHECK-SAME:     (%[[arg0:.+]]: {{.*}})
//       CHECK:     return %[[arg0]] : tensor<0xi32>

// -----

func.func @concat_remove_zero_extent_segments(%arg0: tensor<?xi32>, %arg1: tensor<0xi32>) -> tensor<?xi32> {
  %0 = stablehlo.concatenate %arg0, %arg0, %arg1, dim = 0 : (tensor<?xi32>, tensor<?xi32>, tensor<0xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @concat_remove_zero_extent_segments
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32>, %[[arg1:.+]]: tensor<0xi32>)
//       CHECK:     %[[v0:.+]] = stablehlo.concatenate %[[arg0]], %[[arg0]], dim = 0 : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
//       CHECK:     return %[[v0]] : tensor<?xi32>

// -----

func.func @concat_simplify_single_operand(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = stablehlo.concatenate %arg0, dim = 0 : (tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @concat_simplify_single_operand
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32>)
//  CHECK-NEXT:     return %[[arg0]]

// -----

func.func @concat_simplify_single_operand_requires_cast(%arg0: tensor<4xi32>) -> tensor<?xi32> {
  %0 = stablehlo.concatenate %arg0, dim = 0 : (tensor<4xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func.func @concat_simplify_single_operand_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>)
//  CHECK-NEXT:     %[[cast:.+]] = tensor.cast %[[arg0]] : tensor<4xi32> to tensor<?xi32>
//  CHECK-NEXT:     return %[[cast]]


// -----

func.func @concatenate_requires_cast(%arg0: tensor<?xi32>) -> tensor<?xf32> {
  %0 = stablehlo.concatenate %arg0, dim = 0 : (tensor<?xi32>) -> tensor<1xi32>
  %1 = stablehlo.dynamic_iota %0, dim=0 : (tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @concatenate_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32>)
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg0]] : tensor<?xi32> to tensor<1xi32>
//       CHECK:     %[[v0:.+]] = stablehlo.dynamic_iota %[[cast]], dim = 0
//       CHECK:     return %[[v0]] : tensor<?xf32>


// -----

func.func @dynamic_broadcast_in_dim_requires_cast(%arg0: tensor<?xf32>, %arg1: tensor<?xi32>) -> tensor<?xf32> {
  %0 = stablehlo.concatenate %arg1, dim = 0 : (tensor<?xi32>) -> tensor<1xi32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims=[0] : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @dynamic_broadcast_in_dim_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<?xi32>)
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg1]] : tensor<?xi32> to tensor<1xi32>
//       CHECK:     %[[v0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[arg0]], %[[cast]], dims = [0]
//       CHECK:     return %[[v0]] : tensor<?xf32>

// -----

func.func @real_dynamic_slice_param_requires_cast(%arg0: tensor<?xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<?xi32>) -> tensor<?xf32> {
  %0 = stablehlo.concatenate %arg3, dim = 0 : (tensor<?xi32>) -> tensor<1xi32>
  %1 = stablehlo.real_dynamic_slice %arg0, %0, %arg1, %arg2 : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @real_dynamic_slice_param_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<1xi32>, %[[arg2:.+]]: tensor<1xi32>, %[[arg3:.+]]: tensor<?xi32>)
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg3]] : tensor<?xi32> to tensor<1xi32>
//       CHECK:     %[[v0:.+]] = stablehlo.real_dynamic_slice %[[arg0]], %[[cast]], %[[arg1]], %[[arg2]]
//       CHECK:     return %[[v0]] : tensor<?xf32>

// -----

func.func @dynamic_pad_requires_cast(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>, %arg2: tensor<?xi32>, %arg3: tensor<2xi32>) -> tensor<?x?xf32> {
  %cst_0 = stablehlo.constant dense<[0, 0]> : tensor<2xi32>
  %0 = stablehlo.concatenate %arg2, dim = 0 : (tensor<?xi32>) -> tensor<2xi32>
  %1 = "stablehlo.dynamic_pad"(%arg0, %arg1, %0, %arg3, %cst_0) : (tensor<?x?xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %1: tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_pad_requires_cast
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<f32>, %[[arg2:.+]]: tensor<?xi32>, %[[arg3:.+]]: tensor<2xi32>) -> tensor<?x?xf32> {
//       CHECK:     %[[constant:.+]] = stablehlo.constant dense<0> : tensor<2xi32>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg2]] : tensor<?xi32> to tensor<2xi32>
//       CHECK:     %[[v0:.+]] = stablehlo.dynamic_pad %[[arg0]], %[[arg1]], %[[cast]], %[[arg3]], %[[constant]]
//       CHECK:     return %[[v0]] : tensor<?x?xf32>

// -----

func.func @dynamic_gather_requires_cast(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?xf32> {
  %0 = stablehlo.iota dim = 0 : tensor<4x6xf32>
  %1 = stablehlo.concatenate %arg1, dim = 0 : (tensor<?xi32>) -> tensor<2xi32>
  %2 = "stablehlo.dynamic_gather"(%0, %arg0, %1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1], collapsed_slice_dims = [0],
      start_index_map = [0], index_vector_dim = 1>
    } : (tensor<4x6xf32>, tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_gather_requires_cast
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<?xi32>, %[[arg1:.+]]: tensor<?xi32>) -> tensor<?x?xf32> {
//       CHECK:     %[[iota:.+]] = stablehlo.iota dim = 0 : tensor<4xf32>
//       CHECK:     %[[broadcast:.+]] = stablehlo.broadcast_in_dim %[[iota]], dims = [0] : (tensor<4xf32>) -> tensor<4x6xf32>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg1]] : tensor<?xi32> to tensor<2xi32>
//       CHECK:     %[[v0:.+]] = "stablehlo.dynamic_gather"(%[[broadcast]], %[[arg0]], %[[cast]]) {{.*}} : (tensor<4x6xf32>, tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xf32>

// -----

func.func @dynamic_reshape_requires_cast(%arg0: tensor<?xf32>, %arg1: tensor<?xi32>) -> tensor<?xf32> {
  %0 = stablehlo.concatenate %arg1, dim = 0 : (tensor<?xi32>) -> tensor<1xi32>
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @dynamic_reshape_requires_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<?xi32>)
//       CHECK:     %[[cast:.+]] = tensor.cast %[[arg1]] : tensor<?xi32> to tensor<1xi32>
//       CHECK:     %[[v0:.+]] = stablehlo.dynamic_reshape %[[arg0]], %[[cast]]
//       CHECK:     return %[[v0]] : tensor<?xf32>

// -----

func.func @cascaded_convert(%arg0: tensor<i1>) -> tensor<i32>{
    %0 = stablehlo.convert %arg0: (tensor<i1>) -> tensor<i8>
    %1 = stablehlo.convert %0: (tensor<i8>) -> tensor<i32>
    return %1 : tensor<i32>
}
// CHECK-LABEL: cascaded_convert
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i1>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.convert %[[arg0]] : (tensor<i1>) -> tensor<i32>
//  CHECK-NEXT: return %[[v0]] : tensor<i32>

// -----

func.func @cascaded_convert_negative(%arg0: tensor<i8>) -> tensor<i32>{
    %0 = stablehlo.convert %arg0: (tensor<i8>) -> tensor<i1>
    %1 = stablehlo.convert %0: (tensor<i1>) -> tensor<i32>
    return %1 : tensor<i32>
}
// CHECK-LABEL: cascaded_convert_negative
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i8>)
//  CHECK-NEXT: %[[v0:.+]] = stablehlo.convert
//  CHECK-NEXT: %[[v1:.+]] = stablehlo.convert
//  CHECK-NEXT: return %[[v1]] : tensor<i32>

// -----

func.func @simplify_reshape_broadcastindim_reshape(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x8x256xf16> {
  %0 = stablehlo.reshape %arg0 : (tensor<1x1x1x256xf16>) -> tensor<1x1x1x1x1x1x1x1x256xf16>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3, 4, 5, 7, 8, 9] : (tensor<1x1x1x1x1x1x1x1x256xf16>) -> tensor<1x1x1x1x1x1x8x1x1x256xf16>
  %2 = stablehlo.reshape %1 : (tensor<1x1x1x1x1x1x8x1x1x256xf16>) -> tensor<1x1x8x256xf16>
  return %2 : tensor<1x1x8x256xf16>
}

// CHECK-LABEL: func.func @simplify_reshape_broadcastindim_reshape
//       CHECK: stablehlo.broadcast_in_dim
//  CHECK-SAME: dims = [0, 1, 2, 3]
//  CHECK-SAME: (tensor<1x1x1x256xf16>) -> tensor<1x1x8x256xf16>
//  CHECK-NEXT: return

// -----

func.func @simplify_reshape_broadcastindim_reshape2(%arg0: tensor<1x1xf32>) -> tensor<8x1xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<1x1x1x1x1xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 3, 4, 5] : (tensor<1x1x1x1x1xf32>) -> tensor<1x1x8x1x1x1xf32>
  %2 = stablehlo.reshape %1 : (tensor<1x1x8x1x1x1xf32>) -> tensor<8x1xf32>
  return %2 : tensor<8x1xf32>
}

// CHECK-LABEL: func.func @simplify_reshape_broadcastindim_reshape2
//       CHECK: stablehlo.broadcast_in_dim
//  CHECK-SAME: dims = [0, 1]
//  CHECK-SAME: (tensor<1x1xf32>) -> tensor<8x1xf32>
//  CHECK-NEXT: return

// -----

func.func @simplify_reshape_broadcastindim_reshape3(%arg0: tensor<1x1x1x1x256xf16>) -> tensor<1x1x8x256xf16> {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x1x1x256xf16>) -> tensor<1x1x1x1x1x1x1x1x256xf16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3, 4, 5, 7, 8, 9] : (tensor<1x1x1x1x1x1x1x1x256xf16>) -> tensor<1x1x1x1x1x1x8x1x1x256xf16>
    %2 = stablehlo.reshape %1 : (tensor<1x1x1x1x1x1x8x1x1x256xf16>) -> tensor<1x1x8x256xf16>
    return %2: tensor<1x1x8x256xf16>
}

// CHECK-LABEL: func.func @simplify_reshape_broadcastindim_reshape3
//       CHECK: stablehlo.reshape
//  CHECK-SAME: (tensor<1x1x1x1x256xf16>) -> tensor<1x1x1x256xf16>
//       CHECK: stablehlo.broadcast_in_dim
//  CHECK-SAME: dims = [0, 1, 2, 3]
//  CHECK-SAME: (tensor<1x1x1x256xf16>) -> tensor<1x1x8x256xf16>
//  CHECK-NEXT: return
