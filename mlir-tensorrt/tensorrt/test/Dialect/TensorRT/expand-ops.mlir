// RUN: tensorrt-opt %s -split-input-file -tensorrt-expand-ops | FileCheck %s

func.func @trt_expand_rank(%arg0: tensor<100x100xf32>) -> tensor<1x1x1x100x100xf32> {
  %0 = tensorrt.expand_rank %arg0 : tensor<100x100xf32> to tensor<1x1x1x100x100xf32>
  return %0 : tensor<1x1x1x100x100xf32>
}

// CHECK-LABEL: @trt_expand_rank
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<{{.+}}>)
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 1, 1, 1, 100, 100>,
//  CHECK-SAME:   second_transpose = array<i64: 0, 1, 2, 3, 4>, zero_is_placeholder = false} ins(%{{.+}} : tensor<100x100xf32>) -> tensor<1x1x1x100x100xf32>

// -----

func.func @trt_collapse_rank(%arg0: tensor<100x1x100xf32>) -> tensor<100x100xf32> {
  %0 = tensorrt.collapse_rank %arg0 : tensor<100x1x100xf32> to tensor<100x100xf32>
  return %0 : tensor<100x100xf32>
}

// CHECK-LABEL: @trt_collapse_rank
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<{{.+}}>)
//       CHECK:   tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>,
//  CHECK-SAME:   reshape = array<i64: 100, 100>, second_transpose = array<i64: 0, 1>,
//  CHECK-SAME:   zero_is_placeholder = false} ins(%arg0 : tensor<100x1x100xf32>) -> tensor<100x100xf32>

// -----

func.func @trt_transpose(%arg0: tensor<100x1x100xf32>) -> tensor<100x100x1xf32> {
  %0 = tensorrt.transpose {
    permutation = affine_map<(d0, d1, d2)->(d0, d2, d1)>
  } %arg0 : tensor<100x1x100xf32> to tensor<100x100x1xf32>
  return %0 : tensor<100x100x1xf32>
}

// CHECK-LABEL: @trt_transpose
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<{{.+}}>)
//       CHECK:   tensorrt.shuffle {first_transpose = array<i64: 0, 2, 1>,
//  CHECK-SAME:   second_transpose = array<i64: 0, 1, 2>,
//  CHECK-SAME:   zero_is_placeholder = false} ins(%arg0 : tensor<100x1x100xf32>) -> tensor<100x100x1xf32>

// -----

func.func @trt_transpose_201(%arg0: tensor<100x1x100xf32>) -> tensor<100x100x1xf32> {
  %0 = tensorrt.transpose {
    permutation = affine_map<(d0, d1, d2)->(d2, d0, d1)>
  } %arg0 : tensor<100x1x100xf32> to tensor<100x100x1xf32>
  return %0 : tensor<100x100x1xf32>
}

// CHECK-LABEL: @trt_transpose_201
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<{{.+}}>)
//       CHECK:   tensorrt.shuffle {first_transpose = array<i64: 2, 0, 1>,
//  CHECK-SAME:   second_transpose = array<i64: 0, 1, 2>,
//  CHECK-SAME:   zero_is_placeholder = false} ins(%arg0 : tensor<100x1x100xf32>) -> tensor<100x100x1xf32>

// -----

func.func @trt_tranpose_full_dynamic(%arg : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>{
    %0 = tensorrt.transpose {
      permutation = affine_map<(d0, d1, d2)->(d2, d0, d1)>
    } %arg : tensor<?x?x?xf32> to tensor<?x?x?xf32>
    return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @trt_tranpose_full_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
//       CHECK: %[[v0:.+]] = tensorrt.shuffle {first_transpose = array<i64: 2, 0, 1>,
//  CHECK-SAME: second_transpose = array<i64: 0, 1, 2>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%[[arg0]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
//       CHECK: return %[[v0]] : tensor<?x?x?xf32>

// -----

func.func @trt_argmin(%arg0: tensor<100x1x100xf32>) -> (tensor<100x1x1xf32>, tensor<100x1x1xi32>) {
  %0, %1 = tensorrt.argmin {
    axis = 2 : i64
  } %arg0 : tensor<100x1x100xf32> -> tensor<100x1x1xf32>, tensor<100x1x1xi32>
  return %0, %1 : tensor<100x1x1xf32>, tensor<100x1x1xi32>
}

// CHECK-LABEL: @trt_argmin
//       CHECK:   tensorrt.top_k <kMIN>
//  CHECK-SAME:     axis = 2 : i64, k = 1 : i64
//  CHECK-SAME:     %{{.+}} : tensor<100x1x100xf32> -> tensor<100x1x1xf32>, tensor<100x1x1xi32>

// -----

func.func @trt_argmax(%arg0: tensor<100x1x100xf32>) -> (tensor<100x1x1xf32>, tensor<100x1x1xi32>) {
  %0, %1 = tensorrt.argmax {
    axis = 2 : i64
  } %arg0 : tensor<100x1x100xf32> -> tensor<100x1x1xf32>, tensor<100x1x1xi32>
  return %0, %1 : tensor<100x1x1xf32>, tensor<100x1x1xi32>
}

// CHECK-LABEL: @trt_argmax
//       CHECK:   tensorrt.top_k <kMAX>
//  CHECK-SAME:     axis = 2 : i64, k = 1 : i64
//  CHECK-SAME:     %{{.+}} : tensor<100x1x100xf32> -> tensor<100x1x1xf32>, tensor<100x1x1xi32>

// -----

func.func @trt_argmin_1d_float_operand(%arg0: tensor<10xf32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %0, %1 = tensorrt.argmin {
    axis = 0 : i64
  } %arg0 : tensor<10xf32> -> tensor<1xf32>, tensor<1xi32>
  return %0, %1 : tensor<1xf32>, tensor<1xi32>
}
// CHECK-LABEL: @trt_argmin_1d_float_operand
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 1, 10>, second_transpose = array<i64: 0, 1>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<10xf32>) -> tensor<1x10xf32>
//       CHECK: tensorrt.top_k <kMIN> {axis = 1 : i64, k = 1 : i64}
//  CHECK-SAME: %{{.+}} : tensor<1x10xf32> -> tensor<1x1xf32>, tensor<1x1xi32>
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 1>, second_transpose = array<i64: 0>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<1x1xf32>) -> tensor<1xf32>
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 1>, second_transpose = array<i64: 0>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<1x1xi32>) -> tensor<1xi32>

// -----

func.func @trt_argmin_1d_int_operand(%arg0: tensor<10xi32>) -> (tensor<1xi32>, tensor<1xi32>) {
  %0, %1 = tensorrt.argmin {
    axis = 0 : i64
  } %arg0 : tensor<10xi32> -> tensor<1xi32>, tensor<1xi32>
  return %0, %1 : tensor<1xi32>, tensor<1xi32>
}
// CHECK-LABEL: trt_argmin_1d_int_operand
//       CHECK: tensorrt.identity %{{.+}} : tensor<10xi32> to tensor<10xf32>
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 1, 10>, second_transpose = array<i64: 0, 1>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<10xf32>) -> tensor<1x10xf32>
//       CHECK: tensorrt.top_k <kMIN> {axis = 1 : i64, k = 1 : i64}
//  CHECK-SAME: %{{.+}} : tensor<1x10xf32> -> tensor<1x1xf32>, tensor<1x1xi32>
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 1>, second_transpose = array<i64: 0>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%values : tensor<1x1xf32>) -> tensor<1xf32>
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 1>, second_transpose = array<i64: 0>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<1x1xi32>) -> tensor<1xi32>
//       CHECK: tensorrt.identity %{{.+}} : tensor<1xf32> to tensor<1xi32>

// -----

func.func @trt_argmin_2d_int_operand(%arg0: tensor<2x10xi32>) -> (tensor<1x10xi32>, tensor<1x10xi32>) {
  %0, %1 = tensorrt.argmin {
    axis = 0 : i64
  } %arg0 : tensor<2x10xi32> -> tensor<1x10xi32>, tensor<1x10xi32>
  return %0, %1 : tensor<1x10xi32>, tensor<1x10xi32>
}
// CHECK-LABEL: trt_argmin_2d_int_operand
//       CHECK: tensorrt.identity %{{.+}} : tensor<2x10xi32> to tensor<2x10xf32>
//       CHECK: tensorrt.top_k <kMIN> {axis = 0 : i64, k = 1 : i64}
//  CHECK-SAME: %{{.+}} : tensor<2x10xf32> -> tensor<1x10xf32>, tensor<1x10xi32>
//       CHECK: tensorrt.identity %{{.+}} : tensor<1x10xf32> to tensor<1x10xi32>

// -----

func.func @trt_argmax_1d_float_operand(%arg0: tensor<10xf32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %0, %1 = tensorrt.argmax {
    axis = 0 : i64
  } %arg0 : tensor<10xf32> -> tensor<1xf32>, tensor<1xi32>
  return %0, %1 : tensor<1xf32>, tensor<1xi32>
}
// CHECK-LABEL: @trt_argmax_1d_float_operand
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 1, 10>, second_transpose = array<i64: 0, 1>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<10xf32>) -> tensor<1x10xf32>
//       CHECK: tensorrt.top_k <kMAX> {axis = 1 : i64, k = 1 : i64}
//  CHECK-SAME: %{{.+}} : tensor<1x10xf32> -> tensor<1x1xf32>, tensor<1x1xi32>
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 1>, second_transpose = array<i64: 0>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<1x1xf32>) -> tensor<1xf32>
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 1>, second_transpose = array<i64: 0>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<1x1xi32>) -> tensor<1xi32>

// -----

func.func @trt_argmax_1d_int_operand(%arg0: tensor<10xi32>) -> (tensor<1xi32>, tensor<1xi32>) {
  %0, %1 = tensorrt.argmax {
    axis = 0 : i64
  } %arg0 : tensor<10xi32> -> tensor<1xi32>, tensor<1xi32>
  return %0, %1 : tensor<1xi32>, tensor<1xi32>
}
// CHECK-LABEL: trt_argmax_1d_int_operand
//       CHECK: tensorrt.identity %{{.+}} : tensor<10xi32> to tensor<10xf32>
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 1, 10>, second_transpose = array<i64: 0, 1>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<10xf32>) -> tensor<1x10xf32>
//       CHECK: tensorrt.top_k <kMAX> {axis = 1 : i64, k = 1 : i64}
//  CHECK-SAME: %{{.+}} : tensor<1x10xf32> -> tensor<1x1xf32>, tensor<1x1xi32>
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 1>, second_transpose = array<i64: 0>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%values : tensor<1x1xf32>) -> tensor<1xf32>
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 1>, second_transpose = array<i64: 0>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<1x1xi32>) -> tensor<1xi32>
//       CHECK: tensorrt.identity %{{.+}} : tensor<1xf32> to tensor<1xi32>

// -----

func.func @trt_argmax_2d_int_operand(%arg0: tensor<2x10xi32>) -> (tensor<1x10xi32>, tensor<1x10xi32>) {
  %0, %1 = tensorrt.argmax {
    axis = 0 : i64
  } %arg0 : tensor<2x10xi32> -> tensor<1x10xi32>, tensor<1x10xi32>
  return %0, %1 : tensor<1x10xi32>, tensor<1x10xi32>
}
// CHECK-LABEL: trt_argmax_2d_int_operand
//       CHECK: tensorrt.identity %{{.+}} : tensor<2x10xi32> to tensor<2x10xf32>
//       CHECK: tensorrt.top_k <kMAX> {axis = 0 : i64, k = 1 : i64}
//  CHECK-SAME: %{{.+}} : tensor<2x10xf32> -> tensor<1x10xf32>, tensor<1x10xi32>
//       CHECK: tensorrt.identity %{{.+}} : tensor<1x10xf32> to tensor<1x10xi32>

// -----

func.func @trt_reshape(%arg0: tensor<200xf32>) -> tensor<2x100xf32> {
  %0 = tensorrt.reshape %arg0 : tensor<200xf32> to tensor<2x100xf32>
  return %0 : tensor<2x100xf32>
}

// CHECK-LABEL: @trt_reshape
//       CHECK: tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 2, 100>,
//  CHECK-SAME:   second_transpose = array<i64: 0, 1>, zero_is_placeholder = false} ins(%{{.+}} : tensor<200xf32>) -> tensor<2x100xf32>

// -----

func.func @trt_reshape_i1(%arg0: tensor<200xi1>) -> tensor<2x100xi1> {
  %0 = tensorrt.reshape %arg0 : tensor<200xi1> to tensor<2x100xi1>
  return %0 : tensor<2x100xi1>
}

// CHECK-LABEL: @trt_reshape_i1
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<200xi1>)
//       CHECK: %[[v0:.+]] = tensorrt.identity %[[arg0]] : tensor<200xi1> to tensor<200xi32>
//       CHECK: %[[v1:.+]] = tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 2, 100>,
//  CHECK-SAME:   second_transpose = array<i64: 0, 1>
//  CHECK-SAME:   zero_is_placeholder = false
//  CHECK-SAME:   ins(%[[v0]] : tensor<200xi32>) -> tensor<2x100xi32>
//       CHECK: tensorrt.identity %[[v1]] : tensor<2x100xi32> to tensor<2x100xi1>

// -----

func.func @trt_reshape_dynamic(%arg0: tensor<200xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = tensorrt.reshape %arg0 shape(%arg1: tensor<2xi32>) : tensor<200xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @trt_reshape_dynamic
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor
//       CHECK: tensorrt.shuffle
//  CHECK-SAME:   first_transpose = array<i64: 0>
//  CHECK-SAME:   second_transpose = array<i64: 0, 1>
//  CHECK-SAME:   zero_is_placeholder = false
//  CHECK-SAME:   ins(%[[arg0]], %[[arg1]] : tensor<200xf32>, tensor<2xi32>) -> tensor<?x?xf32>

// -----

func.func @trt_reshape_dynamic_i1(%arg0: tensor<200xi1>, %arg1: tensor<2xi32>) -> tensor<?x?xi1> {
  %0 = tensorrt.reshape %arg0 shape(%arg1: tensor<2xi32>) : tensor<200xi1> to tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}

// CHECK-LABEL: @trt_reshape_dynamic_i1
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<{{.+}}>, %[[arg1:.+]]: tensor
//       CHECK: %[[v0:.+]] = tensorrt.identity %[[arg0]] : tensor<200xi1> to tensor<200xi32>
//       CHECK: %[[v1:.+]] = tensorrt.shuffle
//  CHECK-SAME:   first_transpose = array<i64: 0>
//  CHECK-SAME:   second_transpose = array<i64: 0, 1>
//  CHECK-SAME:   zero_is_placeholder = false
//  CHECK-SAME:   ins(%[[v0]], %[[arg1]] : tensor<200xi32>, tensor<2xi32>) -> tensor<?x?xi32>
//       CHECK: tensorrt.identity %[[v1]] : tensor<?x?xi32> to tensor<?x?xi1>


// -----

func.func @trt_reshape_dynamic_single_unknown_dim(%arg0: tensor<?xi32>) -> tensor<2x?xi32> {
  %0 = tensorrt.reshape %arg0 : tensor<?xi32> to tensor<2x?xi32>
  return %0 : tensor<2x?xi32>
}

// CHECK-LABEL: func.func @trt_reshape_dynamic_single_unknown_dim
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32>)
//       CHECK:     %[[v0:.+]] = tensorrt.shuffle
//  CHECK-SAME:       reshape = array<i64: 2, -1>
//  CHECK-SAME:       ins(%[[arg0]] : tensor<?xi32>) -> tensor<2x?xi32>
//       CHECK:     return %[[v0]] : tensor<2x?xi32>

// -----

// Check case no transpose/shuffle required.

func.func @broadcast_to_slice(%arg0: tensor<1x2xf32>) -> tensor<128x2xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> : tensor<1x2xf32> to tensor<128x2xf32>
  return %0 : tensor<128x2xf32>
}

// CHECK-LABEL: @broadcast_to_slice
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<1x2xf32>)
//       CHECK:   %[[v0:.+]] = tensorrt.slice %[[arg0]][0, 0][128, 2][1, 1] {mode = #tensorrt.slice_mode<kWRAP>} : tensor<1x2xf32> to tensor<128x2xf32>
//       CHECK:   return %[[v0]]

// -----

// Check scalar case.

func.func @broadcast_to_slice(%arg0: tensor<f32>) -> tensor<10xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<> : tensor<f32> to tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @broadcast_to_slice(
//  CHECK-SAME:   %[[arg0:.+]]: tensor<f32>) -> tensor<10xf32> {
//       CHECK:   %[[v0:.+]] = tensorrt.shuffle {
//  CHECK-SAME:       first_transpose = array<i64>
//  CHECK-SAME:       reshape = array<i64: 1>
//  CHECK-SAME:       second_transpose = array<i64: 0>
//  CHECK-SAME:       zero_is_placeholder = false
//  CHECK-SAME:       ins(%[[arg0]] : tensor<f32>) -> tensor<1xf32>
//       CHECK:   %[[v1:.+]] = tensorrt.slice %[[v0]][0][10][1] {mode = #tensorrt.slice_mode<kWRAP>} : tensor<1xf32> to tensor<10xf32>
//       CHECK:   return %[[v1]] : tensor<10xf32>

// -----

// Check permuted dims + reshape case.

func.func @broadcast_to_slice_perm(%arg0: tensor<1x10xf32>) -> tensor<4x10x4x28xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<3, 1>
                : tensor<1x10xf32> to tensor<4x10x4x28xf32>
  return %0 : tensor<4x10x4x28xf32>
}

// CHECK-LABEL: @broadcast_to_slice_perm(
//  CHECK-SAME:   %[[arg0:.+]]: tensor<1x10xf32>) -> tensor<4x10x4x28xf32> {
//       CHECK:   %[[v0:.+]] = tensorrt.shuffle {
//  CHECK-SAME:     first_transpose = array<i64: 1, 0>
//  CHECK-SAME:     reshape = array<i64: 1, 10, 1, 1>
//  CHECK-SAME:     second_transpose = array<i64: 0, 1, 2, 3>
//  CHECK-SAME:     zero_is_placeholder = false
//  CHECK-SAME:     ins(%[[arg0]] : tensor<1x10xf32>) -> tensor<1x10x1x1xf32>
//       CHECK:   %[[v1:.+]] = tensorrt.slice %[[v0]][0, 0, 0, 0][4, 10, 4, 28][1, 1, 1, 1] {mode = #tensorrt.slice_mode<kWRAP>} : tensor<1x10x1x1xf32> to tensor<4x10x4x28xf32>
//       CHECK:   return %[[v1]] : tensor<4x10x4x28xf32>

// -----

func.func @broadcast_to_slice_expand(%arg0: tensor<1x1xi32>) -> tensor<1x1x384xi32> {
  %1 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> : tensor<1x1xi32> to tensor<1x1x384xi32>
  return %1 : tensor<1x1x384xi32>
}

// CHECK-LABEL: @broadcast_to_slice_expand
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x1xi32>) -> tensor<1x1x384xi32> {
//       CHECK:     %[[v0:.+]] = tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 1, 1, 1>, second_transpose = array<i64: 0, 1, 2>, zero_is_placeholder = false} ins(%[[arg0]] : tensor<1x1xi32>) -> tensor<1x1x1xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.slice %[[v0]][0, 0, 0][1, 1, 384][1, 1, 1] {mode = #tensorrt.slice_mode<kWRAP>} : tensor<1x1x1xi32> to tensor<1x1x384xi32>
//       CHECK:     return %[[v1]] : tensor<1x1x384xi32>


// -----

func.func @broadcast_to_slice_static_to_dynamic(%arg0: tensor<1x1xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> shape(%arg1: tensor<2xi32>) : tensor<1x1xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @broadcast_to_slice_static_to_dynamic
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<1x1xf32>, %[[arg1:.+]]: tensor<2xi32>)
//       CHECK:   %[[v0:.+]] = tensorrt.slice %[[arg0]][0, 0][%[[arg1]]: tensor<2xi32>][1, 1]
//  CHECK-SAME:   mode = #tensorrt.slice_mode<kWRAP>
//  CHECK-SAME:    : tensor<1x1xf32> to tensor<?x?xf32>
//       CHECK:   return %[[v0]]

// -----

func.func @broadcast_to_slice_static_to_dynamic_perm(%arg0: tensor<1x1xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<1, 0> shape(%arg1: tensor<2xi32>) : tensor<1x1xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @broadcast_to_slice_static_to_dynamic_perm
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1x1xf32>, %[[arg1:.+]]: tensor<2xi32>) -> tensor<?x?xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.shuffle {first_transpose = array<i64: 1, 0>, second_transpose = array<i64: 0, 1>, zero_is_placeholder = false} ins(%[[arg0]] : tensor<1x1xf32>) -> tensor<1x1xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.slice %[[v0]][0, 0][%[[arg1:.+]]: tensor<2xi32>][1, 1] {mode = #tensorrt.slice_mode<kWRAP>} : tensor<1x1xf32> to tensor<?x?xf32>
//       CHECK:     return %[[v1]] : tensor<?x?xf32>

// -----

func.func @broadcast_to_slice_static_to_dynamic_expand(%arg0: tensor<1xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<1> shape(%arg1: tensor<2xi32>) : tensor<1xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @broadcast_to_slice_static_to_dynamic_expand
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xf32>, %[[arg1:.+]]: tensor<2xi32>) -> tensor<?x?xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 1, 1>, second_transpose = array<i64: 0, 1>, zero_is_placeholder = false} ins(%[[arg0]] : tensor<1xf32>) -> tensor<1x1xf32>
//       CHECK:     %[[v1:.+]] = tensorrt.slice %[[v0]][0, 0][%[[arg1:.+]]: tensor<2xi32>][1, 1] {mode = #tensorrt.slice_mode<kWRAP>} : tensor<1x1xf32> to tensor<?x?xf32>
//       CHECK:     return %[[v1]] : tensor<?x?xf32>

// -----

func.func @broadcast_to_slice_dynamic_to_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<1xi32>) -> tensor<?xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0> shape(%arg1: tensor<1xi32>) : tensor<?xf32> to tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @broadcast_to_slice_dynamic_to_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<1xi32>) -> tensor<?xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.slice %[[arg0]][0][%[[arg1:.+]]: tensor<1xi32>][1] {mode = #tensorrt.slice_mode<kWRAP>} : tensor<?xf32> to tensor<?xf32>
//       CHECK:     return %[[v0]] : tensor<?xf32>

// -----

func.func @broadcast_to_slice_dynamic_to_dynamic_expand(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0> shape(%arg1: tensor<2xi32>) : tensor<?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @broadcast_to_slice_dynamic_to_dynamic_expand
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<2xi32>) -> tensor<?x?xf32> {
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<1> : tensor<1xi32>
//       CHECK:     %[[v0:.+]] = tensorrt.shape %[[arg0]] : tensor<?xf32> -> tensor<1xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.slice %[[v0]][0][1][1] : tensor<1xi32> to tensor<1xi32>
//       CHECK:     %[[v2:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[v1]], %[[cst_i32]] : tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
//       CHECK:     %[[v3:.+]] = tensorrt.shuffle {first_transpose = array<i64: 0>, second_transpose = array<i64: 0, 1>, zero_is_placeholder = false} ins(%[[arg0]], %[[v2]] : tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
//       CHECK:     %[[v4:.+]] = tensorrt.slice %[[v3]][0, 0][%[[arg1:.+]]: tensor<2xi32>][1, 1] {mode = #tensorrt.slice_mode<kWRAP>} : tensor<?x1xf32> to tensor<?x?xf32>
//       CHECK:     return %[[v4]] : tensor<?x?xf32>


// -----


func.func @broadcast_to_slice_dynamic_to_dynamic_expand2(%arg0: tensor<?x?xf32>, %arg1: tensor<3xi32>) -> tensor<?x?x?xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 2> shape(%arg1: tensor<3xi32>) : tensor<?x?xf32> to tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @broadcast_to_slice_dynamic_to_dynamic_expand2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<3xi32>) -> tensor<?x?x?xf32> {
//       CHECK:     %[[cst_i32:.+]] = tensorrt.constant dense<1> : tensor<1xi32>
//       CHECK:     %[[v0:.+]] = tensorrt.shape %[[arg0]] : tensor<?x?xf32> -> tensor<2xi32>
//       CHECK:     %[[v1:.+]] = tensorrt.slice %[[v0]][0][1][1] : tensor<2xi32> to tensor<1xi32>
//       CHECK:     %[[v2:.+]] = tensorrt.slice %[[v0]][1][1][1] : tensor<2xi32> to tensor<1xi32>
//       CHECK:     %[[v3:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%[[v1]], %[[cst_i32]], %[[v2]] : tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
//       CHECK:     %[[v4:.+]] = tensorrt.shuffle {first_transpose = array<i64: 0, 1>, second_transpose = array<i64: 0, 1, 2>, zero_is_placeholder = false} ins(%[[arg0]], %[[v3]] : tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x1x?xf32>
//       CHECK:     %[[v5:.+]] = tensorrt.slice %[[v4]][0, 0, 0][%[[arg1:.+]]: tensor<3xi32>][1, 1, 1] {mode = #tensorrt.slice_mode<kWRAP>} : tensor<?x1x?xf32> to tensor<?x?x?xf32>
//       CHECK:     return %[[v5]] : tensor<?x?x?xf32>
