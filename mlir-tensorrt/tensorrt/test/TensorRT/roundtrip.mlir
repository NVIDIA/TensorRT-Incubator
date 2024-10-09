// RUN: tensorrt-opt -split-input-file %s | tensorrt-opt -split-input-file | FileCheck %s

// -----

func.func @trt_transpose(%arg0 : tensor<10x20x30xf32>) -> tensor<20x10x30xf32> {
  %0 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2)->(d1, d0, d2)>}
        %arg0 : tensor<10x20x30xf32> to tensor<20x10x30xf32>
  return %0 : tensor<20x10x30xf32>
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-LABEL: @trt_transpose
//  CHECK-NEXT: tensorrt.transpose {permutation = #[[$map]]} %{{.+}} : tensor<10x20x30xf32> to tensor<20x10x30xf32>

// -----

func.func @trt_transpose(%arg0 : tensor<10x20x30xf32>) -> tensor<?x?x?xf32> {
  %0 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2)->(d1, d0, d2)>}
        %arg0 : tensor<10x20x30xf32> to tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @trt_transpose
//  CHECK-NEXT: tensorrt.transpose {{.*}} : tensor<10x20x30xf32> to tensor<?x?x?xf32>

// -----

func.func @trt_einsum(%arg0: tensor<?x?xf32>) -> tensor<?xf32> {
  %0 = tensorrt.einsum {
    equation = "ij->i"
  } ins(%arg0 : tensor<?x?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @trt_einsum
//  CHECK-NEXT:  tensorrt.einsum {equation = "ij->i"} ins(%{{.+}} : tensor<?x?xf32>)

// -----

func.func @trt_fill_linspace() -> tensor<1024xf32> {
  %0 = tensorrt.linspace [0.0][static][1.0] : tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

// -----

func.func @trt_fill_linspace_dynamic(%start: tensor<f32>, %shape: tensor<4xi32>, %step: tensor<4xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tensorrt.linspace [%start: tensor<f32>][%shape: tensor<4xi32>][%step: tensor<4xf32>] : tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @trt_fill_linspace_dynamic(%start: tensor<f32>, %shape: tensor<4xi32>, %step: tensor<4xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tensorrt.linspace [%start: tensor<f32>][%shape: tensor<4xi32>][%step: tensor<4xf32>] : tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @trt_expand_rank(%arg0: tensor<100x100xf32>) -> tensor<1x1x1x100x100xf32> {
  %0 = tensorrt.expand_rank %arg0 : tensor<100x100xf32> to tensor<1x1x1x100x100xf32>
  return %0 : tensor<1x1x1x100x100xf32>
}

// CHECK-LABEL: @trt_expand_rank
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<{{.+}}>)
//       CHECK: tensorrt.expand_rank %[[arg0]] : tensor<100x100xf32> to tensor<1x1x1x100x100xf32>

// -----

func.func @trt_expand_rank_append_one(%arg0: tensor<100xf32>) -> tensor<100x1xf32> {
  %0 = tensorrt.expand_rank %arg0 : tensor<100xf32> to tensor<100x1xf32>
  return %0 : tensor<100x1xf32>
}

// CHECK-LABEL: @trt_expand_rank
//       CHECK: tensorrt.expand_rank %{{.+}} : tensor<100xf32> to tensor<100x1xf32>

// -----

func.func @trt_expand_rank_insert_one(%arg0: tensor<100x10xf32>) -> tensor<100x1x10xf32> {
  %0 = tensorrt.expand_rank %arg0 : tensor<100x10xf32> to tensor<100x1x10xf32>
  return %0 : tensor<100x1x10xf32>
}

// CHECK-LABEL: @trt_expand_rank
//       CHECK: tensorrt.expand_rank %{{.+}} : tensor<100x10xf32> to tensor<100x1x10xf32>


// -----

func.func @trt_collapse_rank(%arg0: tensor<100x1x100xf32>) -> tensor<100x100xf32> {
  %0 = tensorrt.collapse_rank %arg0 : tensor<100x1x100xf32> to tensor<100x100xf32>
  return %0 : tensor<100x100xf32>
}

// CHECK-LABEL: @trt_collapse_rank
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<{{.+}}>)
//       CHECK: tensorrt.collapse_rank %[[arg0]] : tensor<100x1x100xf32> to tensor<100x100xf32>

// -----

func.func @trt_collapse_rank(%arg0: tensor<100x1x1x100xf32>) -> tensor<100x1x100xf32> {
  %0 = tensorrt.collapse_rank %arg0 : tensor<100x1x1x100xf32> to tensor<100x1x100xf32>
  return %0 : tensor<100x1x100xf32>
}

// CHECK-LABEL: @trt_collapse_rank
//  CHECK-SAME:    (%[[arg0:.+]]: tensor<{{.+}}>)
//       CHECK: tensorrt.collapse_rank %[[arg0]] : tensor<100x1x1x100xf32> to tensor<100x1x100xf32>

// -----

func.func @trt_reshape(%arg0: tensor<100x10xf32>) -> tensor<1000xf32> {
  %0 = tensorrt.reshape %arg0 : tensor<100x10xf32> to tensor<1000xf32>
  return %0 : tensor<1000xf32>
}

// CHECK-LABEL: @trt_reshape
//  CHECK-NEXT:   tensorrt.reshape %{{.+}} : tensor<100x10xf32> to tensor<1000xf32>

// -----

func.func @trt_reshape_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<3xi32>) -> tensor<?x?x?xf32> {
  %0 = tensorrt.reshape %arg0 shape(%arg1: tensor<3xi32>) : tensor<?x?xf32> to tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @trt_reshape_dynamic
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<3xi32>)
//  CHECK-NEXT:   tensorrt.reshape %[[arg0]] shape(%[[arg1]]: tensor<3xi32>) : tensor<?x?xf32> to tensor<?x?x?xf32>

// -----

// Positive test for `tensorrt.element_wise`. Inputs are ranked tensors with unknown dimensions.
func.func @trt_element_wise(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensorrt.element_wise <kSUM>(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: trt_element_wise
//       CHECK: <kSUM>(%{{.+}}, %{{.+}} : tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

// LHS dimension is known and RHS dimension is unknown.
func.func @trt_element_wise_dynamic(%arg0: tensor<2x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<2x?x?xf32> {
  %0 = tensorrt.element_wise <kSUM>(%arg0, %arg1 : tensor<2x?x?xf32>, tensor<?x?x?xf32>) -> tensor<2x?x?xf32>
  return %0 : tensor<2x?x?xf32>
}

// CHECK-LABEL: trt_element_wise_dynamic
//       CHECK: <kSUM>(%{{.+}}, %{{.+}} : tensor<2x?x?xf32>, tensor<?x?x?xf32>) -> tensor<2x?x?xf32>

// -----

func.func @broadcast_ewise_shape_infer_compatible(%arg0: tensor<10xf32>, %arg1: tensor<1xf32>) -> tensor<?xf32> {
  %0 = tensorrt.element_wise <kSUM>(%arg0, %arg1 : tensor<10xf32>, tensor<1xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @broadcast_ewise_shape_infer_compatible
//  CHECK-NEXT:  tensorrt.element_wise

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x128x128xf32> {
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}
// CHECK-LABEL: @trt_convolution
//       CHECK: tensorrt.convolution

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>) -> tensor<?x64x128x128xf32> {
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}
// CHECK-LABEL: @trt_convolution
//       CHECK: tensorrt.convolution

// -----

// Positive test for `tensorrt.activation`. Input is ranked tensor with unknown dimensions
func.func @trt_activation(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %arg0 : tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: @trt_activation
//       CHECK: tensorrt.activation
//  CHECK-SAME: activationType = #tensorrt.activation_type<kRELU>
//  CHECK-NOT: alpha
//  CHECK-NOT: beta
//  CHECK-SAME: %{{.+}} : tensor<?x?x?x?xf32>
// -----

// Positive test for `tensorrt.reduce`. Input is ranked tensor with unknown dimensions
func.func @trt_reduce(%arg0: tensor<2x3x224x224xf32>) -> tensor<2x224x224xf32> {
  %0 = tensorrt.reduce <kMAX> %arg0 {
    reduceAxes = array<i64: 1>,
    keepDimensions = false
  } : tensor<2x3x224x224xf32> -> tensor<2x224x224xf32>
  return %0 : tensor<2x224x224xf32>
}
// CHECK-LABEL: @trt_reduce
//       CHECK: tensorrt.reduce
//  CHECK-SAME: <kMAX> %{{.+}} {
//  CHECK-SAME: reduceAxes = array<i64: 1>
//   CHECK-NOT: keepDimensions
//  CHECK-SAME: : tensor<2x3x224x224xf32> -> tensor<2x224x224xf32>
// -----

func.func @trt_pooling_dynamic(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 2, 2>,
    prePadding = array<i64: 1, 1>,
    postPadding = array<i64: 1, 1>
  } ins(%arg0 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: @trt_pooling
//       CHECK: tensorrt.pooling

// -----

func.func @trt_pooling_static(%arg0: tensor<10x64x112x112xf32>) -> tensor<10x64x56x56xf32> {
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 2, 2>,
    prePadding = array<i64: 1, 1>,
    postPadding = array<i64: 1, 1>
  } ins(%arg0 : tensor<10x64x112x112xf32>) -> tensor<10x64x56x56xf32>
  return %0 : tensor<10x64x56x56xf32>
}
// CHECK-LABEL: @trt_pooling
//       CHECK: tensorrt.pooling
//  CHECK-SAME: {poolingType = #tensorrt.pooling_type<kMAX>, postPadding = array<i64: 1, 1>,
//  CHECK-SAME:  prePadding = array<i64: 1, 1>, stride = array<i64: 2, 2>,
//  CHECK-SAME:   windowSize = array<i64: 3, 3>} ins(%arg0 : tensor<10x64x112x112xf32>) -> tensor<10x64x56x56xf32>

// -----

func.func @trt_pooling_static_fp16(%arg0: tensor<1x64x112x112xf16>) -> tensor<1x64x56x56xf16> {
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX>,
    windowSize = array<i64: 3, 3>,
    stride = array<i64: 2, 2>,
    prePadding = array<i64: 1, 1>,
    postPadding = array<i64: 0, 0>
  } ins(%arg0 : tensor<1x64x112x112xf16>) -> tensor<1x64x56x56xf16>
  return %0 : tensor<1x64x56x56xf16>
}
// CHECK-LABEL: @trt_pooling
//       CHECK: tensorrt.pooling
//  CHECK-SAME: {poolingType = #tensorrt.pooling_type<kMAX>, postPadding = array<i64: 0, 0>,
//  CHECK-SAME:  prePadding = array<i64: 1, 1>, stride = array<i64: 2, 2>,
//  CHECK-SAME:  windowSize = array<i64: 3, 3>} ins(%arg0 : tensor<1x64x112x112xf16>) -> tensor<1x64x56x56xf16>

// -----

func.func @trt_shuffle(%arg0: tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 3, 1, 2>,
    reshape = array<i64: 2, 3, 224, 224>,
    second_transpose = array<i64: 0, 1, 2, 3>,
    zero_is_placeholder = true
  } ins(%arg0 : tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32>

  return %0 : tensor<2x3x224x224xf32>
}
// CHECK-LABEL: @trt_shuffle
//       CHECK: tensorrt.shuffle
//  CHECK-SAME: first_transpose = array<i64: 0, 3, 1, 2>
//  CHECK-SAME: reshape = array<i64: 2, 3, 224, 224>
//  CHECK-SAME: second_transpose = array<i64: 0, 1, 2, 3>
//  CHECK-SAME: ins(%{{.+}} : tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32>

// -----

func.func @trt_shuffle_reshape(%arg0: tensor<2x224x224x3xf32>) -> tensor<2x3x224x224x1xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 3, 1, 2>,
    reshape = array<i64: 2, 3, 224, 224, 1>,
    second_transpose = array<i64: 0, 1, 2, 3, 4>,
    zero_is_placeholder = true
  } ins(%arg0 : tensor<2x224x224x3xf32>) -> tensor<2x3x224x224x1xf32>

  return %0 : tensor<2x3x224x224x1xf32>
}
// CHECK-LABEL: @trt_shuffle_reshape
//       CHECK: tensorrt.shuffle
//  CHECK-SAME: first_transpose = array<i64: 0, 3, 1, 2>
//  CHECK-SAME: reshape = array<i64: 2, 3, 224, 224, 1>
//  CHECK-SAME: second_transpose = array<i64: 0, 1, 2, 3, 4>
//  CHECK-SAME: ins(%{{.+}} : tensor<2x224x224x3xf32>) -> tensor<2x3x224x224x1xf32>

// -----

func.func @trt_shuffle_only_first_trans(%arg0: tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 3, 1, 2>,
    second_transpose = array<i64: 0, 1, 2, 3>
  } ins(%arg0 : tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32>

  return %0 : tensor<2x3x224x224xf32>
}
// CHECK-LABEL: @trt_shuffle_only_first_trans
//       CHECK: tensorrt.shuffle
//  CHECK-SAME: {first_transpose = array<i64: 0, 3, 1, 2>, second_transpose = array<i64: 0, 1, 2, 3>}
//  CHECK-SAME: ins(%{{.+}} : tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32>

// -----

func.func @trt_shuffle_only_second_trans(%arg0: tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    second_transpose = array<i64: 0, 3, 1, 2>
  } ins(%arg0 : tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32>

  return %0 : tensor<2x3x224x224xf32>
}
// CHECK-LABEL: @trt_shuffle_only_second_trans
//       CHECK: tensorrt.shuffle
//  CHECK-SAME: {first_transpose = array<i64: 0, 1, 2, 3>, second_transpose = array<i64: 0, 3, 1, 2>}
//  CHECK-SAME: ins(%{{.+}} : tensor<2x224x224x3xf32>) -> tensor<2x3x224x224xf32>

// -----

func.func @trt_shuffle_reshape_negative_infer(%arg0: tensor<3x4x6xf32>) -> tensor<3x24xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2>,
    reshape = array<i64: 3, -1>,
    second_transpose = array<i64: 0, 1>
  } ins(%arg0 : tensor<3x4x6xf32>) -> tensor<3x24xf32>

  return %0 : tensor<3x24xf32>
}
// CHECK-LABEL: @trt_shuffle_reshape_negative_infer
//       CHECK: tensorrt.shuffle
//  CHECK-SAME: {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 3, -1>, second_transpose = array<i64: 0, 1>}
//  CHECK-SAME: ins(%{{.+}} : tensor<3x4x6xf32>) -> tensor<3x24xf32>

// -----

func.func @trt_shuffle_reshape_negative_infer2(%arg0: tensor<2x3x4x6xf32>) -> tensor<6x4x6xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: 6, -1, 6>,
    second_transpose = array<i64: 0, 1, 2>
  } ins(%arg0 : tensor<2x3x4x6xf32>) -> tensor<6x4x6xf32>

  return %0 : tensor<6x4x6xf32>
}
// CHECK-LABEL: @trt_shuffle_reshape_negative_infer2
//       CHECK: tensorrt.shuffle
//  CHECK-SAME: {first_transpose = array<i64: 0, 1, 2, 3>, reshape = array<i64: 6, -1, 6>, second_transpose = array<i64: 0, 1, 2>}
//  CHECK-SAME: ins(%{{.+}} : tensor<2x3x4x6xf32>) -> tensor<6x4x6xf32>

// -----

func.func @trt_shuffle_reshape_copy(%arg0: tensor<1x10x20x40xf32>) -> tensor<1x10x800xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: 0, 0, 800>,
    second_transpose = array<i64: 0, 1, 2>
  } ins(%arg0 : tensor<1x10x20x40xf32>) -> tensor<1x10x800xf32>

  return %0 : tensor<1x10x800xf32>
}
// CHECK-LABEL: @trt_shuffle_reshape_copy
//       CHECK: tensorrt.shuffle
//  CHECK-SAME: {first_transpose = array<i64: 0, 1, 2, 3>, reshape = array<i64: 0, 0, 800>, second_transpose = array<i64: 0, 1, 2>}
//  CHECK-SAME: ins(%{{.+}} : tensor<1x10x20x40xf32>) -> tensor<1x10x800xf32>

// -----

func.func @trt_shuffle_reshape(%arg0: tensor<30x20x10x1xf32>) -> tensor<30x20x10xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: -1, 0, 0>,
    second_transpose = array<i64: 0, 1, 2>
  } ins(%arg0 : tensor<30x20x10x1xf32>) -> tensor<30x20x10xf32>
  return %0 : tensor<30x20x10xf32>
}
// CHECK-LABEL: @trt_shuffle_reshape
//       CHECK: tensorrt.shuffle
//  CHECK-SAME: first_transpose = array<i64: 0, 1, 2, 3>
//  CHECK-SAME: reshape = array<i64: -1, 0, 0>
//  CHECK-SAME: second_transpose = array<i64: 0, 1, 2>}
//  CHECK-SAME: ins(%{{.+}} : tensor<30x20x10x1xf32>) -> tensor<30x20x10xf32>

// -----

func.func @trt_shuffle_from_scalar(%arg0: tensor<f32>) -> tensor<1x1x1xf32> {
    %1 = tensorrt.shuffle {
        first_transpose = array<i64>,
        reshape = array<i64: 1, 1, 1>,
        second_transpose = array<i64: 0, 1, 2>,
        zero_is_placeholder = false
    } ins(%arg0 : tensor<f32>) -> tensor<1x1x1xf32>
    return %1 : tensor<1x1x1xf32>
}
// CHECK-LABEL: @trt_shuffle_from_scalar
//  CHECK-NEXT: tensorrt.shuffle
//  CHECK-SAME: {first_transpose = array<i64>, reshape = array<i64: 1, 1, 1>, second_transpose = array<i64: 0, 1, 2>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<f32>) -> tensor<1x1x1xf32>

// -----

func.func @trt_shuffle_to_scalar(%arg0: tensor<1x1x1xf32>) -> tensor<f32> {
    %1 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2>,
        reshape = array<i64>,
        second_transpose = array<i64>,
        zero_is_placeholder = false
    } ins(%arg0 : tensor<1x1x1xf32>) -> tensor<f32>

    return %1 : tensor<f32>
}
// CHECK-LABEL: @trt_shuffle_to_scalar
//  CHECK-NEXT: tensorrt.shuffle
//  CHECK-SAME: {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64>, second_transpose = array<i64>, zero_is_placeholder = false}
//  CHECK-SAME: ins(%{{.+}} : tensor<1x1x1xf32>) -> tensor<f32>

// -----

func.func @trt_zero_reshape(%arg0: tensor<0xf32>) -> tensor<10x0xf32> {
    %1 = tensorrt.shuffle {
        first_transpose = array<i64: 0>,
        reshape = array<i64: 10, 0>,
        second_transpose = array<i64: 0, 1>,
        zero_is_placeholder = false
    } ins(%arg0 : tensor<0xf32>) -> tensor<10x0xf32>
    return %1 : tensor<10x0xf32>
}
// CHECK-LABEL: @trt_zero_reshape
//  CHECK-NEXT: tensorrt.shuffle
//  CHECK-SAME: first_transpose = array<i64: 0>
//  CHECK-SAME: reshape = array<i64: 10, 0>
//  CHECK-SAME: second_transpose = array<i64: 0, 1>
//  CHECK-SAME: zero_is_placeholder = false
//  CHECK-SAME: ins(%{{.+}} : tensor<0xf32>) -> tensor<10x0xf32>

// -----

func.func @trt_zero_reshape_dynamic(%arg0: tensor<?x?x1xf32>) -> tensor<?x?xf32> {
    %1 = tensorrt.shuffle {
        first_transpose = array<i64: 0, 1, 2>,
        reshape = array<i64: 0, 0>,
        second_transpose = array<i64: 0, 1>,
        zero_is_placeholder = true
    } ins(%arg0 : tensor<?x?x1xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @trt_zero_reshape_dynamic
//  CHECK-NEXT: tensorrt.shuffle
//  CHECK-SAME: first_transpose = array<i64: 0, 1, 2>
//  CHECK-SAME: reshape = array<i64: 0, 0>
//  CHECK-SAME: second_transpose = array<i64: 0, 1>
//  CHECK-SAME: ins(%{{.+}} : tensor<?x?x1xf32>) -> tensor<?x?xf32>

// -----

func.func @trt_slice_static(%arg0: tensor<?x?xf32>) -> tensor<2x2xf32> {
  %0 = tensorrt.slice %arg0 [0, 0][2, 2][1, 1] : tensor<?x?xf32> to tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: @trt_slice_static
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<?x?xf32>)
//  CHECK-NEXT:   tensorrt.slice %[[arg0]][0, 0][2, 2][1, 1] : tensor<?x?xf32> to tensor<2x2xf32>

// -----

// Static result with dynamic size param is legal.

func.func @trt_slice_static_with_dynamic_param(%arg0: tensor<?x?xf32>, %arg1: tensor<2xi32>) -> tensor<2x2xf32> {
  %0 = tensorrt.slice %arg0 [0, 0][%arg1: tensor<2xi32>][1, 1] : tensor<?x?xf32> to tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: @trt_slice_static_with_dynamic_param
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor
//  CHECK-NEXT:   tensorrt.slice %[[arg0]][0, 0][%[[arg1]]: tensor<2xi32>][1, 1] : tensor<?x?xf32> to tensor<2x2xf32>

// -----

// Unknown shapes with static 'sizes' is legal.

func.func @trt_slice_static_dynamic_result(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensorrt.slice %arg0 [0, 0][2, 2][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: @trt_slice_static_dynamic_result
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<?x?xf32>)
//  CHECK-NEXT:   tensorrt.slice %[[arg0]][0, 0][2, 2][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

// -----

func.func @trt_slice_dynamic(%arg0: tensor<?x?xf32>, %start: tensor<2xi32>, %stride: tensor<2xi32>, %size: tensor<2xi32>) -> tensor<2x2xf32> {
  %0 = tensorrt.slice %arg0 [%start: tensor<2xi32>][%stride: tensor<2xi32>][%size: tensor<2xi32>] : tensor<?x?xf32> to tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: @trt_slice_dynamic
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<?x?xf32>, %[[start:.+]]: tensor<2xi32>, %[[stride:.+]]: tensor<2xi32>, %[[size:.+]]: tensor<2xi32>)
//  CHECK-NEXT:   tensorrt.slice %[[arg0]][%[[start]]: tensor<2xi32>][%[[stride]]: tensor<2xi32>][%[[size]]: tensor<2xi32>] : tensor<?x?xf32> to tensor<2x2xf32>

// -----

func.func @trt_slice_tile(%arg0: tensor<1x4xf32>, %arg1: tensor<f32>) -> tensor<16x4xf32> {
  %0 = tensorrt.slice %arg0 [0, 0][16, 4][0, 1] : tensor<1x4xf32> to tensor<16x4xf32>
  return %0 : tensor<16x4xf32>
}

// CHECK-LABEL: @trt_slice_tile
//  CHECK-NEXT:   tensorrt.slice %{{.+}}[0, 0][16, 4][0, 1] :

// -----


func.func @trt_slice_pad(%arg0: tensor<128x128xf32>) -> tensor<130x130xf32> {
  %cst = tensorrt.constant dense<0.0> : tensor<1xf32>
  %0 = tensorrt.slice %arg0[-1, -1][130, 130][1, 1] fill(%cst : tensor<1xf32>) {
    mode = #tensorrt.slice_mode<kFILL>
  }: tensor<128x128xf32> to tensor<130x130xf32>
  return %0 : tensor<130x130xf32>
}

// CHECK-LABEL: @trt_slice_pad
//  CHECK-NEXT: tensorrt.constant
//  CHECK-NEXT: tensorrt.slice %{{.+}}[-1, -1][130, 130][1, 1] fill(%{{.+}} : tensor<1xf32>)

// -----

func.func @trt_topk_1(%arg0: tensor<?x?xf32>) -> (tensor<?x1xf32>, tensor<?x1xi32>) {
  %0, %1 = tensorrt.top_k <kMAX> {
    k = 1 : i64,
    axis = 1 : i64
  } %arg0 : tensor<?x?xf32> -> tensor<?x1xf32>, tensor<?x1xi32>
  return %0, %1 : tensor<?x1xf32>, tensor<?x1xi32>
}

// CHECK-LABEL: @trt_topk_1
//  CHECK-NEXT: tensorrt.top_k <kMAX>
//  CHECK-SAME:  axis = 1 : i64
//  CHECK-SAME:  k = 1 : i64
//  CHECK-SAME:  %{{.+}} : tensor<?x?xf32> -> tensor<?x1xf32>, tensor<?x1xi32>

// -----

func.func @trt_topk_32(%arg0: tensor<?x?xf32>) -> (tensor<?x32xf32>, tensor<?x32xi32>) {
  %0, %1 = tensorrt.top_k <kMAX> {
    k = 32 : i64,
    axis = 1 : i64
  } %arg0 : tensor<?x?xf32> -> tensor<?x32xf32>, tensor<?x32xi32>
  return %0, %1 : tensor<?x32xf32>, tensor<?x32xi32>
}

// CHECK-LABEL: @trt_topk_32
//  CHECK-NEXT: tensorrt.top_k <kMAX>
//  CHECK-SAME:  axis = 1 : i64
//  CHECK-SAME:  k = 32 : i64
//  CHECK-SAME:  %{{.+}} : tensor<?x?xf32> -> tensor<?x32xf32>, tensor<?x32xi32>

// -----

func.func @trt_argmin(%arg0: tensor<?x10xf32>) -> (tensor<?x1xf32>, tensor<?x1xi32>) {
  // expected-error @below {{'tensorrt.argmin' op "axis" attribute is -1, but this is out of bounds for input of rank 2}}
  %0, %1 = tensorrt.argmin {
    axis = 1 : i64
  } %arg0 : tensor<?x10xf32> -> tensor<?x1xf32>, tensor<?x1xi32>
  return %0, %1 : tensor<?x1xf32>, tensor<?x1xi32>
}

// CHECK-LABEL: @trt_argmin
//       CHECK:   tensorrt.argmin
//  CHECK-SAME:     axis = 1 : i64
//  CHECK-SAME:     %{{.+}} : tensor<?x10xf32> -> tensor<?x1xf32>, tensor<?x1xi32>

// -----

func.func @trt_argmax(%arg0: tensor<?x10xf32>) -> (tensor<?x1xf32>, tensor<?x1xi32>) {
  %0, %1 = tensorrt.argmax {
    axis = 1 : i64
  } %arg0 : tensor<?x10xf32> -> tensor<?x1xf32>, tensor<?x1xi32>
  return %0, %1 : tensor<?x1xf32>, tensor<?x1xi32>
}

// CHECK-LABEL: @trt_argmax
//       CHECK:   tensorrt.argmax
//  CHECK-SAME:     axis = 1 : i64
//  CHECK-SAME:     %{{.+}} : tensor<?x10xf32> -> tensor<?x1xf32>, tensor<?x1xi32>

// -----

func.func @trt_broadcast(%arg0: tensor<1024x1024xf32>) -> tensor<10x1024x1024xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<1, 2> : tensor<1024x1024xf32> to tensor<10x1024x1024xf32>
  return %0 : tensor<10x1024x1024xf32>
}

// CHECK-LABEL: @trt_broadcast
//  CHECK-NEXT:  tensorrt.broadcast %{{.+}} broadcast_dims<1, 2> : tensor<1024x1024xf32> to tensor<10x1024x1024xf32>

// -----

func.func @trt_broadcast(%arg0: tensor<1024x1024xf32>) -> tensor<10x1024x1024xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<2, 1> : tensor<1024x1024xf32> to tensor<10x1024x1024xf32>
  return %0 : tensor<10x1024x1024xf32>
}

// CHECK-LABEL: @trt_broadcast
//  CHECK-NEXT:  tensorrt.broadcast %{{.+}} broadcast_dims<2, 1> : tensor<1024x1024xf32> to tensor<10x1024x1024xf32>

// -----

func.func @trt_broadcast(%arg0: tensor<1x1xf32>) -> tensor<1x10xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> : tensor<1x1xf32> to tensor<1x10xf32>
  return %0 : tensor<1x10xf32>
}

// CHECK-LABEL: @trt_broadcast
//  CHECK-NEXT:  tensorrt.broadcast %{{.+}} broadcast_dims<0, 1> : tensor<1x1xf32> to tensor<1x10xf32>

// -----

func.func @trt_broadcast(%arg0: tensor<f32>) -> tensor<10x1xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<> : tensor<f32> to tensor<10x1xf32>
  return %0 : tensor<10x1xf32>
}

// CHECK-LABEL: @trt_broadcast
//  CHECK-NEXT:  tensorrt.broadcast %{{.+}} broadcast_dims<> : tensor<f32> to tensor<10x1xf32>

// -----


func.func @trt_broadcast(%arg0: tensor<1x1xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> shape(%arg1: tensor<2xi32>) : tensor<1x1xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @trt_broadcast
//  CHECK-NEXT:  tensorrt.broadcast %{{.+}} broadcast_dims<0, 1> shape(%{{.+}} : tensor<2xi32>)
//  CHECK-SAME:   : tensor<1x1xf32> to tensor<?x?xf32>

// -----

// Static shape with dynamic tensor is legal if ranks align.

func.func @trt_broadcast(%arg0: tensor<1x1xf32>, %arg1: tensor<2xi32>) -> tensor<10x10xf32> {
  %0 = tensorrt.broadcast %arg0 broadcast_dims<0, 1> shape(%arg1: tensor<2xi32>) : tensor<1x1xf32> to tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_broadcast
//  CHECK-NEXT:  tensorrt.broadcast %{{.+}} broadcast_dims<0, 1> shape(%{{.+}} : tensor<2xi32>)
//  CHECK-SAME:   : tensor<1x1xf32> to tensor<10x10xf32>

// -----

func.func @trt_identity(%arg0: tensor<10xf32>) -> tensor<10xf16> {
  %0 = tensorrt.identity %arg0 : tensor<10xf32> to tensor<10xf16>
  return %0 : tensor<10xf16>
}

// CHECK-LABEL: @trt_identity
//  CHECK-NEXT:  tensorrt.identity %{{.+}} : tensor<10xf32> to tensor<10xf16>

// -----

func.func @trt_identity1(%arg0: tensor<10xi1>) -> tensor<10xf32> {
  %0 = tensorrt.identity %arg0 : tensor<10xi1> to tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @trt_identity1
//  CHECK-NEXT:  tensorrt.identity %{{.+}} : tensor<10xi1> to tensor<10xf32>

// -----

func.func @trt_identity84(%arg0: tensor<10xi1>) -> tensor<10xf32> {
  %0 = tensorrt.identity84 %arg0 : tensor<10xi1> to tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @trt_identity84
//  CHECK-NEXT:  tensorrt.identity84 %{{.+}} : tensor<10xi1> to tensor<10xf32>

// -----

func.func @trt_concatenation(%arg0: tensor<1x10xf32>, %arg1: tensor<3x10xf32>) -> tensor<4x10xf32> {
  %0 = tensorrt.concatenation {axis = 0 : i32} ins(%arg0, %arg1: tensor<1x10xf32>, tensor<3x10xf32>)
    -> tensor<4x10xf32>
  return %0 : tensor<4x10xf32>
}

// CHECK-LABEL: @trt_concatenation
//  CHECK-NEXT:  tensorrt.concatenation {axis = 0 : i32} ins({{.+}}) -> tensor<4x10xf32>

// -----

func.func @trt_concatenation_dynamic(%arg0: tensor<?x10xf32>, %arg1: tensor<3x10xf32>, %arg2: tensor<?x10xf32>) -> tensor<10x10xf32> {
  %0 = tensorrt.concatenation {axis = 0 : i32}
    ins(%arg0, %arg1, %arg2: tensor<?x10xf32>, tensor<3x10xf32>, tensor<?x10xf32>)
    -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_concatenation_dynamic
//       CHECK:     %[[v0:.+]] = tensorrt.concatenation {axis = 0 : i32} ins(%{{.+}}, %{{.+}}, %{{.+}} : tensor<?x10xf32>, tensor<3x10xf32>, tensor<?x10xf32>) -> tensor<10x10xf32>
//       CHECK:     return %[[v0]] : tensor<10x10xf32>

// -----

func.func @trt_concatenation_dynamic_2(%arg0: tensor<?x?x768xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x768xf32> {
  %0 = tensorrt.concatenation {axis = 1 : i32}
    ins(%arg0, %arg1: tensor<?x?x768xf32>, tensor<?x?x?xf32>)
    -> tensor<?x?x768xf32>
  return %0 : tensor<?x?x768xf32>
}

// CHECK-LABEL: @trt_concatenation_dynamic_2
//       CHECK: %[[v0:.+]] = tensorrt.concatenation {axis = 1 : i32} ins(%{{.+}}, %{{.+}} : tensor<?x?x768xf32>, tensor<?x?x?xf32>) -> tensor<?x?x768xf32>
//       CHECK: return %[[v0]] : tensor<?x?x768xf32>
// -----

func.func @trt_select(%arg0: tensor<10x10xi1>, %arg1: tensor<1x10xf32>, %arg2: tensor<10x1xf32>) -> tensor<10x10xf32> {
  %0 = tensorrt.select ins(%arg0, %arg1, %arg2: tensor<10x10xi1>, tensor<1x10xf32>, tensor<10x1xf32>)
    -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_select
//  CHECK-NEXT:  tensorrt.select ins(%{{.+}}, %{{.+}}, %{{.+}} : tensor<10x10xi1>, tensor<1x10xf32>, tensor<10x1xf32>) -> tensor<10x10xf32>

// -----

func.func @trt_select2(%arg0: tensor<10x10xi1>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xf32>) -> tensor<10x10xf32> {
  %0 = tensorrt.select ins(%arg0, %arg1, %arg2: tensor<10x10xi1>, tensor<1x1xf32>, tensor<1x1xf32>)
    -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_select2
//  CHECK-NEXT:  tensorrt.select ins(%{{.+}}, %{{.+}}, %{{.+}} : tensor<10x10xi1>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<10x10xf32>

// -----

func.func @trt_softmax(%arg0 : tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
  %0 = tensorrt.softmax {axis = 2 : i64} %arg0 : tensor<10x10x10xf32>
  return %0 : tensor<10x10x10xf32>
}

// CHECK-LABEL: @trt_softmax
//  CHECK-NEXT: tensorrt.softmax {axis = 2 : i64} %{{.+}} : tensor<10x10x10xf32>

// -----

func.func @trt_padding1(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x1x12x12xf32> {
  %0 = tensorrt.padding {
    prePadding = array<i64: 1, 1>,
    postPadding = array<i64: 1, 1>
  } ins(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x1x12x12xf32>

  return %0 : tensor<1x1x12x12xf32>
}

// CHECK-LABEL: @trt_padding1
//       CHECK: tensorrt.padding
//  CHECK-SAME: ins(%{{.+}} : tensor<1x1x10x10xf32>) -> tensor<1x1x12x12xf32>

// -----

func.func @trt_padding2(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x1x8x8xf32> {
  %0 = tensorrt.padding {
    prePadding = array<i64: -1, -1>,
    postPadding = array<i64: -1, -1>
  } ins(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x1x8x8xf32>

  return %0 : tensor<1x1x8x8xf32>
}

// CHECK-LABEL: @trt_padding2
//       CHECK: tensorrt.padding
//  CHECK-SAME: ins(%{{.+}} : tensor<1x1x10x10xf32>) -> tensor<1x1x8x8xf32>

// -----

func.func @trt_padding3(%arg0 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tensorrt.padding {
    prePadding = array<i64: -1, -1>,
    postPadding = array<i64: -1, -1>
  } ins(%arg0 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @trt_padding3
//       CHECK: tensorrt.padding
//  CHECK-SAME: ins(%{{.+}} : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

// -----

func.func @trt_onehot_innermost(%indices: tensor<3xi32>, %values: tensor<2xi32>, %depth: tensor<i32>) -> tensor<3x?xi32> {
  %0 = tensorrt.one_hot {
    axis = -1 : si64
  } ins(%indices, %values, %depth : tensor<3xi32>, tensor<2xi32>, tensor<i32>) -> tensor<3x?xi32>

  return %0 : tensor<3x?xi32>
}
// CHECK-LABEL: @trt_onehot_innermost
//       CHECK: tensorrt.one_hot
//  CHECK-SAME: [[axis:.+]] = -1
//  CHECK-SAME: ins(%[[indices:.+]], %[[values:.+]], %[[depth:.+]] : tensor<3xi32>, tensor<2xi32>, tensor<i32>) -> tensor<3x?xi32>

// -----

func.func @trt_onehot_innermost2(%indices: tensor<3x2xi32>, %values: tensor<2xf32>, %depth: tensor<i32>) -> tensor<3x2x?xf32> {
  %0 = tensorrt.one_hot {
    axis = 2 : si64 // rank(indices)
  } ins(%indices, %values, %depth : tensor<3x2xi32>, tensor<2xf32>, tensor<i32>) -> tensor<3x2x?xf32>

  return %0 : tensor<3x2x?xf32>
}
// CHECK-LABEL: @trt_onehot_innermost
//       CHECK: tensorrt.one_hot
//  CHECK-SAME: [[axis:.+]] = 2
//  CHECK-SAME: ins(%[[indices:.+]], %[[values:.+]], %[[depth:.+]] : tensor<3x2xi32>, tensor<2xf32>, tensor<i32>) -> tensor<3x2x?xf32>

// -----

func.func @trt_onehot_outermost(%indices: tensor<3xi32>, %values: tensor<2xi32>, %depth: tensor<i32>) -> tensor<?x3xi32> {
  %0 = tensorrt.one_hot {
    axis = 0 : si64
  } ins(%indices, %values, %depth : tensor<3xi32>, tensor<2xi32>, tensor<i32>) -> tensor<?x3xi32>

  return %0 : tensor<?x3xi32>
}
// CHECK-LABEL: @trt_onehot_outermost
//       CHECK: tensorrt.one_hot
//  CHECK-SAME: [[axis:.+]] = 0
//  CHECK-SAME: ins(%[[indices:.+]], %[[values:.+]], %[[depth:.+]] : tensor<3xi32>, tensor<2xi32>, tensor<i32>) -> tensor<?x3xi32>

// -----

func.func @trt_ragged_softmax(%input: tensor<1x3x5xf32>, %bounds: tensor<1x3x1xi32>) -> tensor<1x3x5xf32> {
  %0 = tensorrt.ragged_softmax ins(%input, %bounds : tensor<1x3x5xf32>, tensor<1x3x1xi32>) -> tensor<1x3x5xf32>
  return %0 : tensor<1x3x5xf32>
}
// CHECK-LABEL: @trt_ragged_softmax
//       CHECK: tensorrt.ragged_softmax
//  CHECK-SAME: ins(%[[input:.+]], %[[bounds:.+]] : tensor<1x3x5xf32>, tensor<1x3x1xi32>) -> tensor<1x3x5xf32>

// -----

func.func @trt_matmul_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: @trt_matmul_dynamic
//       CHECK: tensorrt.matrix_multiply
//  CHECK-SAME: [[op0:.+]] = #tensorrt.matrix_operation<kNONE>
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

func.func @trt_matmul_transpose(%arg0: tensor<20x10xf32>, %arg1: tensor<20x40xf32>) -> tensor<10x40xf32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kTRANSPOSE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<20x10xf32>, tensor<20x40xf32>) -> tensor<10x40xf32>

  return %0 : tensor<10x40xf32>
}
// CHECK-LABEL: @trt_matmul_transpose
//       CHECK: tensorrt.matrix_multiply
//  CHECK-SAME: [[op0:.+]] = #tensorrt.matrix_operation<kTRANSPOSE>
//  CHECK-SAME: [[op1:.+]] = #tensorrt.matrix_operation<kNONE>
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<20x10xf32>, tensor<20x40xf32>) -> tensor<10x40xf32>

// -----

func.func @trt_matmul_vector(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<f32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kVECTOR>,
    op1 = #tensorrt.matrix_operation<kVECTOR>
  } ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>) -> tensor<f32>

  return %0 : tensor<f32>
}
// CHECK-LABEL: @trt_matmul_vector
//       CHECK: tensorrt.matrix_multiply
//  CHECK-SAME: {[[op0:.+]] = #tensorrt.matrix_operation<kVECTOR>, [[op1:.+]] = #tensorrt.matrix_operation<kVECTOR>}
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<10xf32>, tensor<10xf32>) -> tensor<f32>

// -----

func.func @trt_matmul(%arg0: tensor<10x20xf32>, %arg1: tensor<20x10xf32>) -> tensor<10x10xf32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<10x20xf32>, tensor<20x10xf32>) -> tensor<10x10xf32>

  return %0 : tensor<10x10xf32>
}
// CHECK-LABEL: @trt_matmul
//       CHECK: tensorrt.matrix_multiply
//  CHECK-SAME: [[op0:.+]] = #tensorrt.matrix_operation<kNONE>
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<10x20xf32>, tensor<20x10xf32>) -> tensor<10x10xf32>

// -----

func.func @trt_matmul_matrix_vector_2d(%arg0: tensor<10x128x64xf32>, %arg1: tensor<1x64xf32>)
            -> tensor<10x128xf32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kVECTOR>
  } ins(%arg0, %arg1 : tensor<10x128x64xf32>, tensor<1x64xf32>) -> tensor<10x128xf32>
  return %0 : tensor<10x128xf32>
}
// CHECK-LABEL: @trt_matmul_matrix_vector_2d
//       CHECK: tensorrt.matrix_multiply
//  CHECK-SAME: {[[op0:.+]] = #tensorrt.matrix_operation<kNONE>, [[op1:.+]] = #tensorrt.matrix_operation<kVECTOR>}
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<10x128x64xf32>, tensor<1x64xf32>) -> tensor<10x128xf32>

// -----

func.func @trt_matmul_vector_matrix_vector(%arg0: tensor<128xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<256xf32>)
            -> tensor<f32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kVECTOR>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<128xf32>, tensor<128x256xf32>) -> tensor<256xf32>
  %1 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kVECTOR>,
    op1 = #tensorrt.matrix_operation<kVECTOR>
  } ins(%0, %arg2 : tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
  return %1 : tensor<f32>
}
// CHECK-LABEL: @trt_matmul_vector_matrix_vector
//       CHECK: %[[out0:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kVECTOR>, op1 = #tensorrt.matrix_operation<kNONE>}
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<128xf32>, tensor<128x256xf32>) -> tensor<256xf32>
//  CHECK-NEXT: %[[out1:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kVECTOR>, op1 = #tensorrt.matrix_operation<kVECTOR>}
//  CHECK-SAME: ins(%[[out0]], %[[arg2:.+]] : tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
// -----

func.func @trt_matmul_matrix_3d(%arg0: tensor<10x128x64xf32>, %arg1: tensor<10x64x256xf32>)
            -> tensor<10x128x256xf32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<10x128x64xf32>, tensor<10x64x256xf32>) -> tensor<10x128x256xf32>
  return %0 : tensor<10x128x256xf32>
}
// CHECK-LABEL: @trt_matmul_matrix_3d
//       CHECK: tensorrt.matrix_multiply
//  CHECK-SAME: {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<10x128x64xf32>, tensor<10x64x256xf32>) -> tensor<10x128x256xf32>
// -----

func.func @trt_matmul_matrix_vector_1d(%arg0: tensor<128x64xf32>, %arg1: tensor<64xf32>)
            -> tensor<128xf32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kVECTOR>
  } ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}
// CHECK-LABEL: @trt_matmul_matrix_vector_1d
//       CHECK: tensorrt.matrix_multiply
//  CHECK-SAME: {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kVECTOR>}
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<128x64xf32>, tensor<64xf32>) -> tensor<128xf32>
// -----

func.func @trt_vector2d_matrix(%arg0: tensor<1x64xf32>, %arg1: tensor<10x64x128xf32>)
            -> tensor<10x128xf32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kVECTOR>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<1x64xf32>, tensor<10x64x128xf32>) -> tensor<10x128xf32>
  return %0 : tensor<10x128xf32>
}
// CHECK-LABEL: @trt_vector2d_matrix
//       CHECK: tensorrt.matrix_multiply
//  CHECK-SAME: {op0 = #tensorrt.matrix_operation<kVECTOR>, op1 = #tensorrt.matrix_operation<kNONE>}
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<1x64xf32>, tensor<10x64x128xf32>) -> tensor<10x128xf32>

// -----

func.func @trt_matrix_multiply_mat_vec(%arg0: tensor<1x1x1x128x64xf32>, %arg1: tensor<1x1x1x64xf32>)
            -> tensor<1x1x1x128xf32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kVECTOR>
  } ins(%arg0, %arg1 : tensor<1x1x1x128x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x128xf32>
  return %0 : tensor<1x1x1x128xf32>
}

// CHECK-LABEL: @trt_matrix_multiply_mat_vec
//       CHECK:   tensorrt.matrix_multiply

// -----

func.func @trt_matrix_multiply_trans_vec(%arg0: tensor<1x1x1x50x10xf32>, %arg1: tensor<1x4x240x50xf32>) -> tensor<1x4x240x10xf32> {
  %0 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kTRANSPOSE>, op1 = #tensorrt.matrix_operation<kVECTOR>}
    ins(%arg0, %arg1 : tensor<1x1x1x50x10xf32>, tensor<1x4x240x50xf32>) -> tensor<1x4x240x10xf32>
  return %0 : tensor<1x4x240x10xf32>
}

// CHECK-LABEL: @trt_matrix_multiply_trans_vec
//       CHECK:   tensorrt.matrix_multiply

// -----

func.func @trt_matrix_multiply_vec_mat(%arg0: tensor<64xf32>, %arg1: tensor<64x128xf32>)
            -> tensor<128xf32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kVECTOR>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<64xf32>, tensor<64x128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: @trt_matrix_multiply_vec_mat
//       CHECK:   tensorrt.matrix_multiply

// -----

func.func @trt_gather(%arg0: tensor<10x20x30xf32>, %arg1: tensor<5xi32>) -> tensor<10x5x30xf32> {
  %0 = tensorrt.gather {
    axis = 1 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<5xi32>) -> tensor<10x5x30xf32>
  return %0 : tensor<10x5x30xf32>
}

// CHECK-LABEL: @trt_gather
//       CHECK:   tensorrt.gather
//  CHECK-SAME:     axis = 1 : i64
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<10x20x30xf32>, tensor<5xi32>)
//  CHECK-SAME:     -> tensor<10x5x30xf32>

// -----

func.func @trt_gather_elements(%arg0: tensor<10x20x30xf32>, %arg1: tensor<10x20x10xi32>) -> tensor<10x20x10xf32> {
  %0 = tensorrt.gather_elements {
    axis = 2 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<10x20x10xi32>) -> tensor<10x20x10xf32>
  return %0 : tensor<10x20x10xf32>
}

// CHECK-LABEL: @trt_gather_elements
//       CHECK:   tensorrt.gather_elements
//  CHECK-SAME:     axis = 2 : i64
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<10x20x30xf32>, tensor<10x20x10xi32>)
//  CHECK-SAME:     -> tensor<10x20x10xf32>

// -----

func.func @trt_nonzero_static(%arg0: tensor<3x7x9xf32>) -> tensor<3x?xi32> {
  %0 = tensorrt.non_zero %arg0 : tensor<3x7x9xf32> -> tensor<3x?xi32>
  return %0 : tensor<3x?xi32>
}

// CHECK-LABEL: @trt_nonzero_static
//       CHECK: tensorrt.non_zero
//  CHECK-SAME: %{{.+}} : tensor<3x7x9xf32> -> tensor<3x?xi32>

// -----

func.func @trt_nonzero_dynamic(%arg0: tensor<?x?x?xf32>) -> tensor<3x?xi32> {
  %0 = tensorrt.non_zero %arg0 : tensor<?x?x?xf32> -> tensor<3x?xi32>
  return %0 : tensor<3x?xi32>
}

// CHECK-LABEL: @trt_nonzero_dynamic
//       CHECK: tensorrt.non_zero
//  CHECK-SAME: %{{.+}} : tensor<?x?x?xf32> -> tensor<3x?xi32>
func.func @trt_if(%arg0: tensor<i1>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %result = tensorrt.if (%arg0: tensor<i1>) -> tensor<10xf32> {
      tensorrt.yield %arg1: tensor<10xf32>
    } else {
      tensorrt.yield %arg1: tensor<10xf32>
    }
  return %result: tensor<10xf32>
}

// -----

// CHECK-LABEL: @trt_if
//       CHECK:   tensorrt.if
//       CHECK:   tensorrt.yield
//       CHECK:   else
//       CHECK:   tensorrt.yield
func.func @trt_if(%arg1: tensor<10xf32>, %arg2: tensor<10xf32>) -> tensor<10xf32> {
  %cond = arith.constant dense<1> : tensor<i1>
  %result = tensorrt.if (%cond: tensor<i1>) -> tensor<10xf32> {
      %add = tensorrt.element_wise <kSUM>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
          -> tensor<10xf32>
      tensorrt.yield %arg1: tensor<10xf32>
    } else {
      %sub = tensorrt.element_wise <kSUB>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
          -> tensor<10xf32>
      tensorrt.yield %arg1: tensor<10xf32>
    }
  return %result: tensor<10xf32>
}

// -----

func.func @trt_shape_bounds(%arg1: tensor<10xf32> {
                              tensorrt.profile = #tensorrt.shape_profile<
                                  min = [10],
                                  opt = [10],
                                  max= [10]
                                >
                              },
                            %arg2: tensor<f32> {
                              tensorrt.profile = #tensorrt.shape_profile<
                                  min = [],
                                  opt = [],
                                  max=  []
                                >
                              }
                            ) -> (tensor<10xf32>, tensor<f32>) {
  return %arg1, %arg2: tensor<10xf32>, tensor<f32>
}

// CHECK-LABEL: @trt_shape_bounds
//  CHECK-SAME:   (%{{.+}}: tensor<10xf32> {tensorrt.profile =
//  CHECK-SAME:      #tensorrt.shape_profile<min = [10], opt = [10], max = [10]>},
//  CHECK-SAME:    %{{arg1}}: tensor<f32> {tensorrt.profile =
//  CHECK-SAME:      #tensorrt.shape_profile<min = [], opt = [], max = []>})

// -----

func.func @trt_shape_op(%arg0: tensor<?x?x?x?xf32>) -> tensor<4xi32> {
  %0 = tensorrt.shape %arg0 : tensor<?x?x?x?xf32> -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: @trt_shape_op
//       CHECK:   tensorrt.shape %{{.+}} : tensor<?x?x?x?xf32> -> tensor<4xi32>

// -----

func.func @trt_shape_op_0d(%arg0: tensor<f32>) -> tensor<0xi32> {
  %0 = tensorrt.shape %arg0 : tensor<f32> -> tensor<0xi32>
  return %0 : tensor<0xi32>
}

// CHECK-LABEL: @trt_shape_op_0d
//       CHECK:   tensorrt.shape %{{.+}} : tensor<f32> -> tensor<0xi32>

// -----

func.func @trt_parametric_relu_op_same_dimensions(%arg0: tensor<2x10xf32>, %arg1: tensor<2x10xf32>) -> tensor<2x10xf32> {
  %0 = tensorrt.parametric_relu ins(%arg0, %arg1 : tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
  return %0 : tensor<2x10xf32>
}

// CHECK-LABEL: @trt_parametric_relu_op_same_dimensions
//       CHECK:   tensorrt.parametric_relu
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>

// -----

func.func @trt_parametric_relu_op_brodcast_dimensions(%arg0: tensor<2x10xf32>, %arg1: tensor<1x1xf32>) -> tensor<2x10xf32> {
  %0 = tensorrt.parametric_relu ins(%arg0, %arg1 : tensor<2x10xf32>, tensor<1x1xf32>) -> tensor<2x10xf32>
  return %0 : tensor<2x10xf32>
}

// CHECK-LABEL: @trt_parametric_relu_op_brodcast_dimensions
//       CHECK:   tensorrt.parametric_relu
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<2x10xf32>, tensor<1x1xf32>) -> tensor<2x10xf32>

// -----

func.func @trt_parametric_relu_op_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensorrt.parametric_relu ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @trt_parametric_relu_op_dynamic
//       CHECK:   tensorrt.parametric_relu
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// -----

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x128x128xf32> {
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}
// CHECK-LABEL: @trt_deconvolution
//       CHECK: tensorrt.deconvolution

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>) -> tensor<?x64x128x128xf32> {
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) -> tensor<?x64x128x128xf32>
  return %0 : tensor<?x64x128x128xf32>
}
// CHECK-LABEL: @trt_deconvolution
//       CHECK: tensorrt.deconvolution

// -----

func.func @trt_resize_nearest(%arg0: tensor<10x10xf32>) -> tensor<20x20xf32> {
  %result = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>
  } %arg0 : (tensor<10x10xf32>) -> tensor<20x20xf32>
  return %result : tensor<20x20xf32>
}

// CHECK-LABEL: @trt_resize_nearest
//       CHECK: tensorrt.resize_nearest
//  CHECK-SAME: coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>
//  CHECK-SAME: nearestRounding = #tensorrt.resize_round_mode<kFLOOR>
//  CHECK-SAME: selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>
//  CHECK-SAME: %{{.+}} : (tensor<10x10xf32>) -> tensor<20x20xf32>

// -----

func.func @trt_resize_nearest_output_shape(%arg0: tensor<?x?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %result = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>
  } %arg0, %arg1 : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
// CHECK-LABEL: @trt_resize_nearest_output_shape
//       CHECK: tensorrt.resize_nearest
//  CHECK-SAME: coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>
//  CHECK-SAME: nearestRounding = #tensorrt.resize_round_mode<kFLOOR>
//  CHECK-SAME: selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>
//  CHECK-SAME: %[[arg0:.+]], %[[arg1:.+]] : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>

// -----

func.func @trt_resize_linear(%arg0: tensor<10x10xf32>) -> tensor<20x20xf32> {
  %result = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>,
    selectorForSinglePixel = #tensorrt.resize_selector<kUPPER>
  } %arg0 : (tensor<10x10xf32>) -> tensor<20x20xf32>
  return %result : tensor<20x20xf32>
}
// CHECK-LABEL: @trt_resize_linear
//       CHECK: tensorrt.resize_linear
//  CHECK-SAME: coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>
//  CHECK-SAME: selectorForSinglePixel = #tensorrt.resize_selector<kUPPER>
//  CHECK-SAME: %{{.+}} : (tensor<10x10xf32>) -> tensor<20x20xf32>

// -----

func.func @trt_resize_linear_output_shape(%arg0: tensor<?x?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %result = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>,
    selectorForSinglePixel = #tensorrt.resize_selector<kUPPER>
  } %arg0, %arg1 : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
// CHECK-LABEL: @trt_resize_linear_output_shape
//       CHECK: tensorrt.resize_linear
//  CHECK-SAME: coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>
//  CHECK-SAME: selectorForSinglePixel = #tensorrt.resize_selector<kUPPER>
//  CHECK-SAME: %[[arg0:.+]], %[[arg1:.+]] : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>

// -----

func.func @trt_resize_cubic(%arg0: tensor<10x10xf32>) -> tensor<20x20xf32> {
  %result = tensorrt.resize_cubic {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kHALF_PIXEL>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>,
    cubicCoeff = -0.75 : f32
  } %arg0 : (tensor<10x10xf32>) -> tensor<20x20xf32>
  return %result : tensor<20x20xf32>
}
// CHECK-LABEL: @trt_resize_cubic
//       CHECK: tensorrt.resize_cubic
//  CHECK-SAME: coordinateTransformation = #tensorrt.resize_coordinate_transformation<kHALF_PIXEL>
//  CHECK-SAME: cubicCoeff = -7.500000e-01 : f32
//  CHECK-SAME: selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>
//  CHECK-SAME: %{{.+}} : (tensor<10x10xf32>) -> tensor<20x20xf32>

// -----

func.func @trt_resize_cubic_output_shape(%arg0: tensor<?x?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %result = tensorrt.resize_cubic {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kHALF_PIXEL>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>,
    cubicCoeff = -0.75 : f32
  } %arg0, %arg1 : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
// CHECK-LABEL: @trt_resize_cubic_output_shape
//       CHECK: tensorrt.resize_cubic
//  CHECK-SAME: coordinateTransformation = #tensorrt.resize_coordinate_transformation<kHALF_PIXEL>
//  CHECK-SAME: cubicCoeff = -7.500000e-01 : f32
//  CHECK-SAME: selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>
//  CHECK-SAME: %[[arg0:.+]], %[[arg1:.+]] : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>

// -----

func.func @trt_matmul_transpose_dynamic(%arg0: tensor<?x10xf32>, %arg1: tensor<20x40xf32>) -> tensor<10x40xf32> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kTRANSPOSE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<?x10xf32>, tensor<20x40xf32>) -> tensor<10x40xf32>

  return %0 : tensor<10x40xf32>
}
// CHECK-LABEL: @trt_matmul_transpose_dynamic
//       CHECK: tensorrt.matrix_multiply
//  CHECK-SAME: [[op0:.+]] = #tensorrt.matrix_operation<kTRANSPOSE>
//  CHECK-SAME: [[op1:.+]] = #tensorrt.matrix_operation<kNONE>
//  CHECK-SAME: ins(%[[arg0:.+]], %[[arg1:.+]] : tensor<?x10xf32>, tensor<20x40xf32>) -> tensor<10x40xf32>

// -----

func.func @trt_for_loop(%lb: tensor<i32>, %ub: tensor<i32>, %step: tensor<i32>, %arg0: tensor<10xf16>, %arg1: tensor<10xf16>)
    -> (tensor<10xf16>, tensor<10xf16>) {
  %0, %1 = tensorrt.for %i = %lb to %ub step %step init(%iter0 = %arg0, %iter1 = %arg1) -> (tensor<10xf16>, tensor<10xf16>) {
    tensorrt.yield %iter0, %iter1 : tensor<10xf16>, tensor<10xf16>
  }
  return %0, %1 : tensor<10xf16>, tensor<10xf16>
}

// CHECK-LABEL: @trt_for_loop
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<i32>, %[[arg1:.+]]: tensor<i32>, %[[arg2:.+]]: tensor<i32>, %[[arg3:.+]]: tensor<10xf16>, %[[arg4:.+]]: tensor<10xf16>)
//       CHECK:     %[[v0:.+]]:2 = tensorrt.for %[[arg5:.+]] = %[[arg0]] to %[[arg1]] step %[[arg2]] init(%[[arg6:.+]] = %[[arg3]], %[[arg7:.+]] = %[[arg4]]) -> (tensor<10xf16>, tensor<10xf16>) {
//       CHECK:     tensorrt.yield %[[arg6]], %[[arg7]] : tensor<10xf16>, tensor<10xf16>
//       CHECK:   return %[[v0]]#0, %[[v0]]#1 : tensor<10xf16>, tensor<10xf16>

// -----

func.func @trt_deconvolution(%arg0: tensor<?x32x?x?xf32>, %arg1: tensor<32x64x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x?x?xf32> {
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<?x32x?x?xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) biasWeights(%arg2 : tensor<64xf32>) -> tensor<?x64x?x?xf32>
  return %0 : tensor<?x64x?x?xf32>
}

// CHECK-LABEL: @trt_deconvolution
//       CHECK: tensorrt.deconvolution

// -----


func.func @trt_deconvolution_kernel(%arg0: tensor<4x4x5x5xf32>, %arg1: tensor<4x4x3x3xf32>, %arg2: tensor<4xf32>) -> tensor<4x4x7x7xf32> {
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 0, 0>,
    post_padding = array<i64: 0, 0>,
    dilation = array<i64: 1, 1>,
    num_groups = 1 : ui32
  } in(%arg0 : tensor<4x4x5x5xf32>) kernelWeights(%arg1: tensor<4x4x3x3xf32>) biasWeights(%arg2 : tensor<4xf32>) -> tensor<4x4x7x7xf32>
  return %0 : tensor<4x4x7x7xf32>
}

// CHECK-LABEL: @trt_deconvolution_kernel
//       CHECK: tensorrt.deconvolution

// -----

func.func @trt_deconvolution_stride(%arg0: tensor<4x4x5x5xf32>, %arg1: tensor<4x4x3x3xf32>, %arg2: tensor<4xf32>) -> tensor<4x4x11x11xf32> {
  %0 = tensorrt.deconvolution {
    stride = array<i64: 2, 2>,
    pre_padding = array<i64: 0, 0>,
    post_padding = array<i64: 0, 0>,
    dilation = array<i64: 1, 1>,
    num_groups = 1 : ui32
  } in(%arg0 : tensor<4x4x5x5xf32>) kernelWeights(%arg1: tensor<4x4x3x3xf32>) biasWeights(%arg2 : tensor<4xf32>) -> tensor<4x4x11x11xf32>
  return %0 : tensor<4x4x11x11xf32>
}

// CHECK-LABEL: @trt_deconvolution_stride
//       CHECK: tensorrt.deconvolution

// -----

func.func @trt_deconvolution_padding(%arg0: tensor<4x4x5x5xf32>, %arg1: tensor<4x4x3x3xf32>, %arg2: tensor<4xf32>) -> tensor<4x4x3x5xf32> {
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 2, 1>,
    post_padding = array<i64: 2, 1>,
    dilation = array<i64: 1, 1>,
    num_groups = 1 : ui32
  } in(%arg0 : tensor<4x4x5x5xf32>) kernelWeights(%arg1: tensor<4x4x3x3xf32>) biasWeights(%arg2 : tensor<4xf32>) -> tensor<4x4x3x5xf32>
  return %0 : tensor<4x4x3x5xf32>
}

// CHECK-LABEL: @trt_deconvolution_padding
//       CHECK: tensorrt.deconvolution

// -----

func.func @trt_deconvolution_stride_padding(%arg0: tensor<4x4x5x5xf32>, %arg1: tensor<4x4x3x3xf32>, %arg2: tensor<4xf32>) -> tensor<4x4x11x13xf32> {
  %0 = tensorrt.deconvolution {
    stride = array<i64: 3, 3>,
    pre_padding = array<i64: 2, 1>,
    post_padding = array<i64: 2, 1>,
    dilation = array<i64: 1, 1>,
    num_groups = 1 : ui32
  } in(%arg0 : tensor<4x4x5x5xf32>) kernelWeights(%arg1: tensor<4x4x3x3xf32>) biasWeights(%arg2 : tensor<4xf32>) -> tensor<4x4x11x13xf32>
  return %0 : tensor<4x4x11x13xf32>
}

// CHECK-LABEL: @trt_deconvolution_stride_padding
//       CHECK: tensorrt.deconvolution

// -----

func.func @trt_deconvolution_stride_padding_dilation(%arg0: tensor<4x4x5x5xf32>, %arg1: tensor<4x4x3x3xf32>, %arg2: tensor<4xf32>) -> tensor<4x4x21x23xf32> {
  %0 = tensorrt.deconvolution {
    stride = array<i64: 3, 3>,
    pre_padding = array<i64: 2, 1>,
    post_padding = array<i64: 2, 1>,
    dilation = array<i64: 6, 6>,
    num_groups = 1 : ui32
  } in(%arg0 : tensor<4x4x5x5xf32>) kernelWeights(%arg1: tensor<4x4x3x3xf32>) biasWeights(%arg2 : tensor<4xf32>) -> tensor<4x4x21x23xf32>
  return %0 : tensor<4x4x21x23xf32>
}

// CHECK-LABEL: @trt_deconvolution_stride_padding_dilation
//       CHECK: tensorrt.deconvolution

// -----

func.func @trt_convolution(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x1x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x130x130xf32> {
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    num_groups = 32 : ui32
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x1x1x1xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x130x130xf32>
  return %0 : tensor<?x64x130x130xf32>
}

// CHECK-LABEL: @trt_convolution
//       CHECK: tensorrt.convolution

// -----

func.func @trt_convolution_dyamic(%arg0: tensor<?x32x?x?xf32>, %arg1: tensor<32x1x1x1xf32>, %arg2: tensor<32xf32>) -> tensor<?x32x129x129xf32> {
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    dilation = array<i64: 1, 1>,
    num_groups = 32 : ui32
  } in(%arg0 : tensor<?x32x?x?xf32>) kernel(%arg1: tensor<32x1x1x1xf32>) bias(%arg2 : tensor<32xf32>) -> tensor<?x32x129x129xf32>
  return %0 : tensor<?x32x129x129xf32>
}

// CHECK-LABEL: @trt_convolution_dyamic
//       CHECK: tensorrt.convolution

// -----

func.func @trt_convolution_stride(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x63x63xf32> {
  %0 = tensorrt.convolution {
    stride = array<i64: 2, 2>,
    pre_padding = array<i64: 0, 0>,
    post_padding = array<i64: 0, 0>,
    dilation = array<i64: 1, 1>,
    num_groups = 1 : ui32
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x63x63xf32>
  return %0 : tensor<?x64x63x63xf32>
}

// CHECK-LABEL: @trt_convolution_stride
//       CHECK: tensorrt.convolution
// -----

func.func @trt_convolution_dilation(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x124x124xf32> {
  %0 = tensorrt.convolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 0, 0>,
    post_padding = array<i64: 0, 0>,
    dilation = array<i64: 2, 2>,
    num_groups = 1 : ui32
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x124x124xf32>
  return %0 : tensor<?x64x124x124xf32>
}

// CHECK-LABEL: @trt_convolution_dilation
//       CHECK: tensorrt.convolution
// -----

func.func @trt_convolution_stride_dilation(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x62x62xf32> {
  %0 = tensorrt.convolution {
    stride = array<i64: 2, 2>,
    pre_padding = array<i64: 0, 0>,
    post_padding = array<i64: 0, 0>,
    dilation = array<i64: 2, 2>,
    num_groups = 1 : ui32
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x62x62xf32>
  return %0 : tensor<?x64x62x62xf32>
}

// CHECK-LABEL: @trt_convolution_stride_dilation
//       CHECK: tensorrt.convolution
// -----

func.func @trt_convolution_stride_dilation_pad(%arg0: tensor<?x32x128x128xf32>, %arg1: tensor<64x32x3x3xf32>, %arg2: tensor<64xf32>) -> tensor<?x64x66x66xf32> {
  %0 = tensorrt.convolution {
    stride = array<i64: 2, 2>,
    pre_padding = array<i64: 4, 4>,
    post_padding = array<i64: 4, 4>,
    dilation = array<i64: 2, 2>,
    num_groups = 1 : ui32
  } in(%arg0 : tensor<?x32x128x128xf32>) kernel(%arg1: tensor<64x32x3x3xf32>) bias(%arg2 : tensor<64xf32>) -> tensor<?x64x66x66xf32>
  return %0 : tensor<?x64x66x66xf32>
}

// CHECK-LABEL: @trt_convolution_stride_dilation_pad
//       CHECK: tensorrt.convolution

// -----

func.func @trt_scatter_nd(%arg0: tensor<4x4x4xf32>, %arg1: tensor<2x1xi32>, %arg2: tensor<2x4x4xf32>) -> tensor<4x4x4xf32> {
  %0 = tensorrt.scatter_nd
    data(%arg0: tensor<4x4x4xf32>)
    indices(%arg1: tensor<2x1xi32>)
    updates(%arg2: tensor<2x4x4xf32>)

    return %0: tensor<4x4x4xf32>
}

// CHECK-LABEL: @trt_scatter_nd
//       CHECK:   tensorrt.scatter_nd
//  CHECK-SAME:     data(%{{.+}} : tensor<4x4x4xf32>) indices(%{{.+}} : tensor<2x1xi32>) updates(%{{.+}}: tensor<2x4x4xf32>)

// -----

func.func @trt_unary_neg(%arg0: tensor<10x10x!quant.uniform<i8:f32, 1.0:0>>)
    -> tensor<10x10x!quant.uniform<i8:f32, 1.0:0>> {
  %0 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNEG>} %arg0 : tensor<10x10x!quant.uniform<i8:f32, 1.0:0>>
  return %0 : tensor<10x10x!quant.uniform<i8:f32, 1.0:0>>
}

// CHECK-LABEL: @trt_unary_neg
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10x10x!quant.uniform<i8:f32, 1.0{{.*}}>>) -> tensor<10x10x!quant.uniform<i8:f32, 1.0{{.*}}>> {
//       CHECK:     %[[v0:.+]] = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kNEG>} %[[arg0]] : tensor<10x10x!quant.uniform
//       CHECK:     return %[[v0]]

// -----

func.func @trt_while_loop() -> (tensor<f32>, tensor<f32>) {
  %one = tensorrt.constant dense<1.0> : tensor<f32>
  %iter_init = tensorrt.constant dense<1.0> : tensor<f32>
  %limit = tensorrt.constant dense<10.0> : tensor<f32>
  %res_init = tensorrt.constant dense<0.0> : tensor<f32>
  %result0, %result1 = tensorrt.while(%iter_init,%res_init : tensor<f32>,tensor<f32>) -> tensor<f32>,tensor<f32>
  {
    // condition
    ^bb0(%iter:tensor<f32>, %result: tensor<f32>):
      %cond = tensorrt.element_wise <kLESS>(%iter, %limit : tensor<f32>, tensor<f32>)
            -> tensor<i1>
      tensorrt.condition(%cond : tensor<i1>) %iter, %result : tensor<f32>,tensor<f32>
  } ,
  {
    // body
    ^bb1(%iter:tensor<f32>, %result: tensor<f32>):
      %new_result = tensorrt.element_wise <kSUM> (%iter, %result : tensor<f32>, tensor<f32>) -> tensor<f32>
      %new_iter = tensorrt.element_wise <kSUM> (%one, %iter : tensor<f32>, tensor<f32>) -> tensor<f32>
      tensorrt.yield %new_iter, %new_result: tensor<f32>, tensor<f32>
  }
  return %result0, %result1 : tensor<f32>,tensor<f32>
}
// CHECK-LABEL: @trt_while_loop
//       CHECK: tensorrt.while
//  CHECK-SAME: (%[[c_f32_0:.+]], %[[c_f32_2:.+]] : tensor<f32>, tensor<f32>) -> tensor<f32>, tensor<f32>
//       CHECK: tensorrt.condition
//       CHECK: tensorrt.yield %[[v2:.+]], %[[v1:.+]] : tensor<f32>, tensor<f32>
//       CHECK: return

// -----

func.func @trt_einsum_1d_view(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tensorrt.einsum {
    equation = "i"
  } ins(%arg0 : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: trt_einsum_1d_view
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "i"} ins(%{{.+}} : tensor<?xf32>) -> tensor<?xf32>
// -----

func.func @trt_einsum_all_sum(%arg0: tensor<?xf32>) -> tensor<f32> {
  %0 = tensorrt.einsum {
    equation = "i->"
  } ins(%arg0 : tensor<?xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: trt_einsum_all_sum
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "i->"} ins(%{{.+}} : tensor<?xf32>) -> tensor<f32>

// -----

func.func @trt_einsum_all_sum_2d(%arg0: tensor<?x?xf32>) -> tensor<f32> {
  %0 = tensorrt.einsum {
    equation = "ii->"
  } ins(%arg0 : tensor<?x?xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: trt_einsum_all_sum_2d
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ii->"} ins(%{{.+}} : tensor<?x?xf32>) -> tensor<f32>

// -----

func.func @trt_einsum_elementwise_mul(%arg0: tensor<4x7xf16>, %arg1: tensor<4x7xf16>) -> tensor<4x7xf16> {
  %0 = tensorrt.einsum {
    equation = "ij, ij -> ij"
  } ins(%arg0, %arg1 : tensor<4x7xf16>, tensor<4x7xf16>) ->tensor<4x7xf16>
  return %0 : tensor<4x7xf16>
}

// CHECK-LABEL: trt_einsum_elementwise_mul
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ij, ij -> ij"} ins(%{{.+}}, %{{.+}} : tensor<4x7xf16>, tensor<4x7xf16>) -> tensor<4x7xf16>

// -----

func.func @trt_einsum_inner_product(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
  %0 = tensorrt.einsum {
    equation = "i,i"
  } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) ->tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: trt_einsum_inner_product
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "i,i"} ins(%{{.+}}, %{{.+}} : tensor<4xf32>, tensor<4xf32>) -> tensor<f32>

// -----

func.func @trt_einsum_outer_product(%arg0: tensor<4xf32>, %arg1: tensor<5xf32>) -> tensor<4x5xf32> {
  %0 = tensorrt.einsum {
    equation = "i,j -> ij"
  } ins(%arg0, %arg1 : tensor<4xf32>, tensor<5xf32>) ->tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// CHECK-LABEL: trt_einsum_outer_product
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "i,j -> ij"} ins(%{{.+}}, %{{.+}} : tensor<4xf32>, tensor<5xf32>) -> tensor<4x5xf32>

// -----

func.func @trt_einsum_2d_view(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = tensorrt.einsum {
    equation = "ij"
  } ins(%arg0 : tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: trt_einsum_2d_view
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ij"} ins(%{{.+}} : tensor<2x3xf32>) -> tensor<2x3xf32>

// -----

func.func @trt_einsum_2d_diagonal(%arg0: tensor<?x?xf32>) -> tensor<?xf32> {
  %0 = tensorrt.einsum {
    equation = "ii->i"
  } ins(%arg0 : tensor<?x?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: trt_einsum_2d_diagonal
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ii->i"} ins(%{{.+}} : tensor<?x?xf32>) -> tensor<?xf32>

// -----

func.func @trt_einsum_transpose(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = tensorrt.einsum {
    equation = "ij->ji"
  } ins(%arg0 : tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: trt_einsum_transpose
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ij->ji"} ins(%{{.+}} : tensor<2x3xf32>) -> tensor<3x2xf32>

// -----

func.func @trt_einsum_sum_horizontal(%arg0: tensor<?x?xf32>) -> tensor<?xf32> {
  %0 = tensorrt.einsum {
    equation = "ij->i"
  } ins(%arg0 : tensor<?x?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: trt_einsum_sum_horizontal
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ij->i"} ins(%{{.+}} : tensor<?x?xf32>) -> tensor<?xf32>

// -----

func.func @trt_einsum_matmul(%arg0: tensor<4x7xf32>, %arg1: tensor<7x9xf32>) -> tensor<4x9xf32> {
  %0 = tensorrt.einsum {
    equation = "ij,jk -> ik"
  } ins(%arg0, %arg1 : tensor<4x7xf32>, tensor<7x9xf32>) ->tensor<4x9xf32>
  return %0 : tensor<4x9xf32>
}

// CHECK-LABEL: trt_einsum_matmul
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ij,jk -> ik"} ins(%{{.+}}, %{{.+}} : tensor<4x7xf32>, tensor<7x9xf32>) -> tensor<4x9xf32>

// -----

func.func @trt_einsum_matmul_transpose(%arg0: tensor<4x7xf32>, %arg1: tensor<7x9xf32>) -> tensor<9x4xf32> {
  %0 = tensorrt.einsum {
    equation = "ij,jk -> ki"
  } ins(%arg0, %arg1 : tensor<4x7xf32>, tensor<7x9xf32>) ->tensor<9x4xf32>
  return %0 : tensor<9x4xf32>
}

// CHECK-LABEL: trt_einsum_matmul_transpose
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ij,jk -> ki"} ins(%{{.+}}, %{{.+}} : tensor<4x7xf32>, tensor<7x9xf32>) -> tensor<9x4xf32>

// -----

func.func @trt_einsum_matrix_inner(%arg0: tensor<4x7xf32>, %arg1: tensor<9x7xf32>) -> tensor<4x9xf32> {
  %0 = tensorrt.einsum {
    equation = "ij,kj ->ik"
  } ins(%arg0, %arg1 : tensor<4x7xf32>, tensor<9x7xf32>) ->tensor<4x9xf32>
  return %0 : tensor<4x9xf32>
}

// CHECK-LABEL: trt_einsum_matrix_inner
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ij,kj ->ik"} ins(%{{.+}}, %{{.+}} : tensor<4x7xf32>, tensor<9x7xf32>) -> tensor<4x9xf32>

// -----

func.func @trt_einsum_matrix_multiply_each_row(%arg0: tensor<4x7xf32>, %arg1: tensor<9x7xf32>) -> tensor<4x9x7xf32> {
  %0 = tensorrt.einsum {
    equation = "ij,kj ->ikj"
  } ins(%arg0, %arg1 : tensor<4x7xf32>, tensor<9x7xf32>) ->tensor<4x9x7xf32>
  return %0 : tensor<4x9x7xf32>
}
// CHECK-LABEL: trt_einsum_matrix_multiply_each_row
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ij,kj ->ikj"} ins(%{{.+}}, %{{.+}} : tensor<4x7xf32>, tensor<9x7xf32>) -> tensor<4x9x7xf32>

// -----

func.func @trt_einsum_matrix_multiply_each_value(%arg0: tensor<4x7xf32>, %arg1: tensor<9x7xf32>) -> tensor<4x7x9x7xf32> {
  %0 = tensorrt.einsum {
    equation = "ij,kl"
  } ins(%arg0, %arg1 : tensor<4x7xf32>, tensor<9x7xf32>) ->tensor<4x7x9x7xf32>
  return %0 : tensor<4x7x9x7xf32>
}

// CHECK-LABEL: trt_einsum_matrix_multiply_each_value
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ij,kl"} ins(%{{.+}}, %{{.+}} : tensor<4x7xf32>, tensor<9x7xf32>) -> tensor<4x7x9x7xf32>

// -----

func.func @trt_einsum_rearrange_order(%arg0: tensor<4x7xf32>, %arg1: tensor<7x9xf32>) -> tensor<9x4xf32> {
  %0 = tensorrt.einsum {
    equation = "zj,jl"
  } ins(%arg0, %arg1 : tensor<4x7xf32>, tensor<7x9xf32>) ->tensor<9x4xf32>
  return %0 : tensor<9x4xf32>
}

// CHECK-LABEL: trt_einsum_rearrange_order
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "zj,jl"} ins(%{{.+}}, %{{.+}} : tensor<4x7xf32>, tensor<7x9xf32>) -> tensor<9x4xf32>
// -----

func.func @trt_einsum_transpose_4d(%arg0: tensor<4x7x8x9xf32>) -> tensor<7x4x9x8xf32> {
  %0 = tensorrt.einsum {
    equation = "ijkl->jilk"
  } ins(%arg0: tensor<4x7x8x9xf32>) ->tensor<7x4x9x8xf32>
  return %0 : tensor<7x4x9x8xf32>
}

// CHECK-LABEL: trt_einsum_transpose_4d
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "ijkl->jilk"} ins(%{{.+}} : tensor<4x7x8x9xf32>)
// -----

func.func @trt_einsum_batch_matmul(%arg0: tensor<2x?x?xf32>, %arg1: tensor<2x?x4xf32>) -> tensor<2x?x4xf32> {
  %0 = tensorrt.einsum {
    equation = "bij,bjk->bik"
  } ins(%arg0, %arg1: tensor<2x?x?xf32>, tensor<2x?x4xf32>) ->tensor<2x?x4xf32>
  return %0 : tensor<2x?x4xf32>
}

// CHECK-LABEL: trt_einsum_batch_matmul
//       CHECK: tensorrt.einsum
//  CHECK-SAME: {equation = "bij,bjk->bik"} ins(%{{.+}}, %{{.+}} : tensor<2x?x?xf32>, tensor<2x?x4xf32>)


// -----

func.func @trt_scatter_elements(%arg0: tensor<3x3xf32>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xf32>) -> tensor<3x3xf32> {
  %0 = tensorrt.scatter_elements
    data(%arg0: tensor<3x3xf32>)
    indices(%arg1: tensor<2x3xi32>)
    updates(%arg2: tensor<2x3xf32>)

    return %0: tensor<3x3xf32>
}

// CHECK-LABEL: @trt_scatter_elements
//       CHECK:   tensorrt.scatter_elements
//  CHECK-SAME:     data(%{{.+}} : tensor<3x3xf32>) indices(%{{.+}} : tensor<2x3xi32>) updates(%{{.+}}: tensor<2x3xf32>)

// -----

func.func @trt_assertion(%condition: tensor<3xi1>) {
  tensorrt.assertion {
    message = "One or more conditions fail."
  } ins(%condition : tensor<3xi1>)

  return
}
// CHECK-LABEL: @trt_assertion
//       CHECK: tensorrt.assertion
//  CHECK-SAME: [[message:.+]] = "One or more conditions fail."
//  CHECK-SAME: ins(%[[condition:.+]] : tensor<3xi1>

// -----

func.func @trt_batch_normalize(%inp: tensor<2x3x2x2xf16>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16> {
    %0 = tensorrt.normalization {
        axis = array<i64: 0>
    } (%inp: tensor<2x3x2x2xf16>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16>
    return %0: tensor<2x3x2x2xf16>
}

// CHECK-LABEL: @trt_batch_normalize
//       CHECK: tensorrt.normalization

// -----

func.func @trt_group_normalize(%inp: tensor<2x24x2x2xf16>, %scale: tensor<1x4x1x1xf16>, %bias: tensor<1x4x1x1xf16>) -> tensor<2x24x2x2xf16> {
    %0 = tensorrt.normalization {
        axis = array<i64: 2, 3>, num_groups = 4 : i32
    } (%inp: tensor<2x24x2x2xf16>, %scale: tensor<1x4x1x1xf16>, %bias: tensor<1x4x1x1xf16>) -> tensor<2x24x2x2xf16>
    return %0: tensor<2x24x2x2xf16>
}

// CHECK-LABEL: @trt_group_normalize
//       CHECK: tensorrt.normalization

// -----

func.func @trt_instance_normalize(%inp: tensor<2x3x2x2xf16>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16> {
    %0 = tensorrt.normalization {
        axis = array<i64: 2, 3>
    } (%inp: tensor<2x3x2x2xf16>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16>
    return %0: tensor<2x3x2x2xf16>
}

// CHECK-LABEL: @trt_instance_normalize
//       CHECK: tensorrt.normalization

// -----

func.func @trt_layer_normalize_last_2d(%inp: tensor<2x3x2x2x2xf16>, %scale: tensor<1x1x1x2x2xf16>, %bias: tensor<1x1x1x2x2xf16>) -> tensor<2x3x2x2x2xf16> {
    %0 = tensorrt.normalization {
        axis = array<i64: 3, 4>
    } (%inp: tensor<2x3x2x2x2xf16>, %scale: tensor<1x1x1x2x2xf16>, %bias: tensor<1x1x1x2x2xf16>) -> tensor<2x3x2x2x2xf16>
    return %0: tensor<2x3x2x2x2xf16>
}

// CHECK-LABEL: @trt_layer_normalize_last_2d
//       CHECK: tensorrt.normalization

// -----

func.func @trt_layer_normalize_last_3d(%inp: tensor<2x3x2x2x2xf16>, %scale: tensor<1x1x2x2x2xf16>, %bias: tensor<1x1x2x2x2xf16>) -> tensor<2x3x2x2x2xf16> {
    %0 = tensorrt.normalization {
        axis = array<i64: 2, 3, 4>
    } (%inp: tensor<2x3x2x2x2xf16>, %scale: tensor<1x1x2x2x2xf16>, %bias: tensor<1x1x2x2x2xf16>) -> tensor<2x3x2x2x2xf16>
    return %0: tensor<2x3x2x2x2xf16>
}

// CHECK-LABEL: @trt_layer_normalize_last_3d
//       CHECK: tensorrt.normalization

// -----

tensorrt.module @engines {
  func.func @trt_callee(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    return %arg0: tensor<?xf32>
  }
}

func.func @trt_call_dynamic(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = tensor.empty() : tensor<10xf32>
  %1 = tensorrt.call @engines::@trt_callee(%arg0 : tensor<10xf32>) outs(%0: tensor<10xf32>)
    -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: @trt_call_dynamic
//       CHECK:     tensorrt.call @engines::@trt_callee(%{{.+}} : tensor<10xf32>) outs(%{{.+}} : tensor<10xf32>) -> tensor<10xf32>

// -----

func.func @random_uniform_static_low_high() -> tensor<1x2x3x4xf16> {
  %0 = tensorrt.random_uniform {
    static_low = 2.0,
    static_high = 4.0
  } ->  tensor<1x2x3x4xf16>
  return %0 : tensor<1x2x3x4xf16>
}

// CHECK-LABEL: @random_uniform_static_low_high
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.random_uniform
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @random_uniform_static_low_high_with_shape(%arg0: tensor<4xi32>) -> tensor<?x?x?x?xf16> {
  %0 = tensorrt.random_uniform {
    static_low = 2.0,
    static_high = 4.0
  } shape(%arg0: tensor<4xi32>) ->  tensor<?x?x?x?xf16>
  return %0 : tensor<?x?x?x?xf16>
}

// CHECK-LABEL: @random_uniform_static_low_high
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.random_uniform
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @random_uniform_all_dynamic(%low: tensor<f16>, %high: tensor<f16>, %shape: tensor<4xi32>) -> tensor<?x?x?x?xf16> {
  %0 = tensorrt.random_uniform low(%low: tensor<f16>) high(%high: tensor<f16>) shape(%shape: tensor<4xi32>) ->  tensor<?x?x?x?xf16>
  return %0 : tensor<?x?x?x?xf16>
}

// CHECK-LABEL: @random_uniform_all_dynamic
//  CHECK-SAME: (%[[low:.+]]: {{.*}}, %[[high:.+]]: {{.*}}, %[[shape:.+]]: {{.*}})
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.random_uniform low(%[[low]] : {{.*}}) high(%[[high]] : {{.*}}) shape(%[[shape]] : {{.*}})
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @random_uniform_low_high_dynamic(%low: tensor<f32>, %high: tensor<f32>) -> tensor<1x2x3x4xf32> {
  %0 = tensorrt.random_uniform low(%low: tensor<f32>) high(%high: tensor<f32>) ->  tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}

// CHECK-LABEL: @random_uniform_low_high_dynamic
//  CHECK-SAME: (%[[low:.+]]: {{.*}}, %[[high:.+]]: {{.*}})
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.random_uniform low(%[[low]] : {{.*}}) high(%[[high]] : {{.*}})
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @random_uniform_default() -> tensor<1x2x3x4xf16> {
  %0 = tensorrt.random_uniform ->  tensor<1x2x3x4xf16>
  return %0 : tensor<1x2x3x4xf16>
}

// CHECK-LABEL: @random_uniform_default
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.random_uniform
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @random_normal_static_mean_std() -> tensor<1x2x3x4xf16> {
  %0 = tensorrt.random_normal {
    static_mean = 2.0,
    static_std = 4.0
  } ->  tensor<1x2x3x4xf16>
  return %0 : tensor<1x2x3x4xf16>
}

// CHECK-LABEL: @random_normal_static_mean_std
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.random_normal
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @random_normal_static_mean_std_with_shape(%arg0: tensor<4xi32>) -> tensor<?x?x?x?xf16> {
  %0 = tensorrt.random_normal {
    static_mean = 2.0,
    static_std = 4.0
  } shape(%arg0: tensor<4xi32>) ->  tensor<?x?x?x?xf16>
  return %0 : tensor<?x?x?x?xf16>
}

// CHECK-LABEL: @random_normal_static_mean_std
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.random_normal
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @random_normal_all_dynamic(%mean: tensor<f16>, %std: tensor<f16>, %shape: tensor<4xi32>) -> tensor<?x?x?x?xf16> {
  %0 = tensorrt.random_normal mean(%mean: tensor<f16>) std(%std: tensor<f16>) shape(%shape: tensor<4xi32>) ->  tensor<?x?x?x?xf16>
  return %0 : tensor<?x?x?x?xf16>
}

// CHECK-LABEL: @random_normal_all_dynamic
//  CHECK-SAME: (%[[mean:.+]]: {{.*}}, %[[std:.+]]: {{.*}}, %[[shape:.+]]: {{.*}})
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.random_normal mean(%[[mean]] : {{.*}}) std(%[[std]] : {{.*}}) shape(%[[shape]] : {{.*}})
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @random_normal_mean_std_dynamic(%mean: tensor<f32>, %std: tensor<f32>) -> tensor<1x2x3x4xf32> {
  %0 = tensorrt.random_normal mean(%mean: tensor<f32>) std(%std: tensor<f32>) ->  tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}

// CHECK-LABEL: @random_normal_mean_std_dynamic
//  CHECK-SAME: (%[[mean:.+]]: {{.*}}, %[[std:.+]]: {{.*}})
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.random_normal mean(%[[mean]] : {{.*}}) std(%[[std]] : {{.*}})
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @random_normal_default() -> tensor<1x2x3x4xf16> {
  %0 = tensorrt.random_normal ->  tensor<1x2x3x4xf16>
  return %0 : tensor<1x2x3x4xf16>
}

// CHECK-LABEL: @random_normal_default
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.random_normal
//  CHECK-NEXT: return %[[v0]]

// -----

func.func @gather_nd(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<16x17x2xi32>) -> tensor<16x17x3x4xf32> {
  %0 = tensorrt.gather_nd data(%arg0) indices(%arg1) : (tensor<1x2x3x4xf32>, tensor<16x17x2xi32>) -> tensor<16x17x3x4xf32>
  return %0 : tensor<16x17x3x4xf32>
}

// CHECK-LABEL: @gather_nd(
//  CHECK-NEXT:   tensorrt.gather_nd data(%{{.+}}) indices(%{{.+}}) : (tensor<1x2x3x4xf32>, tensor<16x17x2xi32>) -> tensor<16x17x3x4xf32>

// -----

func.func @gather_nd_erased_dims(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<16x17x2xi32>) -> tensor<?x?x?x?xf32> {
  %0 = tensorrt.gather_nd data(%arg0) indices(%arg1) : (tensor<1x2x3x4xf32>, tensor<16x17x2xi32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @gather_nd_erased_dims
//       CHECK:   tensorrt.gather_nd data(%{{.+}}) indices(%{{.+}}) : (tensor<1x2x3x4xf32>, tensor<16x17x2xi32>) -> tensor<?x?x?x?xf32>

// -----

func.func @gather_nd_scalar(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<16x17x4xi32>) -> tensor<16x17xf32> {
  %0 = tensorrt.gather_nd data(%arg0) indices(%arg1) : (tensor<1x2x3x4xf32>, tensor<16x17x4xi32>) -> tensor<16x17xf32>
  return %0 : tensor<16x17xf32>
}

// CHECK-LABEL: @gather_nd_scalar
//       CHECK:   tensorrt.gather_nd data(%{{.+}}) indices(%{{.+}}) : (tensor<1x2x3x4xf32>, tensor<16x17x4xi32>) -> tensor<16x17xf32>

// -----

func.func @plugin_shape_inference(%arg0: tensor<?x4x?x?xf32>) -> tensor<41x?x?x16xf32> {
  // Check that more refined result type than what type inference
  // produces is OK with type verifier.
  %0 = tensorrt.opaque_plugin {
      creator_params = {},
      plugin_name = "TestInferShapePlugin",
      plugin_namespace = "",
      plugin_version = "0"}(%arg0) : (tensor<?x4x?x?xf32>) -> tensor<41x?x?x16xf32> {
  ^bb0(%arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64):
    %1 = arith.constant 41 : i64
    tensorrt.yield %1, %arg2, %arg3, %arg4 : i64, i64, i64, i64
  }
  return %0 : tensor<41x?x?x16xf32>
}


// CHECK-LABEL: func.func @plugin_shape_inference
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x4x?x?xf32>) -> tensor<41x?x?x16xf32> {
//       CHECK:     %[[v0:.+]] = tensorrt.opaque_plugin {{.*}}(%[[arg0]]) : (tensor<?x4x?x?xf32>) -> tensor<41x?x?x16xf32>
//  CHECK-NEXT:     ^bb0(%[[arg1:.+]]: i64, %[[arg2:.+]]: i64, %[[arg3:.+]]: i64, %[[arg4:.+]]: i64):
//  CHECK-NEXT:       %[[c41_i64:.+]] = arith.constant 41 : i64
//  CHECK-NEXT:       tensorrt.yield %[[c41_i64]], %[[arg2]], %[[arg3]], %[[arg4]] : i64, i64, i64, i64
//       CHECK:     return %[[v0]] : tensor<41x?x?x16xf32>

// -----

func.func @plugin_shape_inference(%arg0: tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // Check that more refined inferred type than the result type is OK
  // with the type verifier.
  %0 = tensorrt.opaque_plugin {
      creator_params = {},
      plugin_name = "TestInferShapePlugin",
      plugin_namespace = "",
      plugin_version = "0"}(%arg0) : (tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32> {
  ^bb0(%arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64):
    %1 = arith.constant 41 : i64
    tensorrt.yield %1, %arg2, %arg3, %arg4 : i64, i64, i64, i64
  }
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func.func @plugin_shape_inference
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32>
//  CHECK-NEXT:     %[[v0:.+]] = tensorrt.opaque_plugin {{.*}}(%[[arg0]]) : (tensor<?x4x?x?xf32>) -> tensor<?x?x?x?xf32>
//  CHECK-NEXT:     ^bb0(%[[arg1:.+]]: i64, %[[arg2:.+]]: i64, %[[arg3:.+]]: i64, %[[arg4:.+]]: i64):
//  CHECK-NEXT:       %[[c41_i64:.+]] = arith.constant 41 : i64
//  CHECK-NEXT:       tensorrt.yield %[[c41_i64]], %[[arg2]], %[[arg3]], %[[arg4]] : i64, i64, i64, i64
//       CHECK:     }
//       CHECK:     return %[[v0]] : tensor<?x?x?x?xf32>

// -----

func.func @dynamic_quantize_dequantize(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi8>) -> tensor<?x?xf32>{
    %0 = "tensorrt.constant"() {weights = dense<0.899999976> : tensor<4xf32>} : () -> tensor<4xf32>
    %1 = "tensorrt.quantize"(%arg0, %0) <{axis = 0 : i32}> : (tensor<?x?xf32>, tensor<4xf32>) -> tensor<?x?xi8>
    %2 = "tensorrt.dequantize"(%arg1, %0) <{axis = 0 : i32}> : (tensor<?x?xi8>, tensor<4xf32>) -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_quantize_dequantize
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<?x?xi8>)
//  CHECK-NEXT: %[[v0:.+]] = tensorrt.constant
//  CHECK-NEXT: %[[v1:.+]] = tensorrt.quantize {axis = 0 : i32} in(%[[arg0]] : tensor<?x?xf32>)
//  CHECK-NEXT: %[[v2:.+]] = tensorrt.dequantize {axis = 0 : i32} in(%[[arg1]] : tensor<?x?xi8>)
//  CHECK-NEXT: return %[[v2]]
