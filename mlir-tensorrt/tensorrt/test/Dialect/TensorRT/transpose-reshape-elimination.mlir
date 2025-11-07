// RUN: tensorrt-opt %s -split-input-file -tensorrt-transpose-reshape-elimination | FileCheck %s


// CHECK: transpose_merge_with_matmul
// CHECK: %[[out1:.+]] = tensorrt.matrix_multiply
// CHECK: return %[[out1]]
func.func @transpose_merge_with_matmul(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x4x5xf32>) -> tensor<2x3x1x5xf32> {
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%arg0, %arg1 : tensor<1x2x3x4xf32>,tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32>
    %2 = tensorrt.shuffle {first_transpose = array<i64: 1, 2, 0, 3>, reshape = array<i64: 2, 3, 1, 5>, second_transpose = array<i64: 0, 1, 2, 3>, zero_is_placeholder = false} ins(%1 : tensor<1x2x3x5xf32>) -> tensor<2x3x1x5xf32>
    return %2 : tensor<2x3x1x5xf32>
}

// -----

// CHECK: @reshape_push_up_through_matmul(%[[arg0:.+]]: tensor<16x1024x1024xbf16>, %[[arg1:.+]]: tensor<1x1024x1024xbf16>)
// CHECK: %[[out:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%[[arg0]], %[[arg1]] : tensor<16x1024x1024xbf16>, tensor<1x1024x1024xbf16>)
// CHECK: return %[[out]]
func.func @reshape_push_up_through_matmul(%arg0: tensor<16x1024x1024xbf16>, %arg1: tensor<1x1024x1024xbf16>) -> tensor<16x1024x1024xbf16> {
    %6 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 1, 16384, 1024>, second_transpose = array<i64: 0, 1, 2>, zero_is_placeholder = false} ins(%arg0 : tensor<16x1024x1024xbf16>) -> tensor<1x16384x1024xbf16>
    %7 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%6, %arg1 : tensor<1x16384x1024xbf16>, tensor<1x1024x1024xbf16>) -> tensor<1x16384x1024xbf16>
    %8 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 16, 1024, 1024>, second_transpose = array<i64: 0, 1, 2>, zero_is_placeholder = false} ins(%7 : tensor<1x16384x1024xbf16>) -> tensor<16x1024x1024xbf16>
    return %8 : tensor<16x1024x1024xbf16>
}

// -----

// CHECK: func.func @reshape_push_to_constant(%[[arg0:.+]]: tensor<4x30xf32>)
// CHECK: %[[const:.+]] = tensorrt.constant dense_resource<__elided__>
// CHECK: %[[out:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%[[arg0]], %[[const]] : {{.*}})
// CHECK: return %[[out]]
func.func @reshape_push_to_constant(%arg0: tensor<4x30xf32>) -> tensor<4x50xf32> {
    %const = tensorrt.constant dense_resource<__elided__> : tensor<5x6x50xf32>
    %1 = tensorrt.reshape %arg0 : tensor<4x30xf32> to tensor<4x5x6xf32>
    %out = tensorrt.einsum {equation = "abc,bcd->ad"} ins(%1, %const : tensor<4x5x6xf32>, tensor<5x6x50xf32>) -> tensor<4x50xf32>
    return %out : tensor<4x50xf32>
}

// -----

// CHECK: @reshape_transpose_push(%[[arg0:.+]]: tensor<2x3x4x5xf32>)
// CHECK: return %[[arg0]]
func.func @reshape_transpose_push(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>{
    %1 = tensorrt.reshape %arg0 : tensor<2x3x4x5xf32> to tensor<6x4x5xf32>
    %2 = tensorrt.transpose { permutation = affine_map<(d0, d1, d2) -> (d0, d2, d1)> } %1 : tensor<6x4x5xf32> to tensor<6x5x4xf32>
    %3 = tensorrt.reshape %2 : tensor<6x5x4xf32> to tensor<2x3x5x4xf32>
    %4 = tensorrt.transpose { permutation = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)> } %3 : tensor<2x3x5x4xf32> to tensor<2x3x4x5xf32>
    return %4 : tensor<2x3x4x5xf32>
}

// -----

// CHECK: @reshape_transpose_cant_push(%[[arg0:.+]]: tensor<6x4x5xf32>)
// CHECK: %[[V1:.+]] = tensorrt.reshape %[[arg0]]
// CHECK: %[[V2:.+]] = tensorrt.transpose {{.*}} %[[V1]]
// CHECK: %[[V3:.+]] = tensorrt.reshape %[[V2]]
// CHECK: return %[[V3]]
func.func @reshape_transpose_cant_push(%arg0: tensor<6x4x5xf32>) -> tensor<6x4x5xf32>{
    %1 = tensorrt.reshape %arg0 : tensor<6x4x5xf32> to tensor<2x3x4x5xf32>
    %2 = tensorrt.transpose { permutation = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)> } %1 : tensor<2x3x4x5xf32> to tensor<2x5x4x3xf32>
    %3 = tensorrt.reshape %2 : tensor<2x5x4x3xf32> to tensor<6x4x5xf32>
    return %3 : tensor<6x4x5xf32>
}

// -----

// CHECK: @unary_push_reshape(%[[arg0:.+]]: tensor<10x3xf32>)
// CHECK: %[[out:.+]] = tensorrt.unary {{.*}} %[[arg0]]
// CHECK: return %[[out]]
func.func @unary_push_reshape(%arg0: tensor<10x3xf32>) -> tensor<10x3xf32> {
    %1 = tensorrt.reshape %arg0 : tensor<10x3xf32> to tensor<5x6xf32>
    %2 = tensorrt.unary { unaryOperation = #tensorrt.unary_operation<kEXP> } %1 : tensor<5x6xf32>
    %3 = tensorrt.reshape %2 : tensor<5x6xf32> to tensor<10x3xf32>
    return %3 : tensor<10x3xf32>
}

// -----

// CHECK: @identity_push_reshape(%[[arg0:.+]]: tensor<10x3xf32>)
// CHECK: %[[out:.+]] = tensorrt.identity %[[arg0]]
// CHECK: return %[[out]]
func.func @identity_push_reshape(%arg0: tensor<10x3xf32>) -> tensor<10x3xf16> {
    %1 = tensorrt.reshape %arg0 : tensor<10x3xf32> to tensor<5x6xf32>
    %2 = tensorrt.identity %1 : tensor<5x6xf32> to tensor<5x6xf16>
    %3 = tensorrt.reshape %2 : tensor<5x6xf16> to tensor<10x3xf16>
    return %3 : tensor<10x3xf16>
}

// -----

// CHECK: @activation_push_reshape(%[[arg0:.+]]: tensor<10x3xf32>)
// CHECK: %[[out:.+]] = tensorrt.activation {{.*}} %[[arg0]]
// CHECK: return %[[out]]
func.func @activation_push_reshape(%arg0: tensor<10x3xf32>) -> tensor<10x3xf32> {
    %1 = tensorrt.reshape %arg0 : tensor<10x3xf32> to tensor<5x6xf32>
    %2 = tensorrt.activation { activationType = #tensorrt.activation_type<kRELU>} %1 : tensor<5x6xf32>
    %3 = tensorrt.reshape %2 : tensor<5x6xf32> to tensor<10x3xf32>
    return %3 : tensor<10x3xf32>
}

// -----

// CHECK: transpose_quantize_dequantize_push(%[[arg0:.+]]: tensor<10x5xf32>, %[[scale:.+]]: tensor<f32>)
// CHECK: %[[V0:.+]] = tensorrt.quantize in(%[[arg0]] : tensor<10x5xf32>) scale(%[[scale]] : tensor<f32>) -> tensor<10x5xi8>
// CHECK: %[[V1:.+]] = tensorrt.dequantize in(%[[V0]] : tensor<10x5xi8>) scale(%[[scale]] : tensor<f32>) -> tensor<10x5xf32>
// CHECK: return %[[V1]]
func.func @transpose_quantize_dequantize_push(%arg0: tensor<10x5xf32>, %scale: tensor<f32>) -> tensor<10x5xf32> {
    %1 = tensorrt.transpose {permutation = affine_map<(d0, d1)->(d1, d0)>} %arg0 : tensor<10x5xf32> to tensor<5x10xf32>
    %2 = tensorrt.quantize in(%1 : tensor<5x10xf32>) scale(%scale : tensor<f32>) -> tensor<5x10xi8>
    %3 = tensorrt.dequantize in(%2 : tensor<5x10xi8>) scale(%scale : tensor<f32>) -> tensor<5x10xf32>
    %4 = tensorrt.transpose {permutation = affine_map<(d0, d1)->(d1, d0)>} %3 : tensor<5x10xf32> to tensor<10x5xf32>
    return %4 : tensor<10x5xf32>
}

// -----

// CHECK: reshape_quantize_dequantize_push(%[[arg0:.+]]: tensor<10x5xf32>, %[[scale:.+]]: tensor<f32>)
// CHECK: %[[V0:.+]] = tensorrt.quantize in(%[[arg0]] : tensor<10x5xf32>) scale(%[[scale]] : tensor<f32>) -> tensor<10x5xi8>
// CHECK: %[[V1:.+]] = tensorrt.dequantize in(%[[V0]] : tensor<10x5xi8>) scale(%[[scale]] : tensor<f32>) -> tensor<10x5xf32>
// CHECK: return %[[V1]]
func.func @reshape_quantize_dequantize_push(%arg0: tensor<10x5xf32>, %scale: tensor<f32>) -> tensor<10x5xf32> {
    %1 = tensorrt.reshape %arg0 : tensor<10x5xf32> to tensor<5x10xf32>
    %2 = tensorrt.quantize in(%1 : tensor<5x10xf32>) scale(%scale : tensor<f32>) -> tensor<5x10xi8>
    %3 = tensorrt.dequantize in(%2 : tensor<5x10xi8>) scale(%scale : tensor<f32>) -> tensor<5x10xf32>
    %4 = tensorrt.reshape %3 : tensor<5x10xf32> to tensor<10x5xf32>
    return %4 : tensor<10x5xf32>
}

// -----

// CHECK: @matrix_multiply_keep(%[[arg0:.+]]: tensor<1x2x3x4xf32>, %[[arg1:.+]]: tensor<1x2x4x5xf32>)
// CHECK: tensorrt.matrix_multiply
func.func @matrix_multiply_keep(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32> {
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%arg0, %arg1 : tensor<1x2x3x4xf32>,tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32>
    return %1 : tensor<1x2x3x5xf32>
}

// -----

// CHECK: @elementwise_push_down_reshape(%[[arg0:.+]]: tensor<1x2x3x4xf32>)
// CHECK: %[[ret:.+]] = tensorrt.element_wise <kSUM>(%[[arg0]], %[[const:.+]])
// CHECK: return %[[ret]]
func.func @elementwise_push_down_reshape(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
    %const = tensorrt.constant dense_resource<__elided__> : tensor<2xf32>
    %const_1 = tensorrt.expand_rank %const : tensor<2xf32> to tensor<1x2xf32>
    %const_2 = tensorrt.broadcast %const_1 broadcast_dims<0, 1> : tensor<1x2xf32> to tensor<12x2xf32>
    %1 = tensorrt.reshape %arg0 : tensor<1x2x3x4xf32> to tensor<12x2xf32>
    %2 = tensorrt.element_wise <kSUM>(%1, %const_2 : tensor<12x2xf32>, tensor<12x2xf32>) -> tensor<12x2xf32>
    %3 = tensorrt.reshape %2 : tensor<12x2xf32> to tensor<1x2x3x4xf32>
    return %3 : tensor<1x2x3x4xf32>
}

// -----

// CHECK: @reshape_with_one(%[[arg0:.+]]: tensor<2x3x4x5xf32>)
// CHECK: %[[const:.+]] = tensorrt.constant dense_resource<__elided__>
// CHECK: %[[V0:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%[[arg0]], %[[const]] : {{.*}}) -> tensor<2x3x4x6xf32>
// CHECK: return %[[V0]]
func.func @reshape_with_one(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x6xf32> {
    %const = tensorrt.constant dense_resource<__elided__> : tensor<1x1x2x3x5x6xf32>
    %1 = tensorrt.reshape %arg0 : tensor<2x3x4x5xf32> to tensor<1x1x2x3x4x5xf32>
    %2 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%1, %const : tensor<1x1x2x3x4x5xf32>, tensor<1x1x2x3x5x6xf32>) -> tensor<1x1x2x3x4x6xf32>
    %3 = tensorrt.reshape %2 : tensor<1x1x2x3x4x6xf32> to tensor<2x3x4x6xf32>
    return %3 : tensor<2x3x4x6xf32>
}

// -----

// CHECK: @elementwise_reshape(%[[arg0:.+]]: tensor<12x3x3xf32>, %[[arg1:.+]]: tensor<12xf32>)
// CHECK: %[[v0:.+]] = tensorrt.expand_rank %[[arg1]] : tensor<12xf32> to tensor<12x1x1xf32>
// CHECK: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[arg0]], %[[v0]] : tensor<12x3x3xf32>, tensor<12x1x1xf32>) -> tensor<12x3x3xf32>
// CHECK: %[[v2:.+]] = tensorrt.transpose {permutation = #map} %[[v1]] : tensor<12x3x3xf32> to tensor<12x3x3xf32>
// CHECK: return %[[v2]]
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
func.func @elementwise_reshape(%arg0: tensor<12x3x3xf32>, %arg1: tensor<12xf32>) -> tensor<12x3x3xf32> {
  %0 = tensorrt.transpose {permutation = #map} %arg0 : tensor<12x3x3xf32> to tensor<12x3x3xf32>
  %1 = tensorrt.expand_rank %arg1 : tensor<12xf32> to tensor<12x1x1xf32>
  %2 = tensorrt.element_wise <kDIV>(%0, %1 : tensor<12x3x3xf32>, tensor<12x1x1xf32>) -> tensor<12x3x3xf32>
  return %2 : tensor<12x3x3xf32>
}

// -----

// CHECK: @matmul_argument_swap(%[[arg0:.+]]: tensor<1x2x4x3x561xf32>, %[[arg1:.+]]: tensor<1x2x4x3x3xf32>) -> tensor<1x2x4x561x3xf32>
// CHECK-DAG: %[[v0:.+]] = tensorrt.[[op1:.+]] %[[arg0]] : tensor[[shape1:.+]]
// CHECK-DAG: %[[v1:.+]] = tensorrt.[[op2:.+]] %[[v0]] : tensor[[shape2:.+]]
// CHECK-DAG: %[[v2:.+]] = tensorrt.[[op3:.+]] %[[arg1]] : tensor[[shape3:.+]]
// CHECK-DAG: %[[v3:.+]] = tensorrt.[[op4:.+]] %[[v2]] : tensor[[shape4:.+]]
// CHECK: %[[v4:.+]] = tensorrt.matrix_multiply [[params:.+]] ins(%[[v3]], %[[v1]] : {{.*}})
// CHECK: return %[[v4]]
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>
func.func @matmul_argument_swap(%arg0: tensor<1x2x4x3x561xf32>, %arg1: tensor<1x2x4x3x3xf32>) -> tensor<1x2x4x561x3xf32> {
  %0 = tensorrt.reshape %arg1 : tensor<1x2x4x3x3xf32> to tensor<8x3x3xf32>
  %1 = tensorrt.reshape %arg0 : tensor<1x2x4x3x561xf32> to tensor<8x3x561xf32>
  %2 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%0, %1 : tensor<8x3x3xf32>, tensor<8x3x561xf32>) -> tensor<8x3x561xf32>
  %3 = tensorrt.reshape %2 : tensor<8x3x561xf32> to tensor<1x2x4x3x561xf32>
  %4 = tensorrt.transpose {permutation = #map} %3 : tensor<1x2x4x3x561xf32> to tensor<1x2x4x561x3xf32>
  return %4 : tensor<1x2x4x561x3xf32>
}

// -----

// CHECK: @transpose_reshape_reorder(%[[arg0:.+]]: tensor<12x256x8x8x16x8xf32>)
// CHECK: %[[v0:.+]] = tensorrt.transpose {permutation = #map} %[[arg0]] : tensor<12x256x8x8x16x8xf32> to tensor<12x8x8x16x8x256xf32>
// CHECK: %[[v1:.+]] = tensorrt.reshape %[[v0]] : tensor<12x8x8x16x8x256xf32> to tensor<12x64x128x256xf32>
// CHECK: return %[[v1]]
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @transpose_reshape_reorder(%arg0: tensor<12x256x8x8x16x8xf32>) -> tensor<12x64x128x256xf32> {
  %0 = tensorrt.reshape %arg0 : tensor<12x256x8x8x16x8xf32> to tensor<12x256x64x128xf32>
  %1 = tensorrt.transpose {permutation = #map} %0 : tensor<12x256x64x128xf32> to tensor<12x64x128x256xf32>
  return %1 : tensor<12x64x128x256xf32>
}

// -----

// CHECK: affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK: @transpose_softmax(%[[arg0:.+]]: tensor<2x3x4x5xf32>)
// CHECK: %[[v0:.+]] = tensorrt.transpose {permutation = #map} %[[arg0]] : tensor<2x3x4x5xf32> to tensor<2x5x3x4xf32>
// CHECK: %[[v1:.+]] = tensorrt.softmax {axis = 2 : i64} %[[v0]] : tensor<2x5x3x4xf32>
// CHECK: return %[[v1]]
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @transpose_softmax(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x5x3x4xf32> {
  %0 = tensorrt.transpose {permutation = #map} %arg0 : tensor<2x3x4x5xf32> to tensor<2x4x5x3xf32>
  %1 = tensorrt.softmax {axis = 3 : i64} %0 : tensor<2x4x5x3xf32>
  %2 = tensorrt.transpose {permutation = #map} %1 : tensor<2x4x5x3xf32> to tensor<2x5x3x4xf32>
  return %2 : tensor<2x5x3x4xf32>
}

// -----

// CHECK: @reshape_softmax(%[[arg0:.+]]: tensor<24x5x6xf32>)
// CHECK: %[[v0:.+]] = tensorrt.softmax {axis = 1 : i64} %[[arg0]] : tensor<24x5x6xf32>
// CHECK: return %[[v0]]
func.func @reshape_softmax(%arg0: tensor<24x5x6xf32>) -> tensor<24x5x6xf32> {
  %0 = tensorrt.reshape %arg0 : tensor<24x5x6xf32> to tensor<2x3x4x5x6xf32>
  %1 = tensorrt.softmax{axis = 3 : i64} %0 : tensor<2x3x4x5x6xf32>
  %2 = tensorrt.reshape %1 : tensor<2x3x4x5x6xf32> to tensor<24x5x6xf32>
  return %2 : tensor<24x5x6xf32>
}

// -----

// CHECK: @reshape_softmax_cant_push(%[[arg0:.+]]: tensor<2x3x4x5x6xf32>)
// CHECK: %[[v0:.+]] = tensorrt.reshape %[[arg0]] : tensor<2x3x4x5x6xf32> to tensor<24x10x3xf32>
// CHECK: %[[v1:.+]] = tensorrt.softmax {axis = 1 : i64} %[[v0]] : tensor<24x10x3xf32>
// CHECK: %[[v2:.+]] = tensorrt.reshape %[[v1]] : tensor<24x10x3xf32> to tensor<2x3x4x5x6xf32>
// CHECK: return %[[v2]]
func.func @reshape_softmax_cant_push(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32> {
  %0 = tensorrt.reshape %arg0 : tensor<2x3x4x5x6xf32> to tensor<24x10x3xf32>
  %1 = tensorrt.softmax {axis = 1 : i64} %0 : tensor<24x10x3xf32>
  %2 = tensorrt.reshape %1 : tensor<24x10x3xf32> to tensor<2x3x4x5x6xf32>
  return %2 : tensor<2x3x4x5x6xf32>
}

// -----

// CHECK: @reshape_transpose_reorder_ones_dim(%[[arg0:.+]]: tensor<2x1x1x1x1xf32>, %[[arg1:.+]]: tensor<1x2x3x3xf32>)
// CHECK: %[[v0:.+]] = tensorrt.collapse_rank %[[arg0]] : tensor<2x1x1x1x1xf32> to tensor<2x1x1x1xf32>
// CHECK: %[[v1:.+]] = tensorrt.deconvolution [[parmas:.+]] in(%[[arg1]] : tensor<1x2x3x3xf32>) kernelWeights(%[[v0]] : tensor<2x1x1x1xf32>) -> tensor<1x2x3x5xf32>
// CHECK: return %[[v1]]
func.func @reshape_transpose_reorder_ones_dim(%arg0: tensor<2x1x1x1x1xf32>, %arg1: tensor<1x2x3x3xf32>) -> tensor<1x2x3x5xf32> {
    %2 = tensorrt.transpose {permutation = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>} %arg0 : tensor<2x1x1x1x1xf32> to tensor<2x1x1x1x1xf32>
    %3 = tensorrt.reshape %2 : tensor<2x1x1x1x1xf32> to tensor<2x1x1x1xf32>
    %4 = tensorrt.deconvolution {dilation = array<i64: 1, 1>, num_groups = 2 : ui32, post_padding = array<i64: 0, 0>, pre_padding = array<i64: 0, 0>, stride = array<i64: 1, 2>} in(%arg1 : tensor<1x2x3x3xf32>) kernelWeights(%3 : tensor<2x1x1x1xf32>) -> tensor<1x2x3x5xf32>
    return %4 : tensor<1x2x3x5xf32>
}

// -----

// CHECK: @push_down_transpose_einsum(%[[arg0:.+]]: {{.*}}, %[[arg1:.+]]: {{.*}})
// CHECK: %[[const0:.+]] = tensorrt.constant dense<1.000000e+00> : {{.*}}
// CHECK: %[[v0:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[arg0]], %[[arg1]] : {{.*}})
// CHECK: %[[v1:.+]] = tensorrt.reshape %[[v0]]
// CHECK: %[[v2:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v1]], %[[const0]] : {{.*}})
// CHECK: return %[[v2]]
func.func @push_down_transpose_einsum(%arg0: tensor<1x6x1500x64xf32>, %arg1: tensor<1x6x1500x1500xf32>) -> tensor<1x1500x384xf32> {
  %cst_f32 = tensorrt.constant dense<1.000000e+00> : tensor<384x384xf32>
  %0 = tensorrt.reshape %arg0 : tensor<1x6x1500x64xf32> to tensor<6x1500x64xf32>
  %1 = tensorrt.reshape %arg1 : tensor<1x6x1500x1500xf32> to tensor<6x1500x1500xf32>
  %2 = tensorrt.einsum {equation = "bcd,bec->ebd"} ins(%0, %1 : tensor<6x1500x64xf32>, tensor<6x1500x1500xf32>) -> tensor<1500x6x64xf32>
  %3 = tensorrt.reshape %2 : tensor<1500x6x64xf32> to tensor<1x1500x6x64xf32>
  %4 = tensorrt.reshape %2 : tensor<1500x6x64xf32> to tensor<1500x384xf32>
  %cst_f32_0 = tensorrt.constant dense<1.000000e+00> : tensor<384x6x64xf32>
  %5 = tensorrt.einsum {equation = "bde,cde->bc"} ins(%2, %cst_f32_0 : tensor<1500x6x64xf32>, tensor<384x6x64xf32>) -> tensor<1500x384xf32>
  %6 = tensorrt.reshape %5 : tensor<1500x384xf32> to tensor<1x1500x384xf32>
  return %6 : tensor<1x1500x384xf32>
}

// -----

// CHECK: @multihead_attention
// CHECK: %[[v0:.+]] = tensorrt.matrix_multiply
// CHECK: %[[v1:.+]] = tensorrt.element_wise <kPROD>(%[[v0]], %[[const0:.+]] : {{.*}})
// CHECK: %[[v2:.+]] = tensorrt.element_wise <kSUM>(%[[v1]], %[[const1:.+]] : {{.*}})
// CHECK: %[[v3:.+]] = tensorrt.softmax {axis = [[axis:.+]] : i64} %[[v2]]
// CHECK: %[[v4:.+]] = tensorrt.matrix_multiply {{.*}} ins(%[[v3]], %[[values:.+]] : {{.*}})
#map3 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
func.func @multihead_attention(%arg0: tensor<566x48x64xf32>, %arg1: tensor<566x48x64xf32>, %arg2: tensor<566x48x64xf32>) -> tensor<566x48x64xf32> {
  %cst_f32_683 = tensorrt.constant dense<1.000000e+00> : tensor<1x1x1xf32>
  %cst_f32_704 = tensorrt.constant dense<0.000000e+00> : tensor<1x1x1xf32>
  %312 = tensorrt.transpose {permutation = #map3} %arg2 : tensor<566x48x64xf32> to tensor<48x566x64xf32>
  %314 = tensorrt.transpose {permutation = #map3} %arg0 : tensor<566x48x64xf32> to tensor<48x566x64xf32>
  %315 = tensorrt.transpose {permutation = #map5} %arg1 : tensor<566x48x64xf32> to tensor<48x64x566xf32>
  %316 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%314, %315 : tensor<48x566x64xf32>, tensor<48x64x566xf32>) -> tensor<48x566x566xf32>
  %317 = tensorrt.element_wise <kPROD>(%316, %cst_f32_683 : tensor<48x566x566xf32>, tensor<1x1x1xf32>) -> tensor<48x566x566xf32>
  %318 = tensorrt.element_wise <kSUM>(%317, %cst_f32_704 : tensor<48x566x566xf32>, tensor<1x1x1xf32>) -> tensor<48x566x566xf32>
  %319 = tensorrt.softmax {axis = 2 : i64} %318 : tensor<48x566x566xf32>
  %320 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%319, %312 : tensor<48x566x566xf32>, tensor<48x566x64xf32>) -> tensor<48x566x64xf32>
  %321 = tensorrt.transpose {permutation = #map3} %320 : tensor<48x566x64xf32> to tensor<566x48x64xf32>
  return %321 : tensor<566x48x64xf32>
}

// -----

// CHECK: @transpose_on_scalar(%[[arg0:.+]]: tensor<4488x4x48xf32>, %[[arg1:.+]]: tensor<f32>)
// CHECK: %[[v0:.+]] = tensorrt.expand_rank %[[arg1]] : tensor<f32> to tensor<1x1x1xf32>
// CHECK: %[[v1:.+]] = tensorrt.element_wise <kDIV>(%[[arg0]], %[[v0]] : tensor<4488x4x48xf32>, tensor<1x1x1xf32>) -> tensor<4488x4x48xf32>
// CHECK: %[[v2:.+]] = tensorrt.transpose {permutation = #map} %[[v1]] : tensor<4488x4x48xf32> to tensor<4x4488x48xf32>
// CHECK: return %[[v2]]
#map = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
func.func @transpose_on_scalar(%arg0: tensor<4488x4x48xf32>, %arg1: tensor<f32>) -> tensor<4x4488x48xf32> {
  %0 = tensorrt.transpose {permutation = #map} %arg0 : tensor<4488x4x48xf32> to tensor<4x4488x48xf32>
  %1 = tensorrt.expand_rank %arg1 : tensor<f32> to tensor<1x1x1xf32>
  %2 = tensorrt.element_wise <kDIV>(%0, %1 : tensor<4x4488x48xf32>, tensor<1x1x1xf32>) -> tensor<4x4488x48xf32>
  return %2 : tensor<4x4488x48xf32>
}

// -----

// CHECK: @einsum_multiply_two_axis(%[[arg0:.+]]: tensor<10x11x12xf32>, %[[arg1:.+]]: tensor<13x11x12xf32>)
// CHECK-DAG: %[[v0:.+]] = tensorrt.reshape %[[arg0]] : tensor<10x11x12xf32> to tensor<10x132xf32>
// CHECK-DAG: %[[v1:.+]] = tensorrt.reshape %[[arg1]] : tensor<13x11x12xf32> to tensor<13x132xf32>
// CHECK: %[[v2:.+]] = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%[[v0]], %[[v1]] : tensor<10x132xf32>, tensor<13x132xf32>) -> tensor<10x13xf32>
// CHECK: return %[[v2]]
func.func @einsum_multiply_two_axis(%arg0: tensor<10x11x12xf32>, %arg1: tensor<13x11x12xf32>) -> tensor<10x13xf32> {
  %0 = tensorrt.einsum {equation = "acd,bcd->ab"} ins(%arg0, %arg1: tensor<10x11x12xf32>, tensor<13x11x12xf32>) -> tensor<10x13xf32>
  return %0 : tensor<10x13xf32>
}

// -----

// CHECK: @can_not_push_reshape_through_einsum(%[[arg0:.+]]: tensor<2x20x12x64xf32>, %[[arg1:.+]]: tensor<2x12x20x1xf32>)
// CHECK: %[[v0:.+]] = tensorrt.einsum {{{.*}}} ins(%[[arg0]], %[[arg1]] : {{.*}})
// CHECK: %[[v1:.+]] = tensorrt.reshape %[[v0]] : tensor<2x12x64xf32> to tensor<2x1x768xf32>
// CHECK: return %[[v1]]
func.func @can_not_push_reshape_through_einsum(%arg0: tensor<2x20x12x64xf32>, %arg1: tensor<2x12x20x1xf32>) -> tensor<2x1x768xf32>{
  %0 = tensorrt.einsum {equation = "acbd,abcd->abd"} ins(%arg0, %arg1 : tensor<2x20x12x64xf32>, tensor<2x12x20x1xf32>) -> tensor<2x12x64xf32>
  %1 = tensorrt.reshape %0 : tensor<2x12x64xf32> to tensor<2x1x768xf32>
  return %1 : tensor<2x1x768xf32>
}

// -----

// CHECK: @push_reshape_broadcast(%[[arg0:.+]]: tensor<6x64x448xf32>, %[[arg1:.+]]: tensor<6x1x448xf32>)
// CHECK: %[[const:.+]] = tensorrt.constant dense_resource<__elided__> : tensor<1x1x384x384xf32>
// CHECK-DAG: %[[v0:.+]] = tensorrt.expand_rank %[[arg0]] : tensor<6x64x448xf32> to tensor<1x1x6x64x448xf32>
// CHECK-DAG: %[[v1:.+]] = tensorrt.expand_rank %[[arg1]] : tensor<6x1x448xf32> to tensor<1x1x6x1x448xf32>
// CHECK: %[[v2:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v0]], %[[v1]] : {{.*}})
// CHECK: %[[v3:.+]] = tensorrt.reshape %[[v2]] : tensor<1x1x6x64xf32> to tensor<1x1x384xf32>
// CHECK: %[[v4:.+]] = tensorrt.matrix_multiply {{{.*}}} ins(%[[v3]], %[[const]] : {{.*}}) -> tensor<1x1x384xf32>
// CHECK: return %[[v4]]
func.func @push_reshape_broadcast(%arg0: tensor<6x64x448xf32>, %arg1: tensor<6x1x448xf32>) -> tensor<1x1x384xf32> {
    %const = tensorrt.constant dense_resource<__elided__> : tensor<384x6x64xf32>
    %1 = tensorrt.einsum {equation = "bdc,bdc->bd"} ins(%arg0, %arg1 : tensor<6x64x448xf32>, tensor<6x1x448xf32>) -> tensor<6x64xf32>
    %2 = tensorrt.einsum {equation = "bd,ebd->e"} ins(%1, %const : tensor<6x64xf32>, tensor<384x6x64xf32>) -> tensor<384xf32>
    %3 = tensorrt.reshape %2 : tensor<384xf32> to tensor<1x1x384xf32>
    return %3 :  tensor<1x1x384xf32>
}