// RUN: tensorrt-opt %s -split-input-file -tensorrt-transpose-reshape-elimination | FileCheck %s


// CHECK: transpose_merge_with_matmul
// CHECK: %[[out1:.+]] = tensorrt.matrix_multiply
// CHECK: %[[out2:.+]] = tensorrt.expand_rank %[[out1]]
// CHECK: return %[[out2]]
func.func @transpose_merge_with_matmul(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x4x5xf32>) -> tensor<2x3x1x5xf32> {
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%arg0, %arg1 : tensor<1x2x3x4xf32>,tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32>
    %2 = tensorrt.shuffle {first_transpose = array<i64: 1, 2, 0, 3>, reshape = array<i64: 2, 3, 1, 5>, second_transpose = array<i64: 0, 1, 2, 3>, zero_is_placeholder = false} ins(%1 : tensor<1x2x3x5xf32>) -> tensor<2x3x1x5xf32>
    return %2 : tensor<2x3x1x5xf32>
}

// -----

// CHECK: reshape_push_up_through_matmul
// CHECK: %[[out:.+]] = tensorrt.einsum
// CHECK: return %[[out]]
func.func @reshape_push_up_through_matmul(%arg0: tensor<16x1024x1024xbf16>, %arg1: tensor<1x1024x1024xbf16>) -> tensor<16x1024x1024xbf16> {
    %6 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 1, 16384, 1024>, second_transpose = array<i64: 0, 1, 2>, zero_is_placeholder = false} ins(%arg0 : tensor<16x1024x1024xbf16>) -> tensor<1x16384x1024xbf16>
    %7 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>} ins(%6, %arg1 : tensor<1x16384x1024xbf16>, tensor<1x1024x1024xbf16>) -> tensor<1x16384x1024xbf16>
    %8 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2>, reshape = array<i64: 16, 1024, 1024>, second_transpose = array<i64: 0, 1, 2>, zero_is_placeholder = false} ins(%7 : tensor<1x16384x1024xbf16>) -> tensor<16x1024x1024xbf16>
    return %8 : tensor<16x1024x1024xbf16>
}

// -----

// CHECK: func.func @reshape_push_down_into_einsum(%[[arg0:.+]]: tensor<4x5x6xf32>)
// CHECK: %[[out:.+]] = tensorrt.einsum [[attr:.+]] ins(%[[arg0]], %[[arg0]] : tensor<4x5x6xf32>, tensor<4x5x6xf32>) -> tensor<4x4xf32>
// CHECK: return %[[out]]
func.func @reshape_push_down_into_einsum(%arg0: tensor<4x5x6xf32>) -> tensor<4x4xf32> {
    %1 = tensorrt.reshape %arg0 : tensor<4x5x6xf32> to tensor<4x30xf32>
    %2 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kTRANSPOSE>} ins(%1, %1 : tensor<4x30xf32>, tensor<4x30xf32>) -> tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
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
// CHECK: %[[V2:.+]] = tensorrt.transpose [[attr:.+]] %[[V1]]
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
// CHECK: %[[out:.+]] = tensorrt.unary [[attrs:.+]] %[[arg0]]
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
// CHECK: %[[out:.+]] = tensorrt.activation [[attr:.+]] %[[arg0]]
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

func.func @matmul_eliminate_reshape_lhs_2(%arg0: tensor<1x2x3x4x5x6xf16>, %arg1: tensor<1x2x6x8xf16>) -> tensor<1x2x3x4x5x8xf16>{
    %0 = tensorrt.reshape %arg0 : tensor<1x2x3x4x5x6xf16> to tensor<1x2x60x6xf16>
    %1 = tensorrt.matrix_multiply {op0 = #tensorrt.matrix_operation<kNONE>, op1 = #tensorrt.matrix_operation<kNONE>}
    ins(%0, %arg1 : tensor<1x2x60x6xf16>, tensor<1x2x6x8xf16>) -> tensor<1x2x60x8xf16>
    %2 = tensorrt.reshape %1 : tensor<1x2x60x8xf16> to tensor<1x2x3x4x5x8xf16>
    return %2: tensor<1x2x3x4x5x8xf16>
}
