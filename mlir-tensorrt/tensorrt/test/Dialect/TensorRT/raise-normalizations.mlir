// RUN: tensorrt-opt %s  --tensorrt-raise-normalizations --split-input-file | FileCheck %s

func.func @raise_inst_norm_nchw(%arg0: tensor<1x3x1x1xf32>, %arg1: tensor<1x3x1x1xf32>, %arg2 : tensor<8x3x224x224xf32>) -> (tensor<8x3x224x224xf32>) {
  %cst_f32 = tensorrt.constant dense<9.99999974E-6> : tensor<1x1x1x1xf32>
  %cst_f32_0 = tensorrt.constant dense<5.017600e+04> : tensor<1x1x1x1xf32>
  %0 = tensorrt.reshape %arg2 : tensor<8x3x224x224xf32> to tensor<8x3x50176xf32>
  %1 = tensorrt.reduce <kSUM> %0 {reduceAxes = array<i64: 2>} : tensor<8x3x50176xf32> -> tensor<8x3xf32>
  %2 = tensorrt.expand_rank %1 : tensor<8x3xf32> to tensor<8x3x1x1xf32>
  %3 = tensorrt.element_wise <kDIV>(%2, %cst_f32_0 : tensor<8x3x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<8x3x1x1xf32>
  %4 = tensorrt.element_wise <kSUB>(%arg2, %3 : tensor<8x3x224x224xf32>, tensor<8x3x1x1xf32>) -> tensor<8x3x224x224xf32>
  %5 = tensorrt.element_wise <kPROD>(%4, %4 : tensor<8x3x224x224xf32>, tensor<8x3x224x224xf32>) -> tensor<8x3x224x224xf32>
  %6 = tensorrt.reshape %5 : tensor<8x3x224x224xf32> to tensor<8x3x50176xf32>
  %7 = tensorrt.reduce <kSUM> %6 {reduceAxes = array<i64: 2>} : tensor<8x3x50176xf32> -> tensor<8x3xf32>
  %8 = tensorrt.expand_rank %7 : tensor<8x3xf32> to tensor<8x3x1x1xf32>
  %9 = tensorrt.element_wise <kDIV>(%8, %cst_f32_0 : tensor<8x3x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<8x3x1x1xf32>
  %10 = tensorrt.element_wise <kSUM>(%9, %cst_f32 : tensor<8x3x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<8x3x1x1xf32>
  %11 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kRECIP>} %10 : tensor<8x3x1x1xf32>
  %12 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %11 : tensor<8x3x1x1xf32>
  %13 = tensorrt.element_wise <kPROD>(%arg1, %12 : tensor<1x3x1x1xf32>, tensor<8x3x1x1xf32>) -> tensor<8x3x1x1xf32>
  %14 = tensorrt.element_wise <kPROD>(%13, %4 : tensor<8x3x1x1xf32>, tensor<8x3x224x224xf32>) -> tensor<8x3x224x224xf32>
  %15 = tensorrt.element_wise <kSUM>(%14, %arg0 : tensor<8x3x224x224xf32>, tensor<1x3x1x1xf32>) -> tensor<8x3x224x224xf32>
  return %15 : tensor<8x3x224x224xf32>
}

// CHECK-LABEL: @raise_inst_norm_nchw
//  CHECK-SAME: %[[arg0:.+]]: tensor<1x3x1x1xf32>, %[[arg1:.+]]: tensor<1x3x1x1xf32>, %[[arg2:.+]]: tensor<8x3x224x224xf32>) -> tensor<8x3x224x224xf32>
//       CHECK: %[[v0:.+]] = tensorrt.normalization {axis = array<i64: 2, 3>}(%[[arg2]] : tensor<8x3x224x224xf32>, %[[arg1]] : tensor<1x3x1x1xf32>, %[[arg0]] : tensor<1x3x1x1xf32>) -> tensor<8x3x224x224xf32>

// -----
 func.func public @neg_raise_nhwc(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<8x224x224x3xf32>) -> (tensor<8x224x224x3xf32>) {
    %cst_f32 = tensorrt.constant dense<9.99999974E-6> : tensor<1x1x1x1xf32>
    %cst_f32_0 = tensorrt.constant dense<5.017600e+04> : tensor<1x1x1x1xf32>
    %0 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2, 3>, reshape = array<i64: 8, 50176, 3>, second_transpose = array<i64: 0, 1, 2>, zero_is_placeholder = false} ins(%arg2 : tensor<8x224x224x3xf32>) -> tensor<8x50176x3xf32>
    %1 = tensorrt.reduce <kSUM> %0 {reduceAxes = array<i64: 1>} : tensor<8x50176x3xf32> -> tensor<8x3xf32>
    %2 = tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 8, 1, 1, 3>, second_transpose = array<i64: 0, 1, 2, 3>, zero_is_placeholder = false} ins(%1 : tensor<8x3xf32>) -> tensor<8x1x1x3xf32>
    %3 = tensorrt.element_wise <kDIV>(%2, %cst_f32_0 : tensor<8x1x1x3xf32>, tensor<1x1x1x1xf32>) -> tensor<8x1x1x3xf32>
    %4 = tensorrt.element_wise <kSUB>(%arg2, %3 : tensor<8x224x224x3xf32>, tensor<8x1x1x3xf32>) -> tensor<8x224x224x3xf32>
    %5 = tensorrt.element_wise <kPROD>(%4, %4 : tensor<8x224x224x3xf32>, tensor<8x224x224x3xf32>) -> tensor<8x224x224x3xf32>
    %6 = tensorrt.shuffle {first_transpose = array<i64: 0, 1, 2, 3>, reshape = array<i64: 8, 50176, 3>, second_transpose = array<i64: 0, 1, 2>, zero_is_placeholder = false} ins(%5 : tensor<8x224x224x3xf32>) -> tensor<8x50176x3xf32>
    %7 = tensorrt.reduce <kSUM> %6 {reduceAxes = array<i64: 1>} : tensor<8x50176x3xf32> -> tensor<8x3xf32>
    %8 = tensorrt.shuffle {first_transpose = array<i64: 0, 1>, reshape = array<i64: 8, 1, 1, 3>, second_transpose = array<i64: 0, 1, 2, 3>, zero_is_placeholder = false} ins(%7 : tensor<8x3xf32>) -> tensor<8x1x1x3xf32>
    %9 = tensorrt.element_wise <kDIV>(%8, %cst_f32_0 : tensor<8x1x1x3xf32>, tensor<1x1x1x1xf32>) -> tensor<8x1x1x3xf32>
    %10 = tensorrt.element_wise <kSUM>(%9, %cst_f32 : tensor<8x1x1x3xf32>, tensor<1x1x1x1xf32>) -> tensor<8x1x1x3xf32>
    %11 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kRECIP>} %10 : tensor<8x1x1x3xf32>
    %12 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %11 : tensor<8x1x1x3xf32>
    %13 = tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 1, 1, 1, 3>, second_transpose = array<i64: 0, 1, 2, 3>, zero_is_placeholder = false} ins(%arg1 : tensor<3xf32>) -> tensor<1x1x1x3xf32>
    %14 = tensorrt.element_wise <kPROD>(%13, %12 : tensor<1x1x1x3xf32>, tensor<8x1x1x3xf32>) -> tensor<8x1x1x3xf32>
    %15 = tensorrt.element_wise <kPROD>(%14, %4 : tensor<8x1x1x3xf32>, tensor<8x224x224x3xf32>) -> tensor<8x224x224x3xf32>
    %16 = tensorrt.shuffle {first_transpose = array<i64: 0>, reshape = array<i64: 1, 1, 1, 3>, second_transpose = array<i64: 0, 1, 2, 3>, zero_is_placeholder = false} ins(%arg0 : tensor<3xf32>) -> tensor<1x1x1x3xf32>
    %17 = tensorrt.element_wise <kSUM>(%15, %16 : tensor<8x224x224x3xf32>, tensor<1x1x1x3xf32>) -> tensor<8x224x224x3xf32>
    return %17 : tensor<8x224x224x3xf32>
}

// CHECK-LABEL: @neg_raise_nhwc
//   CHECK-NOT: tensorrt.normalization

// -----

// CHECK: @raise_layer_norm_pytorch(%[[arg0:.+]]: tensor<16x1024x1024xf32>)
// CHECK: %[[ret:.+]] = tensorrt.normalization [[attr:.+]](%[[arg0]] : tensor<16x1024x1024xf32>, %[[gamma:.+]] : tensor<1x1x1024xf32>, %[[beta:.+]] : tensor<1x1x1024xf32>)
// CHECK: return %[[ret]]
func.func @raise_layer_norm_pytorch(%arg0: tensor<16x1024x1024xf32>) -> tensor<16x1024x1024xf32> {
    %cst_i64 = tensorrt.constant dense<1024> : tensor<i64>
    %cst_f32 = tensorrt.constant dense<9.99999974E-6> : tensor<1x1x1xf32>

    %cst_bf16_1 = tensorrt.constant dense_resource<__elided__> : tensor<1024xbf16>  // beta (added)
    %cst_bf16_2 = tensorrt.constant dense_resource<__elided__> : tensor<1024xbf16>   // gamma (multiplied)

    %6 = tensorrt.reduce <kSUM> %arg0 {keepDimensions = true, reduceAxes = array<i64: 2>} : tensor<16x1024x1024xf32> -> tensor<16x1024x1xf32>
    %7 = tensorrt.cast %cst_i64 : tensor<i64> to tensor<f32>
    %8 = tensorrt.expand_rank %7 : tensor<f32> to tensor<1x1x1xf32>
    %9 = tensorrt.element_wise <kDIV>(%6, %8 : tensor<16x1024x1xf32>, tensor<1x1x1xf32>) -> tensor<16x1024x1xf32>
    %10 = tensorrt.element_wise <kSUB>(%arg0, %9 : tensor<16x1024x1024xf32>, tensor<16x1024x1xf32>) -> tensor<16x1024x1024xf32>
    %11 = tensorrt.element_wise <kPROD>(%10, %10 : tensor<16x1024x1024xf32>, tensor<16x1024x1024xf32>) -> tensor<16x1024x1024xf32>
    %12 = tensorrt.reduce <kSUM> %11 {keepDimensions = true, reduceAxes = array<i64: 2>} : tensor<16x1024x1024xf32> -> tensor<16x1024x1xf32>
    %13 = tensorrt.element_wise <kDIV>(%12, %8 : tensor<16x1024x1xf32>, tensor<1x1x1xf32>) -> tensor<16x1024x1xf32>  // Var[x]
    %15 = tensorrt.reduce <kAVG> %arg0 {keepDimensions = true, reduceAxes = array<i64: 2>} : tensor<16x1024x1024xf32> -> tensor<16x1024x1xf32>  // E[x]
    %16 = tensorrt.element_wise <kSUM>(%13, %cst_f32 : tensor<16x1024x1xf32>, tensor<1x1x1xf32>) -> tensor<16x1024x1xf32> // Var[x] + epsilon
    %17 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kRECIP>} %16 : tensor<16x1024x1xf32>
    %18 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %17 : tensor<16x1024x1xf32>    // compute 1/sqrt(...)
    %19 = tensorrt.element_wise <kSUB>(%arg0, %15 : tensor<16x1024x1024xf32>, tensor<16x1024x1xf32>) -> tensor<16x1024x1024xf32>
    %20 = tensorrt.element_wise <kPROD>(%19, %18 : tensor<16x1024x1024xf32>, tensor<16x1024x1xf32>) -> tensor<16x1024x1024xf32>  // multiply for division
    %21 = tensorrt.cast %cst_bf16_2 : tensor<1024xbf16> to tensor<1024xf32>
    %22 = tensorrt.expand_rank %21 : tensor<1024xf32> to tensor<1x1x1024xf32>
    %23 = tensorrt.element_wise <kPROD>(%20, %22 : tensor<16x1024x1024xf32>, tensor<1x1x1024xf32>) -> tensor<16x1024x1024xf32> // multiply gamma
    %24 = tensorrt.cast %cst_bf16_1 : tensor<1024xbf16> to tensor<1024xf32>
    %25 = tensorrt.expand_rank %24 : tensor<1024xf32> to tensor<1x1x1024xf32>
    %26 = tensorrt.element_wise <kSUM>(%23, %25 : tensor<16x1024x1024xf32>, tensor<1x1x1024xf32>) -> tensor<16x1024x1024xf32> // add beta
    return %26 : tensor<16x1024x1024xf32>
}