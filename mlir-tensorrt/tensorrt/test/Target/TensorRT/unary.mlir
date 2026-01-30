// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(tensorrt-expand-ops,translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_unary_exp_op
//  CHECK-SAME: tensorrt.engine
func.func @trt_unary_exp_op(%arg0: tensor<10x128x64xf32>) -> tensor<10x128x64xf32> {
  %0 = tensorrt.unary {
    unaryOperation = #tensorrt.unary_operation<kEXP>
  } %arg0 : tensor<10x128x64xf32>
  return %0 : tensor<10x128x64xf32>
}


// CHECK-LABEL: @trt_unary_scalar
//  CHECK-SAME: tensorrt.engine
func.func @trt_unary_scalar() -> tensor<f32> {
    %inp = tensorrt.constant dense<4.0> : tensor<f32>
    %expanded = tensorrt.expand_rank %inp : tensor<f32> to tensor<1xf32>
    %0 = tensorrt.unary {unaryOperation = #tensorrt.unary_operation<kSQRT>} %expanded : tensor<1xf32>
    %collapse = tensorrt.collapse_rank %0 : tensor<1xf32> to tensor<f32>
    return %collapse : tensor<f32>
}

// CHECK-LABEL: @trt_unary_abs_i8_op
//  CHECK-SAME: tensorrt.engine
func.func @trt_unary_abs_i8_op(%arg0: tensor<2x10xi8>) -> tensor<2x10xi8> {
  %abs = tensorrt.unary {
    unaryOperation = #tensorrt.unary_operation<kABS>
  } %arg0 : tensor<2x10xi8>
  return %abs : tensor<2x10xi8>
}
