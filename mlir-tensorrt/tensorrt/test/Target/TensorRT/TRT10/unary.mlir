// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_unary_exp_op_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_unary_exp_op_fp8(%arg0: tensor<10x128x64xf8E4M3FN>) -> tensor<10x128x64xf32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<10x128x64xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10x128x64xf32>
  %0 = tensorrt.unary {
    unaryOperation = #tensorrt.unary_operation<kEXP>
  } %dq_arg0 : tensor<10x128x64xf32>
  return %0 : tensor<10x128x64xf32>
}

// -----

// CHECK-LABEL: @trt_unary_exp_op_bf16
//  CHECK-SAME: tensorrt.engine
func.func @trt_unary_exp_op_bf16(%arg0: tensor<10x128x64xbf16>) -> tensor<10x128x64xbf16> {
  %0 = tensorrt.unary {
    unaryOperation = #tensorrt.unary_operation<kEXP>
  } %arg0 : tensor<10x128x64xbf16>
  return %0 : tensor<10x128x64xbf16>
}
