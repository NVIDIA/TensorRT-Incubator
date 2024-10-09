// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s


func.func @trt_parametric_relu_fp8(%arg0: tensor<2x10xf8E4M3FN>, %arg1: tensor<2x10xf8E4M3FN>) -> tensor<2x10xf32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<2x10xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<2x10xf32>
  %dq_arg1 = tensorrt.dequantize in (%arg1: tensor<2x10xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<2x10xf32>
  %0 = tensorrt.parametric_relu ins(%dq_arg0, %dq_arg1 : tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
  return %0 : tensor<2x10xf32>
}

// CHECK-LABEL: @trt_parametric_relu_fp8
//  CHECK-SAME: tensorrt.engine
