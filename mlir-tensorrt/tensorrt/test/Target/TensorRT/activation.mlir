// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_relu
//  CHECK-SAME: tensorrt.engine
func.func @trt_relu(%arg1: tensor<2x10xf16>, %arg2: tensor<2x10xf32>) -> (tensor<2x10xf16>, tensor<2x10xf32>) {
  %1 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %arg1 : tensor<2x10xf16>
  %2 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %arg2 : tensor<2x10xf32>
  return %1, %2: tensor<2x10xf16>, tensor<2x10xf32>
}
// -----

// CHECK-LABEL: @trt_relu_i8
//  CHECK-SAME: tensorrt.engine
func.func @trt_relu_i8(%arg0: tensor<2x10xi8>) -> tensor<2x10xf16> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %0 = tensorrt.dequantize in (%arg0: tensor<2x10xi8>) scale (%scale: tensor<f32>) -> tensor<2x10xf16>
  %1 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %0 : tensor<2x10xf16>
  return %1 : tensor<2x10xf16>
}
