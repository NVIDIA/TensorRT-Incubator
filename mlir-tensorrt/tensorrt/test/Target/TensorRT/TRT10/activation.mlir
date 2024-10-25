// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s

// CHECK-LABEL: @trt_relu_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_relu_fp8(%arg0: tensor<2x10xf8E4M3FN>) -> tensor<2x10xf16> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %0 = tensorrt.dequantize in (%arg0: tensor<2x10xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<2x10xf16>
  %1 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %0 : tensor<2x10xf16>
  return %1 : tensor<2x10xf16>
}

// -----

func.func @trt_relu_bf16(%arg0: tensor<2x10xbf16>) -> tensor<2x10xbf16> {
  %1 = tensorrt.activation {
    activationType = #tensorrt.activation_type<kRELU>
  } %arg0 : tensor<2x10xbf16>
  return %1 : tensor<2x10xbf16>
}

// CHECK-LABEL: @trt_relu_bf16
//  CHECK-SAME: tensorrt.engine
