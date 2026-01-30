// FIX: tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// FIX:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @trt_padding_fp8(%arg0: tensor<1x1x10x10xf8E4M3FN>) -> (tensor<1x1x8x8xf32>) {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<1x1x10x10xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<1x1x10x10xf32>
  %0 = tensorrt.padding {
    prePadding = array<i64: -1, -1>,
    postPadding = array<i64: -1, -1>
  } ins(%dq : tensor<1x1x10x10xf32>) -> tensor<1x1x8x8xf32>

  return %0 : tensor<1x1x8x8xf32>
}

// CHECK-LABEL: @trt_padding_fp8
//  CHECK-SAME: tensorrt.engine
