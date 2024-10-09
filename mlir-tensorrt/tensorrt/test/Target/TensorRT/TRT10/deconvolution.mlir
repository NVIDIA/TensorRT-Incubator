// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @trt_deconvolution_fp8(%arg0: tensor<1x32x128x128xf8E4M3FN>, %arg1: tensor<32x64x3x3xf32>) -> tensor<1x64x128x128xf32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<1x32x128x128xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<1x32x128x128xf32>
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%dq : tensor<1x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) -> tensor<1x64x128x128xf32>
  return %0 : tensor<1x64x128x128xf32>
}

// CHECK-LABEL: @trt_deconvolution_fp8
//  CHECK-SAME: tensorrt.engine
