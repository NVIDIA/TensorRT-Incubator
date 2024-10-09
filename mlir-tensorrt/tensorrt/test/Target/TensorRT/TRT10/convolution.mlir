// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @trt_2d_fp8_convolution(%arg0: tensor<1x32x128x128xf8E4M3FN>) -> tensor<1x64x128x128xf16> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<1x32x128x128xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<1x32x128x128xf16>
  %0 = tensorrt.convolution {
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    stride = array<i64: 1, 1>,
    kernelStatic = dense<0.1>:tensor<64x32x3x3xf16>
  } in (%dq : tensor<1x32x128x128xf16>) -> tensor<1x64x128x128xf16>
  return %0 : tensor<1x64x128x128xf16>
}

// CHECK-LABEL: @trt_2d_fp8_convolution
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_2d_bf16_convolution(%arg0: tensor<1x32x128x128xbf16>) -> tensor<1x64x128x128xbf16> {
  %0 = tensorrt.convolution {
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    stride = array<i64: 1, 1>,
    kernelStatic = dense<0.1>:tensor<64x32x3x3xbf16>
  } in (%arg0 : tensor<1x32x128x128xbf16>) -> tensor<1x64x128x128xbf16>
  return %0 : tensor<1x64x128x128xbf16>
}

// CHECK-LABEL: @trt_2d_bf16_convolution
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_2d_int4_convolution(%arg0: tensor<1x32x128x128xf16>) -> tensor<1x64x128x128xf16> {
  %k = tensorrt.constant dense<2> : tensor<64x32x3x3xi4>
  %scale = tensorrt.constant dense<1.0> : tensor<f32>
  %dq_k = tensorrt.dequantize in (%k: tensor<64x32x3x3xi4>) scale (%scale: tensor<f32>) -> tensor<64x32x3x3xf16>
  %0 = tensorrt.convolution {
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    stride = array<i64: 1, 1>
  } in (%arg0 : tensor<1x32x128x128xf16>) kernel(%dq_k: tensor<64x32x3x3xf16>) -> tensor<1x64x128x128xf16>
  return %0 : tensor<1x64x128x128xf16>
}

// CHECK-LABEL: @trt_2d_int4_convolution
//  CHECK-SAME: tensorrt.engine
