// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

// CHECK-LABEL: @trt_2d_convolution
//  CHECK-SAME: tensorrt.engine
func.func @trt_2d_convolution(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x64x128x128xf32> {
  %0 = tensorrt.convolution {
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    stride = array<i64: 1, 1>,
    biasStatic = dense<0.1>:tensor<64xf32>,
    kernelStatic = dense<0.1>:tensor<64x32x3x3xf32>
  } in (%arg0: tensor<1x32x128x128xf32>) -> tensor<1x64x128x128xf32>
  return %0 : tensor<1x64x128x128xf32>
}

// -----

// CHECK-LABEL: @trt_2d_f16_convolution
//  CHECK-SAME: tensorrt.engine
func.func @trt_2d_f16_convolution(%arg0: tensor<1x32x128x128xf16>) -> tensor<1x64x128x128xf16> {
  %0 = tensorrt.convolution {
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>,
    stride = array<i64: 1, 1>,
    kernelStatic = dense<0.1>:tensor<64x32x3x3xf16>
  } in (%arg0 : tensor<1x32x128x128xf16>) -> tensor<1x64x128x128xf16>
  return %0 : tensor<1x64x128x128xf16>
}
