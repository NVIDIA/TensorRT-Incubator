// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)"  %s | FileCheck %s

func.func @trt_deconvolution(%arg0: tensor<1x32x128x128xf32>, %arg1: tensor<32x64x3x3xf32>) -> tensor<1x64x128x128xf32> {
  %0 = tensorrt.deconvolution {
    stride = array<i64: 1, 1>,
    pre_padding = array<i64: 1, 1>,
    post_padding = array<i64: 1, 1>
  } in(%arg0 : tensor<1x32x128x128xf32>) kernelWeights(%arg1: tensor<32x64x3x3xf32>) -> tensor<1x64x128x128xf32>
  return %0 : tensor<1x64x128x128xf32>
}

// CHECK-LABEL: @trt_deconvolution
//  CHECK-SAME: tensorrt.engine
