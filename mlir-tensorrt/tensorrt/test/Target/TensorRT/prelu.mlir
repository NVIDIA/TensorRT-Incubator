// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 %s | FileCheck %s


func.func @trt_parametric_relu(%arg0: tensor<2x10xf32>, %arg1: tensor<2x10xf32>) -> tensor<2x10xf32> {
  %0 = tensorrt.parametric_relu ins(%arg0, %arg1 : tensor<2x10xf32>, tensor<2x10xf32>) -> tensor<2x10xf32>
  return %0 : tensor<2x10xf32>
}

// CHECK-LABEL: @trt_parametric_relu
//  CHECK-SAME: tensorrt.engine
