// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

// CHECK-LABEL: @random_uniform_static_low_high
//  CHECK-SAME: tensorrt.engine
func.func @random_uniform_static_low_high() -> tensor<1x2x3x4xf32> {
  %0 = tensorrt.random_uniform {
    static_low = 2.0,
    static_high = 4.0
  } ->  tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}

// -----

// CHECK-LABEL: @random_uniform_low_high_dynamic
//  CHECK-SAME: tensorrt.engine
func.func @random_uniform_low_high_dynamic(%low: tensor<f32>, %high: tensor<f32>) -> tensor<1x2x3x4xf32> {
  %0 = tensorrt.random_uniform low(%low: tensor<f32>)
  high(%high: tensor<f32>) ->  tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}

// -----

// CHECK-LABEL: @random_uniform_default
//  CHECK-SAME: tensorrt.engine
func.func @random_uniform_default() -> tensor<1x2x3x4xf32> {
  %0 = tensorrt.random_uniform ->  tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}
