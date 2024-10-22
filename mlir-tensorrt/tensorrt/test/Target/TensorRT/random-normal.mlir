// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

// CHECK-LABEL: @random_normal_static_mean_std
//  CHECK-SAME: tensorrt.engine
func.func @random_normal_static_mean_std() -> tensor<1x2x3x4xf32> {
  %0 = tensorrt.random_normal {
    static_mean = 2.0,
    static_std = 4.0
  } ->  tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}

// -----

// CHECK-LABEL: @random_normal_mean_std_dynamic
//  CHECK-SAME: tensorrt.engine
func.func @random_normal_mean_std_dynamic(%mean: tensor<f32>, %std: tensor<f32>) -> tensor<1x2x3x4xf32> {
  %0 = tensorrt.random_normal mean(%mean: tensor<f32>)
  std(%std: tensor<f32>) ->  tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}

// -----

// CHECK-LABEL: @random_normal_default
//  CHECK-SAME: tensorrt.engine
func.func @random_normal_default() -> tensor<1x2x3x4xf32> {
  %0 = tensorrt.random_normal ->  tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}
