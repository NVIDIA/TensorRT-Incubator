// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @onehot_innermost_f32(%indices: tensor<3xi32>, %values: tensor<2xf32>) -> tensor<3x5xf32> {
  %depth = tensorrt.constant dense <5> : tensor<i32>
  %0 = tensorrt.one_hot {
    axis = -1 : si64
  } ins(%indices, %values, %depth : tensor<3xi32>, tensor<2xf32>, tensor<i32>) -> tensor<3x5xf32>

  return %0 : tensor<3x5xf32>
}

// CHECK-LABEL: onehot_innermost_f32
//  CHECK-SAME: tensorrt.engine

// -----

func.func @onehot_innermost_f16(%indices: tensor<3xi32>, %values: tensor<2xf16>) -> tensor<3x5xf16> {
  %depth = tensorrt.constant dense <5> : tensor<i32>
  %0 = tensorrt.one_hot {
    axis = -1 : si64
  } ins(%indices, %values, %depth : tensor<3xi32>, tensor<2xf16>, tensor<i32>) -> tensor<3x5xf16>

  return %0 : tensor<3x5xf16>
}

// CHECK-LABEL: onehot_innermost_f16
//  CHECK-SAME: tensorrt.engine
