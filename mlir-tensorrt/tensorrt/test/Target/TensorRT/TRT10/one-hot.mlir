// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s


func.func @onehot_innermost_f8(%indices: tensor<3xi32>, %values: tensor<2xf8E4M3FN>) -> tensor<3x5xf32> {
  %depth = tensorrt.constant dense <5> : tensor<i32>
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq_values = tensorrt.dequantize in (%values: tensor<2xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<2xf32>
  %0 = tensorrt.one_hot {
    axis = -1 : si64
  } ins(%indices, %dq_values, %depth : tensor<3xi32>, tensor<2xf32>, tensor<i32>) -> tensor<3x5xf32>

  return %0 : tensor<3x5xf32>
}

// CHECK-LABEL: onehot_innermost_f8
//  CHECK-SAME: tensorrt.engine

// -----


func.func @onehot_innermost_bf16(%indices: tensor<3xi32>, %values: tensor<2xbf16>) -> tensor<3x5xbf16> {
  %depth = tensorrt.constant dense <5> : tensor<i32>
  %0 = tensorrt.one_hot {
    axis = -1 : si64
  } ins(%indices, %values, %depth : tensor<3xi32>, tensor<2xbf16>, tensor<i32>) -> tensor<3x5xbf16>

  return %0 : tensor<3x5xbf16>
}

// CHECK-LABEL: onehot_innermost_bf16
//  CHECK-SAME: tensorrt.engine
