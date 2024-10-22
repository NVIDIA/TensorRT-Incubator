// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @reduce_f16(%arg0: tensor<2x3x4xf16>) -> tensor<2x4xf16>{
  %0 = tensorrt.reduce <kSUM> %arg0 {reduceAxes=array<i64: 1>} : tensor<2x3x4xf16> -> tensor<2x4xf16>
  return %0 : tensor<2x4xf16>
}

// CHECK-LABEL: @reduce_f16
//  CHECK-SAME: tensorrt.engine

// -----

func.func @reduce_f16_keep_dims(%arg0: tensor<2x3x4xf16>) -> tensor<2x1x4xf16>{
  %0 = tensorrt.reduce <kSUM> %arg0 { keepDimensions=true, reduceAxes=array<i64: 1>} : tensor<2x3x4xf16> -> tensor<2x1x4xf16>
  return %0 : tensor<2x1x4xf16>
}

// CHECK-LABEL: @reduce_f16_keep_dims
//  CHECK-SAME: tensorrt.engine

// -----

func.func @reduce_f32(%arg0: tensor<2x3x4xf32>) -> tensor<2x4xf32>{
  %0 = tensorrt.reduce <kSUM> %arg0 {reduceAxes=array<i64: 1>} : tensor<2x3x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: @reduce_f32
//  CHECK-SAME: tensorrt.engine

// -----

func.func @reduce_f32_keep_dims(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x4xf32>{
  %0 = tensorrt.reduce <kSUM> %arg0 { keepDimensions=true, reduceAxes=array<i64: 1>} : tensor<2x3x4xf32> -> tensor<2x1x4xf32>
  return %0 : tensor<2x1x4xf32>
}

// CHECK-LABEL: @reduce_f32_keep_dims
//  CHECK-SAME: tensorrt.engine
