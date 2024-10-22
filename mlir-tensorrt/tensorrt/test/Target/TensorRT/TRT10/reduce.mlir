// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @reduce_bf16(%arg0: tensor<2x3x4xbf16>) -> tensor<2x4xbf16>{
  %0 = tensorrt.reduce <kSUM> %arg0 {reduceAxes=array<i64: 1>} : tensor<2x3x4xbf16> -> tensor<2x4xbf16>
  return %0 : tensor<2x4xbf16>
}

// CHECK-LABEL: @reduce_bf16
//  CHECK-SAME: tensorrt.engine

// -----

func.func @reduce_bf16_keep_dims(%arg0: tensor<2x3x4xbf16>) -> tensor<2x1x4xbf16>{
  %0 = tensorrt.reduce <kSUM> %arg0 { keepDimensions=true, reduceAxes=array<i64: 1>} : tensor<2x3x4xbf16> -> tensor<2x1x4xbf16>
  return %0 : tensor<2x1x4xbf16>
}

// CHECK-LABEL: @reduce_bf16_keep_dims
//  CHECK-SAME: tensorrt.engine
