// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @topk_fp8(%arg0: tensor<128x128xf8E4M3FN>) -> (tensor<128x1xf16>, tensor<128x1xi32>) {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<128x128xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<128x128xf16>
  %0, %1 = tensorrt.top_k <kMAX> {
    axis = 1 : i64,
    k = 1 : i64
  } %dq_arg0: tensor<128x128xf16> -> tensor<128x1xf16>, tensor<128x1xi32>
  return %0, %1 : tensor<128x1xf16>, tensor<128x1xi32>
}

// CHECK-LABEL: @topk_fp8
//  CHECK-SAME: tensorrt.engine

// -----

func.func @topk_bf16(%arg0: tensor<128x128xbf16>) -> (tensor<128x1xbf16>, tensor<128x1xi32>) {
  %0, %1 = tensorrt.top_k <kMAX> {
    axis = 1 : i64,
    k = 1 : i64
  } %arg0: tensor<128x128xbf16> -> tensor<128x1xbf16>, tensor<128x1xi32>
  return %0, %1 : tensor<128x1xbf16>, tensor<128x1xi32>
}

// CHECK-LABEL: @topk_bf16
//  CHECK-SAME: tensorrt.engine
