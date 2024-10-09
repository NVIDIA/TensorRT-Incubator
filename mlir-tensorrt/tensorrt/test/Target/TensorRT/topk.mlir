// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -tensorrt-builder-opt-level=0 -mlir-elide-elementsattrs-if-larger=32 %s | FileCheck %s

func.func @topk(%arg0: tensor<128x128xf32>) -> (tensor<128x1xf32>, tensor<128x1xi32>) {
  %0, %1 = tensorrt.top_k <kMAX> {
    axis = 1 : i64,
    k = 1 : i64
  } %arg0: tensor<128x128xf32> -> tensor<128x1xf32>, tensor<128x1xi32>
  return %0, %1 : tensor<128x1xf32>, tensor<128x1xi32>
}

// CHECK-LABEL: @topk
//  CHECK-SAME: tensorrt.engine


func.func @topk_dim0(%arg0: tensor<128x128xf32>) -> (tensor<1x128xf32>, tensor<1x128xi32>) {
  %0, %1 = tensorrt.top_k <kMAX> {
    axis = 0 : i64,
    k = 1 : i64
  } %arg0: tensor<128x128xf32> -> tensor<1x128xf32>, tensor<1x128xi32>
  return %0, %1 : tensor<1x128xf32>, tensor<1x128xi32>
}

// CHECK-LABEL: @topk_dim0
//  CHECK-SAME: tensorrt.engine
