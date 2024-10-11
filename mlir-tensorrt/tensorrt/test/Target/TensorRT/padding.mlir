// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN:  --mlir-elide-elementsattrs-if-larger=32 %s | FileCheck %s


func.func @trt_padding(%arg0: tensor<1x1x10x10xf32>) -> (tensor<1x1x8x8xf32>) {
  %0 = tensorrt.padding {
    prePadding = array<i64: -1, -1>,
    postPadding = array<i64: -1, -1>
  } ins(%arg0 : tensor<1x1x10x10xf32>) -> tensor<1x1x8x8xf32>

  return %0 : tensor<1x1x8x8xf32>
}

// CHECK-LABEL: @trt_padding
//  CHECK-SAME: tensorrt.engine
