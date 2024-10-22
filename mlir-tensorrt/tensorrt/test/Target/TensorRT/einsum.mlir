// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s


// CHECK-LABEL: @trt_einsum
//  CHECK-SAME: tensorrt.engine
func.func @trt_einsum(%arg0: tensor<1x4x8x16xf32>, %arg1: tensor<1x4x16x32xf32>,
  %arg2: tensor<1x4x8x32xf32>) -> tensor<1x4x8x32xf32> {
  %1 = tensorrt.einsum {equation = "bcdz,bcza->bcda"} ins(%arg0, %arg1 : tensor<1x4x8x16xf32>, tensor<1x4x16x32xf32>) -> tensor<1x4x8x32xf32>
  %2 = tensorrt.element_wise <kSUM>(%1, %arg2 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>)
    -> tensor<1x4x8x32xf32>
  return %2 : tensor<1x4x8x32xf32>
}
