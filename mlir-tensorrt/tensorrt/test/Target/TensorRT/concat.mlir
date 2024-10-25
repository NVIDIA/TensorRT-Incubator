// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

// CHECK-LABEL: @trt_concat
//  CHECK-SAME: tensorrt.engine
func.func @trt_concat(%arg0: tensor<1x128x64xf32>, %arg1: tensor<1x128x64xf32>) -> tensor<2x128x64xf32> {
  %0 = tensorrt.concatenation {axis = 0 : i32} ins(%arg0, %arg1 : tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<2x128x64xf32>
  return %0 : tensor<2x128x64xf32>
}
