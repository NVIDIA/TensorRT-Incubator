// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -tensorrt-builder-opt-level=0 -mlir-elide-elementsattrs-if-larger=32 %s | FileCheck %s

#profile = #tensorrt.shape_profile<min=[1, 1024], opt=[5, 1024], max=[10, 1024]>

// CHECK-LABEL: @trt_dynamic_input_shape
//  CHECK-SAME: tensorrt.engine
func.func @trt_dynamic_input_shape(
    %arg0: tensor<?x1024xf32> {tensorrt.shape_profile = #profile},
    %arg1: tensor<?x1024xf32> {tensorrt.shape_profile = #profile}) -> tensor<?x1024xf32> {
  %0 = tensorrt.element_wise <kSUM>(%arg0, %arg1 : tensor<?x1024xf32>, tensor<?x1024xf32>) -> tensor<?x1024xf32>
  return %0 : tensor<?x1024xf32>
}
