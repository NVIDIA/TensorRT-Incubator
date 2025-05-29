// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_dim_names
// CHECK: setDimensionName(0, "batch")
// CHECK: setDimensionName(1, "features")
func.func @trt_dim_names(%arg0: tensor<2x10xf32> {tensorrt.dimension_names = {0 = "batch", 1 = "features"}}) -> tensor<2x10xf32> {
  // Identity op to force input/output
  %0 = tensorrt.identity %arg0 : tensor<2x10xf32>
  return %0 : tensor<2x10xf32>
}
