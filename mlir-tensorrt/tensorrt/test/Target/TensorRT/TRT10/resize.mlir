// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @trt_resize_nearest_fp8(%arg0: tensor<10x10xf8E4M3FN>) -> tensor<20x20xf32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<10x10xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10x10xf32>
  %result = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>
  } %dq : tensor<10x10xf32> to tensor<20x20xf32>
  return %result : tensor<20x20xf32>
}

// CHECK-LABEL: @trt_resize_nearest_fp8
//  CHECK-SAME: tensorrt.engine
