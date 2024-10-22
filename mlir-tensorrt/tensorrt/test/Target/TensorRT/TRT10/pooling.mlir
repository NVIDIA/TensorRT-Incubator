// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_average_pool_2d_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_average_pool_2d_fp8(%arg0: tensor<4x128x200x1xf8E4M3FN>) -> tensor<4x128x20x1xf16> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<4x128x200x1xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<4x128x200x1xf16>
  %0 = tensorrt.pooling {
    averageCountExcludesPadding = true,
    poolingType = #tensorrt.pooling_type<kAVERAGE>,
    postPadding = array<i64: 0, 0>,
    prePadding = array<i64: 0, 0>,
    stride = array<i64: 10, 1>,
    windowSize = array<i64: 10, 1>
  } ins(%dq : tensor<4x128x200x1xf16>) -> tensor<4x128x20x1xf16>
  return %0 : tensor<4x128x20x1xf16>
}
