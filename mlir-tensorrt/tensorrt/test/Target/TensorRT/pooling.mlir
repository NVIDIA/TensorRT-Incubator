// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

// CHECK-LABEL: @trt_max_pool
//  CHECK-SAME: tensorrt.engine
func.func @trt_max_pool(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kMAX>,
    postPadding = array<i64: 1, 1>,
    prePadding = array<i64: 1, 1>,
    stride = array<i64: 1, 1>,
    windowSize = array<i64: 3, 3>
  } ins(%arg0 : tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
  return %0 : tensor<1x32x128x128xf32>
}


// CHECK-LABEL: @trt_avg_pool
//  CHECK-SAME: tensorrt.engine
func.func @trt_avg_pool(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
  %0 = tensorrt.pooling {
    poolingType = #tensorrt.pooling_type<kAVERAGE>,
    postPadding = array<i64: 1, 1>,
    prePadding = array<i64: 1, 1>,
    stride = array<i64: 1, 1>,
    windowSize = array<i64: 3, 3>,
    averageCountExcludesPadding = true
  } ins(%arg0 : tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
  return %0 : tensor<1x32x128x128xf32>
}


// CHECK-LABEL: @trt_average_pool_2d
//  CHECK-SAME: tensorrt.engine
func.func @trt_average_pool_2d(%arg0: tensor<4x128x200x1xf32>) -> tensor<4x128x20x1xf32> {
  %0 = tensorrt.pooling {
    averageCountExcludesPadding = true,
    poolingType = #tensorrt.pooling_type<kAVERAGE>,
    postPadding = array<i64: 0, 0>,
    prePadding = array<i64: 0, 0>,
    stride = array<i64: 10, 1>,
    windowSize = array<i64: 10, 1>
  } ins(%arg0 : tensor<4x128x200x1xf32>) -> tensor<4x128x20x1xf32>
  return %0 : tensor<4x128x20x1xf32>
}


// CHECK-LABEL: @trt_max_pool_3d
//  CHECK-SAME: tensorrt.engine
func.func @trt_max_pool_3d(%arg0: tensor<4x128x200x1xf32>) -> tensor<4x1x128x20x1xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    second_transpose = array<i64: 0, 1, 2, 3, 4>,
    zero_is_placeholder = false,
    reshape = array<i64: 4, 1, 128, 200, 1>
  } ins(%arg0 : tensor<4x128x200x1xf32>) -> tensor<4x1x128x200x1xf32>
  %1 = tensorrt.pooling {
    // averageCountExcludesPadding = true,
    poolingType = #tensorrt.pooling_type<kMAX>,
    postPadding = array<i64: 0, 0, 0>,
    prePadding = array<i64: 0, 0, 0>,
    stride = array<i64: 1, 10, 1>,
    windowSize = array<i64: 1, 10, 1>
  } ins(%0 : tensor<4x1x128x200x1xf32>) -> tensor<4x1x128x20x1xf32>
  return %1 : tensor<4x1x128x20x1xf32>
}

// CHECK-LABEL: @trt_average_pool_2d_fp16
//  CHECK-SAME: tensorrt.engine
func.func @trt_average_pool_2d_fp16(%arg0: tensor<4x128x200x1xf16>) -> tensor<4x128x20x1xf16> {
  %0 = tensorrt.pooling {
    averageCountExcludesPadding = true,
    poolingType = #tensorrt.pooling_type<kAVERAGE>,
    postPadding = array<i64: 0, 0>,
    prePadding = array<i64: 0, 0>,
    stride = array<i64: 10, 1>,
    windowSize = array<i64: 10, 1>
  } ins(%arg0 : tensor<4x128x200x1xf16>) -> tensor<4x128x20x1xf16>
  return %0 : tensor<4x128x20x1xf16>
}
