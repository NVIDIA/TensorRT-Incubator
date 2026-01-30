// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s


// CHECK-LABEL: @trt_shuffle_infer
//  CHECK-sAME: tensorrt.engine
func.func @trt_shuffle_infer(%arg0: tensor<30x20x10x1xf32>) -> tensor<30x20x10xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: -1, 0, 0>,
    second_transpose = array<i64: 0, 1, 2>,
    zero_is_placeholder = true
  } ins(%arg0 : tensor<30x20x10x1xf32>) -> tensor<30x20x10xf32>
  return %0 : tensor<30x20x10xf32>
}


// This tests a case where the compiler is unable to determine shapes or fold
// the constant reshape dimensions. This is trivial to fix, but the translation
// logic should still handle this gracefully.

// CHECK-LABEL: @dynamic_shuffle_ewise
//  CHECK-SAME: tensorrt.engine
func.func @dynamic_shuffle_ewise(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1x1024x1024xf32>) -> tensor<1x?x1024xf32> {
  %reshapeDims = tensorrt.constant dense<[1, -1, 1024]> : tensor<3xi32>
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1>,
    second_transpose = array<i64: 0, 1, 2>,
    zero_is_placeholder = false
  } ins(%arg0, %reshapeDims : tensor<1024x1024xf32>, tensor<3xi32>) -> tensor<1x1024x1024xf32>
  %1 = tensorrt.element_wise <kSUM>(%0, %arg1 : tensor<1x1024x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x?x1024xf32>
  return %1 : tensor<1x?x1024xf32>
}


// CHECK-LABEL: @trt_shuffle_ewise
//  CHECK-SAME: tensorrt.engine
func.func @trt_shuffle_ewise(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1x1024x1024xf32>) -> tensor<1x1024x1024xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1>,
    reshape = array<i64: 1, 1024, 1024>,
    second_transpose = array<i64: 0, 1, 2>,
    zero_is_placeholder = false
  } ins(%arg0 : tensor<1024x1024xf32>) -> tensor<1x1024x1024xf32>
  %1 = tensorrt.element_wise <kSUM>(%0, %arg1 : tensor<1x1024x1024xf32>, tensor<1x1024x1024xf32>) -> tensor<1x1024x1024xf32>
  return %1 : tensor<1x1024x1024xf32>
}


// The below two tests verify the TRT inference procedure for identifying which
// dims correspond to '0' in the reshape spec when "zero_is_placeholder" is true.

// CHECK-LABEL: @trt_shuffle_infer2
//  CHECK-SAME: tensorrt.engine
func.func @trt_shuffle_infer2(%arg0: tensor<30x20x10x1xf32>) -> tensor<300x20xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: -1, 0>,
    second_transpose = array<i64: 0, 1>
  } ins(%arg0 : tensor<30x20x10x1xf32>) -> tensor<300x20xf32>
  return %0 : tensor<300x20xf32>
}

// CHECK-LABEL: @trt_shuffle_infer3
//  CHECK-SAME: tensorrt.engine
func.func @trt_shuffle_infer3(%arg0: tensor<1x2x3x4x5x6xf32>) -> tensor<1x8x3x30xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3, 4, 5>,
    reshape = array<i64: 0, -1, 0, 30>,
    second_transpose = array<i64: 0, 1, 2, 3>
  } ins(%arg0 : tensor<1x2x3x4x5x6xf32>) -> tensor<1x8x3x30xf32>
  return %0 : tensor<1x8x3x30xf32>
}


// CHECK-LABEL: @trt_shuffle_infer4
//  CHECK-SAME: tensorrt.engine
func.func @trt_shuffle_infer4(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x1x3x4xf32> {
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1, 2, 3>,
    reshape = array<i64: 0, 0, 1, 3, 4>,
    second_transpose = array<i64: 0, 1, 2, 3, 4>
  } ins(%arg0 : tensor<1x2x3x4xf32>) -> tensor<1x2x1x3x4xf32>
  return %0 : tensor<1x2x1x3x4xf32>
}

