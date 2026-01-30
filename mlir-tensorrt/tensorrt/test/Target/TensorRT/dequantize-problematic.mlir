// REQUIRES: tensorrt-version-lt-9.0
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

// CHECK-LABEL: @trt_shuffle_i8
//  CHECK-SAME: tensorrt.engine
func.func @trt_shuffle_i8(%arg0: tensor<10x10xi8>) -> tensor<5x2x10xf32> {
  %c1 = tensorrt.constant dense<1.0>:tensor<f32>
  %0 = tensorrt.shuffle {
    first_transpose = array<i64: 0, 1>,
    reshape = array<i64: 10, 2, 5>,
    second_transpose = array<i64: 2, 1, 0>
  } ins(%arg0 : tensor<10x10xi8>) -> tensor<5x2x10xi8>
  %1 = tensorrt.dequantize in(%0 : tensor<5x2x10xi8>)
    scale(%c1 : tensor<f32>) -> tensor<5x2x10xf32>
  return %1 : tensor<5x2x10xf32>
}

// CHECK-LABEL: @constant_splat_int8
//  CHECK-SAME: tensorrt.engine
func.func @constant_splat_int8() -> tensor<10xi8> {
  %cst_i8 = tensorrt.constant dense<5> : tensor<10xi8>

  // Use a dummy Quantize node to force Q/DQ mode.
  %c1 = tensorrt.constant dense<1.0>:tensor<f32>
  %1 = tensorrt.dequantize in(%cst_i8 : tensor<10xi8>) scale(%c1 : tensor<f32>)->tensor<10xf32>
  %2 = tensorrt.quantize in(%1 : tensor<10xf32>) scale(%c1 : tensor<f32>)->tensor<10xi8>
  return %cst_i8 : tensor<10xi8>
}
