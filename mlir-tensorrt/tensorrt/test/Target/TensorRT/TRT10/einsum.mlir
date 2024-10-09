// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s


// CHECK-LABEL: @trt_einsum_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_einsum_fp8(%arg0: tensor<1x4x8x16xf8E4M3FN>, %arg1: tensor<1x4x16x32xf8E4M3FN>,
  %arg2: tensor<1x4x8x32xf32>) -> tensor<1x4x8x32xf32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<1x4x8x16xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<1x4x8x16xf32>
  %dq_arg1 = tensorrt.dequantize in (%arg1: tensor<1x4x16x32xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<1x4x16x32xf32>
  %1 = tensorrt.einsum {equation = "bcdz,bcza->bcda"} ins(%dq_arg0, %dq_arg1 : tensor<1x4x8x16xf32>, tensor<1x4x16x32xf32>) -> tensor<1x4x8x32xf32>
  %2 = tensorrt.element_wise <kSUM>(%1, %arg2 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>)
    -> tensor<1x4x8x32xf32>
  return %2 : tensor<1x4x8x32xf32>
}

// -----

// CHECK-LABEL: @trt_einsum_fp16
//  CHECK-SAME: tensorrt.engine
func.func @trt_einsum_fp16(%arg0: tensor<1x4x8x16xbf16>, %arg1: tensor<1x4x16x32xbf16>,
  %arg2: tensor<1x4x8x32xbf16>) -> tensor<1x4x8x32xbf16> {
  %1 = tensorrt.einsum {equation = "bcdz,bcza->bcda"} ins(%arg0, %arg1 : tensor<1x4x8x16xbf16>, tensor<1x4x16x32xbf16>) -> tensor<1x4x8x32xbf16>
  %2 = tensorrt.element_wise <kSUM>(%1, %arg2 : tensor<1x4x8x32xbf16>, tensor<1x4x8x32xbf16>)
    -> tensor<1x4x8x32xbf16>
  return %2 : tensor<1x4x8x32xbf16>
}
