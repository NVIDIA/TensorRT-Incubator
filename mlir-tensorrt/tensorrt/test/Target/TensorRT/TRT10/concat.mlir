// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_concat_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_concat_fp8(%arg0: tensor<1x128x64xf8E4M3FN>, %arg1: tensor<1x128x64xf8E4M3FN>) -> tensor<2x128x64xf8E4M3FN> {
  %0 = tensorrt.concatenation {axis = 0 : i32} ins(%arg0, %arg1 : tensor<1x128x64xf8E4M3FN>, tensor<1x128x64xf8E4M3FN>) -> tensor<2x128x64xf8E4M3FN>
  return %0 : tensor<2x128x64xf8E4M3FN>
}

// -----

// CHECK-LABEL: @trt_concat_fp8_dq
//  CHECK-SAME: tensorrt.engine
func.func @trt_concat_fp8_dq(%arg0: tensor<1x128x64xf8E4M3FN>, %arg1: tensor<1x128x64xf8E4M3FN>) -> tensor<2x128x64xf16> {
  %cst_f32 = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %0 = tensorrt.dequantize in (%arg0: tensor<1x128x64xf8E4M3FN>) scale (%cst_f32:tensor<f32>) -> tensor<1x128x64xf16>
  %1 = tensorrt.dequantize in (%arg1: tensor<1x128x64xf8E4M3FN>) scale (%cst_f32:tensor<f32>) -> tensor<1x128x64xf16>
  %2 = tensorrt.concatenation {axis = 0 : i32} ins(%0, %1 : tensor<1x128x64xf16>, tensor<1x128x64xf16>) -> tensor<2x128x64xf16>
  return %2 : tensor<2x128x64xf16>
}

// -----

func.func @trt_concat_bf16(%arg0: tensor<1x128x64xbf16>, %arg1: tensor<1x128x64xbf16>) -> tensor<2x128x64xbf16> {
  %0 = tensorrt.concatenation {axis = 0 : i32} ins(%arg0, %arg1 : tensor<1x128x64xbf16>, tensor<1x128x64xbf16>) -> tensor<2x128x64xbf16>
  return %0 : tensor<2x128x64xbf16>
}

// CHECK-LABEL: @trt_concat_bf16
//  CHECK-SAME: tensorrt.engine
