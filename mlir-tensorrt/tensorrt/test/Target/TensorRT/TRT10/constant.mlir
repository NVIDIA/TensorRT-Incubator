// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @constant_splat_int8() -> tensor<10xf16> {
  %cst_i8 = tensorrt.constant dense<5> : tensor<10xi8>
  %cst_f32 = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %0 = tensorrt.dequantize in (%cst_i8:tensor<10xi8>) scale (%cst_f32:tensor<f32>) -> tensor<10xf16>
  return %0 : tensor<10xf16>
}

// CHECK-LABEL: @constant_splat_int8
//  CHECK-SAME: tensorrt.engine

// -----

func.func @constant_fp8() -> (tensor<10xf16>, tensor<4xf16>) {
  %cst_f32 = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %cst_f8_splat = tensorrt.constant dense<0.4> : tensor<10xf8E4M3FN>
  %0 = tensorrt.dequantize in (%cst_f8_splat:tensor<10xf8E4M3FN>) scale (%cst_f32:tensor<f32>) -> tensor<10xf16>
  %cst_f8 = tensorrt.constant dense<[0.4, 0.3, 0.24, 0.56]> : tensor<4xf8E4M3FN>
  %1 = tensorrt.dequantize in (%cst_f8:tensor<4xf8E4M3FN>) scale (%cst_f32:tensor<f32>) -> tensor<4xf16>
  return %0, %1 : tensor<10xf16>, tensor<4xf16>
}

// CHECK-LABEL: @constant_fp8
//  CHECK-SAME: tensorrt.engine

// -----

func.func @constant_bf16() -> (tensor<10xbf16>, tensor<4xbf16>) {
  %cst_bf16_splat = tensorrt.constant dense<0.4> : tensor<10xbf16>
  %cst_bf16 = tensorrt.constant dense<[0.4, 0.3, 0.24, 0.56]> : tensor<4xbf16>
  return %cst_bf16_splat, %cst_bf16 : tensor<10xbf16>, tensor<4xbf16>
}

// CHECK-LABEL: @constant_bf16
//  CHECK-SAME: tensorrt.engine
