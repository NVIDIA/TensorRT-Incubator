// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @quantize_fp8(%arg0: tensor<10xf16>) -> tensor<10xf8E4M3FN>{
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %q = tensorrt.quantize in (%arg0: tensor<10xf16>) scale (%scale: tensor<f32>) -> tensor<10xf8E4M3FN>
  return %q: tensor<10xf8E4M3FN>
}

// CHECK-LABEL: @quantize_fp8
//  CHECK-SAME: tensorrt.engine

// -----

func.func @quantize_bf16_to_f8(%arg0: tensor<10xbf16>) -> tensor<10xf8E4M3FN>{
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %q = tensorrt.quantize in (%arg0: tensor<10xbf16>) scale (%scale: tensor<f32>) -> tensor<10xf8E4M3FN>
  return %q: tensor<10xf8E4M3FN>
}

// CHECK-LABEL: @quantize_bf16_to_f8
//  CHECK-SAME: tensorrt.engine

// -----

func.func @quantize_bf16_to_int8(%arg0: tensor<10xbf16>) -> tensor<10xi8>{
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %q = tensorrt.quantize in (%arg0: tensor<10xbf16>) scale (%scale: tensor<f32>) -> tensor<10xi8>
  return %q: tensor<10xi8>
}

// CHECK-LABEL: @quantize_bf16_to_int8
//  CHECK-SAME: tensorrt.engine

// -----

func.func @quantize_bf16_to_int8_bf16_scale(%arg0: tensor<10xbf16>) -> tensor<10xi8>{
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<bf16>
  %q = tensorrt.quantize in (%arg0: tensor<10xbf16>) scale (%scale: tensor<bf16>) -> tensor<10xi8>
  return %q: tensor<10xi8>
}

// CHECK-LABEL: @quantize_bf16_to_int8
//  CHECK-SAME: tensorrt.engine
