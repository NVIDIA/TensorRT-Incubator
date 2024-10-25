// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @dequantize_fp8(%arg0: tensor<10xf8E4M3FN>) -> tensor<10xf16>{
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<10xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10xf16>
  return %dq: tensor<10xf16>
}

// CHECK-LABEL: @dequantize_fp8
//  CHECK-SAME: tensorrt.engine

// -----

func.func @dequantize_to_bf16(%arg0: tensor<10xf8E4M3FN>) -> tensor<10xbf16>{
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<10xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10xbf16>
  return %dq: tensor<10xbf16>
}

// CHECK-LABEL: @dequantize_to_bf16
//  CHECK-SAME: tensorrt.engine

// -----

func.func @dequantize_to_bf16_bf16_scale(%arg0: tensor<10xf8E4M3FN>) -> tensor<10xbf16>{
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<bf16>
  %dq = tensorrt.dequantize in (%arg0: tensor<10xf8E4M3FN>) scale (%scale: tensor<bf16>) -> tensor<10xbf16>
  return %dq: tensor<10xbf16>
}

// CHECK-LABEL: @dequantize_to_bf16_bf16_scale
//  CHECK-SAME: tensorrt.engine
