// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_identity_noop_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_identity_noop_fp8(%arg0: tensor<10x128x64xf8E4M3FN>) -> tensor<10x128x64xf32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<10x128x64xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10x128x64xf32>
  %0 = tensorrt.identity %dq : tensor<10x128x64xf32> to tensor<10x128x64xf32>
  return %0 : tensor<10x128x64xf32>
}

// -----

// CHECK-LABEL: @trt_identity_fp8_cast_f32_i32_f32
//  CHECK-SAME: tensorrt.engine
func.func @trt_identity_fp8_cast_f32_i32_f32(%arg0: tensor<10x128x64xf8E4M3FN>) -> tensor<10x128x64xf32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<10x128x64xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10x128x64xf32>
  %0 = tensorrt.identity %dq : tensor<10x128x64xf32> to tensor<10x128x64xi32>
  %1 = tensorrt.identity %0 : tensor<10x128x64xi32> to tensor<10x128x64xf32>
  return %1 : tensor<10x128x64xf32>
}

// -----

func.func @trt_identity_from_bf16(%arg0: tensor<10x128x64xbf16>) -> (tensor<10x128x64xf32>, tensor<10x128x64xf16>, tensor<10x128x64xi32>) {
  %1 = tensorrt.identity %arg0 : tensor<10x128x64xbf16> to tensor<10x128x64xf32>
  %2 = tensorrt.identity %arg0 : tensor<10x128x64xbf16> to tensor<10x128x64xf16>
  %3 = tensorrt.identity %arg0 : tensor<10x128x64xbf16> to tensor<10x128x64xi32>
  return %1, %2, %3 : tensor<10x128x64xf32>, tensor<10x128x64xf16>, tensor<10x128x64xi32>
}

// CHECK-LABEL: @trt_identity_from_bf16
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_identity_to_bf16(%arg0: tensor<10x128x64xf32>, %arg1: tensor<10x128x64xf16>, %arg2: tensor<10x128x64xi32>) -> (tensor<10x128x64xbf16>, tensor<10x128x64xbf16>, tensor<10x128x64xbf16>) {
  %1 = tensorrt.identity %arg0 : tensor<10x128x64xf32> to tensor<10x128x64xbf16>
  %2 = tensorrt.identity %arg1 : tensor<10x128x64xf16> to tensor<10x128x64xbf16>
  %3 = tensorrt.identity %arg2 : tensor<10x128x64xi32> to tensor<10x128x64xbf16>
  return %1, %2, %3 : tensor<10x128x64xbf16>, tensor<10x128x64xbf16>, tensor<10x128x64xbf16>
}

// CHECK-LABEL: @trt_identity_to_bf16
//  CHECK-SAME: tensorrt.engine
