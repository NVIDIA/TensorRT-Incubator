// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @trt_gather_default_fp8(%arg0: tensor<10x20x30xf8E4M3FN>, %arg1: tensor<5xi32>, %arg2: tensor<10x5x30xf32>) -> tensor<10x5x30xf32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<10x20x30xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10x20x30xf32>
  %0 = tensorrt.gather {
    axis = 1 : i64
  } ins(%dq, %arg1 : tensor<10x20x30xf32>, tensor<5xi32>) -> tensor<10x5x30xf32>

  %q_arg2 = tensorrt.quantize in (%arg2: tensor<10x5x30xf32>) scale (%scale: tensor<f32>) -> tensor<10x5x30xf8E4M3FN>
  %dq_arg2 = tensorrt.dequantize in (%q_arg2: tensor<10x5x30xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10x5x30xf32>

  // Add an elementwise to confirm shape
  %1 = tensorrt.element_wise <kSUM>(%0, %dq_arg2 : tensor<10x5x30xf32>, tensor<10x5x30xf32>) -> tensor<10x5x30xf32>
  return %1 : tensor<10x5x30xf32>
}

// CHECK-LABEL: @trt_gather_default_fp8
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_gather_default_bf16(%arg0: tensor<10x20x30xbf16>, %arg1: tensor<5xi32>, %arg2: tensor<10x5x30xbf16>) -> tensor<10x5x30xbf16> {
  %0 = tensorrt.gather {
    axis = 1 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xbf16>, tensor<5xi32>) -> tensor<10x5x30xbf16>
  // Add an elementwise to confirm shape
  %1 = tensorrt.element_wise <kSUM>(%0, %arg2 : tensor<10x5x30xbf16>, tensor<10x5x30xbf16>) -> tensor<10x5x30xbf16>
  return %1 : tensor<10x5x30xbf16>
}

// CHECK-LABEL: @trt_gather_default_bf16
//  CHECK-SAME: tensorrt.engine

// -----

func.func @gather_nd_scalar(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<16x17x4xi32>) -> tensor<16x17xf32> {
  %0 = tensorrt.gather_nd data(%arg0) indices(%arg1) : (tensor<1x2x3x4xf32>, tensor<16x17x4xi32>) -> tensor<16x17xf32>
  return %0 : tensor<16x17xf32>
}

// CHECK-LABEL: @gather_nd_scalar
//  CHECK-SAME: tensorrt.engine

// -----


func.func @gather_nd(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<16x17x2xi32>) -> tensor<16x17x3x4xf32> {
  %0 = tensorrt.gather_nd data(%arg0) indices(%arg1) : (tensor<1x2x3x4xf32>, tensor<16x17x2xi32>) -> tensor<16x17x3x4xf32>
  return %0 : tensor<16x17x3x4xf32>
}


// CHECK-LABEL: @gather_nd(
//  CHECK-SAME: tensorrt.engine
