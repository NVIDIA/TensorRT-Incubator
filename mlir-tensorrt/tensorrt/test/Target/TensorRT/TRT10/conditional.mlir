// FIX: tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// FIX:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s


func.func @trt_fp8_if(%cond: tensor<i1>, %arg1: tensor<10xf8E4M3FN>, %arg2: tensor<10xf8E4M3FN>) -> tensor<10xf16> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %0 = tensorrt.dequantize in (%arg1: tensor<10xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10xf16>
  %1 = tensorrt.dequantize in (%arg2: tensor<10xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10xf16>
  %result = tensorrt.if (%cond: tensor<i1>) -> tensor<10xf16> {
      %add = tensorrt.element_wise <kSUM>(%0, %1 : tensor<10xf16>, tensor<10xf16>)
          -> tensor<10xf16>
      %add1 = tensorrt.element_wise <kSUM>(%add, %add : tensor<10xf16>, tensor<10xf16>)
          -> tensor<10xf16>
      tensorrt.yield %add1: tensor<10xf16>
    } else {
      %sub = tensorrt.element_wise <kSUB>(%0, %1 : tensor<10xf16>, tensor<10xf16>)
          -> tensor<10xf16>
      tensorrt.yield %sub: tensor<10xf16>
    }
  return %result: tensor<10xf16>
}

// CHECK-LABEL: @trt_fp8_if
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_bf16_if(%cond: tensor<i1>, %arg1: tensor<10xbf16>, %arg2: tensor<10xbf16>) -> tensor<10xbf16> {
  %result = tensorrt.if (%cond: tensor<i1>) -> tensor<10xbf16> {
      %add = tensorrt.element_wise <kSUM>(%arg1, %arg2 : tensor<10xbf16>, tensor<10xbf16>)
          -> tensor<10xbf16>
      %add1 = tensorrt.element_wise <kSUM>(%add, %add : tensor<10xbf16>, tensor<10xbf16>)
          -> tensor<10xbf16>
      tensorrt.yield %add1: tensor<10xbf16>
    } else {
      %sub = tensorrt.element_wise <kSUB>(%arg1, %arg2 : tensor<10xbf16>, tensor<10xbf16>)
          -> tensor<10xbf16>
      tensorrt.yield %sub: tensor<10xbf16>
    }
  return %result: tensor<10xbf16>
}

// CHECK-LABEL: @trt_bf16_if
//  CHECK-SAME: tensorrt.engine
