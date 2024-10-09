// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @trt_batch_normalize_fp8(%inp: tensor<2x3x2x2xf8E4M3FN>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16> {
    %qdq_scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
    %dq = tensorrt.dequantize in (%inp: tensor<2x3x2x2xf8E4M3FN>) scale (%qdq_scale: tensor<f32>) -> tensor<2x3x2x2xf16>
    %0 = tensorrt.normalization {
        axis = array<i64: 0>
    } (%dq: tensor<2x3x2x2xf16>, %scale: tensor<1x3x1x1xf16>, %bias: tensor<1x3x1x1xf16>) -> tensor<2x3x2x2xf16>
    return %0: tensor<2x3x2x2xf16>
}

// CHECK-LABEL: @trt_batch_normalize_fp8
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_batch_normalize_bf16(%inp: tensor<2x3x2x2xbf16>, %scale: tensor<1x3x1x1xbf16>, %bias: tensor<1x3x1x1xbf16>) -> tensor<2x3x2x2xbf16> {
    %0 = tensorrt.normalization {
        axis = array<i64: 0>
    } (%inp: tensor<2x3x2x2xbf16>, %scale: tensor<1x3x1x1xbf16>, %bias: tensor<1x3x1x1xbf16>) -> tensor<2x3x2x2xbf16>
    return %0: tensor<2x3x2x2xbf16>
}

// CHECK-LABEL: @trt_batch_normalize_bf16
//  CHECK-SAME: tensorrt.engine

