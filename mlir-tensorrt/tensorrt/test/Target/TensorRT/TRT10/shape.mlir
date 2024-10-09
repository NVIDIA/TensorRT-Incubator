// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @tensorrt_shape_op_fp8
//  CHECK-SAME: tensorrt.engine
func.func @tensorrt_shape_op_fp8(
    %arg0: tensor<10x?xf8E4M3FN> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[10, 128], opt=[10, 256], max=[10, 512]>},
  %arg1: tensor<1x1xf8E4M3FN>) -> tensor<2xi32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<10x?xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10x?xf32>
  %dq_arg1 = tensorrt.dequantize in (%arg1: tensor<1x1xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<1x1xf32>
  %0 = tensorrt.element_wise <kSUM> (%dq_arg0, %dq_arg1 : tensor<10x?xf32>, tensor<1x1xf32>) -> tensor<10x?xf32>
  %q = tensorrt.quantize in (%0: tensor<10x?xf32>) scale (%scale: tensor<f32>) -> tensor<10x?xf8E4M3FN>
  %1 = tensorrt.shape %q : tensor<10x?xf8E4M3FN> -> tensor<2xi32>
  return %1 : tensor<2xi32>
}

// -----

// CHECK-LABEL: @tensorrt_shape_op_bf16
//  CHECK-SAME: tensorrt.engine
func.func @tensorrt_shape_op_bf16(
    %arg0: tensor<10x?xbf16> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[10, 128], opt=[10, 256], max=[10, 512]>},
  %arg1: tensor<1x1xbf16>) -> tensor<2xi32> {
  %0 = tensorrt.element_wise <kSUM> (%arg0, %arg1 : tensor<10x?xbf16>, tensor<1x1xbf16>) -> tensor<10x?xbf16>
  %1 = tensorrt.shape %0 : tensor<10x?xbf16> -> tensor<2xi32>
  return %1 : tensor<2xi32>
}

