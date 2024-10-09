// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s


// CHECK-LABEL: @trt_softmax_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_softmax_fp8(%arg0: tensor<1x3x224x224xf8E4M3FN>) -> tensor<1x3x224x224xf16> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<1x3x224x224xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<1x3x224x224xf16>
  %0 = tensorrt.softmax {
    axis = 1
  } %dq_arg0 : tensor<1x3x224x224xf16>
  return %0 : tensor<1x3x224x224xf16>
}

// -----

// CHECK-LABEL: @trt_softmax_bf16
//  CHECK-SAME: tensorrt.engine
func.func @trt_softmax_bf16(%arg0: tensor<1x3x224x224xbf16>) -> tensor<1x3x224x224xbf16> {
  %0 = tensorrt.softmax {
    axis = 1
  } %arg0 : tensor<1x3x224x224xbf16>
  return %0 : tensor<1x3x224x224xbf16>
}

// -----

// CHECK-LABEL: @trt_ragged_softmax_bf16
//  CHECK-SAME: tensorrt.engine
func.func @trt_ragged_softmax_bf16(%arg0: tensor<1x3x5xbf16>, %bounds: tensor<1x3x1xi32>) -> tensor<1x3x5xbf16> {
  %0 = tensorrt.ragged_softmax ins (%arg0, %bounds : tensor<1x3x5xbf16>, tensor<1x3x1xi32>) -> tensor<1x3x5xbf16>
  return %0 : tensor<1x3x5xbf16>
}
