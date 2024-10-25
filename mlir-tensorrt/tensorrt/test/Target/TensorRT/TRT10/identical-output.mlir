// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_return_identical_results_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_return_identical_results_fp8(%arg0: tensor<10x20x30xf8E4M3FN>) -> (tensor<10x20x30xf16>, tensor<10x20x30xf16>) {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<10x20x30xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10x20x30xf16>
  %0 = tensorrt.unary {
    unaryOperation = #tensorrt.unary_operation<kABS>
  } %dq : tensor<10x20x30xf16>
  return %0, %0 : tensor<10x20x30xf16>, tensor<10x20x30xf16>
}

// -----

// CHECK-LABEL: @trt_return_identical_results_bf16
//  CHECK-SAME: tensorrt.engine
func.func @trt_return_identical_results_bf16(%arg0: tensor<10x20x30xbf16>) -> (tensor<10x20x30xbf16>, tensor<10x20x30xbf16>) {
  %0 = tensorrt.unary {
    unaryOperation = #tensorrt.unary_operation<kABS>
  } %arg0 : tensor<10x20x30xbf16>
  return %0, %0 : tensor<10x20x30xbf16>, tensor<10x20x30xbf16>
}
