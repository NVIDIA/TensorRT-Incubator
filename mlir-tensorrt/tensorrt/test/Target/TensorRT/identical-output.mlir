// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s


// CHECK-LABEL: @trt_return_identical_results_f32
//  CHECK-SAME: tensorrt.engine
func.func @trt_return_identical_results_f32(%arg0: tensor<10x128x64xf32>) -> (tensor<10x128x64xf32>, tensor<10x128x64xf32>) {
  %0 = tensorrt.unary {
    unaryOperation = #tensorrt.unary_operation<kEXP>
  } %arg0 : tensor<10x128x64xf32>
  return %0, %0 : tensor<10x128x64xf32>, tensor<10x128x64xf32>
}


// CHECK-LABEL: @trt_return_identical_results_f16
//  CHECK-SAME: tensorrt.engine
func.func @trt_return_identical_results_f16(%arg0: tensor<10x128x64xf16>) -> (tensor<10x128x64xf16>, tensor<10x128x64xf16>) {
  %0 = tensorrt.unary {
    unaryOperation = #tensorrt.unary_operation<kABS>
  } %arg0 : tensor<10x128x64xf16>
  return %0, %0 : tensor<10x128x64xf16>, tensor<10x128x64xf16>
}
