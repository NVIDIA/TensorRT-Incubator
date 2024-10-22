// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_select_op_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_select_op_fp8(%arg0: tensor<128x128xf8E4M3FN>, %arg1: tensor<128x128xf8E4M3FN>) -> tensor<128x128xf32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<128x128xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<128x128xf32>
  %dq_arg1 = tensorrt.dequantize in (%arg0: tensor<128x128xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<128x128xf32>
  %0 = tensorrt.element_wise <kGREATER>(%dq_arg0, %dq_arg1 : tensor<128x128xf32>, tensor<128x128xf32>)->tensor<128x128xi1>
  %1 = tensorrt.select ins(%0, %dq_arg0, %dq_arg1: tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>)->tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: @trt_select_op_bf16
//  CHECK-SAME: tensorrt.engine
func.func @trt_select_op_bf16(%arg0: tensor<128x128xbf16>, %arg1: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
  %0 = tensorrt.element_wise <kGREATER>(%arg0, %arg1 : tensor<128x128xbf16>, tensor<128x128xbf16>)->tensor<128x128xi1>
  %1 = tensorrt.select ins(%0, %arg0, %arg1: tensor<128x128xi1>, tensor<128x128xbf16>, tensor<128x128xbf16>)->tensor<128x128xbf16>
  return %1 : tensor<128x128xbf16>
}
