// REQUIRES: tensorrt-version-ge-10.9
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s

func.func @trt_subbyte_dequantize_even_final_dim(%arg0: tensor<3x5xf16>) -> tensor<4x5xf16> {
  %k = tensorrt.constant dense<2> : tensor<4x3xi4>
  %scale = tensorrt.constant dense<1.0> : tensor<2x3xf16>
  %dq_k = tensorrt.dequantize in (%k: tensor<4x3xi4>) scale (%scale: tensor<2x3xf16>) -> tensor<4x3xf16>
  %r = tensorrt.matrix_multiply {
                op0 = #tensorrt.matrix_operation<kNONE>,
                op1 = #tensorrt.matrix_operation<kNONE>
                } ins(%dq_k, %arg0 : tensor<4x3xf16>, tensor<3x5xf16>) -> tensor<4x5xf16>
  return %r : tensor<4x5xf16>
}

// CHECK-LABEL: @trt_subbyte_dequantize_even_final_dim
//  CHECK-SAME: tensorrt.engine
