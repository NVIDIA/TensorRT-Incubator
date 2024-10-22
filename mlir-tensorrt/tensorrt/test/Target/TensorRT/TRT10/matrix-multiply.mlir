// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_matrix_multiply_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_matrix_multiply_fp8(%arg0: tensor<10x128x64xf8E4M3FN>, %arg1: tensor<10x64x256xf8E4M3FN>)
            -> tensor<10x128x256xf32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq_lhs = tensorrt.dequantize in (%arg0: tensor<10x128x64xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10x128x64xf32>
  %dq_rhs = tensorrt.dequantize in (%arg1: tensor<10x64x256xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10x64x256xf32>
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%dq_lhs, %dq_rhs : tensor<10x128x64xf32>, tensor<10x64x256xf32>) -> tensor<10x128x256xf32>
  return %0 : tensor<10x128x256xf32>
}

// -----

// CHECK-LABEL: @trt_matrix_multiply_bf16
//  CHECK-SAME: tensorrt.engine
func.func @trt_matrix_multiply_bf16(%arg0: tensor<10x128x64xbf16>, %arg1: tensor<10x64x256xbf16>)
            -> tensor<10x128x256xbf16> {
  %0 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
  } ins(%arg0, %arg1 : tensor<10x128x64xbf16>, tensor<10x64x256xbf16>) -> tensor<10x128x256xbf16>
  return %0 : tensor<10x128x256xbf16>
}

// -----

func.func @trt_matrix_multiply_i4(%rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %scale = tensorrt.constant dense<1.0> : tensor<f32>
    %lhs = tensorrt.constant dense<[[1, 2],[3, 2]]> : tensor<2x2xi4>
    %dq_lhs = tensorrt.dequantize in (%lhs: tensor<2x2xi4>) scale (%scale: tensor<f32>) -> tensor<2x2xf32>
    %2 = tensorrt.matrix_multiply {
    op0 = #tensorrt.matrix_operation<kNONE>,
    op1 = #tensorrt.matrix_operation<kNONE>
    } ins(%dq_lhs, %rhs : tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
}

// CHECK-LABEL: @trt_matrix_multiply_i4
//  CHECK-SAME:   tensorrt.engine
