// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @dequantize_fp8(%arg0: tensor<10xf8E4M3FN>) -> tensor<10xf16>{
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<10xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10xf16>
  return %dq: tensor<10xf16>
}

// CHECK-LABEL: @dequantize_fp8
//  CHECK-SAME: tensorrt.engine

func.func @dequantize_to_bf16(%arg0: tensor<10xf8E4M3FN>) -> tensor<10xbf16>{
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq = tensorrt.dequantize in (%arg0: tensor<10xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<10xbf16>
  return %dq: tensor<10xbf16>
}

// CHECK-LABEL: @dequantize_to_bf16
//  CHECK-SAME: tensorrt.engine

func.func @dequantize_to_bf16_bf16_scale(%arg0: tensor<10xf8E4M3FN>) -> tensor<10xbf16>{
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<bf16>
  %dq = tensorrt.dequantize in (%arg0: tensor<10xf8E4M3FN>) scale (%scale: tensor<bf16>) -> tensor<10xbf16>
  return %dq: tensor<10xbf16>
}

// CHECK-LABEL: @dequantize_to_bf16_bf16_scale
//  CHECK-SAME: tensorrt.engine

func.func @int4_quantize_dequantize() -> tensor<4x8xf32>{
    %weight = tensorrt.constant dense<4.0> : tensor<4x8xf32>
    %scale = tensorrt.constant dense<1.0> : tensor<2x8xf32>
    %quantized_i4 = tensorrt.quantize in(%weight : tensor<4x8xf32>) scale(%scale : tensor<2x8xf32>) -> tensor<4x8xi4>
    %dequantize_i4 = tensorrt.dequantize in(%quantized_i4 : tensor<4x8xi4>) scale(%scale : tensor<2x8xf32>) -> tensor<4x8xf32>
    return %dequantize_i4 : tensor<4x8xf32>
}

// CHECK-LABEL: @int4_quantize_dequantize
//  CHECK-SAME: tensorrt.engine

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