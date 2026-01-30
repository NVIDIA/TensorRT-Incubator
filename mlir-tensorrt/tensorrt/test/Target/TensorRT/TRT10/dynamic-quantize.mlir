// REQUIRES: tensorrt-version-ge-10.9
// REQUIRES: has-gpu-sm-gte-10.0
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

func.func @dynamic_quantization() -> tensor<2x32xf32>{
    %input = tensorrt.constant dense<3.9> : tensor<2x32xf32>
    %double_quant_scale = tensorrt.constant dense<1.0> : tensor<f32>
    %out_f4, %scale_f8 = tensorrt.dynamic_quantize {axis = 1 : i32} in(%input : tensor<2x32xf32>) double_quant_scale(%double_quant_scale : tensor<f32>) -> tensor<2x32xf4E2M1FN>, tensor<2x2xf8E4M3FN>
    %dequantize_scale = tensorrt.dequantize in(%scale_f8 : tensor<2x2xf8E4M3FN>) scale(%double_quant_scale : tensor<f32>) -> tensor<2x2xf32>
    %dequantize_data = tensorrt.dequantize in(%out_f4 : tensor<2x32xf4E2M1FN>) scale(%dequantize_scale : tensor<2x2xf32>) -> tensor<2x32xf32>
    return %dequantize_data : tensor<2x32xf32>
}

// CHECK-LABEL: @dynamic_quantization
//  CHECK-SAME: tensorrt.engine