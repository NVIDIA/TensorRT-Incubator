// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s


func.func @trt_dequantize_per_tensor(%arg0: tensor<10x10xi8>) -> tensor<10x10xf32> {
  %scale = tensorrt.constant dense<1.0>:tensor<f32>
  %result = tensorrt.dequantize in(%arg0 : tensor<10x10xi8>) scale(%scale :  tensor<f32>) -> tensor<10x10xf32>
  return %result : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_dequantize_per_tensor
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_dequantize_constant() -> tensor<2xf32> {
  %input = tensorrt.constant dense<[1, 2]>:tensor<2xi8>
  %scale = tensorrt.constant dense<1.0>:tensor<f32>
  %result = tensorrt.dequantize in(%input : tensor<2xi8>) scale(%scale :  tensor<f32>) -> tensor<2xf32>
  return %result : tensor<2xf32>
}

// CHECK-LABEL: @trt_dequantize_constant
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_dequantize_per_axis(%arg0: tensor<10x10xi8>) -> tensor<10x10xf32> {
  %scale = tensorrt.constant dense<[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]>:tensor<10xf32>
  %result = tensorrt.dequantize {
    axis = 1 : i32
  } in(%arg0 : tensor<10x10xi8>) scale(%scale : tensor<10xf32>) -> tensor<10x10xf32>
  return %result : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_dequantize_per_axis
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_dequantize_quant_type(%arg0: tensor<10x10x!quant.uniform<i8:f32, 1.0:0>>) -> tensor<10x10xf32> {
  %scale = tensorrt.constant dense<1.0>:tensor<f32>
  %result = tensorrt.dequantize in(%arg0 : tensor<10x10x!quant.uniform<i8:f32, 1.0:0>>) scale(%scale :  tensor<f32>) -> tensor<10x10xf32>
  return %result : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_dequantize_quant_type
//  CHECK-SAME: tensorrt.engine

// -----

func.func @trt_dequantize_per_tensor_f16_scale(%arg0: tensor<10x10xi8>) -> tensor<10x10xf32> {
  %scale = tensorrt.constant dense<1.0>:tensor<f16>
  %result = tensorrt.dequantize in(%arg0 : tensor<10x10xi8>) scale(%scale :  tensor<f16>) -> tensor<10x10xf32>
  return %result : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_dequantize_per_tensor_f16_scale
//  CHECK-SAME: tensorrt.engine
