// RUN: tensorrt-opt -split-input-file %s | tensorrt-opt | FileCheck %s

// -----

func.func @trt_quantize(%arg0: tensor<10x10xf32>, %arg1: tensor<f32>) -> tensor<10x10xi8> {
  %result = tensorrt.quantize in(%arg0 : tensor<10x10xf32>) scale(%arg1 :  tensor<f32>) -> tensor<10x10xi8>
  return %result : tensor<10x10xi8>
}

// CHECK-LABEL: @trt_quantize
//       CHECK: tensorrt.quantize
//  CHECK-SAME: in(%[[arg0:.+]] : tensor<10x10xf32>) scale(%[[arg1:.+]] : tensor<f32>) -> tensor<10x10xi8>

// -----

func.func @trt_quantize_per_axis(%arg0: tensor<10x10xf32>, %arg1: tensor<10xf32>) -> tensor<10x10xi8> {
  %result = tensorrt.quantize {
    axis = 1 : i32
  } in(%arg0 : tensor<10x10xf32>) scale(%arg1 : tensor<10xf32>) -> tensor<10x10xi8>
  return %result : tensor<10x10xi8>
}

// CHECK-LABEL: @trt_quantize_per_axis
//       CHECK: tensorrt.quantize
//  CHECK-SAME: axis = 1 : i32
//  CHECK-SAME: in(%[[arg0:.+]] : tensor<10x10xf32>) scale(%[[arg1:.+]] : tensor<10xf32>) -> tensor<10x10xi8>

// -----

func.func @trt_quantize_quant_type(%arg0: tensor<16x16xf32>, %arg1: tensor<f32>) -> tensor<16x16x!quant.uniform<i8:f32, 1.0:0>> {
  %0 = tensorrt.quantize in(%arg0 : tensor<16x16xf32>) scale(%arg1 : tensor<f32>) -> tensor<16x16x!quant.uniform<i8:f32, 1.0:0>>
  return %0 : tensor<16x16x!quant.uniform<i8:f32, 1.0:0>>
}

// CHECK-LABEL: @trt_quantize_quant_type
//       CHECK: tensorrt.quantize
//  CHECK-SAME: in(%[[arg0:.+]] : tensor<16x16xf32>) scale(%[[arg1:.+]] : tensor<f32>) -> tensor<16x16x!quant.uniform<i8:f32, 1.000000e+00>>

// -----

func.func @trt_dequantize(%arg0: tensor<10x10xi8>, %arg1: tensor<f32>) -> tensor<10x10xf32> {
  %result = tensorrt.dequantize in(%arg0 : tensor<10x10xi8>) scale(%arg1 :  tensor<f32>) -> tensor<10x10xf32>
  return %result : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_dequantize
//       CHECK: tensorrt.dequantize
//  CHECK-SAME: in(%[[arg0:.+]] : tensor<10x10xi8>) scale(%[[arg1:.+]] : tensor<f32>) -> tensor<10x10xf32>

// -----

func.func @trt_dequantize_per_axis(%arg0: tensor<10x10xi8>, %arg1: tensor<10xf32>) -> tensor<10x10xf32> {
  %result = tensorrt.dequantize {
    axis = 1 : i32
  } in(%arg0 : tensor<10x10xi8>) scale(%arg1 : tensor<10xf32>) -> tensor<10x10xf32>
  return %result : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_dequantize_per_axis
//       CHECK: tensorrt.dequantize
//  CHECK-SAME: axis = 1 : i32
//  CHECK-SAME: in(%[[arg0:.+]] : tensor<10x10xi8>) scale(%[[arg1:.+]] : tensor<10xf32>) -> tensor<10x10xf32>

// -----

func.func @trt_dequantize_quant_type(%arg0: tensor<10x10x!quant.uniform<i8:f32, 1.0:0>>, %arg1: tensor<f32>) -> tensor<10x10xf32> {
  %result = tensorrt.dequantize in(%arg0 : tensor<10x10x!quant.uniform<i8:f32, 1.0:0>>) scale(%arg1 :  tensor<f32>) -> tensor<10x10xf32>
  return %result : tensor<10x10xf32>
}

// CHECK-LABEL: @trt_dequantize_quant_type
//       CHECK: tensorrt.dequantize
//  CHECK-SAME: in(%[[arg0:.+]] : tensor<10x10x!quant.uniform<i8:f32, 1.000000e+00>>) scale(%[[arg1:.+]] : tensor<f32>) -> tensor<10x10xf32>

