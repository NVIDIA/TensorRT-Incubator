// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 -tensorrt-strongly-typed %s | FileCheck %s
// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_slice_static_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_slice_static_fp8(%arg0: tensor<1024x1024xf8E4M3FN>) -> tensor<128x128xf32> {
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<1024x1024xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<1024x1024xf32>
  %0 = tensorrt.slice %dq_arg0[512, 512][128, 128][2, 2] : tensor<1024x1024xf32> to tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: @trt_slice_dynamic_fp8
//  CHECK-SAME: tensorrt.engine
func.func @trt_slice_dynamic_fp8(%arg0: tensor<1024x1024xf8E4M3FN>) -> tensor<?x?xf32> {
  %size = tensorrt.constant dense<[128, 128]> : tensor<2xi32>
  // expected-warning @below {{MLIR Tensor has type: tensor<?x?xf32>, TRT ITensor has type tensor<128x128xf32>}}
  // expected-note @below {{TensorRT does some type inference while constructing the TensorRT network; consider running canonicalization prior to TensorRT translation in order to run type inference to potentially eliminate these differences}}
  %scale = tensorrt.constant dense<1.000000e+00> : tensor<f32>
  %dq_arg0 = tensorrt.dequantize in (%arg0: tensor<1024x1024xf8E4M3FN>) scale (%scale: tensor<f32>) -> tensor<1024x1024xf32>
  %0 = tensorrt.slice %dq_arg0[512, 512][%size: tensor<2xi32>][2, 2] : tensor<1024x1024xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @trt_slice_dynamic_bf16
//  CHECK-SAME: tensorrt.engine
func.func @trt_slice_dynamic_bf16(%arg0: tensor<1024x1024xbf16>) -> tensor<?x?xbf16> {
  %size = tensorrt.constant dense<[128, 128]> : tensor<2xi32>
  // expected-warning @below {{MLIR Tensor has type: tensor<?x?xf32>, TRT ITensor has type tensor<128x128xf32>}}
  // expected-note @below {{TensorRT does some type inference while constructing the TensorRT network; consider running canonicalization prior to TensorRT translation in order to run type inference to potentially eliminate these differences}}
  %0 = tensorrt.slice %arg0[512, 512][%size: tensor<2xi32>][2, 2] : tensor<1024x1024xbf16> to tensor<?x?xbf16>
  return %0 : tensor<?x?xbf16>
}
