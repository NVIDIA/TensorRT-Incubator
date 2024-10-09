// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine{})" \
// RUN:  -tensorrt-builder-opt-level=0 -mlir-elide-elementsattrs-if-larger=32 %s | FileCheck %s

// CHECK-LABEL: @trt_slice_static
//  CHECK-SAME: tensorrt.engine
func.func @trt_slice_static(%arg0: tensor<1024x1024xf32>) -> tensor<128x128xf32> {
  %0 = tensorrt.slice %arg0[512, 512][128, 128][2, 2] : tensor<1024x1024xf32> to tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}


// CHECK-LABEL: @trt_slice_dynamic
//  CHECK-SAME: tensorrt.engine
func.func @trt_slice_dynamic(%arg0: tensor<1024x1024xf32>) -> tensor<?x?xf32> {
  %size = tensorrt.constant dense<[128, 128]> : tensor<2xi32>
  // expected-warning @below {{MLIR Tensor has type: tensor<?x?xf32>, TRT ITensor has type tensor<128x128xf32>}}
  // expected-note @below {{TensorRT does some type inference while constructing the TensorRT network; consider running canonicalization prior to TensorRT translation in order to run type inference to potentially eliminate these differences}}
  %0 = tensorrt.slice %arg0[512, 512][%size: tensor<2xi32>][2, 2] : tensor<1024x1024xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}


// CHECK-LABEL: @trt_slice_tile
//  CHECK-SAME: tensorrt.engine
func.func @trt_slice_tile(%arg0: tensor<1x128xf32>) -> tensor<128x128xf32> {
  %0 = tensorrt.slice %arg0[0, 0][128, 128][0, 1] : tensor<1x128xf32> to tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}


// CHECK-LABEL: @trt_slice_tile_wrap
//  CHECK-SAME: tensorrt.engine
func.func @trt_slice_tile_wrap(%arg0: tensor<1x1xf32>) -> tensor<128x128xf32> {
  %0 = tensorrt.slice %arg0[0, 0][128, 128][1, 1] {mode = #tensorrt.slice_mode<kWRAP>} : tensor<1x1xf32> to tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}


// CHECK-LABEL: @trt_slice_pad
//  CHECK-SAME: tensorrt.engine
func.func @trt_slice_pad(%arg0: tensor<128x128xf32>) -> tensor<130x130xf32> {
  %cst = tensorrt.constant dense<0.0> : tensor<1xf32>
  %0 = tensorrt.slice %arg0[-1, -1][130, 130][1, 1] fill(%cst : tensor<1xf32>) {
    mode = #tensorrt.slice_mode<kFILL>
  } : tensor<128x128xf32> to tensor<130x130xf32>
  return %0 : tensor<130x130xf32>
}


// This test exposed a bug regarding constant duplication: the network encoder
// should be able to handle duplicated constants gracefully. Keep the test to
// prevent regression.

// CHECK-LABEL: @trt_slice_const
//  CHECK-SAME: tensorrt.engine
func.func @trt_slice_const(%arg0: tensor<128x128xf32>) -> tensor<4x4xf32> {
  %cst = tensorrt.constant dense<0.0> : tensor<1xf32>
  %cst1 = tensorrt.constant dense<0> : tensor<2xi32>
  %cst2 = tensorrt.constant dense<1> : tensor<2xi32>
  %cst3 = tensorrt.constant dense<2> : tensor<2xi32>
  %cst4 = tensorrt.constant dense<2> : tensor<2xi32>
  %add = tensorrt.element_wise <kSUM> (%cst3, %cst4 : tensor<2xi32> ,tensor<2xi32>) -> tensor<2xi32>
  %0 = tensorrt.slice %arg0[%cst1: tensor<2xi32>][%add: tensor<2xi32>][%cst2: tensor<2xi32>] fill(%cst : tensor<1xf32>) {
    mode = #tensorrt.slice_mode<kFILL>
  } : tensor<128x128xf32> to tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
