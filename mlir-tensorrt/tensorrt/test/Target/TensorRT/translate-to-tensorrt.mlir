// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" \
// RUN:  -mlir-elide-elementsattrs-if-larger=32 -tensorrt-builder-opt-level=0 %s | FileCheck %s

// CHECK-LABEL: @trt_softmax
//  CHECK-SAME: tensorrt.engine
func.func @trt_softmax(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>) -> (tensor<1x128x128xf32>, tensor<1x128x128xf32>) {
  %0 = tensorrt.softmax {axis = 2 : i64} %arg0 : tensor<1x128x128xf32>
  %1 = tensorrt.softmax {axis = 1 : i64} %arg1 : tensor<1x128x128xf32>
  return %0, %1 : tensor<1x128x128xf32>, tensor<1x128x128xf32>
}


// CHECK-LABEL: @trt_dynamic_input_shape
//  CHECK-SAME: tensorrt.engine
func.func @trt_dynamic_input_shape(
    %arg0: tensor<?x1024xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[5, 1024], opt=[10,1024], max=[10, 1024]>},
    %arg1: tensor<?x1024xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[5, 1024], opt=[10,1024], max=[10, 1024]>}) -> tensor<?x1024xf32> {
  %0 = tensorrt.element_wise <kSUM>(%arg0, %arg1 : tensor<?x1024xf32>, tensor<?x1024xf32>) -> tensor<?x1024xf32>
  return %0 : tensor<?x1024xf32>
}



// CHECK-LABEL: @trt_element_wise
//  CHECK-SAME: tensorrt.engine
func.func @trt_element_wise(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %0 = tensorrt.element_wise <kSUM>(%arg0, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}


// CHECK-LABEL: @trt_element_wise_constant
//  CHECK-SAME: tensorrt.engine
func.func @trt_element_wise_constant(%arg0: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %rhs = tensorrt.constant dense<1.0> : tensor<1x1xf32>
  %0 = tensorrt.element_wise <kSUM>(%arg0, %rhs : tensor<1024x1024xf32>, tensor<1x1xf32>) -> tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}


// CHECK-LABEL: @trt_reduce
//  CHECK-SAME: tensorrt.engine
func.func @trt_reduce(%arg0: tensor<1024x1024xf32>) -> tensor<1024x1xf32> {
  %0 = tensorrt.reduce <kSUM> %arg0 {reduceAxes = array<i64: 1>, keepDimensions = true} : tensor<1024x1024xf32> -> tensor<1024x1xf32>
  return %0 : tensor<1024x1xf32>
}


// Regression test for issue #591 - TensorRT does not allow passing
// inputs to outputs, so translation should automatically insert an identity.

// CHECK-LABEL: @input_passthrough
//  CHECK-SAME: tensorrt.engine
func.func @input_passthrough(%arg0: tensor<1xf32>, %arg1: tensor<1xf16>, %arg2: tensor<1xi32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<1xf16>, tensor<1xi32>) {
  return %arg0, %arg0, %arg1, %arg2: tensor<1xf32>, tensor<1xf32>, tensor<1xf16>, tensor<1xi32>
}
