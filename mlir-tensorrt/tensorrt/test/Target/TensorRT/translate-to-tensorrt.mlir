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


// CHECK-LABEL: @trt_dim_names
//  CHECK-SAME: tensorrt.engine
func.func @trt_dim_names(
  %arg0: tensor<?x?xf32> {tensorrt.dimension_names = {"0" = "batch", "1" = "features"}, tensorrt.shape_profile = #tensorrt.shape_profile<min=[2, 2], opt=[5, 5], max=[10, 10]>},
  %arg1: tensor<?x?xf32> {tensorrt.dimension_names = {"0" = "batch", "1" = "features"}, tensorrt.shape_profile = #tensorrt.shape_profile<min=[2, 2], opt=[5, 5], max=[10, 10]>},
  %arg2: tensor<2x10xf32>) -> tensor<?x?xf32> {
  %0 = tensorrt.identity %arg0 : tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @trt_host_input
//  CHECK-SAME: tensorrt.engine
func.func @trt_host_input(%arg0: tensor<?x4xf32> {tensorrt.dimension_names = {}, tensorrt.shape_profile = #tensorrt.shape_profile<min = [2, 4], opt = [4, 4], max = [6, 4]>}, %arg1: tensor<i32> {plan.memory_space = #plan.memory_space<host>, tensorrt.value_bounds = #tensorrt.shape_profile<min = [1], opt = [2], max = [3]>}) -> tensor<?x?xf32> {
    %0 = tensorrt.element_wise <kSUM>(%arg0, %arg0 : tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
    %1 = tensorrt.shape %0 : tensor<?x4xf32> -> tensor<2xi32>
    %2 = tensorrt.slice %1[0][1][1] : tensor<2xi32> to tensor<1xi32>
    %3 = tensorrt.collapse_rank %2 : tensor<1xi32> to tensor<i32>
    %cst_i32 = tensorrt.constant dense<1> : tensor<i32>
    %4 = tensorrt.element_wise <kPROD>(%3, %cst_i32 : tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = tensorrt.slice %1[1][1][1] : tensor<2xi32> to tensor<1xi32>
    %6 = tensorrt.collapse_rank %5 : tensor<1xi32> to tensor<i32>
    %7 = tensorrt.element_wise <kPROD>(%4, %6 : tensor<i32>, tensor<i32>) -> tensor<i32>
    %cst_i32_0 = tensorrt.constant dense<1> : tensor<i32>
    %8 = tensorrt.element_wise <kPROD>(%arg1, %cst_i32_0 : tensor<i32>, tensor<i32>) -> tensor<i32>
    %9 = tensorrt.element_wise <kFLOOR_DIV>(%7, %8 : tensor<i32>, tensor<i32>) -> tensor<i32>
    %cst_i32_1 = tensorrt.constant dense<1> : tensor<1xi32>
    %10 = tensorrt.reshape %9 shape(%cst_i32_1: tensor<1xi32>) : tensor<i32> to tensor<?xi32>
    %cst_i32_2 = tensorrt.constant dense<1> : tensor<1xi32>
    %11 = tensorrt.reshape %arg1 shape(%cst_i32_2: tensor<1xi32>) : tensor<i32> to tensor<?xi32>
    %12 = tensorrt.concatenation {axis = 0 : i32} ins(%10, %11 : tensor<?xi32>, tensor<?xi32>) -> tensor<2xi32>
    %13 = tensorrt.reshape %0 shape(%12: tensor<2xi32>) : tensor<?x4xf32> to tensor<?x?xf32>
    return %13 : tensor<?x?xf32>
}
