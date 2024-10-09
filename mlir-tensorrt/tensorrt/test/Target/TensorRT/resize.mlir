// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)"  %s | FileCheck %s


func.func @trt_resize_nearest(%arg0: tensor<10x10xf32>) -> tensor<20x20xf32> {
  %result = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>
  } %arg0 : (tensor<10x10xf32>) -> tensor<20x20xf32>
  return %result : tensor<20x20xf32>
}

// CHECK-LABEL: @trt_resize_nearest
//  CHECK-SAME: tensorrt.engine


func.func @trt_resize_nearest_dynamic(
  %arg0: tensor<10x?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[10, 1], opt=[10, 5], max=[10, 10]>}) -> tensor<20x?xf32> {
  %result = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    scales = array<f32: 2.0, 3.0>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>
  } %arg0 : (tensor<10x?xf32>) -> tensor<20x?xf32>
  return %result : tensor<20x?xf32>
}

// CHECK-LABEL: @trt_resize_nearest_dynamic
//  CHECK-SAME: tensorrt.engine

func.func @trt_resize_nearest_output_shape(
  %arg0: tensor<10x10xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %result = tensorrt.resize_nearest {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kASYMMETRIC>,
    nearestRounding = #tensorrt.resize_round_mode<kFLOOR>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>
  } %arg0, %arg1 : (tensor<10x10xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// CHECK-LABEL: @trt_resize_nearest_output_shape
//  CHECK-SAME: tensorrt.engine


func.func @trt_resize_linear(%arg0: tensor<10x10xf32>) -> tensor<20x20xf32> {
  %result = tensorrt.resize_linear {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kALIGN_CORNERS>,
    selectorForSinglePixel = #tensorrt.resize_selector<kUPPER>
  } %arg0 : (tensor<10x10xf32>) -> tensor<20x20xf32>
  return %result : tensor<20x20xf32>
}

// CHECK-LABEL: @trt_resize_linear
//  CHECK-SAME: tensorrt.engine

func.func @trt_resize_cubic(%arg0: tensor<10x10xf32>) -> tensor<20x20xf32> {
  %result = tensorrt.resize_cubic {
    coordinateTransformation = #tensorrt.resize_coordinate_transformation<kHALF_PIXEL>,
    selectorForSinglePixel = #tensorrt.resize_selector<kFORMULA>,
    cubicCoeff = -0.75 : f32
  } %arg0 : (tensor<10x10xf32>) -> tensor<20x20xf32>
  return %result : tensor<20x20xf32>
}

// CHECK-LABEL: @trt_resize_cubic
//  CHECK-SAME: tensorrt.engine
