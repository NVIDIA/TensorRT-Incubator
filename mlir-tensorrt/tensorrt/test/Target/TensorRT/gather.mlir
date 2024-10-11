// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

func.func @trt_gather_default(%arg0: tensor<10x20x30xf32>, %arg1: tensor<5xi32>, %arg2: tensor<10x5x30xf32>) -> tensor<10x5x30xf32> {
  %0 = tensorrt.gather {
    axis = 1 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<5xi32>) -> tensor<10x5x30xf32>

  // Add an elementwise to confirm shape
  %1 = tensorrt.element_wise <kSUM>(%0, %arg2 : tensor<10x5x30xf32>, tensor<10x5x30xf32>) -> tensor<10x5x30xf32>
  return %1 : tensor<10x5x30xf32>
}

// CHECK-LABEL: @trt_gather_default
//  CHECK-SAME: tensorrt.engine


func.func @trt_gather_default1(%arg0: tensor<10x20x30xf32>, %arg1: tensor<2x5xi32>,
                %arg2: tensor<10x2x5x30xf32>) -> tensor<10x2x5x30xf32> {
  %0 = tensorrt.gather {
    axis = 1 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<2x5xi32>) -> tensor<10x2x5x30xf32>

  // Add an elementwise to confirm shape
  %1 = tensorrt.element_wise <kSUM>(%0, %arg2 : tensor<10x2x5x30xf32>, tensor<10x2x5x30xf32>) -> tensor<10x2x5x30xf32>
  return %1 : tensor<10x2x5x30xf32>
}

// CHECK-LABEL: @trt_gather_default1
//  CHECK-SAME: tensorrt.engine

func.func @trt_gather_default_i32(%arg0: tensor<10x20x30xi32>, %arg1: tensor<2x5xi32>,
                %arg2: tensor<10x2x5x30xi32>) -> tensor<10x2x5x30xi32> {
  %0 = tensorrt.gather {
    axis = 1 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xi32>, tensor<2x5xi32>) -> tensor<10x2x5x30xi32>

  // Add an elementwise to confirm shape
  %1 = tensorrt.element_wise <kSUM>(%0, %arg2 : tensor<10x2x5x30xi32>, tensor<10x2x5x30xi32>) -> tensor<10x2x5x30xi32>
  return %1 : tensor<10x2x5x30xi32>
}

// CHECK-LABEL: @trt_gather_default_i32
//  CHECK-SAME: tensorrt.engine

func.func @trt_gather_default_broadcast(%arg0: tensor<10x20x30xf32>, %arg1: tensor<10x20xi32>,
                %arg2: tensor<10x20x30xf32>) -> tensor<10x20x30xf32> {
  %0 = tensorrt.gather {
    axis = 1 : i64,
    numBroadcastDims = 1 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<10x20xi32>) -> tensor<10x20x30xf32>

  // Add an elementwise to confirm shape
  %1 = tensorrt.element_wise <kSUM>(%0, %arg2 : tensor<10x20x30xf32>, tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
  return %1 : tensor<10x20x30xf32>
}

// CHECK-LABEL: @trt_gather_default_broadcast
//  CHECK-SAME: tensorrt.engine


func.func @trt_gather_default_broadcast1(%arg0: tensor<10x20x30xf32>, %arg1: tensor<10xi32>,
                %arg2: tensor<10x30xf32>) -> tensor<10x30xf32> {
  %0 = tensorrt.gather {
    axis = 1 : i64,
    numBroadcastDims = 1 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<10xi32>) -> tensor<10x30xf32>

  // Add an elementwise to confirm shape
  %1 = tensorrt.element_wise <kSUM>(%0, %arg2 : tensor<10x30xf32>, tensor<10x30xf32>) -> tensor<10x30xf32>
  return %1 : tensor<10x30xf32>
}

// CHECK-LABEL: @trt_gather_default_broadcast1
//  CHECK-SAME: tensorrt.engine


func.func @trt_gather_elements(%arg0: tensor<10x20x30xf32>, %arg1: tensor<10x20x10xi32>,
                %arg2: tensor<10x20x10xf32>) -> tensor<10x20x10xf32> {
  %0 = tensorrt.gather_elements {
    axis = 2 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xf32>, tensor<10x20x10xi32>) -> tensor<10x20x10xf32>

  // Add an elementwise to confirm shape
  %1 = tensorrt.element_wise <kSUM>(%0, %arg2 : tensor<10x20x10xf32>, tensor<10x20x10xf32>) -> tensor<10x20x10xf32>
  return %1 : tensor<10x20x10xf32>
}

// CHECK-LABEL: @trt_gather_elements
//  CHECK-SAME: tensorrt.engine

func.func @trt_gather_elements_i32(%arg0: tensor<10x20x30xi32>, %arg1: tensor<10x20x10xi32>,
                %arg2: tensor<10x20x10xi32>) -> tensor<10x20x10xi32> {
  %0 = tensorrt.gather_elements {
    axis = 2 : i64
  } ins(%arg0, %arg1 : tensor<10x20x30xi32>, tensor<10x20x10xi32>) -> tensor<10x20x10xi32>

  // Add an elementwise to confirm shape
  %1 = tensorrt.element_wise <kSUM>(%0, %arg2 : tensor<10x20x10xi32>, tensor<10x20x10xi32>) -> tensor<10x20x10xi32>
  return %1 : tensor<10x20x10xi32>
}

// CHECK-LABEL: @trt_gather_elements_i32
//  CHECK-SAME: tensorrt.engine

