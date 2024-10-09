// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

// CHECK-LABEL: @tensorrt_shape_op
//  CHECK-SAME: tensorrt.engine
func.func @tensorrt_shape_op(
    %arg0: tensor<10x?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[10, 128], opt=[10, 256], max=[10, 512]>},
  %arg1: tensor<1x1xf32>) -> tensor<2xi32> {
  %0 = tensorrt.element_wise <kSUM> (%arg0, %arg1 : tensor<10x?xf32>, tensor<1x1xf32>) -> tensor<10x?xf32>
  %1 = tensorrt.shape %0 : tensor<10x?xf32> -> tensor<2xi32>
  return %1 : tensor<2xi32>
}



// CHECK-LABEL: @tensorrt_shape_op_0d
//  CHECK-SAME: tensorrt.engine
func.func @tensorrt_shape_op_0d(
    %arg0: tensor<10x?xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<
      min=[10, 128], opt=[10, 256], max=[10, 512]>
    }) -> tensor<0xi32> {
  %0 = tensorrt.reduce <kSUM> %arg0 {reduceAxes = array<i64: 0, 1>} : tensor<10x?xf32> -> tensor<f32>
  %1 = tensorrt.shape %0 : tensor<f32> -> tensor<0xi32>
  return %1 : tensor<0xi32>
}

