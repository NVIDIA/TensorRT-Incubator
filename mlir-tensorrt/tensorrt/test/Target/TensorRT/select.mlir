// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)"  %s | FileCheck %s

// CHECK-LABEL: @trt_select_op
//  CHECK-SAME: tensorrt.engine
func.func @trt_select_op(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = tensorrt.element_wise <kGREATER>(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)->tensor<128x128xi1>
  %1 = tensorrt.select ins(%0, %arg0, %arg1: tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>)->tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}


// CHECK-LABEL: @trt_select_op_broadcast_to_cond
//  CHECK-SAME: tensorrt.engine
func.func @trt_select_op_broadcast_to_cond(%arg0: tensor<1x128xf32>, %arg1: tensor<1x128xf32>, %arg2: tensor<128x128xi1>) -> tensor<128x128xf32> {
  %1 = tensorrt.select ins(%arg2, %arg0, %arg1: tensor<128x128xi1>, tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}
