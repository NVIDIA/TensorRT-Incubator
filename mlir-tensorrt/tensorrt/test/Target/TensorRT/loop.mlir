// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

func.func @trt_for_loop() -> tensor<10xf32> {
  %lb = tensorrt.constant dense<0> : tensor<i32>
  %ub = tensorrt.constant dense<10> : tensor<i32>
  %step = tensorrt.constant dense<1> : tensor<i32>
  %zeros = tensorrt.constant dense<0.0> : tensor<10xf32>
  %ones = tensorrt.constant dense<1.0> : tensor<10xf32>
  %0 = tensorrt.for %i = %lb to %ub step %step init(%iter0 = %zeros) -> tensor<10xf32> {
    %add = tensorrt.element_wise <kSUM>(%iter0, %ones : tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    tensorrt.yield %add : tensor<10xf32>
  }
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @trt_for_loop
//  CHECK-SAME: tensorrt.engine


func.func @trt_while_loop() -> (tensor<f32>, tensor<f32>) {
  %one = tensorrt.constant dense<1.0> : tensor<f32>
  %iter_init = tensorrt.constant dense<1.0> : tensor<f32>
  %limit = tensorrt.constant dense<10.0> : tensor<f32>
  %res_init = tensorrt.constant dense<0.0> : tensor<f32>
  %result0, %result1 = tensorrt.while(%iter_init,%res_init : tensor<f32>,tensor<f32>) -> tensor<f32>,tensor<f32>
  {
    // condition
    ^bb0(%iter:tensor<f32>, %result: tensor<f32>):
      %cond = tensorrt.element_wise <kLESS>(%iter, %limit : tensor<f32>, tensor<f32>)
            -> tensor<i1>
      tensorrt.condition(%cond : tensor<i1>) %iter, %result : tensor<f32>,tensor<f32>
  } ,
  {
    // body
    ^bb1(%iter:tensor<f32>, %result: tensor<f32>):
      %new_result = tensorrt.element_wise <kSUM> (%iter, %result : tensor<f32>, tensor<f32>) -> tensor<f32>
      %new_iter = tensorrt.element_wise <kSUM> (%one, %iter : tensor<f32>, tensor<f32>) -> tensor<f32>
      tensorrt.yield %new_iter, %new_result: tensor<f32>, tensor<f32>
  }
  return %result0, %result1 : tensor<f32>,tensor<f32>
}
// CHECK-LABEL: @trt_while_loop
//  CHECK-SAME: tensorrt.engine
