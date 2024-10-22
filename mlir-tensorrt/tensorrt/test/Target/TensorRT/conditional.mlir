// RUN: %pick-one-gpu tensorrt-opt -split-input-file -pass-pipeline="builtin.module(translate-tensorrt-to-engine)" -tensorrt-builder-opt-level=0 \
// RUN: --mlir-elide-elementsattrs-if-larger=32  %s | FileCheck %s

// CHECK-LABEL: @trt_if
//  CHECK-SAME: tensorrt.engine
func.func @trt_if(%cond: tensor<i1>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>) -> tensor<10xf32> {
  %result = tensorrt.if (%cond: tensor<i1>) -> tensor<10xf32> {
      %add = tensorrt.element_wise <kSUM>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
          -> tensor<10xf32>
      tensorrt.yield %add: tensor<10xf32>
    } else {
      %sub = tensorrt.element_wise <kSUB>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
          -> tensor<10xf32>
      tensorrt.yield %sub: tensor<10xf32>
    }
  return %result: tensor<10xf32>
}


// CHECK-LABEL: @trt_if_two_layer
//  CHECK-SAME: tensorrt.engine
func.func @trt_if_two_layer(%cond0: tensor<i1>,
                            %cond1: tensor<i1>,
                            %arg1: tensor<10xf32>,
                            %arg2: tensor<10xf32>) -> tensor<10xf32> {
  %r0 = tensorrt.if (%cond0: tensor<i1>) -> tensor<10xf32> {
    // If true, then....
    %r1 = tensorrt.if (%cond1: tensor<i1>) -> tensor<10xf32> {
      %add = tensorrt.element_wise <kSUM>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
               -> tensor<10xf32>
      tensorrt.yield %add: tensor<10xf32>
    } else {
      %sub = tensorrt.element_wise <kSUB>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
                -> tensor<10xf32>
      tensorrt.yield %sub: tensor<10xf32>
    }
    tensorrt.yield %r1: tensor<10xf32>
  } else {
    // If false, then....
    %r1 = tensorrt.if (%cond1: tensor<i1>) -> tensor<10xf32> {
      %prod = tensorrt.element_wise <kPROD>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
                -> tensor<10xf32>
      tensorrt.yield %prod: tensor<10xf32>
    } else {
      %div = tensorrt.element_wise <kDIV>(%arg1, %arg2 : tensor<10xf32>, tensor<10xf32>)
                -> tensor<10xf32>
      tensorrt.yield %div: tensor<10xf32>
    }
    tensorrt.yield %r1: tensor<10xf32>
  }
  return %r0: tensor<10xf32>
}


func.func @trt_f16_if(%cond: tensor<i1>, %arg1: tensor<10xf16>, %arg2: tensor<10xf16>) -> tensor<10xf16> {
  %result = tensorrt.if (%cond: tensor<i1>) -> tensor<10xf16> {
      %add = tensorrt.element_wise <kSUM>(%arg1, %arg2 : tensor<10xf16>, tensor<10xf16>)
          -> tensor<10xf16>
      %add1 = tensorrt.element_wise <kSUM>(%add, %add : tensor<10xf16>, tensor<10xf16>)
          -> tensor<10xf16>
      tensorrt.yield %add1: tensor<10xf16>
    } else {
      %sub = tensorrt.element_wise <kSUB>(%arg1, %arg2 : tensor<10xf16>, tensor<10xf16>)
          -> tensor<10xf16>
      tensorrt.yield %sub: tensor<10xf16>
    }
  return %result: tensor<10xf16>
}

// CHECK-LABEL: @trt_f16_if
//  CHECK-SAME: tensorrt.engine
