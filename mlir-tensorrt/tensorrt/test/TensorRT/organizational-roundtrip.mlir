// RUN: tensorrt-opt -split-input-file %s | tensorrt-opt -split-input-file | FileCheck %s

tensorrt.module @trt_engines {
  func.func @trt_func1(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    return %arg0: tensor<10xf32>
  }
}

func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %0 = tensorrt.call @trt_engines::@trt_func1(%arg0: tensor<10xf32>) outs(%arg1: tensor<10xf32>)
    -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

//       CHECK: tensorrt.module @trt_engines
// CHECK-LABEL:   @trt_func1
//  CHECK-SAME:   (%[[arg0:.+]]: tensor<10xf32>) -> tensor<10xf32> {
//       CHECK:       return %[[arg0]] : tensor<10xf32>
// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>) -> tensor<10xf32>
//       CHECK:     %[[v0:.+]] = tensorrt.call @trt_engines::@trt_func1(%[[arg0]] : tensor<10xf32>) outs(%[[arg1]] : tensor<10xf32>) -> tensor<10xf32>
//       CHECK:     return %[[v0]] : tensor<10xf32>

