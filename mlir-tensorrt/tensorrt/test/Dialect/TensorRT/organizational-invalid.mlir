// RUN: tensorrt-opt -allow-unregistered-dialect -split-input-file --verify-diagnostics %s

tensorrt.module @trt_engines {
  func.func @trt_func1(%arg0: tensor<20xf32>) -> tensor<10xf32> {
    %0 = tensorrt.constant dense<1.0> : tensor<10xf32>
    return %0: tensor<10xf32>
  }
}

func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error @below {{'tensorrt.call' op callee has function type '(tensor<20xf32>) -> tensor<10xf32>' which is not compatible with input/result types of call}}
  %0 = tensorrt.call @trt_engines::@trt_func1(%arg0: tensor<10xf32>) outs(%arg1: tensor<10xf32>)
    -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
