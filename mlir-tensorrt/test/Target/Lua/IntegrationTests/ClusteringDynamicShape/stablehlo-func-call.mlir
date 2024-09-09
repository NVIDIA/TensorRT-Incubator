// RUN: mlir-tensorrt-opt %s  \
// RUN: -pass-pipeline="builtin.module(stablehlo-preprocessing-pipeline{disable-inliner},\
// RUN: stablehlo-clustering-pipeline, \
// RUN: post-clustering-pipeline, \
// RUN: executor-lowering-pipeline)" \
// RUN: | mlir-tensorrt-translate -mlir-to-runtime-executable -allow-unregistered-dialect |  mlir-tensorrt-runner -input-type=rtexe --use-custom-allocator

func.func @add(%0: tensor<1xf32>, %1: tensor<1xf32>) -> tensor<1xf32> {
    %2 = stablehlo.add %0, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %2 : tensor<1xf32>
}

func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32>  {
  %1 = func.call @add(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}