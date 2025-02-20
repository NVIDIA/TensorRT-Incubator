// RUN: mlir-tensorrt-opt %s -split-input-file -inline | FileCheck %s

trtrt.compiled_func @tensorrt_cluster_region dense<0> : vector<2xi8>

func.func private @tensorrt_cluster(%arg0: tensor<3072x768xf32>, %arg1: !cuda.stream) -> tensor<3072x768xf16> attributes {cluster.tensorrt} {
  %0 = bufferization.alloc_tensor() {memory_space = #executor.memory_type<device>} : tensor<3072x768xf16>
  %1 = trtrt.get_function @tensorrt_cluster_region : !trtrt.context
  %3 = trtrt.enqueue %1 stream(%arg1) (%arg0) outs(%0) : (tensor<3072x768xf32>) -> tensor<3072x768xf16>
  return %3 : tensor<3072x768xf16>
}
func.func public @test_inliner(%arg0: tensor<3072x768xf32>, %arg1: !cuda.stream) -> tensor<3072x768xf16> {
  %0 = func.call @tensorrt_cluster(%arg0, %arg1) : (tensor<3072x768xf32>, !cuda.stream) -> tensor<3072x768xf16>
  return %0 : tensor<3072x768xf16>
}

// CHECK-LABEL:  func.func public @test_inliner
//  CHECK-SAME:  (%[[arg0:.+]]: tensor<3072x768xf32>, %[[arg1:.+]]: !cuda.stream)
//       CHECK:     %[[v0:.+]] = bufferization.alloc_tensor()
//       CHECK:     %[[v1:.+]] = trtrt.get_function
//       CHECK:     %[[v2:.+]] = trtrt.enqueue %[[v1]] stream(%[[arg1]]) (%[[arg0]]) outs(%[[v0]])
//       CHECK:     return %[[v2]] : tensor<3072x768xf16>
