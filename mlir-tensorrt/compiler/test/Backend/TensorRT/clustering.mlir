// RUN: mlir-tensorrt-opt %s -split-input-file \
// RUN:  -plan-clustering="entrypoint=" -plan-create-closed-regions -plan-outline-clusters \
// RUN: | FileCheck %s

builtin.module attributes {
  plan.backends = [#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>]
} {

  func.func @cluster_and_outline_test(%arg0: tensor<4xf32>) -> (tensor<2xf32>) {
    %1 = stablehlo.slice %arg0[0:2] : (tensor<4xf32>) -> tensor<2xf32>
    return %1: tensor<2xf32>
  }
}

// CHECK-LABEL: func.func @cluster_and_outline_test
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>)
//       CHECK:     %[[v0:.+]] = stablehlo.slice %[[arg0]]
//       CHECK:     return %[[v0]] : tensor<2xf32>

