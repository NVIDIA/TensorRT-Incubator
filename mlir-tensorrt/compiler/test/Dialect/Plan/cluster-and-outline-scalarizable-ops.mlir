// RUN: mlir-tensorrt-opt %s -split-input-file \
// RUN:  -plan-clustering="entrypoint=" -plan-outline-clusters \
// RUN: | FileCheck %s

builtin.module attributes {
  plan.cluster_kinds = [#plan.host_cluster<benefit=1>]
} {

  func.func @cluster_and_outline_test(%arg0: tensor<4xf32>) -> (tensor<2x2x1xf32> {tensorrt.host_tensor}) {
    %1 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> (tensor<4xf32>)
    %2 = "stablehlo.add"(%1, %arg0) : (
      tensor<4xf32>, tensor<4xf32>
    ) -> (tensor<4xf32>)
    %3 = stablehlo.reshape %2 : (tensor<4xf32>) -> tensor<2x2x1xf32>
    return %3: tensor<2x2x1xf32>
  }
}

// CHECK-LABEL: func.func @cluster_and_outline_test
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>)
//       CHECK:     %[[v0:.+]] = call @host_cluster(%[[arg0]]) : (tensor<4xf32>) -> tensor<2x2x1xf32>
//       CHECK:     return %[[v0]] : tensor<2x2x1xf32>

// CHECK-LABEL: func.func private @host_cluster
//       CHECK:     %[[v0:.+]] = stablehlo.iota dim = 0 : tensor<4xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.add %[[v0]], %[[arg0]] : tensor<4xf32>
//       CHECK:     %[[v2:.+]] = stablehlo.reshape %[[v1]] : (tensor<4xf32>) -> tensor<2x2x1xf32>
//       CHECK:     return %[[v2]] : tensor<2x2x1xf32>

// -----

builtin.module attributes {
  plan.cluster_kinds = [#plan.host_cluster<benefit=1>]
} {

func.func @cluster_and_outline_test2(%arg0: tensor<f32>) -> (tensor<1xf32> {tensorrt.host_tensor}) {
  %1 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> (tensor<1xf32>)
  %2 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64>} : (tensor<f32>) -> tensor<1xf32>
  %3 = "stablehlo.add"(%1, %2) : (
    tensor<1xf32>, tensor<1xf32>
  ) -> (tensor<1xf32>)
  return %3: tensor<1xf32>
}

}

// CHECK-LABEL: func.func @cluster_and_outline_test2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<f32>)
//       CHECK:     %[[v0:.+]] = call @host_cluster(%[[arg0]])
//       CHECK:     return %[[v0]]
// CHECK-LABEL: func.func private @host_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<f32>)
//       CHECK:     %[[v0:.+]] = stablehlo.iota
//       CHECK:     %[[v1:.+]] = stablehlo.broadcast_in_dim %[[arg0]],
//       CHECK:     %[[v2:.+]] = stablehlo.add %[[v0]], %[[v1]]
//       CHECK:     return %[[v2]]

// -----

func.func @const_only_dont_cluster(%arg0: tensor<4xf32>) -> (tensor<f32> {tensorrt.host_tensor}, tensor<1xf32> {tensorrt.host_tensor}) {
  %cst = stablehlo.constant dense<-0.000000e+00> : tensor<f32>
  %cst1 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
  return %cst, %cst1 : tensor<f32>, tensor<1xf32>
}

// CHECK-LABEL: func.func @const_only_dont_cluster
//  CHECK-NEXT:     %[[cst:.+]] = stablehlo.constant
//  CHECK-NEXT:     %[[cst_0:.+]] = stablehlo.constant
//  CHECK-NEXT:     return %[[cst]], %[[cst_0]]

// -----

builtin.module attributes {
  plan.cluster_kinds = [#plan.host_cluster<benefit=1>]
} {


func.func @conversion_ops(%arg0: tensor<4xf32>) -> (tensor<4xf32> {tensorrt.host_tensor}) {
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<4xf32>) -> tensor<4xi32>
  %1 = stablehlo.convert %0 : (tensor<4xi32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

}

// CHECK-LABEL: func.func @conversion_ops
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>)
//  CHECK-NEXT:     %[[v0:.+]] = call @host_cluster(%[[arg0]])
//  CHECK-NEXT:     return %[[v0]]
// CHECK-LABEL: func.func private @host_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>)
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.bitcast_convert %[[arg0]] :
//  CHECK-NEXT:     %[[v1:.+]] = stablehlo.convert %[[v0]]
//  CHECK-NEXT:     return %[[v1]]