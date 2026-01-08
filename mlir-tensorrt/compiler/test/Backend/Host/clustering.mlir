// RUN: mlir-tensorrt-opt %s -split-input-file \
// RUN:  -plan-clustering -plan-outline-clusters \
// RUN: | FileCheck %s

builtin.module attributes {
  plan.backends = [#plan.host_backend<benefit=1>]
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
//       CHECK:     %[[v0:.+]] = call @host_backend(%[[arg0]]) : (tensor<4xf32>) -> tensor<2x2x1xf32>
//       CHECK:     return %[[v0]] : tensor<2x2x1xf32>

// CHECK-LABEL: func.func private @host_backend
//       CHECK:     %[[v0:.+]] = stablehlo.iota dim = 0 : tensor<4xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.add %[[v0]], %[[arg0]] : tensor<4xf32>
//       CHECK:     %[[v2:.+]] = stablehlo.reshape %[[v1]] : (tensor<4xf32>) -> tensor<2x2x1xf32>
//       CHECK:     return %[[v2]] : tensor<2x2x1xf32>

// -----

builtin.module attributes {
  plan.backends = [#plan.host_backend<benefit=1>]
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
//       CHECK:     %[[v0:.+]] = call @host_backend(%[[arg0]])
//       CHECK:     return %[[v0]]
// CHECK-LABEL: func.func private @host_backend
//  CHECK-SAME: (%[[arg0:.+]]: tensor<f32>)
//       CHECK:     %[[v0:.+]] = stablehlo.iota
//       CHECK:     %[[v1:.+]] = stablehlo.broadcast_in_dim %[[arg0]],
//       CHECK:     %[[v2:.+]] = stablehlo.add %[[v0]], %[[v1]]
//       CHECK:     return %[[v2]]

// -----

// This test checks that we clone constants during outlining rather than
// clustering them into initial regions. We don't want other clusters
// to consume small constants as outputs of other clusters. The effect
// should be as if the constant was duplicated prior to clustering.

builtin.module attributes {
  plan.backends = [#plan.host_backend<benefit=1>]
} {


func.func @clone_constants(%arg0: tensor<4xf32>) -> (tensor<f32>, tensor<f32>, tensor<1xf32>) {
  %cst = stablehlo.constant dense<-0.000000e+00> : tensor<f32>
  %cst1 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
  %reshape = stablehlo.reshape %cst1 : (tensor<1xf32>) -> tensor<f32>
  %add = stablehlo.add %cst, %reshape : tensor<f32>
  return %add, %cst, %cst1 : tensor<f32>, tensor<f32>, tensor<1xf32>
}
}

// CHECK-LABEL: func.func @clone_constants
//  CHECK-SAME: (%[[arg0:.+]]:
//   CHECK-DAG:     %[[cst:.+]] = stablehlo.constant dense<-0.000000e+00>
//   CHECK-DAG:     %[[cst_0:.+]] = stablehlo.constant dense<1.000000e+00>
//   CHECK-DAG:     %[[v0:.+]] = call @host_backend() : () -> tensor<f32>
//   CHECK-DAG:     return %[[v0]], %[[cst]], %[[cst_0]]

// CHECK-LABEL: func.func private @host_backend
//   CHECK-DAG:     %[[cst:.+]] = stablehlo.constant dense<-0.000000e+00>
//   CHECK-DAG:     %[[cst_0:.+]] = stablehlo.constant dense<1.000000e+00>
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.reshape %[[cst_0]]
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.add %[[cst]], %[[v0]]
//   CHECK-DAG:     return %[[v1]]


// -----

builtin.module attributes {
  plan.backends = [#plan.host_backend<benefit=1>]
} {


func.func @conversion_ops(%arg0: tensor<4xf32>) -> (tensor<4xf32> {tensorrt.host_tensor}) {
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<4xf32>) -> tensor<4xi32>
  %1 = stablehlo.convert %0 : (tensor<4xi32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

}

// CHECK-LABEL: func.func @conversion_ops
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>)
//  CHECK-NEXT:     %[[v0:.+]] = call @host_backend(%[[arg0]])
//  CHECK-NEXT:     return %[[v0]]
// CHECK-LABEL: func.func private @host_backend
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>)
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.bitcast_convert %[[arg0]] :
//  CHECK-NEXT:     %[[v1:.+]] = stablehlo.convert %[[v0]]
//  CHECK-NEXT:     return %[[v1]]
