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
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<4xf32>
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c1]]] : tensor<4xf32>
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[arg0]][%[[c2]]] : tensor<4xf32>
//       CHECK:     %[[c3:.+]] = arith.constant 3 : index
//       CHECK:     %[[extracted_2:.+]] = tensor.extract %[[arg0]][%[[c3]]] : tensor<4xf32>
//       CHECK:     %[[v0:.+]]:4 = call @host_cluster(%[[extracted]], %[[extracted_0]], %[[extracted_1]], %[[extracted_2]])
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v0]]#0, %[[v0]]#1, %[[v0]]#2, %[[v0]]#3 : tensor<2x2x1xf32>
//       CHECK:     return %[[from_elements]]
// CHECK-LABEL: private @host_cluster
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32, %[[arg2:.+]]: f32, %[[arg3:.+]]: f32)
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]] : tensor<4xf32>
//       CHECK:     %[[v0:.+]] = stablehlo.iota dim = 0 : tensor<4xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.add %[[v0]], %[[from_elements]] : tensor<4xf32>
//       CHECK:     %[[v2:.+]] = stablehlo.reshape %[[v1]] : (tensor<4xf32>) -> tensor<2x2x1xf32>
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c0_0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c0_1:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[v2]][%[[c0]], %[[c0_0]], %[[c0_1]]]
//       CHECK:     %[[c0_2:.+]] = arith.constant 0 : index
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[c0_3:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted_4:.+]] = tensor.extract %[[v2]][%[[c0_2]], %[[c1]], %[[c0_3]]]
//       CHECK:     %[[c1_5:.+]] = arith.constant 1 : index
//       CHECK:     %[[c0_6:.+]] = arith.constant 0 : index
//       CHECK:     %[[c0_7:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted_8:.+]] = tensor.extract %[[v2]][%[[c1_5]], %[[c0_6]], %[[c0_7]]]
//       CHECK:     %[[c1_9:.+]] = arith.constant 1 : index
//       CHECK:     %[[c1_10:.+]] = arith.constant 1 : index
//       CHECK:     %[[c0_11:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted_12:.+]] = tensor.extract %[[v2]][%[[c1_9]], %[[c1_10]], %[[c0_11]]]
//       CHECK:     return %[[extracted]], %[[extracted_4]], %[[extracted_8]], %[[extracted_12]]

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
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][] : tensor<f32>
//       CHECK:     %[[v0:.+]] = call @host_cluster(%[[extracted]]) : (f32) -> f32
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v0]] : tensor<1xf32>
//       CHECK:     return %[[from_elements]] : tensor<1xf32>
//       CHECK: func.func private @host_cluster
//  CHECK-SAME: (%[[arg0:.+]]: f32)
//  CHECK-SAME: attributes {cluster.host}
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[arg0]] : tensor<f32>
//       CHECK:     %[[v0:.+]] = stablehlo.iota dim = 0 : tensor<1xf32>
//       CHECK:     %[[v1:.+]] = stablehlo.broadcast_in_dim %[[from_elements]]
//       CHECK:     %[[v2:.+]] = stablehlo.add %[[v0]], %[[v1]] : tensor<1xf32>
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[v2]][%[[c0]]] : tensor<1xf32>
//       CHECK:     return %[[extracted]] : f32

// -----

func.func @const_only_dont_cluster(%arg0: tensor<4xf32>) -> (tensor<f32> {tensorrt.host_tensor}, tensor<1xf32> {tensorrt.host_tensor}) {
  %cst = stablehlo.constant dense<-0.000000e+00> : tensor<f32>
  %cst1 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
  return %cst, %cst1 : tensor<f32>, tensor<1xf32>
}

// CHECK-LABEL: @const_only_dont_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>)
//  CHECK-NEXT:     %[[v0:.+]] = stablehlo.constant dense<-0.000000e+00> : tensor<f32>
//  CHECK-NEXT:     %[[v1:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
//  CHECK-NEXT:     return %[[v0]], %[[v1]] : tensor<f32>, tensor<1xf32>

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

// CHECK-LABEL: @conversion_ops
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>)
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<4xf32>
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c1]]] : tensor<4xf32>
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[arg0]][%[[c2]]] : tensor<4xf32>
//       CHECK:     %[[c3:.+]] = arith.constant 3 : index
//       CHECK:     %[[extracted_2:.+]] = tensor.extract %[[arg0]][%[[c3]]] : tensor<4xf32>
//       CHECK:     %[[v0:.+]]:4 = call @host_cluster(%[[extracted]], %[[extracted_0]], %[[extracted_1]], %[[extracted_2]]) : (f32, f32, f32, f32) -> (f32, f32, f32, f32)
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v0]]#0, %[[v0]]#1, %[[v0]]#2, %[[v0]]#3 : tensor<4xf32>
//       CHECK:     return %[[from_elements]] : tensor<4xf32>

//       CHECK: func.func private @host_cluster
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32, %[[arg2:.+]]: f32, %[[arg3:.+]]: f32) -> (f32, f32, f32, f32) attributes {cluster.host} {
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]] : tensor<4xf32>
//       CHECK:     %[[v0:.+]] = stablehlo.bitcast_convert %[[from_elements]] : (tensor<4xf32>) -> tensor<4xi32>
//       CHECK:     %[[v1:.+]] = stablehlo.convert %[[v0]] : (tensor<4xi32>) -> tensor<4xf32>
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[v1]][%[[c0]]] : tensor<4xf32>
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[v1]][%[[c1]]] : tensor<4xf32>
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[v1]][%[[c2]]] : tensor<4xf32>
//       CHECK:     %[[c3:.+]] = arith.constant 3 : index
//       CHECK:     %[[extracted_2:.+]] = tensor.extract %[[v1]][%[[c3]]] : tensor<4xf32>
//       CHECK:     return %[[extracted]], %[[extracted_0]], %[[extracted_1]], %[[extracted_2]] : f32, f32, f32, f32
