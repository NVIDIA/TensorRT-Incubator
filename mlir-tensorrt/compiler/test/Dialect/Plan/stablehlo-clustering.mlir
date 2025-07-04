// RUN: mlir-tensorrt-opt -split-input-file \
// RUN:  -plan-clustering %s | FileCheck %s

// Check that we can recognize `stablehlo.dynamic_gather` using `plan.with_shape|plan.with_values` to prove required shape/value equivalence
// propositions.

builtin.module attributes {
  plan.cluster_kinds = [
    #plan.tensorrt_cluster<benefit = 1, disallow_shape_tensor_calculations=false, tensorrt_major_version = 10>,
    #plan.host_cluster<benefit = 0>
  ]
} {

func.func @simple_gather_dynamic(%arg0: tensor<?x?x256x256xi32>, %arg1: tensor<?xi32>, %arg2: tensor<4xi32>) -> tensor<?x?x256x256xi32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : i32
  %dim.0 = tensor.dim %arg0, %c0 : tensor<?x?x256x256xi32>
  %dim = tensor.dim %arg0, %c1 : tensor<?x?x256x256xi32>
  %dim.1 = arith.index_cast %dim : index to i32
  %c1_i32 = arith.constant 1 : i32
  %data = plan.with_shape %arg0(%dim.0, %dim.1, %c256, %c256) : (tensor<?x?x256x256xi32>, index, i32, i32, i32) -> tensor<?x?x256x256xi32>
  %slice_sizes = plan.with_values %arg2(%c1_i32, %dim.1, %c256, %c256) : tensor<4xi32>
  %0 = "stablehlo.dynamic_gather"(%data, %arg1, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    indices_are_sorted = false, slice_sizes = array<i64: 1>
  } : (tensor<?x?x256x256xi32>, tensor<?xi32>, tensor<4xi32>) -> tensor<?x?x256x256xi32>
  return %0 : tensor<?x?x256x256xi32>
}

}

// CHECK-LABEL: func.func @simple_gather_dynamic(
//       CHECK:     %[[v1:.+]] = plan.inline_group target(#plan.tensorrt_cluster<
//  CHECK-NEXT:       with_shape
//  CHECK-NEXT:       with_values
//  CHECK-NEXT:       stablehlo.dynamic_gather
//  CHECK-NEXT:       yield

// -----

// Test that interleaved `plan.with_values` and `arith` dialect operations don't disrupt
// the clustering of stablehlo ops that can be put into host clusters.

builtin.module @host_clusters_with_values attributes {
  plan.cluster_kinds = [
    #plan.tensorrt_cluster<benefit = 1, disallow_shape_tensor_calculations=true>,
    #plan.host_cluster<benefit = 0>
  ]
} {

func.func @test(%arg0: tensor<4xi32>, %arg1: tensor<i32>)
    -> (tensor<i32> {tensorrt.host_tensor}, tensor<i1> {tensorrt.host_tensor}) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = stablehlo.constant dense<0> : tensor<i32>

  %c0_i32 = arith.constant 0 : i32
  %3 = stablehlo.compare EQ, %2, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %extract = tensor.extract %arg1[] : tensor<i32>
  %cmp = arith.cmpi eq, %c0_i32, %extract : i32
  %with_values = plan.with_values %3(%cmp) : tensor<i1>

  %4 = stablehlo.reduce(%arg0 init: %2) across dimensions = [0] : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
    reducer(%arg6: tensor<i32>, %arg7: tensor<i32>)  {
    %27 = stablehlo.add %arg6, %arg7 : tensor<i32>
    stablehlo.return %27 : tensor<i32>
  }
  return %4, %with_values : tensor<i32>, tensor<i1>
}

}

// CHECK-LABEL: module @host_clusters_with_values attributes
// CHECK-LABEL: func.func @test
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>, %[[arg1:.+]]: tensor<i32>)
//   CHECK-DAG:     %[[cst:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
//   CHECK-DAG:     %[[cst_0:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
//   CHECK-DAG:     %[[c:.+]] = stablehlo.constant dense<0> : tensor<i32>
//   CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg1]][] : tensor<i32>
//   CHECK-DAG:     %[[v0:.+]] = arith.cmpi eq
//   CHECK-DAG:     %[[v1:.+]]:2 = plan.inline_group target(#plan.host_cluster<benefit = 0>)
//   CHECK-DAG:       %[[v2:.+]] = stablehlo.compare  EQ, %[[c]], %[[arg1]] :
//   CHECK-DAG:       %[[v3:.+]] = with_values %[[v2]](%[[v0]]) : tensor<i1>
//   CHECK-DAG:       %[[v4:.+]] = stablehlo.reduce(%[[arg0]] init: %[[c]])
//   CHECK-DAG:       yield %[[v3]], %[[v4]] : tensor<i1>, tensor<i32>
//   CHECK-DAG:     return %[[v1]]#1, %[[v1]]#0 : tensor<i32>, tensor<i1>

// -----

// Ensure that we don't create clusters containing 'plan.with_values' or
// 'plan.with_shape' operations. This can cause some un-intended side-effects
// if the compiler introduces extra ops to outline these clusters (e.g. scalar
// host clusters create extra `tensor.extract` and `tensor.from_elements` at
// the boundaries). These extra ops can block optimizations and
// reduce performance.

builtin.module attributes {
  plan.cluster_kinds = [
    #plan.host_cluster<benefit = 0>
  ]
} {

func.func @host_cluster_filtering(%arg0: tensor<i32>, %arg1: i32)
    -> (tensor<i32> {tensorrt.host_tensor}, tensor<i32> {tensorrt.host_tensora}) {
  %0 = plan.with_values %arg0 (%arg1) : tensor<i32>
  %1 = stablehlo.constant dense<1> : tensor<i32>
  %c1_i32 = arith.constant 1 : i32
  %2 = plan.with_values %1(%c1_i32) : tensor<i32>
  return %0, %2 : tensor<i32>, tensor<i32>
}

}

// CHECK-LABEL: func.func @host_cluster_filtering
//   CHECK-NOT:  plan.inline_group

// -----

builtin.module attributes {
  plan.cluster_kinds = [
    #plan.tensorrt_cluster<benefit = 1, disallow_shape_tensor_calculations=false>
  ]
} {

func.func @tensorrt_cluster_filtering(%arg0: tensor<?xf32>, %arg1: i32, %arg2: tensor<i32>)
    -> (tensor<?xf32>, tensor<i32> {tensorrt.host_tensor}) {
  %0 = plan.with_shape %arg0 (%arg1) : (tensor<?xf32>, i32) -> tensor<?xf32>
  %1 = plan.with_values %arg2 (%arg1) : tensor<i32>
  return %0, %1 : tensor<?xf32>, tensor<i32>
}

}

// CHECK-LABEL: func.func @tensorrt_cluster_filtering
//   CHECK-NOT:  plan.inline_group

// -----

// This test forces a split into "host" and "tensorrt" clusters
// where some integer tensor is used on both host and device. It
// verifies that the data flow lattice state is updated correctly
// for the results of the host clusters.

builtin.module attributes {
  plan.cluster_kinds = [
    #plan.tensorrt_cluster<benefit = 1, disallow_shape_tensor_calculations=true>,
    #plan.host_cluster<benefit = 0>
  ]
} {

func.func @test_data_flow_state_update(
    %arg0: tensor<10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xf32>)
    -> (tensor<1xi32>, tensor<1xf32>, tensor<1xf32>) {
  %cst_i32 = stablehlo.constant dense<1> : tensor<1xi32>
  %1 = stablehlo.add %cst_i32, %arg1  : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %offset = stablehlo.reshape %1  : (tensor<1xi32>) -> tensor<i32>
  %0 = "stablehlo.dynamic_slice"(%arg0, %offset) {
    slice_sizes = array<i64: 1>
  } : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
  %2 = stablehlo.add %0, %arg2  : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = stablehlo.convert %2  : (tensor<1xf32>) -> tensor<1xi32>
  %4 = stablehlo.reshape %3 : (tensor<1xi32>) -> tensor<i32>
  %5 = "stablehlo.dynamic_slice"(%arg0, %4) {
    slice_sizes = array<i64: 1>
  } : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
  return %1, %0, %5 : tensor<1xi32>, tensor<1xf32>, tensor<1xf32>
}

}

// CHECK-LABEL: func.func @test_data_flow_state_update
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<1xi32>, %[[arg2:.+]]:
//   CHECK-DAG:     %[[c:.+]] = stablehlo.constant dense<1> : tensor<1xi32>
//   CHECK-DAG:     %[[v0:.+]]:2 = plan.inline_group target(#plan.host_cluster
//   CHECK-DAG:       %[[v4:.+]] = stablehlo.add %[[c]], %[[arg1]]
//   CHECK-DAG:       %[[v5:.+]] = stablehlo.reshape %[[v4]]
//   CHECK-DAG:       yield %[[v4]], %[[v5]] :
//   CHECK-DAG:     %[[v1:.+]] = plan.inline_group target(#plan.tensorrt_cluster
//   CHECK-DAG:       %[[v4:.+]] = stablehlo.dynamic_slice %[[arg0]], %[[v0]]#1,
//   CHECK-DAG:       yield %[[v4]]
//   CHECK-DAG:     %[[v2:.+]] = plan.inline_group target(#plan.host_cluster
//   CHECK-DAG:       %[[v4:.+]] = stablehlo.add %[[v1]], %[[arg2]]
//   CHECK-DAG:       %[[v5:.+]] = stablehlo.convert %[[v4]]
//   CHECK-DAG:       %[[v6:.+]] = stablehlo.reshape %[[v5]]
//   CHECK-DAG:       yield %[[v6]] : tensor<i32>
//   CHECK-DAG:     %[[v3:.+]] = plan.inline_group target(#plan.tensorrt_cluster
//   CHECK-DAG:       %[[v4:.+]] = stablehlo.dynamic_slice %[[arg0]], %[[v2]],
//   CHECK-DAG:       yield %[[v4]] : tensor<1xf32>
//   CHECK-DAG:     return %[[v0]]#0, %[[v1]], %[[v3]] :
