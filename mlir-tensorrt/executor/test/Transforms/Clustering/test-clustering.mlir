// RUN: executor-opt %s -allow-unregistered-dialect -split-input-file \
// RUN:    -test-clustering="merge-independent-clusters disable-bfs-clustering horizontal-merge-arith-op-limit=99" \
// RUN: | FileCheck %s --check-prefix=ONLYM
// RUN: executor-opt %s -allow-unregistered-dialect -split-input-file -test-clustering | FileCheck %s
// RUN: executor-opt %s -allow-unregistered-dialect -split-input-file \
// RUN:  -test-clustering=merge-independent-clusters | FileCheck %s --check-prefix=MERGE
// RUN: executor-opt %s -allow-unregistered-dialect -split-input-file \
// RUN:  -test-clustering="bfs-root-traversal=post" | FileCheck %s --check-prefix=BFSPOST

func.func @test_cluster_simple(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %c1 = arith.constant dense<1.> : tensor<10xf32>
  %0 = arith.addf %arg0, %c1 : tensor<10xf32>
  %1 = "some_dialect.some_op"(%0) : (tensor<10xf32>) -> tensor<10xf32>
  %2 = arith.addf %1, %c1 : tensor<10xf32>
  %3 = arith.addf %1, %0 : tensor<10xf32>
  return %2, %3 : tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: @test_cluster_simple
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>)
//       CHECK:     %[[cst:.+]] = arith.constant
//       CHECK:     %[[v0:.+]] = call @cluster(%[[arg0]], %[[cst]])
//       CHECK:     %[[v1:.+]] = "some_dialect.some_op"(%[[v0]])
//       CHECK:     %[[v2:.+]] = call @cluster_0(%[[v1]], %[[cst]])
//       CHECK:     %[[v3:.+]] = call @cluster_1(%[[v1]], %[[v0]])
//       CHECK:     return %[[v2]], %[[v3]]
// CHECK-LABEL: private @cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>)
//       CHECK:     %[[v0:.+]] = arith.addf %[[arg0]], %[[arg1]]
//       CHECK:     return %[[v0]]
// CHECK-LABEL: private @cluster_0
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>)
//       CHECK:     %[[v0:.+]] = arith.addf %[[arg0]], %[[arg1]]
//       CHECK:     return %[[v0]]
// CHECK-LABEL: private @cluster_1
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>)
//       CHECK:     %[[v0:.+]] = arith.addf %[[arg0]], %[[arg1]]
//       CHECK:     return %[[v0]]

// MERGE-LABEL: @test_cluster_simple
//  MERGE-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>)
//       MERGE:     %[[cst:.+]] = arith.constant
//       MERGE:     %[[v0:.+]] = call @cluster(%[[arg0]], %[[cst]])
//       MERGE:     %[[v1:.+]] = "some_dialect.some_op"(%[[v0]])
//       MERGE:     %[[v2:.+]]:2 = call @cluster_0(%[[v1]], %[[cst]], %[[v0]])
//       MERGE:     return %[[v2]]#0, %[[v2]]#1
// MERGE-LABEL: private @cluster
//  MERGE-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>)
//       MERGE:     %[[v0:.+]] = arith.addf %[[arg0]], %[[arg1]]
//       MERGE:     return %[[v0]]
// MERGE-LABEL: private @cluster_0
//  MERGE-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>, %[[arg2:.+]]: tensor<10xf32>)
//       MERGE:     %[[v0:.+]] = arith.addf %[[arg0]], %[[arg1]]
//       MERGE:     %[[v1:.+]] = arith.addf %[[arg0]], %[[arg2]]
//       MERGE:     return %[[v0]], %[[v1]]

// -----

// Logic in the test pass says "if a constant is used across multiple clusters, pass by argument instead of cloning".

func.func @test_cluster_constant_handling_by_arg(%arg0: tensor<10x3xf32>, %arg1: tensor<10x3xf32>)
    -> (tensor<10x3xf32>, tensor<10x3xf32>) {
  %c1 = arith.constant dense<1.> : tensor<10x3xf32>
  %2 = arith.addf %arg0, %c1 : tensor<10x3xf32>
  %3 = arith.addf %arg1, %c1 : tensor<10x3xf32>
  return %2, %3 : tensor<10x3xf32>, tensor<10x3xf32>
}

// CHECK-LABEL: @test_cluster_constant_handling_by_arg
//       CHECK:     %[[cst:.+]] = arith.constant dense<1.000000e+00> : tensor<10x3xf32>
//       CHECK:     call @cluster(%{{.+}}, %[[cst]])
//       CHECK:     call @cluster_0(%{{.+}}, %[[cst]])


// -----


// The test pass contains logic to test the horizonatal merge callback option. We only allow unioning independent
// clusters if together they will form a cluster with at most four arithmetic ops.

func.func @test_horizontal_merge_callback(%arg0: tensor<10xf32>)
    -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>,
        tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) {
  %c1 = arith.constant dense<1.> : tensor<10xf32>
  %c2 = arith.constant dense<2.> : tensor<10xf32>
  %c3 = arith.constant dense<3.> : tensor<10xf32>
  %c4 = arith.constant dense<4.> : tensor<10xf32>
  %2 = arith.addf %arg0, %c1 : tensor<10xf32>
  %3 = arith.addf %arg0, %c2 : tensor<10xf32>
  %4 = arith.addf %arg0, %c3 : tensor<10xf32>
  %5 = arith.addf %arg0, %c4 : tensor<10xf32>
  %6 = arith.addf %arg0, %c1 : tensor<10xf32>
  %7 = arith.addf %arg0, %c2 : tensor<10xf32>
  %8 = arith.addf %arg0, %c3 : tensor<10xf32>
  %9 = arith.addf %arg0, %c4 : tensor<10xf32>
  return %2, %3, %4, %5, %6, %7, %8, %9
    : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>,
      tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
}

//    CHECK-LABEL: @test_horizontal_merge_callback

//    MERGE-LABEL: @test_horizontal_merge_callback
//          MERGE:   call @cluster(
//     MERGE-NEXT:   call @cluster_0(
//      MERGE-NOT:   call @cluster


// -----


// Test horizontal merge cycle detection edge case. In this example, there is no conflict
// since the insertion point of clusters is always at the root.

func.func @test_horizontal_merge_cycle_detection(%arg0: tensor<10xf32>)
    -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) {
  %c1 = arith.constant dense<1.> : tensor<10xf32>
  %c2 = arith.constant dense<2.> : tensor<10xf32>
  %c3 = arith.constant dense<3.> : tensor<10xf32>
  %c4 = arith.constant dense<4.> : tensor<10xf32>
  %0 = arith.addf %arg0, %c1 {__cluster_id__ = 0 : i64} : tensor<10xf32>
  %1 = arith.addf %arg0, %c2 {__cluster_id__ = 2 : i64} : tensor<10xf32>
  %2 = arith.addf %1, %c3 {__cluster_id__ = 1 : i64} : tensor<10xf32>
  %3 = arith.addf %0, %c4 {__cluster_id__ = 1 : i64} : tensor<10xf32>
  %4 = arith.addf %0, %c2 {__cluster_id__ = 2 : i64} : tensor<10xf32>
  %5 = arith.addf %1, %c1 {__cluster_id__ = 1 : i64} : tensor<10xf32>
  return %5, %2, %3, %4 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
}


// CHECK-LABEL: @test_horizontal_merge_cycle_detection
//       CHECK:     call @cluster(
//  CHECK-NEXT:     call @cluster_0(
//  CHECK-NEXT:     return

// MERGE-LABEL: @test_horizontal_merge_cycle_detection

//   ONLYM-LABEL: @test_horizontal_merge_cycle_detection
// ONLYM-COUNT-1: %{{.+}}:4 = call @cluster
//     ONLYM-NOT: call @cluster

// -----

func.func @bfs_root_traversal_order_sensitive_multiple_return(%arg0: tensor<10xf32>)
    -> (tensor<10xf32>, tensor<10xf32>) {
  %0 = arith.addf %arg0, %arg0 : tensor<10xf32>
  %1 = arith.addf %0, %0 : tensor<10xf32>
  %2 = arith.addf %0, %arg0 : tensor<10xf32>
  return %1, %2 : tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: @bfs_root_traversal_order_sensitive_multiple_return
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>)
//       CHECK:     %[[v0:.+]] = call @cluster(%[[arg0]]) : (tensor<10xf32>) -> tensor<10xf32>
//       CHECK:     %[[v1:.+]] = call @cluster_0(%[[v0]]) : (tensor<10xf32>) -> tensor<10xf32>
//       CHECK:     %[[v2:.+]] = call @cluster_1(%[[v0]], %[[arg0]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
//       CHECK:     return %[[v1]], %[[v2]]

// MERGE-LABEL: @bfs_root_traversal_order_sensitive_multiple_return
//  MERGE-NEXT:     call @cluster(
//  MERGE-NEXT:     return

// ONLYM-LABEL: @bfs_root_traversal_order_sensitive_multiple_return
//  ONLYM-NEXT:     call @cluster(
//  ONLYM-NEXT:     return

// -----

func.func @bfs_root_traversal_order_sensitive_diamond(%arg0: tensor<10xf32>)
    -> tensor<10xf32> {
  %0 = arith.addf %arg0, %arg0 : tensor<10xf32>
  %1 = arith.addf %0, %0 : tensor<10xf32>
  %2 = arith.addf %0, %arg0 : tensor<10xf32>
  %3 = arith.addf %1, %2 : tensor<10xf32>
  return %3 : tensor<10xf32>
}

// CHECK-LABEL: @bfs_root_traversal_order_sensitive_diamond
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>) -> tensor<10xf32> {
//       CHECK:     %[[v0:.+]] = call @cluster(%[[arg0]])
//       CHECK:     %[[v1:.+]] = call @cluster_0(%[[v0]], %[[arg0]])
//       CHECK:     return %[[v1]]

// BFSPOST-LABEL: @bfs_root_traversal_order_sensitive_diamond
//  BFSPOST-SAME: (%[[arg0:.+]]: tensor<10xf32>) -> tensor<10xf32> {
//       BFSPOST:     %[[v0:.+]] = call @cluster(%[[arg0]]) : (tensor<10xf32>) -> tensor<10xf32>
//       BFSPOST:     return %[[v0]] : tensor<10xf32>

// MERGE-LABEL: @bfs_root_traversal_order_sensitive_diamond
//  MERGE-SAME: (%[[arg0:.+]]: tensor<10xf32>) -> tensor<10xf32> {
//       MERGE:     %[[v0:.+]] = call @cluster(%[[arg0]]) : (tensor<10xf32>) -> tensor<10xf32>
//       MERGE:     return %[[v0]] : tensor<10xf32>

// -----

func.func @merge_independent_clusters_user_in_region(%arg0: tensor<10xf32>,
                                                     %arg1: tensor<10xf32>,
                                                     %arg2: i1, %arg3: tensor<10xf32>)
     -> (tensor<10xf32>, tensor<10xf32>) {
  // Here the cluster (%0, %1) has a user in the `scf.if` region. This should prevent
  // merging with the cluster (%3, %4) since the root of the resulting cluster (%4) will
  // not dominate the use in the `scf.if` region.
  %0 = arith.addf %arg0, %arg1 : tensor<10xf32>
  %1 = arith.mulf %0, %arg1 : tensor<10xf32>
  %2 = scf.if %arg2 -> tensor<10xf32> {
    scf.yield %1: tensor<10xf32>
  } else {
    scf.yield %arg3 : tensor<10xf32>
  }
  %3 = arith.addf %1, %1 : tensor<10xf32>
  %4 = arith.mulf %3, %arg3 : tensor<10xf32>
  return %3, %4 : tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: @merge_independent_clusters_user_in_region
//       CHECK:  call @cluster
//       CHECK:  scf.if
//       CHECK:  scf.yield
//       CHECK:  scf.yield
//       CHECK:  call @cluster_0
//       CHECK:  call @cluster_1
//       CHECK: return

// MERGE-LABEL: @merge_independent_clusters_user_in_region
//       MERGE:  call @cluster
//       MERGE:  scf.if
//       MERGE:  scf.yield
//       MERGE:  scf.yield
//       MERGE:  call @cluster_0
//       MERGE: return

// -----

func.func @nested_clusters(%arg0: tensor<10xf32>,
                           %arg1: tensor<10xf32>,
                           %arg2: i1, %arg3: tensor<10xf32>)
     -> tensor<10xf32> {
  %0 = arith.addf %arg0, %arg1 : tensor<10xf32>
  %1 = arith.mulf %0, %arg1 : tensor<10xf32>
  %6, %7 = scf.while (%arg4 = %1, %arg5 = %1) : (tensor<10xf32>, tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
    %3 = arith.addf %1, %arg3 : tensor<10xf32>
    %33 = arith.subf %1, %arg3 : tensor<10xf32>
    %4 = "some_op"(%33, %3) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %5 = arith.addf %4, %4 : tensor<10xf32>
    %6 = arith.subf %4, %4 : tensor<10xf32>
    %7 = "some_op"(%5, %6) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %8 = arith.addf %7, %7 : tensor<10xf32>
    %9 = arith.subf %7, %7 : tensor<10xf32>
    scf.condition (%arg2) %8, %9 : tensor<10xf32>, tensor<10xf32>
  } do {
  ^bb0(%arg5: tensor<10xf32>, %arg6: tensor<10xf32>):
    %3 = arith.mulf %arg5, %arg5 : tensor<10xf32>
    %33 = arith.subf %arg5, %arg5 : tensor<10xf32>
    %4 = "some_op"(%3, %33) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %5 = arith.mulf %4, %4 : tensor<10xf32>
    %6 = arith.divf %4, %4 : tensor<10xf32>
    %7 = "some_op"(%5, %6) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %8 = arith.mulf %7, %7 : tensor<10xf32>
    %9 = arith.divf %7, %7 : tensor<10xf32>
    scf.yield %8, %9 : tensor<10xf32>, tensor<10xf32>
  }
  return %6 : tensor<10xf32>
}

// This test should be specifically checked for ordering of the cluster
// names `@cluster_[digit]`. They should appear always in increasing order.
// This is how we can verify that the sort algorithm internal to the clustering utilities
// is working correctly.


// CHECK-LABEL: @nested_clusters
//       CHECK: call @cluster
//       CHECK: call @cluster_0
//       CHECK: call @cluster_1
//       CHECK: call @cluster_2
//       CHECK: call @cluster_3
//       CHECK: call @cluster_4
//       CHECK: call @cluster_5
//       CHECK: call @cluster_6
//       CHECK: call @cluster_7
//       CHECK: call @cluster_8
//       CHECK: call @cluster_9
//       CHECK: call @cluster_10
//       CHECK: call @cluster_11
//       CHECK: return

// MERGE-LABEL: @nested_clusters
//       MERGE: call @cluster
//       MERGE: call @cluster_0
//       MERGE: call @cluster_1
//       MERGE: call @cluster_2
//       MERGE: call @cluster_3
//       MERGE: call @cluster_4
//       MERGE: call @cluster_5
//       MERGE: return

