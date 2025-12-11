// RUN: mlir-tensorrt-opt %s -split-input-file -mtrt-scf-detensorize -allow-unregistered-dialect | FileCheck %s

func.func @detensorize_while(%arg0: tensor<i32>, %arg1: tensor<1xi32>)
    -> (tensor<i32> {tensorrt.host_tensor}, tensor<1xi32> {tensorrt.host_tensor}) {
  %c0 = arith.constant 0 : index
  %0, %1 = scf.while(%arg2 = %arg0, %arg3 = %arg1) : (tensor<i32>, tensor<1xi32>)
      -> (tensor<i32>, tensor<1xi32>) {
    %a = tensor.extract %arg2[] : tensor<i32>
    %b = tensor.extract %arg3[%c0] : tensor<1xi32>
    %c = arith.addi %a, %b : i32
    %d = arith.trunci %c : i32 to i1
    scf.condition (%d) %arg2, %arg3 : tensor<i32>, tensor<1xi32>
  } do {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<1xi32>):
    %a = tensor.extract %arg2[] : tensor<i32>
    %b = tensor.extract %arg3[%c0] : tensor<1xi32>
    %c = arith.addi %a, %b : i32
    %d = tensor.from_elements %c : tensor<i32>
    %e = tensor.from_elements %c : tensor<1xi32>
    scf.yield %d, %e : tensor<i32>, tensor<1xi32>
  }
  return %0, %1 : tensor<i32>, tensor<1xi32>
}

// CHECK-LABEL: @detensorize_while
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i32>, %[[arg1:.+]]: tensor<1xi32>)
//  CHECK-NEXT:     %[[c0:.+]] = arith.constant 0 : index
//  CHECK-NEXT:     %[[extracted:.+]] = tensor.extract %[[arg0]][]
//  CHECK-NEXT:     %[[extracted_0:.+]] = tensor.extract %[[arg1]][%[[c0]]]
//  CHECK-NEXT:     %[[v0:.+]]:2 = scf.while (%[[arg2:.+]] = %[[extracted]], %[[arg3:.+]] = %[[extracted_0]]) : (i32, i32) -> (i32, i32)
//  CHECK-NEXT:       %[[v1:.+]] = arith.addi %[[arg2]], %[[arg3]] : i32
//  CHECK-NEXT:       %[[v2:.+]] = arith.trunci %[[v1]] : i32 to i1
//  CHECK-NEXT:       scf.condition(%[[v2]]) %[[arg2]], %[[arg3]] : i32, i32
//  CHECK-NEXT:     } do {
//  CHECK-NEXT:     ^bb0(%[[arg2:.+]]: i32, %[[arg3:.+]]: i32):
//  CHECK-NEXT:       %[[v1:.+]] = arith.addi %[[arg2]], %[[arg3]] : i32
//  CHECK-NEXT:       scf.yield %[[v1]], %[[v1]] : i32, i32
//  CHECK-DAG:     %[[from_elements:.+]] = tensor.from_elements %[[v0]]#0 : tensor<i32>
//   CHECK-DAG:     %[[from_elements_1:.+]] = tensor.from_elements %[[v0]]#1 : tensor<1xi32>
//  CHECK-NEXT:     return %[[from_elements]], %[[from_elements_1]] : tensor<i32>, tensor<1xi32>

// -----

func.func @detensorize_while_negative(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %c0 = arith.constant 0 : index
  %0 = scf.while(%arg2 = %arg0) : (tensor<1xi32>) -> tensor<1xi32> {
    %1 = tensor.extract %arg2[%c0] : tensor<1xi32>
    %2 = arith.trunci %1 : i32 to i1
    scf.condition (%2) %arg2 : tensor<1xi32>
  } do {
  ^bb0(%arg2: tensor<1xi32>):
    %2 = arith.addi %arg2, %arg2 : tensor<1xi32>
    scf.yield %2 : tensor<1xi32>
  }
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: @detensorize_while_negative
//       CHECK:     scf.while {{.+}} : (tensor<1xi32>) -> tensor<1xi32>

// -----

// In this test, we perform a non insert/extract operation on the value in the
// "after" region, but by marking the func result as being on the host,
// we ensure that the block argument is recognized as a host tensor.

func.func @detensorize_while_analysis(%arg0: tensor<1xi32>) -> (tensor<1xi32> {tensorrt.host_tensor}) {
  %c0 = arith.constant 0 : index
  %0 = scf.while(%arg2 = %arg0) : (tensor<1xi32>) -> tensor<1xi32> {
    %1 = tensor.extract %arg2[%c0] : tensor<1xi32>
    %2 = arith.trunci %1 : i32 to i1
    scf.condition (%2) %arg2 : tensor<1xi32>
  } do {
  ^bb0(%arg2: tensor<1xi32>):
    %2 = arith.addi %arg2, %arg2 : tensor<1xi32>
    scf.yield %2 : tensor<1xi32>
  }
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: func.func @detensorize_while_analysis
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>)
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<1xi32>
//       CHECK:     %[[v0:.+]] = scf.while (%[[arg1]] = %[[extracted]]) : (i32) -> i32 {
//  CHECK-NEXT:       %[[v1:.+]] = arith.trunci %[[arg1]] : i32 to i1
//  CHECK-NEXT:       scf.condition(%[[v1]]) %[[arg1]] : i32
//  CHECK-NEXT:     } do {
//  CHECK-NEXT:     ^bb0(%[[arg1:.+]]: i32):
//  CHECK-NEXT:       %[[from_elements_0:.+]] = tensor.from_elements %[[arg1]] : tensor<1xi32>
//  CHECK-NEXT:       %[[v1:.+]] = arith.addi %[[from_elements_0]], %[[from_elements_0]] : tensor<1xi32>
//  CHECK-NEXT:       %[[extracted_1:.+]] = tensor.extract %[[v1]][%[[c0]]] : tensor<1xi32>
//  CHECK-NEXT:       scf.yield %[[extracted_1]] : i32
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v0]] : tensor<1xi32>
//       CHECK:     return %[[from_elements]] : tensor<1xi32>

// -----

func.func @detensorize_while_mixed(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<2xi32>) {
  %c0 = arith.constant 0 : index
  %0:2 = scf.while(%arg2 = %arg0, %arg3 = %arg1) : (tensor<1xi32>, tensor<2xi32>) -> (tensor<1xi32>, tensor<2xi32>) {
    %1 = tensor.extract %arg2[%c0] : tensor<1xi32>
    %2 = arith.trunci %1 : i32 to i1
    scf.condition (%2) %arg2, %arg3 : tensor<1xi32>, tensor<2xi32>
  } do {
  ^bb0(%arg2: tensor<1xi32>, %arg3: tensor<2xi32>):
    %3 = "test.foo"(%arg2) : (tensor<1xi32>) -> (tensor<1xi32>)
    %4 = "test.foo"(%arg3) : (tensor<2xi32>) -> (tensor<2xi32>)
    scf.yield %3, %4 : tensor<1xi32>, tensor<2xi32>
  }
  return %0#0, %0#1 : tensor<1xi32>, tensor<2xi32>
}

// CHECK-LABEL: @detensorize_while_mixed
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>, %[[arg1:.+]]: tensor<2xi32>)
//       CHECK:     scf.while {{.+}} : (i32, tensor<2xi32>) -> (i32, tensor<2xi32>)

// -----

// CHECK-LABEL: func @hoist_matching_extract_insert(
//  CHECK-SAME:     %[[arg:.*]]: tensor<?xf32>
func.func @hoist_matching_extract_insert(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : index
  %add = arith.addi %c0, %c1 : index
  %sub = arith.subi %add, %c1 : index

  // CHECK-DAG: %[[c9:.+]] = arith.constant 9 : index
  // CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index

  // CHECK: %[[extract:.*]] = tensor.extract %[[arg]][%[[c0]]]
  // CHECK: %[[for:.*]] = scf.for {{.*}} iter_args(%[[hoisted:.*]] = %[[extract]])
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {


    %standalone = tensor.extract %t[%c9] : tensor<?xf32>
    "test.foo"(%standalone) : (f32) -> ()

    %1 = tensor.extract %t[%c0] : tensor<?xf32>
    // CHECK: %[[foo:.*]] = "test.foo"(%[[hoisted]])
    %2 = "test.foo"(%1) : (f32) -> (f32)
    // Obfuscate the IR by inserting at offset %sub instead of 0; both of them
    // have the same value.
    %3 = tensor.insert %2 into %t[%sub] : tensor<?xf32>
    // CHECK: scf.yield %[[foo]]
    scf.yield %3 : tensor<?xf32>
  }
  // CHECK: %[[insert:.*]] = tensor.insert %[[for]] into %[[arg]][%[[c0]]]

  // CHECK: return %[[insert]]
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @subset_of_subset
// CHECK-SAME:     (%[[arg:.*]]: tensor
func.func @subset_of_subset(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)
  %c1 = arith.constant 1 : index

  // CHECK: %[[extract1:.*]] = tensor.extract_slice %[[arg]]
  // CHECK: %[[extract2:.*]] = tensor.extract_slice %[[extract1]]
  // CHECK: %[[extract3:.+]] = tensor.extract %[[extract2]]
  // CHECK: %[[for:.*]] = scf.for {{.*}} iter_args(%[[hoisted2:.*]] = %[[extract3]])
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    %extract1 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
    %extract2 = tensor.extract_slice %extract1[1][2][1] : tensor<5xf32> to tensor<2xf32>
    %extract3 = tensor.extract %extract2[%c1] : tensor<2xf32>

    // CHECK: %[[foo:.*]] = "test.foo"(%[[hoisted2]])
    %2 = "test.foo"(%extract3) : (f32) -> (f32)

    %insert0 = tensor.insert %2 into %extract2[%c1] : tensor<2xf32>
    %insert1 = tensor.insert_slice %insert0 into %extract1[1][2][1] : tensor<2xf32> into tensor<5xf32>
    %insert2 = tensor.insert_slice %insert1 into %t[0][5][1] : tensor<5xf32> into tensor<?xf32>

    // CHECK: scf.yield %[[foo]]
    scf.yield %insert2 : tensor<?xf32>
  }
  // CHECK: %[[inserted:.+]] = tensor.insert %[[for]] into %[[extract2]]
  // CHECK: %[[insert2:.*]] = tensor.insert_slice %[[inserted]] into %[[extract1]][1] [2] [1]
  // CHECK: %[[insert1:.*]] = tensor.insert_slice %[[insert2]] into %[[arg]]

  // CHECK: return %[[insert1]]
  return %0 : tensor<?xf32>
}
