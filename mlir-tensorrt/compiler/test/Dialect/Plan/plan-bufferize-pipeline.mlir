// This set of tests is meant to capture the effects of our end-to-end host module bufferization,
// buffer optimization, and deallocation pipeline.

// RUN: mlir-tensorrt-opt %s -split-input-file -plan-bufferize-pipeline | FileCheck %s

func.func @from_elements_staging_buffer(%arg0: f32, %arg1: f32) -> tensor<2xf32> {
  %0 = tensor.from_elements %arg0, %arg1 : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @from_elements_staging_buffer
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32, %[[arg2:.+]]: memref<2xf32, #plan.memory_space<device>> {plan.result_arg}) {
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[alloc:.+]] = memref.alloc() {alignment = 16 : i64} : memref<2xf32, #plan.memory_space<host_pinned>>
//   CHECK-DAG:     memref.store %[[arg0]], %[[alloc]][%[[c0]]] : memref<2xf32, #plan.memory_space<host_pinned>>
//   CHECK-DAG:     memref.store %[[arg1]], %[[alloc]][%[[c1]]] : memref<2xf32, #plan.memory_space<host_pinned>>
//       CHECK:     memref.copy %[[alloc]], %[[arg2]] : memref<2xf32, #plan.memory_space<host_pinned>> to memref<2xf32, #plan.memory_space<device>>
//  CHECK-NEXT:     memref.dealloc %[[alloc]] : memref<2xf32, #plan.memory_space<host_pinned>>
//  CHECK-NEXT:     return

// -----

func.func @small_host_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<?x?x?x?xf32>) {
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
  %1 = tensor.reshape %arg0 (%0) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// There should be a copy since `%arg0` is not writable under our default settings.
// We can only avoid a copy if we use `force-entrypoints-return-allocs`.

//       CHECK: memref.global "private" constant
// CHECK-LABEL: func.func @small_host_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x?xf32, #plan.memory_space<device>>, %[[arg1:.+]]: memref<?x?x?x?xf32, #plan.memory_space<device>> {plan.result_arg}) {
//       CHECK:     %[[v0:.+]] = memref.get_global
//       CHECK:     %[[alloc:.+]] = memref.alloc() {{.*}} : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     memref.copy %[[v0]], %[[alloc]]
//       CHECK:     %[[reshape:.+]] = memref.reshape %[[arg0]](%[[alloc]])
//       CHECK:     memref.copy %[[reshape]], %[[arg1]]
//       CHECK:     memref.dealloc %[[alloc]]


// -----

func.func @small_host_and_device_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<?x?x?x?xf32>, tensor<4xindex>) {
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
  %1 = tensor.reshape %arg0 (%0) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  return %1, %0 : tensor<?x?x?x?xf32>, tensor<4xindex>
}

//       CHECK:   memref.global "private" constant @__constant_4xindex
// CHECK-LABEL: func.func @small_host_and_device_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x?xf32, #plan.memory_space<device>>, %[[arg1:.+]]: memref<?x?x?x?xf32, #plan.memory_space<device>> {plan.result_arg}, %[[arg2:.+]]: memref<4xindex, #plan.memory_space<device>> {plan.result_arg})
//       CHECK:     %[[v0:.+]] = memref.get_global {{.*}} #plan.memory_space<device>>
//       CHECK:     memref.copy %[[v0]], %[[arg2]] :
//       CHECK:     %[[alloc:.+]] = memref.alloc() {{.*}} #plan.memory_space<host>
//       CHECK:     memref.copy %[[arg2]], %[[alloc]]
//       CHECK:     %[[reshape:.+]] = memref.reshape %[[arg0]](%[[alloc]])
//       CHECK:     memref.copy %[[reshape]], %[[arg1]]
//       CHECK:     memref.dealloc %[[alloc]]
//       CHECK:     return

// -----

module @while_loop {

func.func private @cond() -> i1

// The test case illustrates a while loop that for whatever reason may not
// have been "detensorized" earlier in the pipeline. The TensorKindAnalysis
// will show that all tensors are "host-only", but currently bufferization
// does not deduce this via its memory space inference logic. Therefore, the
// loop will be bufferized so that the buffers are in the device
// space at branch points, which means lots of copies are inserted. Before
// adding the 'plan-assign-memory-spaces' pass, we would get a failure here
// due to mixed types of init arg and yielded value inferred by bufferization.
// In the future, we can optimize this case by adding support for rewriting
// the encoding attribute of loop-carried tensors to be host for this case.

func.func @while_loop_host_tensor_carried(%arg0: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %1 = tensor.from_elements %arg0  : tensor<1xf32>
  %2 = scf.while (%arg1 = %1) : (tensor<1xf32>) -> tensor<1xf32> {
    %cond = func.call @cond() : () -> i1
    %e = tensor.extract %arg1[%c0] : tensor<1xf32>
    %f = arith.addf %e, %e : f32
    %3 = tensor.from_elements %f : tensor<1xf32>
    scf.condition(%cond) %3 : tensor<1xf32>
  } do {
  ^bb0(%arg1: tensor<1xf32>):
    %extract = tensor.extract %arg1[%c0] : tensor<1xf32>
    %3 = arith.addf %extract, %extract : f32
    %4 = tensor.from_elements %3 : tensor<1xf32>
    scf.yield %4 : tensor<1xf32>
  }
  %3 = tensor.extract %2[%c0] : tensor<1xf32>
  return %3 : f32
}

}

// CHECK-LABEL: func.func @while_loop_host_tensor_carried
//         CHECK:     scf.while : () -> ()
// CHECK-COUNT-1:       memref.copy
//         CHECK:       scf.condition
// CHECK-COUNT-1:       memref.copy
//         CHECK:       scf.yield
//     CHECK-NOT:     memref.copy

// -----

// This test checks that if we create a function with specific constraints,
// then we should not insert unnecessary copies to tranfer between other spaces.

module @shape_func_with_constraints {

func.func @shape_func_with_constraints(
      %arg0: tensor<2xindex, #plan.memory_space<host>>,
      %arg1: tensor<2xindex, #plan.memory_space<host>>)
        -> tensor<2xindex, #plan.memory_space<host>> attributes {
            plan.memory_space = #plan.memory_space<host>
    } {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %extracted = tensor.extract %arg0[%c0] : tensor<2xindex, #plan.memory_space<host>>
  %extracted_0 = tensor.extract %arg1[%c0] : tensor<2xindex, #plan.memory_space<host>>
  %0 = arith.index_cast %extracted : index to i32
  %1 = arith.index_cast %extracted_0 : index to i32
  %2 = arith.maxsi %0, %1 : i32
  %3 = arith.index_cast %2 : i32 to index
  %from_elements = tensor.from_elements %3, %c2 : tensor<2xindex, #plan.memory_space<host>>
  return %from_elements : tensor<2xindex, #plan.memory_space<host>>
}

}

// CHECK-LABEL: func.func @shape_func_with_constraints
//  CHECK-SAME: (%[[arg0:.+]]: memref<2xindex, #plan.memory_space<host>>, %[[arg1:.+]]: memref<2xindex, #plan.memory_space<host>>, %[[arg2:.+]]: memref<2xindex, #plan.memory_space<host>> {plan.result_arg}) attributes {plan.memory_space = #plan.memory_space<host>} {
//    CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//    CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//    CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//    CHECK-DAG:     %[[v0:.+]] = memref.load %[[arg0]][%[[c0]]]
//    CHECK-DAG:     %[[v1:.+]] = memref.load %[[arg1]][%[[c0]]]
//    CHECK-DAG:     %[[v2:.+]] = arith.index_cast %[[v0]] : index to i32
//    CHECK-DAG:     %[[v3:.+]] = arith.index_cast %[[v1]] : index to i32
//    CHECK-DAG:     %[[v4:.+]] = arith.maxsi %[[v2]], %[[v3]] : i32
//    CHECK-DAG:     %[[v5:.+]] = arith.index_cast %[[v4]] : i32 to index
//    CHECK-DAG:     %[[alloc:.+]] = memref.alloc()
//    CHECK-DAG:     memref.store %[[v5]], %[[alloc]][%[[c0]]]
//    CHECK-DAG:     memref.store %[[c2]], %[[alloc]][%[[c1]]]
//    CHECK-DAG:     memref.copy %[[alloc]], %[[arg2]]
//    CHECK-DAG:     memref.dealloc %[[alloc]]
//        CHECK:     return

// -----

// This test checks that we don't produce incorrect IR when using
// `bufferization.alloc_tensor` to allocate a tensor in a different space.
// Currently `bufferization.alloc_tensor` also requires a cast on the result.
// TODO: remove when upstream is fixed.

func.func @test_alloc_tensor_copy_to_space(%arg0: tensor<2xindex, #plan.memory_space<device>>)
                                           -> tensor<2xindex, #plan.memory_space<host>> {
  %0 = bufferization.alloc_tensor () copy (%arg0) {
    memory_space = #plan.memory_space<host>
  } : tensor<2xindex, #plan.memory_space<device>>
  %1 = tensor.cast %0
    : tensor<2xindex, #plan.memory_space<device>> to tensor<2xindex, #plan.memory_space<host>>
  return %1 : tensor<2xindex, #plan.memory_space<host>>
}

// CHECK-LABEL: func.func @test_alloc_tensor_copy_to_space
//  CHECK-SAME: (%[[arg0:.+]]: memref<2xindex, #plan.memory_space<device>>,
//  CHECK-SAME:  %[[arg1:.+]]: memref<2xindex, #plan.memory_space<host>> {plan.result_arg})
//       CHECK:     memref.copy %[[arg0]], %[[arg1]]
//  CHECK-NEXT:     return

// -----

// This test checks that we produce bufferized IR where:
// - The `scf.for` loops don't have unnecessary transfers inside the
//   loop bodies.
// - We produce correct result when "bufferization.alloc_tensor" is used
//   at the frontend and the memory space constraint is specified using
//   the `memory_space` attribute.

func.func @fill_buffers_using_for_loops() -> (tensor<2x128xf32>, tensor<128xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index

  // Create input tensor in host space and fill it with values.
  %0 = bufferization.alloc_tensor() {
    memory_space = #plan.memory_space<host>
  } : tensor<2x128xf32>

  // Create the second tensor in device space, but it is also
  // filled with values in a loop. Optimization should make this
  // a host tensor allocation.
  %01 = bufferization.alloc_tensor() {
    memory_space = #plan.memory_space<device>
  } : tensor<128xf32>

  %lhs_host = scf.for %i = %c0 to %c256 step %c1 iter_args(%iter = %0) -> (tensor<2x128xf32>) {
    %coords:2 = affine.delinearize_index %i into(%c2, %c128) : index, index
    %v = arith.index_cast %i : index to i32
    %vf = arith.sitofp %v : i32 to f32
    %y = tensor.insert %vf into %iter[%coords#0, %coords#1] : tensor<2x128xf32>
    scf.yield %y : tensor<2x128xf32>
  }
  %rhs_host = scf.for %i = %c0 to %c128 step %c1 iter_args(%iter = %01) -> (tensor<128xf32>) {
    %coords:2 = affine.delinearize_index %i into(%c2, %c128) : index, index
    %v = arith.index_cast %i : index to i32
    %vf = arith.sitofp %v : i32 to f32
    %y = tensor.insert %vf into %iter[%i] : tensor<128xf32>
    scf.yield %y : tensor<128xf32>
  }

  // Clone input host tensor into device memory space.
  %lhs = bufferization.alloc_tensor() copy(%lhs_host) {
    memory_space = #plan.memory_space<device>
  } : tensor<2x128xf32>
  %rhs = bufferization.alloc_tensor() copy(%rhs_host) {
    memory_space = #plan.memory_space<device>
  } : tensor<128xf32>

  return %lhs, %rhs : tensor<2x128xf32>, tensor<128xf32>
}

// CHECK-LABEL: func.func @fill_buffers_using_for_loops
//  CHECK-SAME: (%[[arg0:.+]]: memref<2x128xf32, #plan.memory_space<device>> {plan.result_arg}, %[[arg1:.+]]: memref<128xf32, #plan.memory_space<device>> {plan.result_arg})
//   CHECK-DAG:     %[[alloc:.+]] = memref.alloc() {{.*}} #plan.memory_space<host>>
//       CHECK:     scf.for %[[arg2:.+]] =
//   CHECK-DAG:       %[[v0]]:2 = affine.delinearize_index %[[arg2]] into (2, 128) : index, index
//   CHECK-DAG:       %[[v1:.+]] = arith.index_cast %[[arg2]] : index to i32
//   CHECK-DAG:       %[[v2:.+]] = arith.sitofp %[[v1]] : i32 to f32
//   CHECK-DAG:       memref.store %[[v2]], %[[alloc]][%[[v0]]#0, %[[v0]]#1]
//       CHECK:     }
//       CHECK:     memref.copy %[[alloc]], %[[arg0]] : memref<2x128xf32, #plan.memory_space<host>> to memref<2x128xf32, #plan.memory_space<device>>
//       CHECK:     %[[alloc_0:.+]] = memref.alloc() {{.*}} #plan.memory_space<host>>
//       CHECK:     scf.for %[[arg2:.+]] =
//   CHECK-DAG:       %[[v0:.+]] = arith.index_cast %[[arg2]] : index to i32
//   CHECK-DAG:       %[[v1:.+]] = arith.sitofp %[[v0]] : i32 to f32
//   CHECK-DAG:       memref.store %[[v1]], %[[alloc_0]][%[[arg2]]] : memref<128xf32, #plan.memory_space<host>>
//       CHECK:     }
//       CHECK:     memref.copy %[[alloc_0]], %[[arg1]] : memref<128xf32, #plan.memory_space<host>> to memref<128xf32, #plan.memory_space<device>>
//       CHECK:     memref.dealloc %[[alloc]] : memref<2x128xf32, #plan.memory_space<host>>
//       CHECK:     memref.dealloc %[[alloc_0]] : memref<128xf32, #plan.memory_space<host>>
//       CHECK:     return
