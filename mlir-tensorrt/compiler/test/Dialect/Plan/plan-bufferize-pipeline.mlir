// This set of tests is meant to capture the effects of our end-to-end host module bufferization,
// buffer optimization, and deallocation pipeline.

// RUN: mlir-tensorrt-opt %s -split-input-file -plan-bufferize-pipeline | FileCheck %s

// RUN: mlir-tensorrt-opt %s -split-input-file -plan-bufferize-pipeline="force-entrypoints-return-allocs=true" \
// RUN: | FileCheck %s --check-prefix=ALLOC

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

// ALLOC-LABEL: func.func @from_elements_staging_buffer


// -----

func.func @small_host_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<?x?x?x?xf32>) {
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
  %1 = tensor.reshape %arg0 (%0) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func.func @small_host_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x?xf32, #plan.memory_space<device>>, %[[arg1:.+]]: memref<?x?x?x?xf32, #plan.memory_space<device>> {plan.result_arg}) {
//       CHECK:     %[[global:.+]] = memref.get_global {{.*}} : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     %[[reshape:.+]] = memref.reshape %[[arg0]](%[[global]])
//       CHECK:     memref.copy %[[reshape]], %[[arg1]]


// ALLOC-LABEL: func.func @small_host_tensor_constant
//  ALLOC-SAME: (%[[arg0:.+]]: memref<?x?xf32, #plan.memory_space<device>>) -> memref<?x?x?x?xf32, #plan.memory_space<device>> {
//   ALLOC-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   ALLOC-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   ALLOC-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   ALLOC-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   ALLOC-DAG:     %[[v0:.+]] = memref.get_global @{{.*}} : memref<4xindex, #plan.memory_space<host>>
//   ALLOC-DAG:     %[[reshape:.+]] = memref.reshape %[[arg0]](%[[v0]]) :
//   ALLOC-DAG:     %[[v1:.+]] = memref.load %[[v0]][%[[c3]]] : memref<4xindex, #plan.memory_space<host>>
//   ALLOC-DAG:     %[[v2:.+]] = memref.load %[[v0]][%[[c2]]] : memref<4xindex, #plan.memory_space<host>>
//   ALLOC-DAG:     %[[v3:.+]] = memref.load %[[v0]][%[[c1]]] : memref<4xindex, #plan.memory_space<host>>
//   ALLOC-DAG:     %[[v4:.+]] = memref.load %[[v0]][%[[c0]]] : memref<4xindex, #plan.memory_space<host>>
//       ALLOC:     %[[alloc:.+]] = memref.alloc(%[[v4]], %[[v3]], %[[v2]], %[[v1]])
//       ALLOC:     memref.copy %[[reshape]], %[[alloc]] :
//  ALLOC-NEXT:     return %[[alloc]] : memref<?x?x?x?xf32, #plan.memory_space<device>>

// -----

func.func @small_host_and_device_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<?x?x?x?xf32>, tensor<4xindex>) {
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
  %1 = tensor.reshape %arg0 (%0) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  return %1, %0 : tensor<?x?x?x?xf32>, tensor<4xindex>
}

// CHECK-LABEL: func.func @small_host_and_device_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x?xf32, #plan.memory_space<device>>, %[[arg1:.+]]: memref<?x?x?x?xf32, #plan.memory_space<device>> {plan.result_arg}, %[[arg2:.+]]: memref<4xindex, #plan.memory_space<device>> {plan.result_arg})
//   CHECK-DAG:     %[[global_device:.+]] = memref.get_global {{.*}} #plan.memory_space<device>>
//   CHECK-DAG:     %[[global_host:.+]] = memref.get_global {{.*}} #plan.memory_space<host>>
//       CHECK:     memref.copy %[[global_device]], %[[arg2]] :
//       CHECK:     %[[reshape:.+]] = memref.reshape %[[arg0]](%[[global_host]])
//       CHECK:     memref.copy %[[reshape]], %[[arg1]]
//       CHECK:     return

// -----

// External user is assumed to require read/write access by default.
func.func private @ext_user(%arg0: tensor<1024xindex> {plan.memory_space = #plan.memory_space<host>})

func.func @large_constant() {
  %0 = arith.constant dense<1> : tensor<1024xindex>
  call @ext_user(%0) : (tensor<1024xindex>) -> ()
  return
}

//       CHECK: memref.global {{.*}} : memref<1024xindex, #plan.memory_space<host>>
// CHECK-LABEL: func.func @large_constant
//   CHECK-DAG:   %[[v0:.+]] = memref.get_global @{{.*}} : memref<1024xindex, #plan.memory_space<host>>
//   CHECK-DAG:   %[[alloc:.+]] = memref.alloc() {{.*}} : memref<1024xindex, #plan.memory_space<host>>
//       CHECK:   memref.copy %[[v0]], %[[alloc]] :
//       CHECK:   call @ext_user(%[[alloc]]) : (memref<1024xindex, #plan.memory_space<host>>) -> ()
//       CHECK:   memref.dealloc %[[alloc]] : memref<1024xindex, #plan.memory_space<host>>

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

module @while_loop {

func.func private @cond() -> i1

// This test checks that we don't bufferize a while loop such that the
// arguments have different memory spaces. That is catastrophic to the
// performance since current bufferization doesn't handle this efficiently.

func.func @while_loop_device_tensor_from_elements(%arg0: f32) -> tensor<1xf32> {
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
  return %2 : tensor<1xf32>
}

}

// CHECK-LABEL: func.func @while_loop_device_tensor_from_elements
//         CHECK:     scf.while : () -> ()
// CHECK-COUNT-1:       memref.copy
//         CHECK:       memref.load
//         CHECK:       memref.store
//         CHECK:       memref.copy
//     CHECK-NOT:       memref.copy
//         CHECK:       scf.condition
//         CHECK:       memref.copy
//         CHECK:       memref.load
//         CHECK:       memref.store
//         CHECK:       memref.copy
//     CHECK-NOT:       memref.copy
//         CHECK:       scf.yield
//     CHECK-NOT:     memref.copy
//         CHECK:     return

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

// -----

// CHECK-LABEL: module @calls_no_inline
module @calls_no_inline {
  // CHECK-LABEL: func.func private @check_eq
  // CHECK-SAME: (%[[arg0:.+]]: memref<5xi8, #plan.memory_space<host>>, %[[arg1:.+]]: memref<5xi8, #plan.memory_space<host>>)
  func.func private @check_eq(%arg0: tensor<5xi8>, %arg1: tensor<5xi8>)
      attributes {no_inline, plan.memory_space = #plan.memory_space<host>} {
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK: %[[alloc:.+]] = memref.alloc() {{.*}} #plan.memory_space<host>
    // CHECK: scf.for
    // CHECK-NEXT: memref.load %[[arg0]]
    // CHECK-NEXT: memref.load %[[arg1]]
    // CHECK-NEXT: arith.cmpi
    // CHECK-NEXT: memref.store {{.*}}, %[[alloc]]
    %0 = tensor.empty() : tensor<5xi1>
    %1 = scf.for %arg2 = %c0 to %c5 step %c1 iter_args(%arg3 = %0) -> (tensor<5xi1>) {
      %extracted_0 = tensor.extract %arg0[%arg2] : tensor<5xi8>
      %extracted_1 = tensor.extract %arg1[%arg2] : tensor<5xi8>
      %5 = arith.cmpi eq, %extracted_0, %extracted_1 : i8
      %inserted = tensor.insert %5 into %arg3[%arg2] : tensor<5xi1>
      scf.yield %inserted : tensor<5xi1>
    }
    // CHECK: %[[alloc_0:.+]] = memref.alloc() {{.*}} #plan.memory_space<host>
    %2 = tensor.empty() : tensor<i1>
    // CHECK-NEXT: linalg.fill
    %3 = linalg.fill ins(%true : i1) outs(%2 : tensor<i1>) -> tensor<i1>
    // CHECK-NEXT: scf.for
    %4 = scf.for %arg2 = %c0 to %c5 step %c1 iter_args(%arg3 = %3) -> (tensor<i1>) {
      // CHECK-NEXT: memref.load
      // CHECK-NEXT: memref.load
      // CHECK-NEXT: arith.andi
      // CHECK-NEXT: memref.store
      %extracted_0 = tensor.extract %1[%arg2] : tensor<5xi1>
      %extracted_1 = tensor.extract %arg3[] : tensor<i1>
      %5 = arith.andi %extracted_1, %extracted_0 : i1
      %inserted = tensor.insert %5 into %arg3[] : tensor<i1>
      scf.yield %inserted : tensor<i1>
    }
    // CHECK: memref.load %[[alloc_0]]
    // CHECK-NEXT: cf.assert
    %extracted = tensor.extract %4[] : tensor<i1>
    // CHECK-DAG: memref.dealloc %[[alloc_0]]
    // CHECK-DAG: memref.dealloc %[[alloc]]
    cf.assert %extracted, "check_eq failed"
    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func private @compute
  // CHECK-SAME: (%[[arg0:.+]]: memref<5xi8, #plan.memory_space<host>>, %[[arg1:.+]]: memref<5xi8, #plan.memory_space<host>>,
  // CHECK-SAME:  %[[arg2:.+]]: memref<5xi8, #plan.memory_space<host>> {plan.result_arg})
  func.func private @compute(%arg0: tensor<5xi8>, %arg1: tensor<5xi8>) -> tensor<5xi8>
     attributes {no_inline, plan.memory_space = #plan.memory_space<host>} {
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<5xi8>
    // CHECK: scf.for
    %1 = scf.for %arg2 = %c0 to %c5 step %c1 iter_args(%arg3 = %0) -> (tensor<5xi8>) {
      // CHECK-NEXT: memref.load %[[arg0]]
      // CHECK-NEXT: memref.load %[[arg1]]
      %extracted = tensor.extract %arg0[%arg2] : tensor<5xi8>
      %extracted_0 = tensor.extract %arg1[%arg2] : tensor<5xi8>
      // CHECK-NEXT: arith.addi
      %2 = arith.addi %extracted, %extracted_0 : i8
      // CHECK-NEXT: memref.store {{.*}}, %[[arg2]]
      %inserted = tensor.insert %2 into %arg3[%arg2] : tensor<5xi8>
      scf.yield %inserted : tensor<5xi8>
    }
    // CHECK-NOT: memref.copy
    // CHECK: return
    return %1 : tensor<5xi8>
  }

  // CHECK-LABEL: func.func @main()
  func.func @main() attributes {plan.memory_space = #plan.memory_space<host>} {
    // CHECK: %[[v0:.+]] = memref.get_global
    // CHECK: %[[v1:.+]] = memref.get_global
    // CHECK: %[[v2:.+]] = memref.get_global
    // CHECK: %[[alloc:.+]] = memref.alloc() {{.*}} #plan.memory_space<host>
    %cst = arith.constant dense<[-128, 0, 16, -18, 127]> : tensor<5xi8>
    %cst_0 = arith.constant dense<[0, 1, 8, -9, 0]> : tensor<5xi8>
    %cst_1 = arith.constant dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>
    // CHECK: call @compute(%[[v1]], %[[v2]], %[[alloc]])
    %0 = call @compute(%cst_0, %cst_1) : (tensor<5xi8>, tensor<5xi8>) -> tensor<5xi8>
    // CHECK: call @check_eq(%[[alloc]], %[[v0]])
    call @check_eq(%0, %cst) : (tensor<5xi8>, tensor<5xi8>) -> ()
    // CHECK: memref.dealloc %[[alloc]]
    return
  }
}

// -----

func.func @test_loop_region_dps_rewrite_while(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<10xf32>
  %r = scf.while(%arg1 = %arg0) : (tensor<10xf32>) -> (tensor<10xf32>) {
    %v0 = tensor.extract %arg1[%c0] : tensor<10xf32>
    %cond = arith.cmpf ogt, %v0, %c0f : f32
    scf.condition(%cond) %arg1 : tensor<10xf32>
  } do {
  ^bb0(%arg2: tensor<10xf32>):
    %1 = linalg.map {math.exp}
      ins(%arg2 : tensor<10xf32>)
      outs(%0 : tensor<10xf32>)
    scf.yield %1 : tensor<10xf32>
  }
  return %r : tensor<10xf32>
}

// CHECK-LABEL: func.func @test_loop_region_dps_rewrite_while
//  CHECK-SAME: (%[[arg0:.+]]: memref<10xf32, #plan.memory_space<device>>,
//  CHECK-SAME:  %[[arg1:.+]]: memref<10xf32, #plan.memory_space<device>> {plan.result_arg})
//   CHECK-DAG:     %[[cst:.+]] = arith.constant 0.0
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[alloc:.+]] = memref.alloc() {{.*}} : memref<10xf32, #plan.memory_space<host_pinned>>
//       CHECK:     scf.while : () -> () {
//       CHECK:       memref.copy %[[arg0]], %[[alloc]]
//       CHECK:       %[[v0:.+]] = memref.load %[[alloc]][%[[c0]]]
//       CHECK:       %[[v1:.+]] = arith.cmpf ogt, %[[v0]], %[[cst]] : f32
//       CHECK:       scf.condition(%[[v1]])
//       CHECK:     } do {
//  CHECK-NEXT:       linalg.map {{.*}} ins(%[[arg0]] : {{.*}}) outs(%[[arg0]] :
//  CHECK-NEXT:       scf.yield
//       CHECK:     memref.copy %[[arg0]], %[[arg1]]
//       CHECK:     memref.dealloc %[[alloc]]
//       CHECK:     return

// -----

func.func @alloc_tensors_from_elements(%arg0: i32) -> (
    tensor<1xi32> {plan.memory_space = #plan.memory_space<host>},
    tensor<1xi32>) {
  %0 = tensor.from_elements %arg0 : tensor<1xi32>
  %1 = tensor.from_elements %arg0 : tensor<1xi32>
  return %0, %1 : tensor<1xi32>, tensor<1xi32>
}

// CHECK-LABEL: func.func @alloc_tensors_from_elements
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: memref<1xi32, #plan.memory_space<host>> {plan.result_arg},
//  CHECK-SAME:  %[[arg2:.+]]: memref<1xi32, #plan.memory_space<device>> {plan.result_arg}) {
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[alloc:.+]] = memref.alloc() {{.*}} : memref<1xi32, #plan.memory_space<host>>
//       CHECK:     memref.store %[[arg0]], %[[alloc]][%[[c0]]]
//       CHECK:     %[[alloc_0:.+]] = memref.alloc() {{.*}} : memref<1xi32, #plan.memory_space<host_pinned>>
//       CHECK:     memref.store %[[arg0]], %[[alloc_0]][%[[c0]]]
//   CHECK-DAG:     memref.copy %[[alloc_0]], %[[arg2]]
//   CHECK-DAG:     memref.copy %[[alloc]], %[[arg1]]
//   CHECK-DAG:     memref.dealloc %[[alloc]]
//   CHECK-DAG:     memref.dealloc %[[alloc_0]]
//  CHECK-NEXT:     return

// ALLOC-LABEL: func.func @alloc_tensors_from_elements
//  ALLOC-SAME: (%[[arg0:.+]]: i32)
//       ALLOC:     %[[c0:.+]] = arith.constant 0 : index
//       ALLOC:     %[[alloc:.+]] = memref.alloc() {{.*}} : memref<1xi32, #plan.memory_space<host>>
//       ALLOC:     memref.store %[[arg0]], %[[alloc]][%[[c0]]]
//       ALLOC:     %[[alloc_0:.+]] = memref.alloc() {{.*}} : memref<1xi32, #plan.memory_space<host_pinned>>
//       ALLOC:     memref.store %[[arg0]], %[[alloc_0]][%[[c0]]]
//       ALLOC:     %[[alloc_1:.+]] = memref.alloc() {{.*}} : memref<1xi32, #plan.memory_space<device>>
//       ALLOC:     memref.copy %[[alloc_0]], %[[alloc_1]]
//       ALLOC:     memref.dealloc %[[alloc_0]]
//       ALLOC:     return %[[alloc]], %[[alloc_1]]


// -----

func.func @device_extract(%arg0: tensor<128xi1>, %arg1: index) -> i1 {
  %1 = tensor.extract %arg0[%arg1] : tensor<128xi1>
  return %1 : i1
}

// CHECK-LABEL: func.func @device_extract
//  CHECK-SAME: (%[[arg0:.+]]: memref<128xi1, #plan.memory_space<device>>, %[[arg1:.+]]: index) -> i1
//       CHECK:     %[[alloc:.+]] = memref.alloc() {{.*}} : memref<128xi1, #plan.memory_space<host_pinned>>
//       CHECK:     memref.copy %[[arg0]], %[[alloc]]
//       CHECK:     %[[v0:.+]] = memref.load %[[alloc]][%[[arg1]]]
//       CHECK:     memref.dealloc %[[alloc]]
//       CHECK:     return %[[v0]]

// -----

// Check that some problematic cases involving space transfers within
// `scf.if` produce functionally correct IR. These cases cannot be
// well optimized until we have a alloc-tensor like operation with
// combined alloc + space transfer semantic. And `plan-alloc-tensors`
// inserts an extra copy at the return that could be eliminated.

module @if_yield_constant_cases {

func.func @if_constant_mat(%cond: i1, %value: f32)
    -> (tensor<1xf32> {plan.memory_space = #plan.memory_space<device>}) {
  %0 = scf.if %cond -> tensor<1xf32> {
    %1 = tensor.from_elements %value : tensor<1xf32>
    scf.yield %1 : tensor<1xf32>
  } else {
    %cst = arith.constant dense<2.0> : tensor<1xf32>
    scf.yield %cst : tensor<1xf32>
  }
  return %0 : tensor<1xf32>
}

// CHECK-LABEL: func.func @if_constant_mat
// CHECK: memref.alloc()
// CHECK: memref.alloc()
// CHECK: if
// CHECK:   memref.alloc
// CHECK:   memref.store
// CHECK:   memref.copy
// CHECK:   memref.dealloc
// CHECK: else
// CHECK:     memref.copy
// CHECK-NOT: memref.copy
// CHECK: memref.copy
// CHECK: memref.dealloc
// CHECK: memref.dealloc
// CHECK: return

func.func @if_constant_mat_hoisted(%cond: i1, %value: f32)
    -> (tensor<1xf32> {plan.memory_space = #plan.memory_space<device>}) {
  %cst = arith.constant dense<2.0> : tensor<1xf32>
  %2 = tensor.from_elements %value : tensor<1xf32>

  %0 = scf.if %cond -> tensor<1xf32> {
    scf.yield %2 : tensor<1xf32>
  } else {
    scf.yield %cst : tensor<1xf32>
  }
  return %0 : tensor<1xf32>
}


// CHECK-LABEL: func.func @if_constant_mat
// CHECK: memref.alloc
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: memref.alloc
// CHECK: scf.if
// CHECK-NEXT: } else {
// CHECK-NEXT: memref.copy
// CHECK: memref.copy
// CHECK: memref.dealloc
// CHECK: memref.dealloc
// CHECK: memref.dealloc
// CHECK: return

func.func @if_else_yield_constant_mat(%cond: i1, %value: f32)
    -> (tensor<1xf32> {plan.memory_space = #plan.memory_space<device>}) {
  %0 = scf.if %cond -> tensor<1xf32> {
    %cst = arith.constant dense<2.0> : tensor<1xf32>
    scf.yield %cst : tensor<1xf32>
  } else {
    %1 = tensor.from_elements %value : tensor<1xf32>
    scf.yield %1 : tensor<1xf32>
  }
  return %0 : tensor<1xf32>
}

// CHECK-LABEL: func.func @if_else_yield_constant_mat
// CHECK: memref.alloc()
// CHECK: memref.alloc()
// CHECK: if
// CHECK:     memref.copy
// CHECK-NOT: memref.copy
// CHECK: else
// CHECK:   memref.alloc
// CHECK:   memref.store
// CHECK:   memref.copy
// CHECK:   memref.dealloc
// CHECK: memref.copy
// CHECK: memref.dealloc
// CHECK: memref.dealloc
// CHECK: return
}