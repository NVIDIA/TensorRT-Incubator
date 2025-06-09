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

// TODO: This test shows that the pre-processing prior to one-shot-bufferization is
// sub-optimal. We allocate two host buffers to hold the `tensor<4xindex>` for some
// reason.

// CHECK-LABEL: func.func @small_host_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x?xf32, #plan.memory_space<device>>, %[[arg1:.+]]: memref<?x?x?x?xf32, #plan.memory_space<device>> {plan.result_arg}) {
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//   CHECK-DAG:     %[[alloc:.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     memref.store %[[c1]], %[[alloc]][%[[c0]]] : memref<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     memref.store %[[c2]], %[[alloc]][%[[c1]]] : memref<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     memref.store %[[c3]], %[[alloc]][%[[c2]]] : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     %[[alloc_0:.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     memref.copy %[[alloc]], %[[alloc_0]] : memref<4xindex, #plan.memory_space<host>> to memref<4xindex, #plan.memory_space<host>>
//       CHECK:     memref.store %[[c4]], %[[alloc_0]][%[[c3]]] : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     %[[reshape:.+]] = memref.reshape %[[arg0]](%[[alloc_0]]) :
//       CHECK:     memref.copy %[[reshape]], %[[arg1]] : memref<?x?x?x?xf32, #plan.memory_space<device>> to memref<?x?x?x?xf32, #plan.memory_space<device>>
//       CHECK:     memref.dealloc %[[alloc]] : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     memref.dealloc %[[alloc_0]] : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     return

// -----

func.func @small_host_and_device_tensor_constant(%arg0: tensor<?x?xf32>) -> (tensor<?x?x?x?xf32>, tensor<4xindex>) {
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xindex>
  %1 = tensor.reshape %arg0 (%0) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  return %1, %0 : tensor<?x?x?x?xf32>, tensor<4xindex>
}

//       CHECK:   memref.global "private" constant @__constant_4xindex : memref<4xindex, #plan.memory_space<device>> = dense<[1, 2, 3, 4]> {alignment = 16 : i64}
// CHECK-LABEL: func.func @small_host_and_device_tensor_constant
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x?xf32, #plan.memory_space<device>>,
//  CHECK-SAME:  %[[arg1:.+]]: memref<?x?x?x?xf32, #plan.memory_space<device>> {plan.result_arg},
//  CHECK-SAME:  %[[arg2:.+]]: memref<4xindex, #plan.memory_space<device>> {plan.result_arg}) {
//   CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//   CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[v0:.+]] = memref.get_global @__constant_4xindex : memref<4xindex, #plan.memory_space<device>>
//       CHECK:     memref.copy %[[v0]], %[[arg2]] : memref<4xindex, #plan.memory_space<device>> to memref<4xindex, #plan.memory_space<device>>
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     memref.store %[[c1]], %[[alloc]][%[[c0]]] : memref<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     memref.store %[[c2]], %[[alloc]][%[[c1]]] : memref<4xindex, #plan.memory_space<host>>
//   CHECK-DAG:     memref.store %[[c3]], %[[alloc]][%[[c2]]] : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     %[[alloc_0:.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     memref.copy %[[alloc]], %[[alloc_0]] : memref<4xindex, #plan.memory_space<host>> to memref<4xindex, #plan.memory_space<host>>
//       CHECK:     memref.store %[[c4]], %[[alloc_0]][%[[c3]]] : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     %[[reshape:.+]] = memref.reshape %[[arg0]](%[[alloc_0]]) : (memref<?x?xf32, #plan.memory_space<device>>, memref<4xindex, #plan.memory_space<host>>)
//       CHECK:     memref.copy %[[reshape]], %[[arg1]] : memref<?x?x?x?xf32, #plan.memory_space<device>> to memref<?x?x?x?xf32, #plan.memory_space<device>>
//       CHECK:     memref.dealloc %[[alloc]] : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     memref.dealloc %[[alloc_0]] : memref<4xindex, #plan.memory_space<host>>
//       CHECK:     return

// -----

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

// CHECK-LABEL: func.func @while_loop_host_tensor_carried
//         CHECK:     scf.while : () -> ()
// CHECK-COUNT-2:       memref.copy
//         CHECK:       scf.condition
// CHECK-COUNT-2:       memref.copy
//         CHECK:       scf.yield
// CHECK-COUNT-1:       memref.copy
//     CHECK-NOT:     memref.copy
