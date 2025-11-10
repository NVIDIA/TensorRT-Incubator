// RUN: mlir-tensorrt-opt %s -split-input-file -executor-generate-abi-wrappers -plan-bufferize-pipeline | FileCheck %s --check-prefixes=CHECK,BOTH
// RUN: mlir-tensorrt-opt %s -split-input-file -executor-generate-abi-wrappers=force-undef-output-args=true -plan-bufferize-pipeline=force-entrypoints-return-allocs=true | FileCheck %s --check-prefixes=ALLOC,BOTH

func.func @tensor_select(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: i1) -> tensor<10xf32> {
  %1 = arith.select %arg2, %arg0, %arg1 : tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: func.func public @tensor_select
// CHECK-SAME: (%[[arg0:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xf32, #plan.memory_space<device>>>},
// CHECK-SAME:  %[[arg1:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xf32, #plan.memory_space<device>>>},
// CHECK-SAME:  %[[arg2:[a-z0-9]+]]: i1,
// CHECK-SAME:  %[[arg3:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32, #plan.memory_space<device>>>, executor.result_slot = 0 : i32})
// CHECK-SAME: attributes {executor.func_abi = (memref<10xf32, #plan.memory_space<device>>, memref<10xf32, #plan.memory_space<device>>, i1) -> memref<10xf32, #plan.memory_space<device>>}
// CHECK-DAG:     %[[v0:.+]] = executor.abi.recv %[[arg3]] : memref<10xf32, #plan.memory_space<device>>
// CHECK-DAG:     %[[v1:.+]] = executor.abi.recv %[[arg0]] : memref<10xf32, #plan.memory_space<device>>
// CHECK-DAG:     %[[v2:.+]] = executor.abi.recv %[[arg1]] : memref<10xf32, #plan.memory_space<device>>
//       CHECK:     %[[v3:.+]] = arith.select %[[arg2]], %[[v1]], %[[v2]] : memref<10xf32, #plan.memory_space<device>>
//       CHECK:     memref.copy %[[v3]], %[[v0]] : memref<10xf32, #plan.memory_space<device>> to memref<10xf32, #plan.memory_space<device>>
//       CHECK:     return

// ALLOC-LABEL: func.func public @tensor_select
// ALLOC-SAME: (%[[arg0:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xf32, #plan.memory_space<device>>>},
// ALLOC-SAME:  %[[arg1:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xf32, #plan.memory_space<device>>>},
// ALLOC-SAME:  %[[arg2:[a-z0-9]+]]: i1,
// ALLOC-SAME:  %[[arg3:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32, #plan.memory_space<device>>, undef>
// ALLOC-SAME: attributes {executor.func_abi = (memref<10xf32, #plan.memory_space<device>>, memref<10xf32, #plan.memory_space<device>>, i1) -> memref<10xf32, #plan.memory_space<device>>}
// ALLOC-DAG:     %[[true:.+]] = arith.constant true
// ALLOC-DAG:     %[[v0:.+]] = executor.abi.recv %[[arg0]] : memref<10xf32, #plan.memory_space<device>>
// ALLOC-DAG:     %[[v1:.+]] = executor.abi.recv %[[arg1]] : memref<10xf32, #plan.memory_space<device>>
//      ALLOC:     %[[v3:.+]] = arith.select %[[arg2]], %[[v0]], %[[v1]] : memref<10xf32, #plan.memory_space<device>>
//      ALLOC:     %[[alloc:.+]] = memref.alloc() : memref<10xf32, #plan.memory_space<device>>
//      ALLOC:     memref.copy %[[v3]], %[[alloc]] : memref<10xf32, #plan.memory_space<device>> to memref<10xf32, #plan.memory_space<device>>
//      ALLOC:     executor.abi.send %[[alloc]] to %[[arg3]] ownership(%[[true]])
//      ALLOC:     return

// -----

func.func @reshape_constant_alias() -> (tensor<10xf32>, tensor<2x5xf32>) {
  %0 = arith.constant dense<0.0> : tensor<10xf32>
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [2, 5] : tensor<10xf32> into tensor<2x5xf32>
  return %0, %1 : tensor<10xf32>, tensor<2x5xf32>
}


// CHECK-LABEL:   func.func public @reshape_constant_alias
//  CHECK-SAME: (%[[arg0:[0-9a-z]+]]: !executor.ptr<host>
//  CHECK-SAME:  %[[arg1:[0-9a-z]+]]: !executor.ptr<host>
//   CHECK-DAG:     %[[v0:.+]] = executor.abi.recv %[[arg1]] :
//   CHECK-DAG:     %[[v1:.+]] = executor.abi.recv %[[arg0]] :
//   CHECK-DAG:     %[[v2:.+]] = memref.get_global {{.*}} : memref<10xf32, #plan.memory_space<device>>
//   CHECK-DAG:     %[[v3:.+]] = memref.get_global {{.*}} : memref<2x5xf32, #plan.memory_space<device>>
//       CHECK:     memref.copy %[[v2]], %[[v1]] :
//       CHECK:     memref.copy %[[v3]], %[[v0]] :
//       CHECK:     return

// ALLOC-LABEL:   func.func public @reshape_constant_alias
// ALLOC-SAME: (%[[arg0:[0-9a-z]+]]: !executor.ptr<host>
// ALLOC-SAME:        executor.abi = #executor.arg<byref, memref<10xf32, #plan.memory_space<device>>, undef>
// ALLOC-SAME:  %[[arg1:[0-9a-z]+]]: !executor.ptr<host>
// ALLOC-SAME:        executor.abi = #executor.arg<byref, memref<2x5xf32, #plan.memory_space<device>>, undef>
//   ALLOC-DAG:     %[[true:.+]] = arith.constant true
//   ALLOC-DAG:     %[[v0:.+]] = memref.get_global {{.*}} : memref<10xf32, #plan.memory_space<device>>
//   ALLOC-DAG:     %[[v1:.+]] = memref.get_global {{.*}} : memref<2x5xf32, #plan.memory_space<device>>
//   ALLOC-DAG:     %[[alloc:.+]] = memref.alloc() : memref<10xf32, #plan.memory_space<device>>
//       ALLOC:     memref.copy %[[v0]], %[[alloc]] : memref<10xf32, #plan.memory_space<device>> to memref<10xf32, #plan.memory_space<device>>
//       ALLOC:     %[[alloc_0:.+]] = memref.alloc() : memref<2x5xf32, #plan.memory_space<device>>
//       ALLOC:     memref.copy %[[v1]], %[[alloc_0]] : memref<2x5xf32, #plan.memory_space<device>> to memref<2x5xf32, #plan.memory_space<device>>
//   ALLOC-DAG:     %[[v2:.+]] = executor.abi.send %[[alloc]] to %[[arg0]] ownership(%[[true]]) : memref<10xf32, #plan.memory_space<device>>
//   ALLOC-DAG:     %[[v3:.+]] = executor.abi.send %[[alloc_0]] to %[[arg1]] ownership(%[[true]]) : memref<2x5xf32, #plan.memory_space<device>>
//       ALLOC:     return

// -----

func.func @test_scalar_complex(%arg0: f32, %arg1: f32) -> (f32, complex<f32>) {
  %0 = arith.addf %arg0, %arg1 : f32
  %1 = arith.subf %arg0, %arg1 : f32
  %2 = complex.create %0, %1 : complex<f32>
  return %0, %2 : f32, complex<f32>
}

// BOTH-LABEL: func.func public @test_scalar_complex
// BOTH-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32,
// BOTH-SAME: %[[arg2:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, f32
// BOTH-SAME: %[[arg3:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, complex<f32>
//       BOTH:     %[[v0:.+]] = arith.addf %[[arg0]], %[[arg1]]
//       BOTH:     %[[v1:.+]] = arith.subf %[[arg0]], %[[arg1]]
//       BOTH:     %[[v2:.+]] = complex.create %[[v0]], %[[v1]]
//       BOTH:     %[[v3:.+]] = executor.abi.send %[[v0]] to %[[arg2]]
//       BOTH:     %[[v4:.+]] = executor.abi.send %[[v2]] to %[[arg3]]
//       BOTH:     return

// -----
module @multiple_dynamic_subviews {

func.func @multiple_dynamic_subviews(%arg0: tensor<?xf32>, %arg1: index, %arg2: index) -> (tensor<?xf32>, tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %size = tensor.dim %arg0, %c0 : tensor<?xf32>
  %empty = tensor.empty(%size) : tensor<?xf32>
  %add = linalg.map {arith.addf} ins(%arg0, %arg0 : tensor<?xf32>, tensor<?xf32>) outs(%empty : tensor<?xf32>)
  %0 = tensor.extract_slice %add[%arg1][%arg2][1] : tensor<?xf32> to tensor<?xf32>
  %1 = tensor.extract_slice %add[10][%arg2][2] : tensor<?xf32> to tensor<?xf32>
  return %0, %1 : tensor<?xf32>, tensor<?xf32>
}

}

// ALLOC-LABEL: func.func public @multiple_dynamic_subviews
//  ALLOC-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<?xf32, #plan.memory_space<device>>>},
//  ALLOC-SAME: %[[arg1:.+]]: index, %[[arg2:.+]]: index,
//  ALLOC-SAME: %[[arg3:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<?xf32, #plan.memory_space<device>>, undef>
//  ALLOC-SAME: %[[arg4:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<?xf32, #plan.memory_space<device>>, undef>
//   ALLOC-DAG:     %[[true:.+]] = arith.constant true
//       ALLOC:     %[[alloc:.+]] = memref.alloc
//       ALLOC:     linalg.map
//   ALLOC-DAG:     %[[subview0:.+]] = memref.subview %[[alloc]][%{{.*}}]
//   ALLOC-DAG:     %[[subview1:.+]] = memref.subview %[[alloc]][10]
//       ALLOC:     %[[out0:.+]] = memref.alloc
//       ALLOC:     memref.copy %[[subview0]], %[[out0]]
//       ALLOC:     %[[out1:.+]] = memref.alloc
//       ALLOC:     memref.copy %[[subview1]], %[[out1]]
//       ALLOC:     memref.dealloc %[[alloc]]
//   ALLOC-DAG:     %[[v1:.+]] = executor.abi.send %[[out0]] to %[[arg3]] ownership(%[[true]]) : memref<?xf32, #plan.memory_space<device>>
//   ALLOC-DAG:     %[[v2:.+]] = executor.abi.send %[[out1]] to %[[arg4]] ownership(%[[true]]) : memref<?xf32, #plan.memory_space<device>>
//       ALLOC:     return

// CHECK-LABEL: func.func public @multiple_dynamic_subviews
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<?xf32, #plan.memory_space<device>>>},
//  CHECK-SAME: %[[arg1:.+]]: index, %[[arg2:.+]]: index,
//  CHECK-SAME: %[[arg3:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<?xf32, #plan.memory_space<device>>>
//  CHECK-SAME: %[[arg4:[a-z0-9]+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<?xf32, #plan.memory_space<device>>>
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[v0:.+]] = executor.abi.recv %[[arg4]] : memref<?xf32, #plan.memory_space<device>>
//   CHECK-DAG:     %[[v1:.+]] = executor.abi.recv %[[arg3]] : memref<?xf32, #plan.memory_space<device>>
//   CHECK-DAG:     %[[v2:.+]] = executor.abi.recv %[[arg0]] : memref<?xf32, #plan.memory_space<device>>
//       CHECK:     %[[alloc:.+]] = memref.alloc
//       CHECK:     linalg.map { arith.addf } ins(%[[v2]], %[[v2]] : {{.*}}) outs(%[[alloc]] : {{.*}})
//   CHECK-DAG:     %[[subview:.+]] = memref.subview %[[alloc]][%[[arg1]]] [%[[arg2]]] [1]
//   CHECK-DAG:     %[[subview_0:.+]] = memref.subview %[[alloc]][10] [%[[arg2]]] [2]
//   CHECK-DAG:     memref.copy %[[subview]], %[[v1]]
//   CHECK-DAG:     memref.copy %[[subview_0]], %[[v0]]
//       CHECK:     memref.dealloc %[[alloc]]
//       CHECK:     return

// -----

func.func @subview_reshape(%arg0: tensor<?xf32>, %arg1: index, %arg2: index) -> (tensor<10xf32>, tensor<2x5xf32>) {
  %empty = tensor.empty() : tensor<10xf32>
  %0 = tensor.extract_slice %arg0[0][10][1] : tensor<?xf32> to tensor<10xf32>
  %add = linalg.map {arith.addf} ins(%0, %0 : tensor<10xf32>, tensor<10xf32>) outs(%empty : tensor<10xf32>)
  %1 = tensor.expand_shape %add [[0, 1]] output_shape [2, 5] : tensor<10xf32> into tensor<2x5xf32>
  return %add, %1 : tensor<10xf32>, tensor<2x5xf32>
}

// ALLOC-LABEL: func.func public @subview_reshape
//  ALLOC-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<?xf32, #plan.memory_space<device>>>},
//  ALLOC-SAME: %[[arg1:.+]]: index, %[[arg2:.+]]: index,
//  ALLOC-SAME: %[[arg3:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32, #plan.memory_space<device>>, undef>
//  ALLOC-SAME: %[[arg4:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<2x5xf32, #plan.memory_space<device>>, undef>
//       ALLOC:     %[[true:.+]] = arith.constant true
//       ALLOC:     %[[v0:.+]] = executor.abi.recv %[[arg0]] : memref<?xf32, #plan.memory_space<device>>
//       ALLOC:     %[[alloc:.+]] = memref.alloc() {alignment = 16 : i64} : memref<10xf32, #plan.memory_space<device>>
//       ALLOC:     %[[subview:.+]] = memref.subview %[[v0]][0] [10] [1] : memref<?xf32, #plan.memory_space<device>> to memref<10xf32, strided<[1]>, #plan.memory_space<device>>
//       ALLOC:     linalg.map { arith.addf } ins(%[[subview]], %[[subview]] : memref<10xf32, strided<[1]>, #plan.memory_space<device>>, memref<10xf32, strided<[1]>, #plan.memory_space<device>>) outs(%[[alloc]] : memref<10xf32, #plan.memory_space<device>>)
//       ALLOC:     %[[expand_shape:.+]] = memref.expand_shape %[[alloc]]
//       ALLOC:     %[[v1:.+]] = executor.abi.send %[[alloc]] to %[[arg3]] ownership(%[[true]])
//       ALLOC:     %[[alloc1:.+]] = memref.alloc()
//       ALLOC:     memref.copy %[[expand_shape]], %[[alloc1]] :
//       ALLOC:     %[[v2:.+]] = executor.abi.send %[[alloc1]] to %[[arg4]] ownership(%[[true]])
//       ALLOC:     return

// CHECK-LABEL: func.func public @subview_reshape
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<?xf32, #plan.memory_space<device>>>},
//  CHECK-SAME: %[[arg1:.+]]: index, %[[arg2:.+]]: index,
//  CHECK-SAME: %[[arg3:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32, #plan.memory_space<device>>>
//  CHECK-SAME: %[[arg4:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<2x5xf32, #plan.memory_space<device>>
//   CHECK-DAG:     %[[v0:.+]] = executor.abi.recv %[[arg4]] : memref<2x5xf32, #plan.memory_space<device>>
//   CHECK-DAG:     %[[v1:.+]] = executor.abi.recv %[[arg3]] : memref<10xf32, #plan.memory_space<device>>
//   CHECK-DAG:     %[[v2:.+]] = executor.abi.recv %[[arg0]] : memref<?xf32, #plan.memory_space<device>>
//       CHECK:     %[[subview:.+]] = memref.subview %[[v2]][0] [10] [1] :
//       CHECK:     linalg.map { arith.addf } ins(%[[subview]], %[[subview]] : {{.*}}) outs(%[[v1]]
//       CHECK:     %[[expand_shape:.+]] = memref.expand_shape %[[v1]]
//       CHECK:     memref.copy %[[expand_shape]], %[[v0]]
//       CHECK:     return
