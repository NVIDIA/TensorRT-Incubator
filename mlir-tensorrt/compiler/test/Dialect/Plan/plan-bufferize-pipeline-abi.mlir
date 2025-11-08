// RUN: mlir-tensorrt-opt %s -split-input-file -executor-generate-abi-wrappers=force-undef-output-args=true -plan-bufferize-pipeline=force-entrypoints-return-allocs=true | FileCheck %s --check-prefix=ALLOC
// RUN: mlir-tensorrt-opt %s -split-input-file -executor-generate-abi-wrappers -plan-bufferize-pipeline | FileCheck %s


func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: i1) -> tensor<10xf32> {
  %1 = arith.select %arg2, %arg0, %arg1 : tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: @main
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

// ALLOC-LABEL: @main
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

func.func @main() -> (tensor<10xf32>, tensor<2x5xf32>) {
  %0 = arith.constant dense<0.0> : tensor<10xf32>
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [2, 5] : tensor<10xf32> into tensor<2x5xf32>
  return %0, %1 : tensor<10xf32>, tensor<2x5xf32>
}


// CHECK-LABEL:   @main
//  CHECK-SAME: (%[[arg0:[0-9a-z]+]]: !executor.ptr<host>
//  CHECK-SAME:  %[[arg1:[0-9a-z]+]]: !executor.ptr<host>
//   CHECK-DAG:     %[[v0:.+]] = executor.abi.recv %[[arg1]] :
//   CHECK-DAG:     %[[v1:.+]] = executor.abi.recv %[[arg0]] :
//   CHECK-DAG:     %[[v2:.+]] = memref.get_global {{.*}} : memref<10xf32, #plan.memory_space<device>>
//   CHECK-DAG:     %[[v3:.+]] = memref.get_global {{.*}} : memref<2x5xf32, #plan.memory_space<device>>
//       CHECK:     memref.copy %[[v2]], %[[v1]] :
//       CHECK:     memref.copy %[[v3]], %[[v0]] :
//       CHECK:     return

// ALLOC-LABEL:   @main
// ALLOC-SAME: (%[[arg0:[0-9a-z]+]]: !executor.ptr<host>
// ALLOC-SAME:        executor.abi = #executor.arg<byref, memref<10xf32, #plan.memory_space<device>>, undef>
// ALLOC-SAME:  %[[arg1:[0-9a-z]+]]: !executor.ptr<host>
// ALLOC-SAME:        executor.abi = #executor.arg<byref, memref<2x5xf32, #plan.memory_space<device>>, undef>
//   ALLOC-DAG:     %[[true:.+]] = arith.constant true
//   ALLOC-DAG:     %[[v0:.+]] = memref.get_global {{.*}} : memref<10xf32, #plan.memory_space<device>>
//   ALLOC-DAG:     %[[v1:.+]] = memref.get_global {{.*}} : memref<2x5xf32, #plan.memory_space<device>>
//   ALLOC-DAG:     %[[alloc:.+]] = memref.alloc() : memref<10xf32, #plan.memory_space<device>>
//       ALLOC:     memref.copy %[[v0]], %[[alloc]] : memref<10xf32, #plan.memory_space<device>> to memref<10xf32, #plan.memory_space<device>>
//       ALLOC:     %[[v2:.+]] = executor.abi.send %[[alloc]] to %[[arg0]] ownership(%[[true]]) : memref<10xf32, #plan.memory_space<device>>
//       ALLOC:     %[[alloc_0:.+]] = memref.alloc() : memref<2x5xf32, #plan.memory_space<device>>
//       ALLOC:     memref.copy %[[v1]], %[[alloc_0]] : memref<2x5xf32, #plan.memory_space<device>> to memref<2x5xf32, #plan.memory_space<device>>
//       ALLOC:     %[[v3:.+]] = executor.abi.send %[[alloc_0]] to %[[arg1]] ownership(%[[true]]) : memref<2x5xf32, #plan.memory_space<device>>
//       ALLOC:     return
