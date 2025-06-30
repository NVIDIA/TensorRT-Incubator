// RUN: mlir-tensorrt-opt %s -split-input-file -plan-buffer-results-to-out-params=ignore-public-functions -canonicalize | FileCheck %s

func.func private @alloc_size() -> index

func.func private @callee(%arg0 : index)
    -> (memref<5xf32> {plan.tag = "foo"}, index, memref<?xf32> {plan.tag = "bar"}) {
  %0 = memref.alloc() : memref<5xf32>
  %size = func.call @alloc_size() : () -> index
  %1 = memref.alloc(%size) : memref<?xf32>
  %c1 = arith.constant 1 : index
  %2 = arith.addi %arg0, %c1 : index
  return %0, %2, %1 : memref<5xf32>, index, memref<?xf32>
}

func.func private @callee_external() -> (memref<5xf32> {plan.tag = "foo"})

func.func @caller() -> (memref<5xf32>, index, memref<?xf32>, memref<5xf32>) {
  %c10 = arith.constant 10 : index
  %0:3 = func.call @callee(%c10) : (index) -> (memref<5xf32>, index, memref<?xf32>)
  %1 = func.call @callee_external() : () -> (memref<5xf32>)
  return %0#0, %0#1, %0#2, %1 : memref<5xf32>, index, memref<?xf32>, memref<5xf32>
}

// CHECK-LABEL: func.func private @callee(
//  CHECK-SAME: %[[arg0:.+]]: index, %[[arg1:.+]]: memref<5xf32> {plan.result_arg, plan.tag = "foo"}) -> (index, memref<?xf32> {plan.tag = "bar"})
//       CHECK:     %[[alloc:.+]] = memref.alloc
//       CHECK:     %[[add:.+]] = arith.addi
//   CHECK-NOT:     memref.copy
//       CHECK:     return %[[add]], %[[alloc]] : index, memref<?xf32>

// CHECK-LABEL: func.func private @callee_external() -> (memref<5xf32> {plan.tag = "foo"})

// CHECK-LABEL: func.func @caller
//   CHECK-DAG:     %[[c10:.+]] = arith.constant 10 : index
//   CHECK-DAG:     %[[alloc:.+]] = memref.alloc() : memref<5xf32>
//   CHECK-DAG:     %[[v0:.+]]:2 = call @callee(%[[c10]], %[[alloc]]) : (index, memref<5xf32>) -> (index, memref<?xf32>)
//   CHECK-DAG:     %[[v1:.+]] = call @callee_external() : () -> memref<5xf32>
//   CHECK-DAG:     return %[[alloc]], %[[v0]]#0, %[[v0]]#1, %[[v1]]

// -----

func.func private @callee_returns_aliasing() -> (memref<10xf32>, memref<5xf32>) {
  %0 = memref.alloc() : memref<10xf32>
  %1 = memref.subview %0[0][5][1] : memref<10xf32> to memref<5xf32>
  return %0, %1 : memref<10xf32>, memref<5xf32>
}

func.func @caller() -> (memref<10xf32>, memref<5xf32>) {
  %0:2 = func.call @callee_returns_aliasing() : () -> (memref<10xf32>, memref<5xf32>)
  return %0#0, %0#1 : memref<10xf32>, memref<5xf32>
}

// CHECK-LABEL: @callee_returns_aliasing() -> (memref<10xf32>, memref<5xf32>)


// -----

func.func private @callee_returns_block_arg(%arg0: memref<5xf32>) -> (memref<5xf32>) {
  return %arg0 : memref<5xf32>
}

func.func @caller(%arg0: memref<5xf32>) -> (memref<5xf32>) {
  %1 = func.call @callee_returns_block_arg(%arg0) : (memref<5xf32>) -> (memref<5xf32>)
  return %1 : memref<5xf32>
}

// CHECK-LABEL: @callee_returns_block_arg
// CHECK-NEXT: return

// CHECK-LABEL: @caller
//   CHECK-NOT:  memref.alloc


// -----

func.func private @callee_aliasing_duplicate_alloc() -> (memref<10xf32>, memref<10xf32>) {
  %0 = memref.alloc() : memref<10xf32>
  return %0, %0 : memref<10xf32>, memref<10xf32>
}

func.func @caller(%arg0: index, %arg1: f32) -> (f32) {
  %0:2 = func.call @callee_aliasing_duplicate_alloc() : () -> (memref<10xf32>, memref<10xf32>)
  memref.store %arg1, %0#0[%arg0] : memref<10xf32>
  %1 = memref.load %0#1[%arg0] : memref<10xf32>
  return %1 : f32
}

// CHECK-LABEL: @callee_aliasing_duplicate_alloc() -> (memref<10xf32>, memref<10xf32>)
// CHECK-LABEL: @callee


// -----

func.func private @multiple_blocks(%arg0: i1, %arg1: memref<5xf32>, %arg2: memref<5xf32>) -> (memref<5xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  return %arg1 : memref<5xf32>
^bb2:
  return %arg2 : memref<5xf32>
}

func.func @caller(%arg0: i1, %arg1: memref<5xf32>, %arg2: memref<5xf32>) -> (memref<5xf32>) {
  %0 = func.call @multiple_blocks(%arg0, %arg1, %arg2) : (i1, memref<5xf32>, memref<5xf32>) -> (memref<5xf32>)
  return %0 : memref<5xf32>
}

// CHECK-LABEL: func.func private @multiple_blocks
//  CHECK-SAME: (%[[arg0:.+]]: i1, %[[arg1:.+]]: memref<5xf32>, %[[arg2:.+]]: memref<5xf32>, %[[arg3:.+]]: memref<5xf32> {plan.result_arg})
//       CHECK:     memref.copy %[[arg1]], %[[arg3]]
//       CHECK:     return
//       CHECK:     memref.copy %[[arg2]], %[[arg3]]
//       CHECK:     return

// CHECK-LABEL: @caller
//       CHECK:  %[[alloc:.+]] = memref.alloc() : memref<5xf32>
//       CHECK:  call @multiple_blocks
//       CHECK:  return %[[alloc]]

// -----

// This test uses a bunch of convoluted code to verify
// that complicated sequence of operations rooted at an allocation
// can be hoisted.

// Note that ultimately the allocation and unused function arguments
// are dropped in follow on passes (canonicalize, remove-dead-values).

!result_type1 = memref<?xf32>
!result_type2 = memref<?xf32, strided<[?], offset: ?>>

func.func private @callee_returns_complicated(
    %arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> (!result_type1, !result_type2) {
  %0 = memref.alloc(%arg0) : memref<?xf32>
  %1 = memref.alloc(%arg1) : memref<?xf32>

  %c0 = arith.constant 0 : index
  %3 = arith.constant 1.0 : f32
  %cond = arith.cmpi slt, %arg3, %arg4 : index
  %4 = scf.if %cond -> !result_type2 {
    %sv0 = memref.subview %1[%arg2][%arg3][%arg4] : memref<?xf32> to !result_type2
    scf.yield %sv0 : !result_type2
  } else {
    %sv1 = memref.subview %1[%arg4][%arg2][%arg3] : memref<?xf32> to !result_type2
    scf.yield %sv1 : !result_type2
  }
  memref.store %3, %0[%c0] : !result_type1
  memref.store %3, %4[%c0] : !result_type2
  return %0, %4 : !result_type1, !result_type2
}

func.func @caller(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index)
    -> (!result_type1, !result_type2) {
  %0:2 = func.call @callee_returns_complicated(%arg0, %arg1, %arg2, %arg3, %arg4)
    : (index, index, index, index, index) -> (!result_type1, !result_type2)
  return %0#0, %0#1 : !result_type1, !result_type2
}

// CHECK-LABEL: func.func private @callee_returns_complicated
//  CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index, %[[arg2:.+]]: index, %[[arg3:.+]]: index, %[[arg4:.+]]: index,
//  CHECK-SAME:  %[[arg5:.+]]: memref<?xf32> {plan.result_arg}, %[[arg6:.+]]: memref<?xf32, strided<[?], offset: ?>> {plan.result_arg}) {
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[cst:.+]] = arith.constant 1.000000e+00 : f32
//   CHECK-DAG:     memref.store %[[cst]], %[[arg5]][%[[c0]]] : memref<?xf32>
//   CHECK-DAG:     memref.store %[[cst]], %[[arg6]][%[[c0]]] : memref<?xf32, strided<[?], offset: ?>>
//       CHECK:     return

// CHECK-LABEL: func.func @caller
//  CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index, %[[arg2:.+]]: index, %[[arg3:.+]]: index, %[[arg4:.+]]: index)
//   CHECK-DAG:     %[[alloc:.+]] = memref.alloc(%[[arg0]]) : memref<?xf32>
//   CHECK-DAG:     %[[alloc_0:.+]] = memref.alloc(%[[arg1]]) : memref<?xf32>
//   CHECK-DAG:     %[[v0:.+]] = arith.cmpi slt, %[[arg3]], %[[arg4]] : index
//   CHECK-DAG:     %[[v1:.+]] = scf.if %[[v0]] -> (memref<?xf32, strided<[?], offset: ?>>) {
//       CHECK:       %[[subview:.+]] = memref.subview %[[alloc_0]][%[[arg2]]] [%[[arg3]]] [%[[arg4]]]
//       CHECK:       scf.yield %[[subview]] :
//       CHECK:     } else {
//       CHECK:       %[[subview:.+]] = memref.subview %[[alloc_0]][%[[arg4]]] [%[[arg2]]] [%[[arg3]]]
//       CHECK:       scf.yield %[[subview]]
//       CHECK:     call @callee_returns_complicated(%[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]], %[[arg4]], %[[alloc]], %[[v1]])
//       CHECK:     return %[[alloc]], %[[v1]]
