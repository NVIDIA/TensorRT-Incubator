// RUN: mlir-tensorrt-opt %s -memref-cast-elimination -split-input-file | FileCheck %s

func.func @simplify_if(%arg0: i1) -> memref<10xf32>{
  %0 = scf.if %arg0  -> memref<10xf32, strided<[?], offset: ?>> {
    %0 = memref.alloc() : memref<10xf32>
    %1 = memref.cast %0 : memref<10xf32> to memref<10xf32, strided<[?], offset: ?>>
    scf.yield %1 : memref<10xf32, strided<[?], offset:?>>
  } else {
    %0 = memref.alloc() : memref<10xf32>
    %1 = memref.cast %0 : memref<10xf32> to memref<10xf32, strided<[?], offset: ?>>
    scf.yield %1 : memref<10xf32, strided<[?], offset:?>>
  }
  %1 = memref.cast %0 : memref<10xf32, strided<[?], offset: ?>> to memref<10xf32>
  return %1 : memref<10xf32>
}

// CHECK-LABEL: @simplify_if
//  CHECK-SAME: (%[[arg0:.+]]: i1) -> memref<10xf32> {
//       CHECK:     %[[v0:.+]] = scf.if %[[arg0]] -> (memref<10xf32>) {
//       CHECK:       %[[alloc:.+]] = memref.alloc() : memref<10xf32>
//       CHECK:       scf.yield %[[alloc]] : memref<10xf32>
//       CHECK:       %[[alloc:.+]] = memref.alloc() : memref<10xf32>
//       CHECK:       scf.yield %[[alloc]] : memref<10xf32>
//       CHECK:     return %[[v0]] : memref<10xf32>

// -----

func.func @simplify_if_multi_result(%arg0: i1) -> (memref<10xf32>, memref<10xf32, strided<[?], offset: ?>>){
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0:3 = scf.if %arg0  -> (memref<10xf32, strided<[?], offset: ?>>, index, memref<10xf32, strided<[?], offset: ?>>) {
    %0 = memref.alloc() : memref<10xf32>
    %1 = memref.alloc() : memref<10xf32>
    %2 = memref.cast %0 : memref<10xf32> to memref<10xf32, strided<[?], offset: ?>>
    %3 = memref.cast %1 : memref<10xf32> to memref<10xf32, strided<[?], offset: ?>>
    scf.yield %2, %c1, %3 : memref<10xf32, strided<[?], offset:?>>, index, memref<10xf32, strided<[?], offset:?>>
  } else {
    %0 = memref.alloc() : memref<10xf32>
    %1 = memref.cast %0 : memref<10xf32> to memref<10xf32, strided<[?], offset: ?>>
    scf.yield %1, %c2, %1 : memref<10xf32, strided<[?], offset:?>>, index, memref<10xf32, strided<[?], offset:?>>
  }
  %2 = memref.cast %0#0 : memref<10xf32, strided<[?], offset: ?>> to memref<10xf32>
  return %2, %0#2 : memref<10xf32>, memref<10xf32, strided<[?], offset: ?>>
}

// CHECK-LABEL: @simplify_if_multi_result
//  CHECK-SAME: (%[[arg0:.+]]: i1)
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[v0:.+]]:3 = scf.if %[[arg0]] -> (memref<10xf32>, index, memref<10xf32>) {
//       CHECK:       %[[alloc:.+]] = memref.alloc() : memref<10xf32>
//       CHECK:       %[[alloc_0:.+]] = memref.alloc() : memref<10xf32>
//       CHECK:       scf.yield %[[alloc]], %[[c1]], %[[alloc_0]] : memref<10xf32>, index, memref<10xf32>
//       CHECK:     } else {
//       CHECK:       %[[alloc:.+]] = memref.alloc() : memref<10xf32>
//       CHECK:       scf.yield %[[alloc]], %[[c2]], %[[alloc]] : memref<10xf32>, index, memref<10xf32>
//       CHECK:     %[[cast:.+]] = memref.cast %[[v0]]#2 : memref<10xf32> to memref<10xf32, strided<[?], offset: ?>>
//       CHECK:     return %[[v0]]#0, %[[cast]] : memref<10xf32>, memref<10xf32, strided<[?], offset: ?>>

// -----

func.func @simplify_if_negative(%arg0: i1, %arg1: index) -> memref<10xf32>{
  %0 = scf.if %arg0  -> memref<10xf32, strided<[?], offset: ?>> {
    %0 = memref.alloc(%arg1) : memref<?xf32>
    %1 = memref.cast %0 : memref<?xf32> to memref<10xf32, strided<[?], offset: ?>>
    scf.yield %1 : memref<10xf32, strided<[?], offset:?>>
  } else {
    %0 = memref.alloc() : memref<10xf32>
    %1 = memref.cast %0 : memref<10xf32> to memref<10xf32, strided<[?], offset: ?>>
    scf.yield %1 : memref<10xf32, strided<[?], offset:?>>
  }
  %1 = memref.cast %0 : memref<10xf32, strided<[?], offset: ?>> to memref<10xf32>
  return %1 : memref<10xf32>
}

// CHECK-LABEL: @simplify_if_negative

//       CHECK:     scf.if
//       CHECK:       memref.alloc
//       CHECK:       memref.cast
//       CHECK:       scf.yield
//       CHECK:     else
//       CHECK:       memref.alloc
//       CHECK:       memref.cast
//       CHECK:       scf.yield
