// REQUIRES: debug-print
// RUN: executor-opt -executor-allocs-to-globals %s -split-input-file \
// RUN:  -debug -debug-only=executor-allocs-to-globals 2>&1 \
// RUN:  | FileCheck %s --check-prefix=DEBUG

func.func @test_disjoint_allocations() {
  %cst0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index

  %0 = memref.alloc() : memref<128xf32>
  memref.store %cst0, %0[%c0] : memref<128xf32>
  memref.dealloc %0 : memref<128xf32>

  %4 = memref.alloc() : memref<128xf32>
  memref.store %cst0, %4[%c0] : memref<128xf32>
  memref.dealloc %4 : memref<128xf32>
  return
}

// DEBUG-LABEL: func ::test_disjoint_allocations:
// DEBUG-DAG: 128xf32             _____x___
// DEBUG-DAG: 128xf32             __x______
// DEBUG-NEXT: memory used: 512
// DEBUG-NEXT: memory saved: 512

//       CHECK:   executor.global @workspace_0 : memref<128xf32> {
//  CHECK-NEXT:     %[[alloc:.+]] = memref.alloc() : memref<128xf32>
//  CHECK-NEXT:     executor.return %[[alloc]] : memref<128xf32>
// CHECK-LABEL: @test_disjoint_allocations
//       CHECK:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[v0:.+]] = executor.get_global @workspace_0 : memref<128xf32>
//       CHECK:     memref.store %[[cst]], %[[v0]][%[[c0]]] : memref<128xf32>
//       CHECK:     %[[v1:.+]] = executor.get_global @workspace_0 : memref<128xf32>
//       CHECK:     memref.store %[[cst]], %[[v1]][%[[c0]]] : memref<128xf32>
//       CHECK:     return

// -----

func.func @test_disjoint_index_allocations() {
  %cst0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index

  %0 = memref.alloc() : memref<128xindex>
  memref.store %c0, %0[%c0] : memref<128xindex>
  memref.dealloc %0 : memref<128xindex>

  %4 = memref.alloc() : memref<128xindex>
  memref.store %c0, %4[%c0] : memref<128xindex>
  memref.dealloc %4 : memref<128xindex>
  return
}

// DEBUG-LABEL: func ::test_disjoint_index_allocations:
//  DEBUG-DAG: 128xindex           _____x___
//  DEBUG-DAG: 128xindex           __x______
//      DEBUG: [executor-allocs-to-globals]  memory used: 1024
// DEBUG-NEXT: [executor-allocs-to-globals]  memory saved: 1024

//       CHECK:   executor.global @workspace_0 : memref<128xindex>
//       CHECK:     %[[alloc:.+]] = memref.alloc() : memref<128xindex>
//       CHECK:     executor.return %[[alloc]] : memref<128xindex>
// CHECK-LABEL: @test_disjoint_index_allocations
//       CHECK:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[v0:.+]] = executor.get_global @workspace_0 : memref<128xindex>
//       CHECK:     memref.store %[[c0]], %[[v0]][%[[c0]]] : memref<128xindex>
//       CHECK:     %[[v1:.+]] = executor.get_global @workspace_0 : memref<128xindex>
//       CHECK:     memref.store %[[c0]], %[[v1]][%[[c0]]] : memref<128xindex>

// -----

func.func @test_aliased_non_disjoint_allocations() {
  %cst0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index

  %0 = memref.alloc() : memref<128xf32>
  %00 = memref.subview %0[0][64][1] : memref<128xf32> to memref<64xf32>

  %1 = memref.alloc() : memref<128xf32>
  memref.store %cst0, %1[%c0] : memref<128xf32>
  memref.store %cst0, %00[%c0] : memref<64xf32>

  memref.dealloc %0 : memref<128xf32>
  memref.dealloc %1 : memref<128xf32>
  return
}

// DEBUG-LABEL: ::test_aliased_non_disjoint_allocations:
//   DEBUG-DAG: 128xf32             ____x_____
//   DEBUG-DAG: 128xf32             __xxxx____
//  DEBUG-NEXT: memory used: 1024
//  DEBUG-NEXT: memory saved: 0

//       CHECK:   executor.global @{{.+}} : memref<128xf32>
//       CHECK:     %[[alloc:.+]] = memref.alloc() : memref<128xf32>
//       CHECK:     executor.return %[[alloc]] : memref<128xf32>
//       CHECK:   executor.global @{{.+}} : memref<128xf32>
//       CHECK:     %[[alloc:.+]] = memref.alloc() : memref<128xf32>
//       CHECK:     executor.return %[[alloc]] : memref<128xf32>
// CHECK-LABEL: @test_aliased_non_disjoint_allocations
//       CHECK:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[v0:.+]] = executor.get_global @{{.+}}
//       CHECK:     %[[subview:.+]] = memref.subview %[[v0]][0] [64] [1]
//       CHECK:     %[[v1:.+]] = executor.get_global @{{.+}} : memref<128xf32>
//       CHECK:     memref.store %[[cst]], %[[v1]][%[[c0]]] : memref<128xf32>
//       CHECK:     memref.store %[[cst]], %[[subview]][%[[c0]]] : memref<64xf32>
