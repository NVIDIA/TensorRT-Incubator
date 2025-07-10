// REQUIRES: host-has-at-least-1-gpus
// RUN: mlir-tensorrt-opt %s -convert-memref-to-cuda -convert-cuda-to-executor -executor-lowering-pipeline \
// RUN:   | mlir-tensorrt-translate -mlir-to-runtime-executable \
// RUN:   | mlir-tensorrt-runner -input-type=rtexe -features=core,cuda | FileCheck %s

func.func @run_with_shape_2d(%arg0: memref<?xindex>, %arg1: memref<2xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %reshaped = memref.reshape %arg0 (%arg1) : (memref<?xindex>, memref<2xindex>) -> (memref<?x?xindex>)

  %el0 = memref.load %arg1[%c0] : memref<2xindex>
  %el1 = memref.load %arg1[%c1] : memref<2xindex>

  scf.for %i = %c0 to %el0 step %c1 {
    scf.for %j = %c0 to %el1 step %c1 {
      %loaded = memref.load %reshaped[%i, %j] : memref<?x?xindex>
      executor.print "result[%d, %d] = %d"(%i, %j, %loaded : index, index, index)
    }
  }
  return
}


func.func @main() -> index {
  %c8 = arith.constant 8 : index
  %1 = memref.alloc(%c8) : memref<?xindex>
  %2 = memref.alloc() : memref<2xindex>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  memref.store %c2, %2[%c0] : memref<2xindex>
  memref.store %c4, %2[%c1] : memref<2xindex>

  %size = memref.dim %1, %c0 : memref<?xindex>
  scf.for %i = %c0 to %size step %c1 {
    memref.store %i, %1[%i] : memref<?xindex>
  }

  func.call @run_with_shape_2d(%1, %2) : (memref<?xindex>, memref<2xindex>) -> ()

  return %c0 : index
}


// CHECK-LABEL: result[0, 0] = 0
//       CHECK: result[0, 1] = 1
//       CHECK: result[0, 2] = 2
//       CHECK: result[0, 3] = 3
//       CHECK: result[1, 0] = 4
//       CHECK: result[1, 1] = 5
//       CHECK: result[1, 2] = 6
//       CHECK: result[1, 3] = 7
