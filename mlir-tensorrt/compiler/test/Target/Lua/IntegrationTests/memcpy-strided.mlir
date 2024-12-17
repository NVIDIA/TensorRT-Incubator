// REQUIRES: host-has-at-least-1-gpus
// RUN: mlir-tensorrt-opt %s -convert-memref-to-cuda -convert-plan-to-executor -convert-cuda-to-executor -executor-lowering-pipeline \
// RUN:   | mlir-tensorrt-translate -mlir-to-runtime-executable \
// RUN:   | mlir-tensorrt-runner -input-type=rtexe | FileCheck %s

func.func @main() -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c99 = arith.constant 99.9 : f32

  %dev = memref.alloc() : memref<4x8xf32, #plan.memory_space<device>>
  %host = memref.alloc() : memref<4x8xf32, #plan.memory_space<host>>
  %host2 = memref.alloc() : memref<4x8xf32, #plan.memory_space<host>>
  %host3 = memref.alloc() : memref<4x8xf32, #plan.memory_space<host>>

  scf.for %j = %c0 to %c4 step %c1 {
    scf.for %i = %c0 to %c8 step %c1 {
      %isi = arith.index_cast %i : index  to i32
      %jsi = arith.index_cast %j : index to i32
      %if = arith.sitofp %isi : i32 to f32
      %jf = arith.sitofp %jsi : i32 to f32
      %st = arith.addf %if, %jf : f32
      memref.store %st, %host[%j, %i] : memref<4x8xf32, #plan.memory_space<host>>
    }
  }

  scf.for %i = %c0 to %c8 step %c1 {
    %host_subview = memref.subview %host[0, %i][4, 1][1, 1] : memref<4x8xf32, #plan.memory_space<host>> to
      memref<4xf32, strided<[8], offset: ?>, #plan.memory_space<host>>
    %host2_subview = memref.subview %host2[0, %i][4, 1][1, 1] : memref<4x8xf32, #plan.memory_space<host>> to
      memref<4xf32, strided<[8], offset: ?>, #plan.memory_space<host>>
    %host3_subview = memref.subview %host3[0, %i][4, 1][1, 1] : memref<4x8xf32, #plan.memory_space<host>> to
      memref<4xf32, strided<[8], offset: ?>, #plan.memory_space<host>>
    %dev_subview = memref.subview %dev[0, %i][4, 1][1, 1] : memref<4x8xf32, #plan.memory_space<device>> to
      memref<4xf32, strided<[8], offset: ?>, #plan.memory_space<device>>
    memref.copy %host_subview, %dev_subview : memref<4xf32, strided<[8], offset: ?>, #plan.memory_space<host>>
      to memref<4xf32, strided<[8], offset: ?>, #plan.memory_space<device>>
    memref.copy %dev_subview, %host2_subview :
      memref<4xf32, strided<[8], offset: ?>, #plan.memory_space<device>> to
      memref<4xf32, strided<[8], offset: ?>, #plan.memory_space<host>>
    memref.copy %host2_subview, %host3_subview :
      memref<4xf32, strided<[8], offset: ?>, #plan.memory_space<host>> to
      memref<4xf32, strided<[8], offset: ?>, #plan.memory_space<host>>
  }

  scf.for %j = %c0 to %c4 step %c1 {
    scf.for %i = %c0 to %c8 step %c1 {
      %ld = memref.load %host3[%j, %i] : memref<4x8xf32, #plan.memory_space<host>>
      executor.print "result[%d, %d] = %f"(%j, %i, %ld : index, index, f32)
    }
  }

  return %c0 : index
}

// CHECK: result[0, 0] = 0.000000
// CHECK: result[0, 1] = 1.000000
// CHECK: result[0, 2] = 2.000000
// CHECK: result[0, 3] = 3.000000
// CHECK: result[0, 4] = 4.000000
// CHECK: result[0, 5] = 5.000000
// CHECK: result[0, 6] = 6.000000
// CHECK: result[0, 7] = 7.000000
// CHECK: result[1, 0] = 1.000000
// CHECK: result[1, 1] = 2.000000
// CHECK: result[1, 2] = 3.000000
// CHECK: result[1, 3] = 4.000000
// CHECK: result[1, 4] = 5.000000
// CHECK: result[1, 5] = 6.000000
// CHECK: result[1, 6] = 7.000000
// CHECK: result[1, 7] = 8.000000
// CHECK: result[2, 0] = 2.000000
// CHECK: result[2, 1] = 3.000000
// CHECK: result[2, 2] = 4.000000
// CHECK: result[2, 3] = 5.000000
// CHECK: result[2, 4] = 6.000000
// CHECK: result[2, 5] = 7.000000
// CHECK: result[2, 6] = 8.000000
// CHECK: result[2, 7] = 9.000000
// CHECK: result[3, 0] = 3.000000
// CHECK: result[3, 1] = 4.000000
// CHECK: result[3, 2] = 5.000000
// CHECK: result[3, 3] = 6.000000
// CHECK: result[3, 4] = 7.000000
// CHECK: result[3, 5] = 8.000000
// CHECK: result[3, 6] = 9.000000
// CHECK: result[3, 7] = 10.000000
