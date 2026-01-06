// REQUIRES: host-has-at-least-1-gpus
// RUN: mlir-tensorrt-compiler %s --phase-start=lowering --disable-all-extensions -o - \
// RUN:   | mlir-tensorrt-runner -input-type=rtexe -features=core,cuda | FileCheck %s

func.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c99_i32 = arith.constant 99 : i32
  %c32 = arith.constant 32.09 : f32
  %c0_f32 = arith.constant 0.0 : f32

  %0 = memref.alloc() : memref<i32, #plan.memory_space<host>>
  %1 = memref.alloc() : memref<i32, #plan.memory_space<host>>
  memref.store %c99_i32, %0[] : memref<i32, #plan.memory_space<host>>
  memref.store %c0_i32, %1[] : memref<i32, #plan.memory_space<host>>
  memref.copy %0, %1 : memref<i32, #plan.memory_space<host>> to memref<i32, #plan.memory_space<host>>
  %load = memref.load %1[] : memref<i32, #plan.memory_space<host>>
  executor.print "i32 memcpy result = %d"(%load : i32)

  %c0_i1 = arith.constant 0 : i1
  %c1_i1 = arith.constant 1 : i1

  %2 = memref.alloc() : memref<i1, #plan.memory_space<host>>
  %3 = memref.alloc() : memref<i1, #plan.memory_space<host>>
  memref.store %c1_i1, %2[] : memref<i1, #plan.memory_space<host>>
  memref.store %c0_i1, %3[] : memref<i1, #plan.memory_space<host>>
  memref.copy %2, %3 : memref<i1, #plan.memory_space<host>> to memref<i1, #plan.memory_space<host>>
  %load_i1 = memref.load %3[] : memref<i1, #plan.memory_space<host>>
  executor.print "i1 memcpy result = %d"(%load_i1 : i1)

  %4 = memref.alloc() : memref<2x2xi1, #plan.memory_space<host>>
  %5 = memref.alloc() : memref<2x2xi1, #plan.memory_space<host>>
  memref.store %c1_i1, %4[%c1, %c1] : memref<2x2xi1, #plan.memory_space<host>>
  memref.store %c0_i1, %5[%c1, %c1] : memref<2x2xi1, #plan.memory_space<host>>
  memref.copy %4, %5 : memref<2x2xi1, #plan.memory_space<host>> to memref<2x2xi1, #plan.memory_space<host>>
  %load_i1_2x2 = memref.load %5[%c1, %c1] : memref<2x2xi1, #plan.memory_space<host>>
  executor.print "i1 2x2 memcpy result = %d"(%load_i1_2x2 : i1)

  %6 = memref.alloc() : memref<4x8xf32, #plan.memory_space<device>>
  %7 = memref.alloc() : memref<4x8xf32, #plan.memory_space<device>>
  %9 = memref.alloc() : memref<4x8xf32, #plan.memory_space<host>>
  %10 = memref.alloc() : memref<4x8xf32, #plan.memory_space<host>>

  memref.store %c32, %9[%c1, %c1] : memref<4x8xf32, #plan.memory_space<host>>
  memref.store %c0_f32, %10[%c1, %c1] : memref<4x8xf32, #plan.memory_space<host>>
  memref.copy %9, %6 : memref<4x8xf32, #plan.memory_space<host>> to memref<4x8xf32, #plan.memory_space<device>>
  memref.copy %10, %7 : memref<4x8xf32, #plan.memory_space<host>> to memref<4x8xf32, #plan.memory_space<device>>

  memref.copy %6, %7 : memref<4x8xf32, #plan.memory_space<device>> to memref<4x8xf32, #plan.memory_space<device>>
  memref.copy %7, %10 : memref<4x8xf32, #plan.memory_space<device>> to memref<4x8xf32, #plan.memory_space<host>>
  %load_f32 = memref.load %10[%c1, %c1] : memref<4x8xf32, #plan.memory_space<host>>
  executor.print "f32 4x8 memcpy result = %f"(%load_f32 : f32)

  return %c0_i32 : i32
}

// CHECK: i32 memcpy result = 99
// CHECK: i1 memcpy result = 1
// CHECK: i1 2x2 memcpy result = 1
// CHECK: f32 4x8 memcpy result = 32.090000
