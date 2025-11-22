// RUN: executor-opt %s --executor-generate-abi-wrappers -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable \
// RUN:   | executor-runner -input-type=rtexe -features=core | FileCheck %s

func.func @main() -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c99 = arith.constant 99.9 : f32


  %host = memref.alloc() : memref<4x8xf32, #executor.memory_type<host>>
  %host1 = memref.alloc() : memref<4x8xf32, #executor.memory_type<host>>

  scf.for %j = %c0 to %c4 step %c1 {
    scf.for %i = %c0 to %c8 step %c1 {
      %isi = arith.index_cast %i : index  to i32
      %jsi = arith.index_cast %j : index to i32
      %if = arith.sitofp %isi : i32 to f32
      %jf = arith.sitofp %jsi : i32 to f32
      %st = arith.addf %if, %jf : f32
      memref.store %st, %host[%j, %i] : memref<4x8xf32, #executor.memory_type<host>>
    }
  }

  scf.for %i = %c0 to %c8 step %c1 {
    %host_subview = memref.subview %host[0, %i][4, 1][1, 1] : memref<4x8xf32, #executor.memory_type<host>> to
      memref<4xf32, strided<[8], offset: ?>, #executor.memory_type<host>>

    %col_perm = executor.bitwise_xori %i, %c1 : index
    %host1_subview = memref.subview %host1[0, %col_perm][4, 1][1, 1] : memref<4x8xf32, #executor.memory_type<host>> to
      memref<4xf32, strided<[8], offset: ?>, #executor.memory_type<host>>

    memref.copy %host_subview, %host1_subview : memref<4xf32, strided<[8], offset: ?>, #executor.memory_type<host>>
      to memref<4xf32, strided<[8], offset: ?>, #executor.memory_type<host>>
  }

  scf.for %j = %c0 to %c4 step %c1 {
    scf.for %i = %c0 to %c8 step %c1 {
      %ld = memref.load %host1[%j, %i] : memref<4x8xf32, #executor.memory_type<host>>
      executor.print "result[%d, %d] = %f"(%j, %i, %ld : index, index, f32)
    }
  }

  return %c0 : index
}

//      CHECK: result[0, 0] = 1.000000
// CHECK-NEXT: result[0, 1] = 0.000000
// CHECK-NEXT: result[0, 2] = 3.000000
// CHECK-NEXT: result[0, 3] = 2.000000
// CHECK-NEXT: result[0, 4] = 5.000000
// CHECK-NEXT: result[0, 5] = 4.000000
// CHECK-NEXT: result[0, 6] = 7.000000
// CHECK-NEXT: result[0, 7] = 6.000000
// CHECK-NEXT: result[1, 0] = 2.000000
// CHECK-NEXT: result[1, 1] = 1.000000
// CHECK-NEXT: result[1, 2] = 4.000000
// CHECK-NEXT: result[1, 3] = 3.000000
// CHECK-NEXT: result[1, 4] = 6.000000
// CHECK-NEXT: result[1, 5] = 5.000000
// CHECK-NEXT: result[1, 6] = 8.000000
// CHECK-NEXT: result[1, 7] = 7.000000
// CHECK-NEXT: result[2, 0] = 3.000000
// CHECK-NEXT: result[2, 1] = 2.000000
// CHECK-NEXT: result[2, 2] = 5.000000
// CHECK-NEXT: result[2, 3] = 4.000000
// CHECK-NEXT: result[2, 4] = 7.000000
// CHECK-NEXT: result[2, 5] = 6.000000
// CHECK-NEXT: result[2, 6] = 9.000000
// CHECK-NEXT: result[2, 7] = 8.000000
// CHECK-NEXT: result[3, 0] = 4.000000
// CHECK-NEXT: result[3, 1] = 3.000000
// CHECK-NEXT: result[3, 2] = 6.000000
// CHECK-NEXT: result[3, 3] = 5.000000
// CHECK-NEXT: result[3, 4] = 8.000000
// CHECK-NEXT: result[3, 5] = 7.000000
// CHECK-NEXT: result[3, 6] = 10.000000
// CHECK-NEXT: result[3, 7] = 9.000000
