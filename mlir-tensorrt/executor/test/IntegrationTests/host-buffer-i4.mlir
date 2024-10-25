// RUN: executor-opt %s -test-executor-bufferization-pipeline -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable \
// RUN:   | executor-runner -input-type=rtexe | FileCheck %s

!memref_type = memref<4xi4, strided<[?], offset: ?>, #executor.memory_type<host>>

func.func private @print_tensor(%arg0: !memref_type) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %el = memref.load %arg0 [%i] : !memref_type
    %el_i32 = arith.extsi %el : i4 to i32
    executor.print "[%d] = %d"(%i, %el_i32 : index, i32)
  }
  return
}

func.func @main() -> i32 {

  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi4>
  %1 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi4>
  %2 = bufferization.alloc_tensor() : tensor<4xi4>
  %alloc1 = bufferization.alloc_tensor() : tensor<4xi4>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  %result:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %2, %acc1 = %alloc1) -> (tensor<4xi4>, tensor<4xi4>) {
    %lhs = tensor.extract %0[%i] : tensor<4xi4>
    %rhs = tensor.extract %1[%i] : tensor<4xi4>
    %add = arith.addi %lhs, %rhs : i4
    %sub = arith.subi %lhs, %rhs : i4

    %updated0 = tensor.insert %add into %acc[%i] : tensor<4xi4>
    %updated1 = tensor.insert %sub into %acc1[%i] : tensor<4xi4>

    scf.yield %updated0, %updated1 : tensor<4xi4>, tensor<4xi4>
  }

  %memref = bufferization.to_memref %result#0 read_only : tensor<4xi4> -> !memref_type
  func.call @print_tensor(%memref) : (!memref_type) -> ()
  %memref1 = bufferization.to_memref %result#1 read_only : tensor<4xi4> -> !memref_type
  func.call @print_tensor(%memref1) : (!memref_type) -> ()

  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}

//      CHECK: [0] = 2
// CHECK-NEXT: [1] = 4
// CHECK-NEXT: [2] = 6
// CHECK-NEXT: [3] = -8

// CHECK-NEXT: [0] = 0
// CHECK-NEXT: [1] = 0
// CHECK-NEXT: [2] = 0
// CHECK-NEXT: [3] = 0
