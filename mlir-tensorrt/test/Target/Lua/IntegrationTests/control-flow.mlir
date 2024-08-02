// RUN: mlir-tensorrt-opt %s -executor-lowering-pipeline \
// RUN:   | mlir-tensorrt-translate -mlir-to-lua \
// RUN:   | mlir-tensorrt-runner -input-type=lua | FileCheck %s

func.func @test_for(%lb: index, %ub: index, %step: index) {
  %c0 = executor.constant 0 : index
  %0 = scf.for %i = %lb to %ub step %step iter_args(%iter = %c0) -> index {
    %acc = executor.addi %iter, %i : index
    executor.print "i = %d"(%i : index)
    scf.yield %acc : index
  }
  executor.print "test_for = %d"(%0 : index)
  return
}

func.func @main() -> i64 {
  %c0 = executor.constant 0 : i64
  %c0_index = executor.constant 0 : index
  %c10 = executor.constant 10 : index
  %c1 = executor.constant 1 : index
  func.call @test_for(%c0_index, %c10, %c1) : (index, index, index) -> ()
  return %c0 : i64
}

//CHECK-LABEL: i = 0
// CHECK-NEXT: i = 1
// CHECK-NEXT: i = 2
// CHECK-NEXT: i = 3
// CHECK-NEXT: i = 4
// CHECK-NEXT: i = 5
// CHECK-NEXT: i = 6
// CHECK-NEXT: i = 7
// CHECK-NEXT: i = 8
// CHECK-NEXT: i = 9
// CHECK-NEXT: test_for = 45
