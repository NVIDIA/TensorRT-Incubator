// RUN: executor-opt %s --executor-generate-abi-wrappers -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable \
// RUN:   | executor-runner -input-type=rtexe --features=core | FileCheck %s

// RUN: executor-opt %s --executor-generate-abi-wrappers -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable -lua-translation-block-arg-coalescing=false \
// RUN:   | executor-runner -input-type=rtexe --features=core | FileCheck %s

func.func @test_for(%lb: index, %ub: index, %step: index) {
  %c0 = executor.constant 0 : index
  %c1 = executor.constant 1 : index
  %0 = scf.for %i = %lb to %ub step %step iter_args(%iter = %c0) -> index {
    %0 = scf.for %j = %lb to %ub step %step iter_args(%iter1 = %iter) -> index {
      %acc = executor.addi %iter1, %c1 : index
      executor.print "i = %d, j = %d, acc = %d"(%i, %j, %acc : index, index, index)
      scf.yield %acc : index
    }
    scf.yield %0 : index
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

// CHECK-LABEL: i = 0, j = 0, acc = 1
// CHECK-NEXT: i = 0, j = 1, acc = 2
// CHECK-NEXT: i = 0, j = 2, acc = 3
// CHECK-NEXT: i = 0, j = 3, acc = 4
// CHECK-NEXT: i = 0, j = 4, acc = 5
// CHECK-NEXT: i = 0, j = 5, acc = 6
// CHECK-NEXT: i = 0, j = 6, acc = 7
// CHECK-NEXT: i = 0, j = 7, acc = 8
// CHECK-NEXT: i = 0, j = 8, acc = 9
// CHECK-NEXT: i = 0, j = 9, acc = 10
// CHECK-NEXT: i = 1, j = 0, acc = 11
// CHECK-NEXT: i = 1, j = 1, acc = 12
// CHECK-NEXT: i = 1, j = 2, acc = 13
// CHECK-NEXT: i = 1, j = 3, acc = 14
// CHECK-NEXT: i = 1, j = 4, acc = 15
// CHECK-NEXT: i = 1, j = 5, acc = 16
// CHECK-NEXT: i = 1, j = 6, acc = 17
// CHECK-NEXT: i = 1, j = 7, acc = 18
// CHECK-NEXT: i = 1, j = 8, acc = 19
// CHECK-NEXT: i = 1, j = 9, acc = 20
// CHECK-NEXT: i = 2, j = 0, acc = 21
// CHECK-NEXT: i = 2, j = 1, acc = 22
// CHECK-NEXT: i = 2, j = 2, acc = 23
// CHECK-NEXT: i = 2, j = 3, acc = 24
// CHECK-NEXT: i = 2, j = 4, acc = 25
// CHECK-NEXT: i = 2, j = 5, acc = 26
// CHECK-NEXT: i = 2, j = 6, acc = 27
// CHECK-NEXT: i = 2, j = 7, acc = 28
// CHECK-NEXT: i = 2, j = 8, acc = 29
// CHECK-NEXT: i = 2, j = 9, acc = 30
// CHECK-NEXT: i = 3, j = 0, acc = 31
// CHECK-NEXT: i = 3, j = 1, acc = 32
// CHECK-NEXT: i = 3, j = 2, acc = 33
// CHECK-NEXT: i = 3, j = 3, acc = 34
// CHECK-NEXT: i = 3, j = 4, acc = 35
// CHECK-NEXT: i = 3, j = 5, acc = 36
// CHECK-NEXT: i = 3, j = 6, acc = 37
// CHECK-NEXT: i = 3, j = 7, acc = 38
// CHECK-NEXT: i = 3, j = 8, acc = 39
// CHECK-NEXT: i = 3, j = 9, acc = 40
// CHECK-NEXT: i = 4, j = 0, acc = 41
// CHECK-NEXT: i = 4, j = 1, acc = 42
// CHECK-NEXT: i = 4, j = 2, acc = 43
// CHECK-NEXT: i = 4, j = 3, acc = 44
// CHECK-NEXT: i = 4, j = 4, acc = 45
// CHECK-NEXT: i = 4, j = 5, acc = 46
// CHECK-NEXT: i = 4, j = 6, acc = 47
// CHECK-NEXT: i = 4, j = 7, acc = 48
// CHECK-NEXT: i = 4, j = 8, acc = 49
// CHECK-NEXT: i = 4, j = 9, acc = 50
// CHECK-NEXT: i = 5, j = 0, acc = 51
// CHECK-NEXT: i = 5, j = 1, acc = 52
// CHECK-NEXT: i = 5, j = 2, acc = 53
// CHECK-NEXT: i = 5, j = 3, acc = 54
// CHECK-NEXT: i = 5, j = 4, acc = 55
// CHECK-NEXT: i = 5, j = 5, acc = 56
// CHECK-NEXT: i = 5, j = 6, acc = 57
// CHECK-NEXT: i = 5, j = 7, acc = 58
// CHECK-NEXT: i = 5, j = 8, acc = 59
// CHECK-NEXT: i = 5, j = 9, acc = 60
// CHECK-NEXT: i = 6, j = 0, acc = 61
// CHECK-NEXT: i = 6, j = 1, acc = 62
// CHECK-NEXT: i = 6, j = 2, acc = 63
// CHECK-NEXT: i = 6, j = 3, acc = 64
// CHECK-NEXT: i = 6, j = 4, acc = 65
// CHECK-NEXT: i = 6, j = 5, acc = 66
// CHECK-NEXT: i = 6, j = 6, acc = 67
// CHECK-NEXT: i = 6, j = 7, acc = 68
// CHECK-NEXT: i = 6, j = 8, acc = 69
// CHECK-NEXT: i = 6, j = 9, acc = 70
// CHECK-NEXT: i = 7, j = 0, acc = 71
// CHECK-NEXT: i = 7, j = 1, acc = 72
// CHECK-NEXT: i = 7, j = 2, acc = 73
// CHECK-NEXT: i = 7, j = 3, acc = 74
// CHECK-NEXT: i = 7, j = 4, acc = 75
// CHECK-NEXT: i = 7, j = 5, acc = 76
// CHECK-NEXT: i = 7, j = 6, acc = 77
// CHECK-NEXT: i = 7, j = 7, acc = 78
// CHECK-NEXT: i = 7, j = 8, acc = 79
// CHECK-NEXT: i = 7, j = 9, acc = 80
// CHECK-NEXT: i = 8, j = 0, acc = 81
// CHECK-NEXT: i = 8, j = 1, acc = 82
// CHECK-NEXT: i = 8, j = 2, acc = 83
// CHECK-NEXT: i = 8, j = 3, acc = 84
// CHECK-NEXT: i = 8, j = 4, acc = 85
// CHECK-NEXT: i = 8, j = 5, acc = 86
// CHECK-NEXT: i = 8, j = 6, acc = 87
// CHECK-NEXT: i = 8, j = 7, acc = 88
// CHECK-NEXT: i = 8, j = 8, acc = 89
// CHECK-NEXT: i = 8, j = 9, acc = 90
// CHECK-NEXT: i = 9, j = 0, acc = 91
// CHECK-NEXT: i = 9, j = 1, acc = 92
// CHECK-NEXT: i = 9, j = 2, acc = 93
// CHECK-NEXT: i = 9, j = 3, acc = 94
// CHECK-NEXT: i = 9, j = 4, acc = 95
// CHECK-NEXT: i = 9, j = 5, acc = 96
// CHECK-NEXT: i = 9, j = 6, acc = 97
// CHECK-NEXT: i = 9, j = 7, acc = 98
// CHECK-NEXT: i = 9, j = 8, acc = 99
// CHECK-NEXT: i = 9, j = 9, acc = 100
// CHECK-NEXT: test_for = 100
