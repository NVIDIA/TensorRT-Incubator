// RUN: executor-opt %s -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-lua \
// RUN:   | executor-runner -input-type=lua -features=core | FileCheck %s

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

func.func @test_forwarded_args_in_entrybranch(%arg0: i64, %arg1: i64, %cond: i1) -> i64 {
  %c1 = executor.constant 1 : i64
  cf.cond_br %cond, ^bb1(%arg0: i64), ^bb2(%arg1: i64)
^bb1(%arg2: i64):  // pred: ^bb0
  %1 = executor.addi %c1, %arg2 : i64
  cf.br ^bb3(%1 : i64)
^bb2(%arg3: i64):  // pred: ^bb0
  %2 = executor.addi %c1, %arg3 : i64
  cf.br ^bb3(%2 : i64)
^bb3(%4: i64):  // 2 preds: ^bb1, ^bb2
  executor.print "test_forwarded_args_in_entrybranch(%d, %d, %d) = %d"(
    %arg0, %arg1, %cond, %4 : i64, i64, i1, i64
  )
  return %4 : i64
}

func.func @test_cf_switch(%arg0: i64, %arg1: i64) -> i64 {
  cf.switch %arg0 : i64, [
    default: ^bb1(%arg1 : i64),
    1: ^bb2(%arg1 : i64),
    2: ^bb3(%arg1 : i64)
  ]
  ^bb1(%arg2: i64):
    %1 = executor.addi %arg2, %arg2 : i64
    cf.br ^bb4(%1 : i64)
  ^bb2(%arg3: i64):
    %2 = executor.muli %arg3, %arg3 : i64
    cf.br ^bb4(%2 : i64)
  ^bb3(%arg4: i64):
    cf.br ^bb4(%arg4 : i64)
  ^bb4(%arg5: i64):
    executor.print "test_cf_switch(%d, %d) = %d"(%arg0, %arg1, %arg5 : i64, i64, i64)
    return %arg5 : i64
}

func.func @main() -> i64 {
  %c0 = executor.constant 0 : i64
  %c0_index = executor.constant 0 : index
  %c10 = executor.constant 10 : index
  %c1 = executor.constant 1 : index
  func.call @test_for(%c0_index, %c10, %c1) : (index, index, index) -> ()

  %c0_i1 = executor.constant 0 : i1
  %c1_i1 = executor.constant 1 : i1
  %c0_i64 = executor.constant 0 : i64
  %c1_i64 = executor.constant 1 : i64
  %c2_i64 = executor.constant 2 : i64
  %c3_i64 = executor.constant 3 : i64
  func.call @test_forwarded_args_in_entrybranch(%c0_i64, %c1_i64, %c0_i1)
    : (i64, i64, i1) -> (i64)
  func.call @test_forwarded_args_in_entrybranch(%c0_i64, %c1_i64, %c1_i1)
    : (i64, i64, i1) -> (i64)
  func.call @test_cf_switch(%c1_i64, %c2_i64) : (i64, i64) -> (i64)
  func.call @test_cf_switch(%c2_i64, %c2_i64) : (i64, i64) -> (i64)
  func.call @test_cf_switch(%c3_i64, %c3_i64) : (i64, i64) -> (i64)

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

// CHECK: test_forwarded_args_in_entrybranch(0, 1, 0) = 2
// CHECK: test_forwarded_args_in_entrybranch(0, 1, 1) = 1

// CHECK: test_cf_switch(1, 2) = 4
// CHECK: test_cf_switch(2, 2) = 2
// CHECK: test_cf_switch(3, 3) = 6
