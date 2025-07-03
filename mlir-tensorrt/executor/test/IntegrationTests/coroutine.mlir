// RUN: executor-opt %s -split-input-file -executor-lowering-pipeline | \
// RUN: executor-translate -mlir-to-lua -split-input-file --output-split-marker="-- -----" \
// RUN:   | executor-runner -features=core -input-type=lua -split-input-file="-- -----" | FileCheck %s

func.func @coro(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %start = arith.index_cast %arg0 : i32 to index
  %stop = arith.index_cast %arg1 : i32 to index
  %step = arith.constant 1 : index
  scf.for %i = %start to %stop step %step {
    %i_i32 = arith.index_cast %i : index to i32
    executor.coro_yield %i_i32, %i_i32 : i32, i32
  }
  %c1_i32  = arith.constant 1 : i32
  %0 = executor.addi %arg1, %c1_i32 : i32
  return %0, %0  : i32, i32
}

func.func @main() -> (i32) {
  %c0 = executor.constant 0 : i32
  %c10 = executor.constant 10 : i32
  %true = executor.constant 1 : i1
  %c1 = executor.constant 1 : i32
  %coro = executor.coro_create @coro : (i32, i32) -> (i32, i32)
  %r:2 = scf.while (%cond = %true, %arg2 = %c0) : (i1, i32) -> (i1, i32) {
    scf.condition(%cond) %cond, %arg2 : i1, i32
  } do {
  ^bb0(%arg3: i1, %arg4: i32):
    %is_first = executor.icmp <eq> %arg4, %c0 : i32
    %count = executor.addi %arg4, %c1 : i32
    executor.print "resuming coroutine iteration #%d"(%arg4: i32)
    %r:3 = scf.if %is_first -> (i1, i32, i32) {
      executor.print "coroutine first"()
      %r:3 = executor.coro_await %coro(%c0, %c10 : i32, i32) : (i32, i32) -> (i32, i32)
      scf.yield %r#0, %r#1, %r#2 : i1, i32, i32
    } else {
      %r:3 = executor.coro_await %coro() : (i32, i32) -> (i32, i32)
      scf.yield %r#0, %r#1, %r#2 : i1, i32, i32
    }
    scf.if %r#0 {
      executor.print "coroutine yielded %d, %d"(%r#1, %r#2 : i32, i32)
    } else {
      executor.print "coroutine terminated"()
    }
    scf.yield %r#0, %count : i1, i32
  }
  executor.print "coroutine done"()
  return %c0 : i32
}

//      CHECK: resuming coroutine iteration #0
// CHECK-NEXT: coroutine first
// CHECK-NEXT: coroutine yielded 0, 0
// CHECK-NEXT: resuming coroutine iteration #1
// CHECK-NEXT: coroutine yielded 1, 1
// CHECK-NEXT: resuming coroutine iteration #2
// CHECK-NEXT: coroutine yielded 2, 2
// CHECK-NEXT: resuming coroutine iteration #3
// CHECK-NEXT: coroutine yielded 3, 3
// CHECK-NEXT: resuming coroutine iteration #4
// CHECK-NEXT: coroutine yielded 4, 4
// CHECK-NEXT: resuming coroutine iteration #5
// CHECK-NEXT: coroutine yielded 5, 5
// CHECK-NEXT: resuming coroutine iteration #6
// CHECK-NEXT: coroutine yielded 6, 6
// CHECK-NEXT: resuming coroutine iteration #7
// CHECK-NEXT: coroutine yielded 7, 7
// CHECK-NEXT: resuming coroutine iteration #8
// CHECK-NEXT: coroutine yielded 8, 8
// CHECK-NEXT: resuming coroutine iteration #9
// CHECK-NEXT: coroutine yielded 9, 9
// CHECK-NEXT: resuming coroutine iteration #10
// CHECK-NEXT: coroutine yielded 11, 11
// CHECK-NEXT: resuming coroutine iteration #11
// CHECK-NEXT: coroutine terminated
// CHECK-NEXT: done
