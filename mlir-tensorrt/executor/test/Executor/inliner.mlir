// RUN: executor-opt %s -split-input-file -inline | FileCheck %s

executor.global @global3 constant : vector<10xf32> {
  %0 = arith.constant dense<0.0> : vector<10xf32>
  executor.return %0 : vector<10xf32>
}

func.func private @global_get() -> vector<10xf32> {
  %0 = executor.get_global @global3 : vector<10xf32>
  return %0 : vector<10xf32>
}

executor.func private @enqueue_variadic(i32, ...)
executor.func private @enqueue_non_variadic(i32, !executor.ptr<device>)

func.func public @test_inliner(%arg0: i32, %arg1: !executor.ptr<device>) -> vector<10xf32> {
  executor.call @enqueue_variadic(%arg0, %arg1) : (i32, !executor.ptr<device>) -> ()
  executor.call @enqueue_non_variadic(%arg0, %arg1) : (i32, !executor.ptr<device>) -> ()
  %0 = func.call @global_get() : () -> (vector<10xf32>)
  return %0 : vector<10xf32>
}

//       CHECK:   executor.global @global3 constant : vector<10xf32>
//       CHECK:     %[[cst:.+]] = arith.constant dense<0.000000e+00> : vector<10xf32>
//       CHECK:     executor.return %[[cst]] : vector<10xf32>
//       CHECK:   executor.func private @enqueue_variadic(i32, ...)
//       CHECK:   executor.func private @enqueue_non_variadic(i32, !executor.ptr<device>)
//       CHECK:   func.func public @test_inliner(%[[arg0:.+]]: i32, %[[arg1:.+]]: !executor.ptr<device>) -> vector<10xf32> {
//       CHECK:     executor.call @enqueue_variadic(%[[arg0]], %[[arg1]]) : (i32, !executor.ptr<device>) -> ()
//       CHECK:     executor.call @enqueue_non_variadic(%[[arg0]], %[[arg1]]) : (i32, !executor.ptr<device>) -> ()
//       CHECK:     %[[v0:.+]] = executor.get_global @global3 : vector<10xf32>
//       CHECK:     return %[[v0]] : vector<10xf32>
