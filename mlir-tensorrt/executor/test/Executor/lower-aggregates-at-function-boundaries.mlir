// RUN: executor-opt %s --executor-lower-aggregates-at-function-boundaries=mode=indirect -split-input-file | FileCheck %s --check-prefix=IND
// RUN: executor-opt %s --executor-lower-aggregates-at-function-boundaries=mode=unpacked -split-input-file | FileCheck %s --check-prefix=UNP
// RUN: executor-opt %s --executor-lower-aggregates-at-function-boundaries=mode=direct -split-input-file | FileCheck %s --check-prefix=DIR

func.func @aggregate_arg(%arg0: !executor.table<!executor.ptr<host>, i32> {executor.slot = 0})
    -> (i32 {executor.slot = 1}) {
  %0 = executor.table.get %arg0[1] : !executor.table<!executor.ptr<host>, i32>
  return %0 : i32
}

// IND-LABEL: func.func @aggregate_arg
//  IND-SAME:  (%[[arg0:.+]]: !executor.ptr<host> {executor.slot = 0 : i64})
//  IND-SAME:  -> (i32 {executor.result = 0 : i64, executor.slot = 1 : i64})
//       IND: %[[v0:.+]] = executor.load %[[arg0]]
//       IND: %[[v1:.+]] = executor.table.get %[[v0]][1]
//       IND: return %[[v1]]

// UNP-LABEL: func.func @aggregate_arg
//  UNP-SAME:  (%[[arg0:.+]]: !executor.ptr<host> {executor.slot = 0 : i64}, %[[arg1:.+]]: i32)
//  UNP-SAME:  -> (i32 {executor.result = 0 : i64, executor.slot = 1 : i64})
//       UNP: %[[v0:.+]] = executor.table.create(%[[arg0]], %[[arg1]] :
//       UNP: %[[v1:.+]] = executor.table.get %[[v0]][1]
//       UNP: return %[[v1]]

// DIR-LABEL: func.func @aggregate_arg
//  DIR-SAME:  (%[[arg0:.+]]: !executor.table<!executor.ptr<host>, i32> {executor.slot = 0 : i64})
//  DIR-SAME:  -> (i32 {executor.slot = 1 : i64})
//       DIR: %[[v0:.+]] = executor.table.get %[[arg0]][1]
//       DIR: return %[[v0]]

// -----

func.func @aggregate_result(%arg0: !executor.ptr<host> {executor.slot = 0},
                            %arg1: i32 {executor.slot = 1})
                            -> (!executor.table<!executor.ptr<host>, i32> {executor.slot = 2}) {
  %0 = executor.table.create(%arg0, %arg1 : !executor.ptr<host>, i32) : !executor.table<!executor.ptr<host>, i32>
  return %0 : !executor.table<!executor.ptr<host>, i32>
}

// IND-LABEL: func.func @aggregate_result
//  IND-SAME:  (%[[arg0:.+]]: !executor.ptr<host> {executor.slot = 0 : i64},
//  IND-SAME:   %[[arg1:.+]]: i32 {executor.slot = 1 : i64},
//  IND-SAME:   %[[arg2:.+]]: !executor.ptr<host> {executor.result = 0 : i64, executor.slot = 2 : i64}) {
//       IND: %[[v0:.+]] = executor.table.create(%[[arg0]], %[[arg1]] :
//       IND: executor.store %[[v0]] to %[[arg2]]
//       IND: return

// UNP-LABEL: func.func @aggregate_result
//  UNP-SAME:  (%[[arg0:.+]]: !executor.ptr<host> {executor.slot = 0 : i64},
//  UNP-SAME:   %[[arg1:.+]]: i32 {executor.slot = 1 : i64})
//  UNP-SAME:  -> (!executor.ptr<host> {executor.result = 0 : i64, executor.slot = 2 : i64}, i32) {
//       UNP: %[[v0:.+]] = executor.table.create(%[[arg0]], %[[arg1]] :
//       UNP: %[[v1:.+]] = executor.table.get %[[v0]][0]
//       UNP: %[[v2:.+]] = executor.table.get %[[v0]][1]
//       UNP: return %[[v1]], %[[v2]] :

// DIR-LABEL: func.func @aggregate_result


// -----


// IND-LABEL: func.func @return_table(
//  IND-SAME:  %[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: i32, %[[arg2:.+]]: !executor.ptr<host> {executor.result = 0 : i64}) -> (i32 {executor.result = 1 : i64})
func.func @return_table(%arg0: !executor.ptr<host>, %arg1: i32) -> (!executor.table<!executor.ptr<host>, i32>, i32) {
  %0 = executor.table.create (%arg0, %arg1 : !executor.ptr<host>, i32) : !executor.table<!executor.ptr<host>, i32>
  // IND: %[[v0:.+]] = executor.table.create
  // IND: executor.store %[[v0]] to %[[arg2]]
  // IND: return %[[arg1]] : i32
  return %0, %arg1 : !executor.table<!executor.ptr<host>, i32>, i32
}

// UNP-LABEL: func.func @caller(

// IND-LABEL: func.func @caller(
//  IND-SAME:  %[[arg0:.+]]: !executor.ptr<host> {executor.slot = 0 : i64},
//  IND-SAME:  %[[arg1:.+]]: i32 {executor.slot = 1 : i64},
//  IND-SAME:  %[[arg2:.+]]: !executor.ptr<host> {executor.result = 1 : i64, executor.slot = 3 : i64})
//  IND-SAME:  -> (i32 {executor.result = 0 : i64, executor.slot = 2 : i64})
func.func @caller(
    %arg0: !executor.ptr<host> {executor.slot = 0},
    %arg1: i32 {executor.slot = 1})
      -> (i32 {executor.slot = 2}, !executor.table<!executor.ptr<host>, i32> {executor.slot = 3})
      attributes {} {
  // IND: %[[alloca:.+]] = executor.alloca
  // IND: %[[v1:.+]] = call @return_table(%[[arg0]], %[[arg1]], %[[alloca]])
  // IND: %[[v2:.+]] = executor.load %[[alloca]]
  // IND: executor.store %[[v2]] to %[[arg2]]
  // IND: return %[[v1]] : i32
  %0:2 = func.call @return_table(%arg0, %arg1) : (!executor.ptr<host>, i32)
                                        -> (!executor.table<!executor.ptr<host>, i32>, i32)
  return %0#1, %0#0 : i32, !executor.table<!executor.ptr<host>, i32>
}

// -----

// Test nested aggregates (tables containing tables)
func.func @nested_aggregate_arg(%arg0: !executor.table<!executor.table<i32, f32>, !executor.ptr<host>>)
    -> (!executor.table<i32, f32>) {
  %0 = executor.table.get %arg0[0] : !executor.table<!executor.table<i32, f32>, !executor.ptr<host>>
  return %0 : !executor.table<i32, f32>
}

// IND-LABEL: func.func @nested_aggregate_arg
//  IND-SAME:  (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: !executor.ptr<host> {executor.result = 0 : i64}) {
//       IND: %[[v0:.+]] = executor.load %[[arg0]]
//       IND: %[[v1:.+]] = executor.table.get %[[v0]][0]
//       IND: executor.store %[[v1]] to %[[arg1]]
//       IND: return

// UNP-LABEL: func.func @nested_aggregate_arg
//  UNP-SAME:  (%[[arg0:.+]]: i32, %[[arg1:.+]]: f32, %[[arg2:.+]]: !executor.ptr<host>)
//  UNP-SAME:  -> (i32 {executor.result = 0 : i64}, f32)
//       UNP: %[[v0:.+]] = executor.table.create(%[[arg0]], %[[arg1]] :
//       UNP: %[[v1:.+]] = executor.table.create(%[[v0]], %[[arg2]] :
//       UNP: %[[v2:.+]] = executor.table.get %[[v1]][0]
//       UNP: %[[v3:.+]] = executor.table.get %[[v2]][0]
//       UNP: %[[v4:.+]] = executor.table.get %[[v2]][1]
//       UNP: return %[[v3]], %[[v4]]

// -----

// Test multiple aggregate arguments and results
func.func @multiple_aggregates(
    %arg0: !executor.table<i32, i64>,
    %arg1: !executor.table<f32, f64>,
    %arg2: i32)
    -> (!executor.table<i32, i64>, !executor.table<f32, f64>, i32) {
  return %arg0, %arg1, %arg2 : !executor.table<i32, i64>, !executor.table<f32, f64>, i32
}

// IND-LABEL: func.func @multiple_aggregates
//  IND-SAME:  (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: !executor.ptr<host>, %[[arg2:.+]]: i32, %[[arg3:.+]]: !executor.ptr<host> {executor.result = 0 : i64}, %[[arg4:.+]]: !executor.ptr<host> {executor.result = 1 : i64}) -> (i32 {executor.result = 2 : i64}) {
//       IND: %[[v0:.+]] = executor.load %[[arg1]]
//       IND: %[[v1:.+]] = executor.load %[[arg0]]
//       IND: executor.store %[[v1]] to %[[arg3]]
//       IND: executor.store %[[v0]] to %[[arg4]]
//       IND: return %[[arg2]]

// UNP-LABEL: func.func @multiple_aggregates
//  UNP-SAME:  (%[[arg0:.+]]: i32, %[[arg1:.+]]: i64,
//  UNP-SAME:   %[[arg2:.+]]: f32, %[[arg3:.+]]: f64,
//  UNP-SAME:   %[[arg4:.+]]: i32)
//  UNP-SAME:  -> (i32 {executor.result = 0 : i64}, i64, f32 {executor.result = 1 : i64}, f64, i32 {executor.result = 2 : i64})
//       UNP: return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[arg4]]

// -----

// Test branch operations with aggregates
func.func @branch_with_aggregate(%cond: i1, %arg0: !executor.table<i32, f32>) -> !executor.table<i32, f32> {
  cf.cond_br %cond, ^bb1(%arg0 : !executor.table<i32, f32>), ^bb2(%arg0 : !executor.table<i32, f32>)
^bb1(%val1: !executor.table<i32, f32>):
  cf.br ^bb3(%val1 : !executor.table<i32, f32>)
^bb2(%val2: !executor.table<i32, f32>):
  cf.br ^bb3(%val2 : !executor.table<i32, f32>)
^bb3(%result: !executor.table<i32, f32>):
  return %result : !executor.table<i32, f32>
}

// IND-LABEL: func.func @branch_with_aggregate
//  IND-SAME:  (%[[arg0:.+]]: i1, %[[arg1:.+]]: !executor.ptr<host>, %[[arg2:.+]]: !executor.ptr<host> {executor.result = 0 : i64}) {
//       IND: %[[v0:.+]] = executor.load %[[arg1]]
//       IND: cf.cond_br %[[arg0]], ^bb1(%[[v0]] : !executor.table<i32, f32>), ^bb2(%[[v0]] : !executor.table<i32, f32>)
//       IND: ^bb3(%[[result:.+]]: !executor.table<i32, f32>):
//       IND: executor.store %[[result]] to %[[arg2]]

// UNP-LABEL: func.func @branch_with_aggregate
//  UNP-SAME:  (%[[cond:.+]]: i1, %[[arg0:.+]]: i32, %[[arg1:.+]]: f32)
//  UNP-SAME:  -> (i32 {executor.result = 0 : i64}, f32)
//       UNP: %[[v0:.+]] = executor.table.create(%[[arg0]], %[[arg1]] : i32, f32)
//       UNP: cf.cond_br %[[cond]], ^bb1(%[[v0]] : !executor.table<i32, f32>), ^bb2(%[[v0]] : !executor.table<i32, f32>)
//       UNP: ^bb3(%[[result:.+]]: !executor.table<i32, f32>):
//       UNP: %[[r0:.+]] = executor.table.get %[[result]][0]
//       UNP: %[[r1:.+]] = executor.table.get %[[result]][1]
//       UNP: return %[[r0]], %[[r1]]

// -----

// Test very large aggregate (stress test for unpacked mode)
func.func @large_aggregate(%arg0: !executor.table<i32, i64, f32, f64, i1, i8, i16, !executor.ptr<host>>)
    -> !executor.table<i32, i64, f32, f64, i1, i8, i16, !executor.ptr<host>> {
  return %arg0 : !executor.table<i32, i64, f32, f64, i1, i8, i16, !executor.ptr<host>>
}

// IND-LABEL: func.func @large_aggregate
//  IND-SAME:  (%[[arg0:.+]]: !executor.ptr<host>,
//  IND-SAME:   %[[arg1:.+]]: !executor.ptr<host> {executor.result = 0 : i64})

// UNP-LABEL: func.func @large_aggregate
//  UNP-SAME:  (%[[arg0:.+]]: i32, %[[arg1:.+]]: i64, %[[arg2:.+]]: f32, %[[arg3:.+]]: f64,
//  UNP-SAME:   %[[arg4:.+]]: i1, %[[arg5:.+]]: i8, %[[arg6:.+]]: i16, %[[arg7:.+]]: !executor.ptr<host>)
//  UNP-SAME:  -> (i32 {executor.result = 0 : i64}, i64, f32, f64, i1, i8, i16, !executor.ptr<host>)

// -----

// Test mixing aggregate and non-aggregate results
func.func @mixed_results(%arg0: i32) -> (i32, !executor.table<i64, f32>, f64, !executor.table<i8>) {
  %c0 = executor.constant 0 : i64
  %c1 = executor.constant 1.0 : f32
  %c2 = executor.constant 2.0 : f64
  %c3 = executor.constant 3 : i8
  %t0 = executor.table.create(%c0, %c1 : i64, f32) : !executor.table<i64, f32>
  %t1 = executor.table.create(%c3 : i8) : !executor.table<i8>
  return %arg0, %t0, %c2, %t1 : i32, !executor.table<i64, f32>, f64, !executor.table<i8>
}

// IND-LABEL: func.func @mixed_results
//  IND-SAME:  (%[[arg0:.+]]: i32,
//  IND-SAME:   %[[arg1:.+]]: !executor.ptr<host> {executor.result = 1 : i64},
//  IND-SAME:   %[[arg2:.+]]: !executor.ptr<host> {executor.result = 3 : i64})
//  IND-SAME:  -> (i32 {executor.result = 0 : i64}, f64 {executor.result = 2 : i64})
//       IND: executor.store %{{.*}} to %[[arg1]]
//       IND: executor.store %{{.*}} to %[[arg2]]
//       IND: return %[[arg0]], %{{.*}} : i32, f64

// UNP-LABEL: func.func @mixed_results
//  UNP-SAME:  (%[[arg0:.+]]: i32)
//  UNP-SAME:  -> (i32 {executor.result = 0 : i64}, i64 {executor.result = 1 : i64}, f32, f64 {executor.result = 2 : i64}, i8 {executor.result = 3 : i64})

// -----

// Test function call with nested aggregates
func.func private @nested_callee(!executor.table<!executor.table<i32>, f32>) -> !executor.table<!executor.table<i32>, f32>

func.func @nested_call_test(%arg0: !executor.table<!executor.table<i32>, f32>) -> !executor.table<!executor.table<i32>, f32> {
  %0 = func.call @nested_callee(%arg0) : (!executor.table<!executor.table<i32>, f32>) -> !executor.table<!executor.table<i32>, f32>
  return %0 : !executor.table<!executor.table<i32>, f32>
}

// IND-LABEL: func.func @nested_call_test
//  IND-SAME:  (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: !executor.ptr<host> {executor.result = 0 : i64}) {
//       IND: %[[alloca:.+]] = executor.alloca
//       IND: call @nested_callee(%[[arg0]], %[[alloca]])
//       IND: %[[v1:.+]] = executor.load %[[alloca]]
//       IND: executor.store %[[v1]] to %[[arg1]]
//       IND: return

// UNP-LABEL: func.func @nested_call_test
//  UNP-SAME:  (%[[arg0:.+]]: i32, %[[arg1:.+]]: f32)
//  UNP-SAME:  -> (i32 {executor.result = 0 : i64}, f32)
//       UNP: %[[v0:.+]]:2 = call @nested_callee(%[[arg0]], %[[arg1]])
//       UNP: return %{{.*}}, %{{.*}}

// -----

// Test switch operation with aggregates (if supported)
func.func @switch_with_aggregate(%idx: i32, %arg0: !executor.table<i64, f64>) -> !executor.table<i64, f64> {
  cf.switch %idx : i32, [
    default: ^bb1(%arg0 : !executor.table<i64, f64>),
    0: ^bb2(%arg0 : !executor.table<i64, f64>),
    1: ^bb3(%arg0 : !executor.table<i64, f64>)
  ]
^bb1(%val1: !executor.table<i64, f64>):
  return %val1 : !executor.table<i64, f64>
^bb2(%val2: !executor.table<i64, f64>):
  return %val2 : !executor.table<i64, f64>
^bb3(%val3: !executor.table<i64, f64>):
  return %val3 : !executor.table<i64, f64>
}

// IND-LABEL: func.func @switch_with_aggregate
//  IND-SAME:  (%[[arg0:.+]]: i32, %[[arg1:.+]]: !executor.ptr<host>, %[[arg2:.+]]: !executor.ptr<host> {executor.result = 0 : i64}) {
//       IND: %[[v0:.+]] = executor.load %[[arg1]]
//       IND: cf.switch %[[arg0]] : i32, [
//       IND:   default: ^bb1(%[[v0]] : !executor.table<i64, f64>),
//       IND:   0: ^bb2(%[[v0]] : !executor.table<i64, f64>),
//       IND:   1: ^bb3(%[[v0]] : !executor.table<i64, f64>)
//       IND: ^bb1(%[[v1:.+]]: !executor.table<i64, f64>):
//       IND: executor.store %[[v1]] to %[[arg2]]

// UNP-LABEL: func.func @switch_with_aggregate
//  UNP-SAME:  (%[[idx:.+]]: i32, %[[arg0:.+]]: i64, %[[arg1:.+]]: f64)
//  UNP-SAME:  -> (i64 {executor.result = 0 : i64}, f64)
//       UNP: %[[v0:.+]] = executor.table.create(%[[arg0]], %[[arg1]] : i64, f64)
//       UNP: cf.switch %[[idx]] : i32, [
//       UNP:   default: ^bb1(%[[v0]] : !executor.table<i64, f64>),
//       UNP:   0: ^bb2(%[[v0]] : !executor.table<i64, f64>),
//       UNP:   1: ^bb3(%[[v0]] : !executor.table<i64, f64>)

// -----

// Test function with no arguments but aggregate results
func.func @no_args_aggregate_results() -> (!executor.table<i32, i64>, f32) {
  %c0 = executor.constant 42 : i32
  %c1 = executor.constant 84 : i64
  %c2 = executor.constant 3.14 : f32
  %t = executor.table.create(%c0, %c1 : i32, i64) : !executor.table<i32, i64>
  return %t, %c2 : !executor.table<i32, i64>, f32
}

// IND-LABEL: func.func @no_args_aggregate_results
//  IND-SAME:  (%[[arg0:.+]]: !executor.ptr<host> {executor.result = 0 : i64})
//  IND-SAME:  -> (f32 {executor.result = 1 : i64})
//       IND: %[[t:.+]] = executor.table.create
//       IND: executor.store %[[t]] to %[[arg0]]
//       IND: return %{{.*}} : f32

// UNP-LABEL: func.func @no_args_aggregate_results
//  UNP-SAME:  () -> (i32 {executor.result = 0 : i64}, i64, f32 {executor.result = 1 : i64})
//       UNP: %[[t:.+]] = executor.table.create
//       UNP: %[[v0:.+]] = executor.table.get %[[t]][0]
//       UNP: %[[v1:.+]] = executor.table.get %[[t]][1]
//       UNP: return %[[v0]], %[[v1]], %{{.*}}

// -----

// Test deeply nested aggregates (3 levels)
func.func @deeply_nested(%arg0: !executor.table<!executor.table<!executor.table<i32>>>)
    -> !executor.table<!executor.table<!executor.table<i32>>> {
  return %arg0 : !executor.table<!executor.table<!executor.table<i32>>>
}

// IND-LABEL: func.func @deeply_nested
//  IND-SAME:  (%[[arg0:.+]]: !executor.ptr<host>,
//  IND-SAME:   %[[arg1:.+]]: !executor.ptr<host> {executor.result = 0 : i64})

// UNP-LABEL: func.func @deeply_nested
//  UNP-SAME:  (%[[arg0:.+]]: i32)
//  UNP-SAME:  -> (i32 {executor.result = 0 : i64})
