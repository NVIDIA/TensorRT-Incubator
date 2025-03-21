// RUN: executor-opt %s -split-input-file -executor-lower-globals -verify-diagnostics | FileCheck %s

executor.global @global1 constant : !executor.ptr<host> {
  %alignment = arith.constant 16 : i32
  %bytes = arith.constant 128 : i32
  %0 = executor.alloc %bytes bytes align(%alignment) : (i32, i32) -> !executor.ptr<host>
  executor.return %0 : !executor.ptr<host>
}

executor.global @global2 : !executor.ptr<host> {
  %alignment = arith.constant 16 : i32
  %bytes = arith.constant 256 : i32
  %0 = executor.alloc %bytes bytes align(%alignment) : (i32, i32) -> !executor.ptr<host>
  executor.return %0 : !executor.ptr<host>
}

// CHECK-LABEL: module attributes {executor.global_init_func = @executor_init_globals} {
//       CHECK:   executor.global @global1 constant : !executor.ptr<host>
//       CHECK:   executor.global @global2 : !executor.ptr<host>
//       CHECK: @executor_init_globals
//       CHECK:     %[[c16:.+]] = arith.constant 16 : i32
//       CHECK:     %[[c128:.+]] = arith.constant 128 : i32
//       CHECK:     %[[v0:.+]] = executor.alloc %[[c128]] bytes align(%[[c16]]) : (i32, i32) -> !executor.ptr<host>
//       CHECK:     executor.set_global %[[v0]], @global1 : !executor.ptr<host>
//       CHECK:     %[[v2:.+]] = executor.alloc
//       CHECK:     executor.set_global %[[v2]], @global2 : !executor.ptr<host>
//       CHECK:     return

// -----

executor.global @global1 constant : !executor.ptr<host> {
  %alignment = arith.constant 16 : i32
  %bytes = arith.constant 128 : i32
  %0 = executor.alloc %bytes bytes align(%alignment) : (i32, i32) -> !executor.ptr<host>
  executor.return %0 : !executor.ptr<host>
}

// test name collision
func.func @executor_init_globals() {
  return
}

// CHECK-LABEL: module attributes {executor.global_init_func = @executor_init_globals_0} {
//       CHECK: executor.global @global1 constant : !executor.ptr<host>
//       CHECK: @executor_init_globals
//       CHECK: @executor_init_globals_0

// -----

// test initializer to global constant resource

executor.global @global3 constant : !executor.ptr<host> attributes {
  initial_value = dense<[1, 2, 3, 4]> : vector<4xi8>
}

// TODO: fix when we have constant initializer de-duplication

executor.global @global4 : !executor.ptr<host> attributes {
  initial_value = dense<[1, 2, 3, 4]> : vector<4xi8>
}

// CHECK-LABEL: module attributes {executor.global_init_func = @executor_init_globals} {
//       CHECK:   executor.global @global3 constant : !executor.ptr<host>
//       CHECK:   executor.global @global4 : !executor.ptr<host>
//       CHECK:   @executor_init_globals
//       CHECK:     %[[v0:.+]] = executor.load_data_segment @global3_initializer : !executor.ptr<host>
//       CHECK:     executor.set_global %[[v0]], @global3 : !executor.ptr<host>
//       CHECK:     %[[v1:.+]] = executor.load_data_segment @global4_initializer : !executor.ptr<host>
//       CHECK:     executor.set_global %[[v1]], @global4 : !executor.ptr<host>
//       CHECK:     return
//       CHECK:   executor.data_segment @global3_initializer constant dense<[1, 2, 3, 4]> : vector<4xi8>
//       CHECK:   executor.data_segment @global4_initializer constant dense<[1, 2, 3, 4]> : vector<4xi8>

// -----

executor.global @global_no_init : !executor.table<!executor.ptr<host>>

// CHECK-LABEL: executor.global @global_no_init
//       CHECK: @executor_init_globals()
//  CHECK-NEXT:   return
