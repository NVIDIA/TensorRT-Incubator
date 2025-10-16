// RUN: executor-opt %s -split-input-file -convert-executor-to-executor="index-bitwidth=32" -verify-diagnostics | FileCheck %s --check-prefix=PACKED

executor.func private @my_func(index, memref<128xf32>) -> index

func.func @executor_func_call(%arg0: index, %arg1: memref<128xf32>) -> index {
  %0 = executor.call @my_func(%arg0, %arg1) : (index, memref<128xf32>) -> index
  return %0 : index
}

// PACKED-LABEL: executor.func private @my_func(i32, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>) -> i32
// PACKED-LABEL: @executor_func_call
//  PACKED-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: memref<128xf32>) -> index {
//   PACKED-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : index to i32
//   PACKED-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<128xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       PACKED:     %[[v2:.+]] = executor.call @my_func(%[[v0]], %[[v1]]) : (i32, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>) -> i32
//       PACKED:     %[[v3:.+]] = builtin.unrealized_conversion_cast %[[v2]] : i32 to index
//       PACKED:     return %[[v3]] : index

// -----

executor.global @global_with_index_type_conversion
    : index {
  %0 = arith.constant 0 : index
  executor.return %0 : index
}

// PACKED-LABEL:   executor.global @global_with_index_type_conversion : i32 {

// -----

func.func @exec_add_index(%arg0: index, %arg1: index) -> index {
  %0 = executor.addi %arg0, %arg1 : index
  return %0 : index
}

// PACKED-LABEL: @exec_add_index
//  PACKED-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index) -> index {
//   PACKED-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : index to i32
//   PACKED-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : index to i32
//       PACKED:     %[[v2:.+]] = executor.addi %[[v0]], %[[v1]] : i32
//       PACKED:     %[[v3:.+]] = builtin.unrealized_conversion_cast %[[v2]] : i32 to index
//       PACKED:     return %[[v3]] : index

// -----

func.func @metadata_conversion(%arg0: i32)  attributes {
  executor.function_metadata = #executor.func_meta<[
      index
    ], [], num_output_args=0>
} {
  return
}

// PACKED-LABEL: @metadata_conversion
//  PACKED-SAME:  attributes {executor.function_metadata = #executor.func_meta<[i32], [], num_output_args = 0>}

// -----

func.func @abi_recv_lowering(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, complex<f32>>})
      attributes {executor.func_abi = (complex<f32>) -> ()} {
  %0 = executor.abi.recv %arg0 : complex<f32>
  return
}

// PACKED-LABEL: @abi_recv_lowering
//  PACKED-SAME: (%[[arg0:.+]]: !executor.ptr<host>
//       PACKED:     %[[c0:.+]] = executor.constant 0 : i32
//       PACKED:     %[[v0:.+]] = executor.load %[[arg0]] + %[[c0]] : (!executor.ptr<host>, i32) -> !executor.table<f32, f32>

// -----

func.func @abi_send_lowering(
    %arg0: i64,
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, i64>})
      attributes {executor.func_abi = (i64) -> (i64)} {
  executor.abi.send %arg0 to %arg1 : i64
  return
}

// PACKED-LABEL: @abi_send_lowering
//  PACKED-SAME: (%[[arg0:.+]]: i64, %[[arg1:.+]]: !executor.ptr<host>
//       PACKED:     %[[c0:.+]] = executor.constant 0 : i32
//       PACKED:     executor.store %[[arg0]] to %[[arg1]] + %[[c0]] : i64, !executor.ptr<host>, i32

// -----

func.func @abi_send_index_lowering(
    %arg0: index,
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, index>})
      attributes {executor.func_abi = (index) -> (index)} {
  executor.abi.send %arg0 to %arg1 : index
  return
}

// PACKED-LABEL: func.func @abi_send_index_lowering
//  PACKED-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, index>}) attributes {executor.func_abi = (index) -> index} {
//       PACKED:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : index to i32
//       PACKED:     %[[c0_i32:.+]] = executor.constant 0 : i32
//       PACKED:     executor.store %[[v0]] to %[[arg1]] + %[[c0_i32]] : i32, !executor.ptr<host>, i32
//       PACKED:     return


// -----

func.func @abi_recv_index_conversion(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xindex>>})
      attributes {executor.func_abi = (memref<10xindex>) -> ()} {
  %0 = executor.abi.recv %arg0 : memref<10xindex>
  return
}

// PACKED-LABEL: @abi_recv_index_conversion
//  PACKED-SAME: (%[[arg0:.+]]: !executor.ptr<host>
//       PACKED:     %[[c0:.+]] = executor.constant 0 : i32
//       PACKED:     %[[v0:.+]] = executor.load %[[arg0]] + %[[c0]] : (!executor.ptr<host>, i32) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>

// -----

func.func @abi_send_index_conversion(
    %arg0: index,
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, index>})
      attributes {executor.func_abi = (index) -> (index)} {
  executor.abi.send %arg0 to %arg1 : index
  return
}

// PACKED-LABEL: @abi_send_index_conversion
//  PACKED-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: !executor.ptr<host>
//       PACKED:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : index to i32
//       PACKED:     %[[c0:.+]] = executor.constant 0 : i32
//       PACKED:     executor.store %[[v0]] to %[[arg1]] + %[[c0]] : i32, !executor.ptr<host>, i32

// -----

// Test that abi.send with memref, undef=true, and ownership=true is lowered to store
func.func @abi_send_memref_undef_ownership(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32>, undef>})
      attributes {executor.func_abi = (memref<10xf32>) -> (memref<10xf32>)} {
  %0 = executor.abi.recv %arg0 : memref<10xf32>
  %true = arith.constant true
  executor.abi.send %0 to %arg1 ownership(%true) : memref<10xf32>
  return
}

// PACKED-LABEL: @abi_send_memref_undef_ownership
//  PACKED-SAME: (%[[arg0:.+]]: !executor.ptr<host>{{.*}}, %[[arg1:.+]]: !executor.ptr<host>
//       PACKED:     %[[c0:.+]] = executor.constant 0 : i32
//       PACKED:     %[[v0:.+]] = executor.load %[[arg0]] + %[[c0]]
//       PACKED:     %[[true:.+]] = arith.constant true
//       PACKED:     %[[c0_0:.+]] = executor.constant 0 : i32
//       PACKED:     executor.store %[[v0]] to %[[arg1]] + %[[c0_0]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>, !executor.ptr<host>, i32

// -----

func.func @abi_send_memref_no_undef(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32>>})
      attributes {executor.func_abi = (memref<10xf32>) -> (memref<10xf32>)} {
  %0 = executor.abi.recv %arg0 : memref<10xf32>
  %true = arith.constant true
  // expected-error @below {{failed to legalize operation 'executor.abi.send' that was explicitly marked illegal}}
  executor.abi.send %0 to %arg1 ownership(%true) : memref<10xf32>
  return
}

// -----

func.func @abi_recv_undef(
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32>, undef>})
      attributes {executor.func_abi = () -> (memref<10xf32>)} {
  // expected-error @below {{failed to legalize operation 'executor.abi.recv' that was explicitly marked illegal}}
  %0 = executor.abi.recv %arg1 : memref<10xf32>
  return
}

// -----

func.func @abi_send_memref_undef_no_ownership(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xf32>, undef>})
      attributes {executor.func_abi = (memref<10xf32>) -> (memref<10xf32>)} {
  %0 = executor.abi.recv %arg0 : memref<10xf32>
  %false = arith.constant false
  // expected-error @+1 {{failed to legalize operation 'executor.abi.send' that was explicitly marked illegal}}
  executor.abi.send %0 to %arg1 ownership(%false) : memref<10xf32>
  return
}
