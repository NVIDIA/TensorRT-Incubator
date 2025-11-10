// REQUIRES: disable
// RUN: executor-opt %s -split-input-file \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries use-encoding-for-memory-space function-boundary-type-conversion=identity-layout-map" -canonicalize \
// RUN:   -buffer-deallocation-pipeline \
// RUN:   | FileCheck %s

// Test 1: Basic ABIRecvOp bufferization - tensor to memref
// CHECK-LABEL: func @abi_recv_tensor_to_memref
func.func @abi_recv_tensor_to_memref(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<10xi32>>})
    -> (tensor<10xi32>)
      attributes {
        executor.func_abi = (tensor<10xi32>) -> (tensor<10xi32>)
      } {
  // CHECK: %[[RECV:.*]] = executor.abi.recv %arg0 : memref<10xi32>
  %0 = executor.abi.recv %arg0 : tensor<10xi32>

  // CHECK: executor.abi.send %[[RECV]] to %arg1 : memref<10xi32>
  %1 = executor.abi.send %0 to %arg1 : tensor<10xi32>
  return %1 : tensor<10xi32>
}

// -----

// Test 2: ABISendOp bufferization with intermediate tensor operations
// CHECK-LABEL: func @abi_send_with_tensor_ops
func.func @abi_send_with_tensor_ops(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<4x8xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<4x8xf32>>})
    -> (tensor<4x8xf32>)
      attributes {
        executor.func_abi = (tensor<4x8xf32>) -> (tensor<4x8xf32>)
      } {
  // CHECK: %[[RECV:.*]] = executor.abi.recv %arg0 : memref<4x8xf32>
  %0 = executor.abi.recv %arg0 : tensor<4x8xf32>

  // The tensor is bufferized to memref
  // CHECK: executor.abi.send %[[RECV]] to %arg1 : memref<4x8xf32>
  %1 = executor.abi.send %0 to %arg1 : tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// -----

// Test 3: ABIRecvOp with device memory space
// CHECK-LABEL: func @abi_recv_device_memspace
func.func @abi_recv_device_memspace(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<10xi32, #executor.memory_type<device>>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<10xi32, #executor.memory_type<device>>>})
    -> (tensor<10xi32, #executor.memory_type<device>>)
      attributes {
        executor.func_abi = (tensor<10xi32>) -> (tensor<10xi32, #executor.memory_type<device>>)
      } {
  // CHECK: %[[RECV:.*]] = executor.abi.recv %arg0 : memref<10xi32, #executor.memory_type<device>>
  %0 = executor.abi.recv %arg0 {memory_space = #executor.memory_type<device>} : tensor<10xi32, #executor.memory_type<device>>

  // CHECK: executor.abi.send %[[RECV]] to %arg1 : memref<10xi32, #executor.memory_type<device>>
  %1 = executor.abi.send %0 to %arg1 : tensor<10xi32, #executor.memory_type<device>>
  return %1 : tensor<10xi32, #executor.memory_type<device>>
}

// -----

// Test 4: Multiple ABIRecvOp operations
// CHECK-LABEL: func @multiple_abi_recv
func.func @multiple_abi_recv(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<20xf32>>},
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<10xi32>>},
    %arg3: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<20xf32>>})
    -> (tensor<10xi32>, tensor<20xf32>)
      attributes {
        executor.func_abi = (tensor<10xi32>, tensor<20xf32>) -> (tensor<10xi32>, tensor<20xf32>)
      } {
  // CHECK: %[[RECV0:.*]] = executor.abi.recv %arg0 : memref<10xi32>
  %0 = executor.abi.recv %arg0 : tensor<10xi32>

  // CHECK: %[[RECV1:.*]] = executor.abi.recv %arg1 : memref<20xf32>
  %1 = executor.abi.recv %arg1 : tensor<20xf32>

  // CHECK: executor.abi.send %[[RECV0]] to %arg2 : memref<10xi32>
  %2 = executor.abi.send %0 to %arg2 : tensor<10xi32>

  // CHECK: executor.abi.send %[[RECV1]] to %arg3 : memref<20xf32>
  %3 = executor.abi.send %1 to %arg3 : tensor<20xf32>
  return %2, %3 : tensor<10xi32>, tensor<20xf32>
}
// -----

// CHECK-LABEL: func @abi_send_undef
func.func @abi_send_undef(
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<10xi32>, undef>},
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<10xi32>>})
    -> (tensor<10xi32>, tensor<10xi32>)
      attributes {
        executor.func_abi = () -> (tensor<10xi32>, tensor<10xi32>)
      } {
  %0 = arith.constant dense<1> : tensor<10xi32>

  // CHECK: executor.abi.send %0 to %arg1 : tensor<10xi32>
  %1 = executor.abi.send %0 to %arg1 : tensor<10xi32>
  %2 = executor.abi.send %0 to %arg2 : tensor<10xi32>
  return %1, %2 : tensor<10xi32>, tensor<10xi32>
}
