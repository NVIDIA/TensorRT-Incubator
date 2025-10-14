// RUN: executor-opt -one-shot-bufferize -split-input-file %s | FileCheck %s

// Test 1: Basic ABIRecvOp bufferization - tensor to memref
// CHECK-LABEL: func @abi_recv_tensor_to_memref
func.func @abi_recv_tensor_to_memref(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<10xi32>>})
      attributes {
        executor.func_abi = (tensor<10xi32>) -> (tensor<10xi32>)
      } {
  // CHECK: %[[RECV:.*]] = executor.abi.recv %arg0 : memref<10xi32>
  %0 = executor.abi.recv %arg0 : tensor<10xi32>

  // CHECK: executor.abi.send %[[RECV]] to %arg1 : memref<10xi32>
  executor.abi.send %0 to %arg1 : tensor<10xi32>
  return
}

// -----

// Test 2: ABISendOp bufferization with intermediate tensor operations
// CHECK-LABEL: func @abi_send_with_tensor_ops
func.func @abi_send_with_tensor_ops(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<4x8xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<4x8xf32>>})
      attributes {
        executor.func_abi = (tensor<4x8xf32>) -> (tensor<4x8xf32>)
      } {
  // CHECK: %[[RECV:.*]] = executor.abi.recv %arg0 : memref<4x8xf32>
  %0 = executor.abi.recv %arg0 : tensor<4x8xf32>

  // The tensor is bufferized to memref
  // CHECK: executor.abi.send %[[RECV]] to %arg1 : memref<4x8xf32>
  executor.abi.send %0 to %arg1 : tensor<4x8xf32>
  return
}

// -----

// Test 3: ABIRecvOp with device memory space
// CHECK-LABEL: func @abi_recv_device_memspace
func.func @abi_recv_device_memspace(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<10xi32>>})
      attributes {
        executor.func_abi = (tensor<10xi32>) -> (tensor<10xi32>)
      } {
  // CHECK: %[[RECV:.*]] = executor.abi.recv %arg0 : memref<10xi32, #executor.memory_type<device>>
  %0 = executor.abi.recv %arg0 {memory_space = #executor.memory_type<device>} : tensor<10xi32>

  // CHECK: executor.abi.send %[[RECV]] to %arg1 : memref<10xi32, #executor.memory_type<device>>
  executor.abi.send %0 to %arg1 : tensor<10xi32>
  return
}

// -----

// Test 4: Multiple ABIRecvOp operations
// CHECK-LABEL: func @multiple_abi_recv
func.func @multiple_abi_recv(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<20xf32>>},
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<10xi32>>},
    %arg3: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<20xf32>>})
      attributes {
        executor.func_abi = (tensor<10xi32>, tensor<20xf32>) -> (tensor<10xi32>, tensor<20xf32>)
      } {
  // CHECK: %[[RECV0:.*]] = executor.abi.recv %arg0 : memref<10xi32>
  %0 = executor.abi.recv %arg0 : tensor<10xi32>

  // CHECK: %[[RECV1:.*]] = executor.abi.recv %arg1 : memref<20xf32>
  %1 = executor.abi.recv %arg1 : tensor<20xf32>

  // CHECK: executor.abi.send %[[RECV0]] to %arg2 : memref<10xi32>
  executor.abi.send %0 to %arg2 : tensor<10xi32>

  // CHECK: executor.abi.send %[[RECV1]] to %arg3 : memref<20xf32>
  executor.abi.send %1 to %arg3 : tensor<20xf32>
  return
}

// -----

// Test 5: ABIRecvOp with pinned_host memory space
// CHECK-LABEL: func @abi_recv_pinned_host
func.func @abi_recv_pinned_host(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<16xf16>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<16xf16>>})
      attributes {
        executor.func_abi = (tensor<16xf16>) -> (tensor<16xf16>)
      } {
  // CHECK: %[[RECV:.*]] = executor.abi.recv %arg0 : memref<16xf16, #executor.memory_type<host_pinned>>
  %0 = executor.abi.recv %arg0 {memory_space = #executor.memory_type<host_pinned>} : tensor<16xf16>

  // CHECK: executor.abi.send %[[RECV]] to %arg1 : memref<16xf16, #executor.memory_type<host_pinned>>
  executor.abi.send %0 to %arg1 : tensor<16xf16>
  return
}

// -----

// Test 8: ABISendOp with non-tensor type (should pass through unchanged)
// CHECK-LABEL: func @abi_send_non_tensor
func.func @abi_send_non_tensor(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>) -> (memref<10xi32>)
      } {
  // CHECK: %[[RECV:.*]] = executor.abi.recv %arg0 : memref<10xi32>
  %0 = executor.abi.recv %arg0 : memref<10xi32>

  // CHECK: executor.abi.send %[[RECV]] to %arg1 : memref<10xi32>
  executor.abi.send %0 to %arg1 : memref<10xi32>
  return
}

