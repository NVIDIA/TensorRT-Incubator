// RUN: executor-opt %s -split-input-file \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries use-encoding-for-memory-space function-boundary-type-conversion=identity-layout-map" -canonicalize \
// RUN:   -buffer-deallocation-pipeline \
// RUN:   | FileCheck %s

// Test 1: Basic buffer_bitcast - tensor to tensor with same element type
// When types match, bufferization creates a clone instead of buffer_bitcast
// CHECK-LABEL: func @buffer_bitcast_same_element_type
func.func @buffer_bitcast_same_element_type(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %[[CLONE:.*]] = bufferization.clone %arg0 : memref<4xi32> to memref<4xi32>
  %0 = executor.buffer_bitcast %arg0 : tensor<4xi32> to tensor<4xi32>
  // CHECK: return %[[CLONE]]
  return %0 : tensor<4xi32>
}

// -----

// Test 2: buffer_bitcast - tensor to tensor with different element types (same bit width)
// CHECK-LABEL: func @buffer_bitcast_different_element_type
func.func @buffer_bitcast_different_element_type(%arg0: tensor<4xi32>) -> tensor<4xf32> {
  // CHECK: %[[CAST:.*]] = executor.buffer_bitcast %arg0 : memref<4xi32> to memref<4xf32>
  %0 = executor.buffer_bitcast %arg0 : tensor<4xi32> to tensor<4xf32>
  // CHECK: %[[CLONE:.*]] = bufferization.clone %[[CAST]] : memref<4xf32> to memref<4xf32>
  // CHECK: return %[[CLONE]]
  return %0 : tensor<4xf32>
}

// -----

// Test 3: buffer_bitcast - memref to memref (should remain as buffer_bitcast)
// CHECK-LABEL: func @buffer_bitcast_memref_to_memref
func.func @buffer_bitcast_memref_to_memref(%arg0: memref<4xi32>) -> memref<4xf32> {
  // CHECK: %[[CAST:.*]] = executor.buffer_bitcast %arg0 : memref<4xi32> to memref<4xf32>
  %0 = executor.buffer_bitcast %arg0 : memref<4xi32> to memref<4xf32>
  return %0 : memref<4xf32>
}

// -----

// Test 4: buffer_bitcast with 2D tensors
// CHECK-LABEL: func @buffer_bitcast_2d_tensor
func.func @buffer_bitcast_2d_tensor(%arg0: tensor<4x4xi32>) -> tensor<4x4xf32> {
  // CHECK: %[[CAST:.*]] = executor.buffer_bitcast %arg0 : memref<4x4xi32> to memref<4x4xf32>
  %0 = executor.buffer_bitcast %arg0 : tensor<4x4xi32> to tensor<4x4xf32>
  // CHECK: %[[CLONE:.*]] = bufferization.clone %[[CAST]] : memref<4x4xf32> to memref<4x4xf32>
  // CHECK: return %[[CLONE]]
  return %0 : tensor<4x4xf32>
}

// -----

// Test 5: buffer_bitcast in a chain with other operations
// CHECK-LABEL: func @buffer_bitcast_chain
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {{.*}}, %[[arg1:.+]]: !executor.ptr<host> {{.*}})
func.func @buffer_bitcast_chain(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<4xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<4xf32>>})
    -> (tensor<4xf32>)
      attributes {
        executor.func_abi = (tensor<4xi32>) -> (tensor<4xf32>)
      } {
  // CHECK-DAG: %[[RECV:.*]] = executor.abi.recv %[[arg0]] : memref<4xi32>
  %0 = executor.abi.recv %arg0 : tensor<4xi32>

  // CHECK-DAG: %[[CAST:.*]] = executor.buffer_bitcast %[[RECV]] : memref<4xi32> to memref<4xf32>
  %1 = executor.buffer_bitcast %0 : tensor<4xi32> to tensor<4xf32>

  // CHECK-DAG: memref.copy %[[CAST]], %[[RECV1:.*]]
  // CHECK-DAG: %[[UB:.+]] = ub.poison
  // CHECK-DAG: %[[CLONE:.+]] = bufferization.clone %[[UB]]
  // CHECK: return %[[CLONE]]
  %2 = executor.abi.send %1 to %arg1 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// Test 6: buffer_bitcast with device memory space
// CHECK-LABEL: func @buffer_bitcast_device_memspace
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {{.*}}, %[[arg1:.+]]: !executor.ptr<host> {{.*}})
func.func @buffer_bitcast_device_memspace(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<4xi32, #executor.memory_type<device>>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<4xf32, #executor.memory_type<device>>>})
    -> (tensor<4xf32, #executor.memory_type<device>>)
      attributes {
        executor.func_abi = (tensor<4xi32>) -> (tensor<4xf32, #executor.memory_type<device>>)
      } {
  // CHECK-DAG: %[[RECV:.*]] = executor.abi.recv %[[arg0]] : memref<4xi32, #executor.memory_type<device>>
  %0 = executor.abi.recv %arg0 {memory_space = #executor.memory_type<device>} : tensor<4xi32, #executor.memory_type<device>>

  // CHECK-DAG: %[[CAST:.*]] = executor.buffer_bitcast %[[RECV]] : memref<4xi32, #executor.memory_type<device>> to memref<4xf32, #executor.memory_type<device>>
  %1 = executor.buffer_bitcast %0 : tensor<4xi32, #executor.memory_type<device>> to tensor<4xf32, #executor.memory_type<device>>

  // CHECK-DAG: memref.copy %[[CAST]], %[[RECV1:.*]]
  // CHECK-DAG: %[[UB:.+]] = ub.poison
  // CHECK-DAG: %[[CLONE:.+]] = bufferization.clone %[[UB]]
  // CHECK: return %[[CLONE]]
  %2 = executor.abi.send %1 to %arg1 : tensor<4xf32, #executor.memory_type<device>>
  return %2 : tensor<4xf32, #executor.memory_type<device>>
}

// -----

// Test 7: Multiple buffer_bitcast operations
// CHECK-LABEL: func @multiple_buffer_bitcast
func.func @multiple_buffer_bitcast(
    %arg0: tensor<4xi32>,
    %arg1: tensor<8xi16>) -> (tensor<4xf32>, tensor<8xf16>) {
  // CHECK: %[[CAST0:.*]] = executor.buffer_bitcast %arg0 : memref<4xi32> to memref<4xf32>
  %0 = executor.buffer_bitcast %arg0 : tensor<4xi32> to tensor<4xf32>

  // CHECK: %[[CAST1:.*]] = executor.buffer_bitcast %arg1 : memref<8xi16> to memref<8xf16>
  %1 = executor.buffer_bitcast %arg1 : tensor<8xi16> to tensor<8xf16>

  // CHECK-DAG: %[[CLONE0:.*]] = bufferization.clone %[[CAST0]] : memref<4xf32> to memref<4xf32>
  // CHECK-DAG: %[[CLONE1:.*]] = bufferization.clone %[[CAST1]] : memref<8xf16> to memref<8xf16>
  // CHECK: return %[[CLONE0]], %[[CLONE1]]
  return %0, %1 : tensor<4xf32>, tensor<8xf16>
}

// -----

// Test 8: buffer_bitcast with i64 to f64 (64-bit types)
// CHECK-LABEL: func @buffer_bitcast_i64_f64
func.func @buffer_bitcast_i64_f64(%arg0: tensor<2xi64>) -> tensor<2xf64> {
  // CHECK: %[[CAST:.*]] = executor.buffer_bitcast %arg0 : memref<2xi64> to memref<2xf64>
  %0 = executor.buffer_bitcast %arg0 : tensor<2xi64> to tensor<2xf64>
  // CHECK: %[[CLONE:.*]] = bufferization.clone %[[CAST]] : memref<2xf64> to memref<2xf64>
  // CHECK: return %[[CLONE]]
  return %0 : tensor<2xf64>
}
