// RUN: executor-opt %s -split-input-file \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries use-encoding-for-memory-space function-boundary-type-conversion=identity-layout-map" -canonicalize \
// RUN:   -buffer-deallocation-pipeline \
// RUN:   | FileCheck %s

// CHECK-LABEL: func @abi_recv_tensor_to_memref
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {{.*}}, %[[arg1:.+]]: !executor.ptr<host> {{.*}})
func.func @abi_recv_tensor_to_memref(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<10xi32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<10xi32>>})
    -> (tensor<10xi32>)
      attributes {
        executor.func_abi = (tensor<10xi32>) -> (tensor<10xi32>)
      } {
  // CHECK-DAG: %[[RECV0:.*]] = executor.abi.recv %arg0 : memref<10xi32>
  // CHECK-DAG: %[[RECV1:.*]] = executor.abi.recv %arg1 : memref<10xi32>
  %0 = executor.abi.recv %arg0 : tensor<10xi32>

  // CHECK-DAG: memref.copy %[[RECV0]], %[[RECV1]]
  // CHECK-DAG: %[[UB:.+]] = ub.poison
  // CHECK-DAG: %[[CLONE:.+]] = bufferization.clone %[[UB]]
  // CHECK: return %[[CLONE]]
  %1 = executor.abi.send %0 to %arg1 : tensor<10xi32>
  return %1 : tensor<10xi32>
}

// -----

// CHECK-LABEL: func @send_with_undef
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {{.*}}, %[[arg1:.+]]: !executor.ptr<host> {{.*}})
func.func @send_with_undef(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<4x8xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<4x8xf32>, undef>})
    -> (tensor<4x8xf32>)
      attributes {
        executor.func_abi = (tensor<4x8xf32>) -> (tensor<4x8xf32>)
      } {
  // CHECK-DAG: %[[RECV0:.*]] = executor.abi.recv %arg0 : memref<4x8xf32>
  %0 = executor.abi.recv %arg0 : tensor<4x8xf32>
  // CHECK-DAG: %[[CLONE:.+]] = bufferization.clone %[[RECV0]]
  // CHECK: return %[[CLONE]]
  %1 = executor.abi.send %0 to %arg1 : tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @send_mixed_undef
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {{.*}}, %[[arg1:.+]]: !executor.ptr<host> {{.*}})
func.func @send_mixed_undef(
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<10xi32>, undef>},
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<10xi32>>})
    -> (tensor<10xi32>, tensor<10xi32>)
      attributes {
        executor.func_abi = () -> (tensor<10xi32>, tensor<10xi32>)
      } {
  %0 = arith.constant dense<1> : tensor<10xi32>
  // CHECK-DAG: %[[UB:.+]] = ub.poison
  // CHECK-DAG: %[[RECV1:.*]] = executor.abi.recv %[[arg1]] : memref<10xi32>
  // CHECK-DAG: %[[GLOBAL:.+]] = memref.get_global
  // CHECK-DAG: memref.copy %[[GLOBAL]], %[[RECV1]]
  // CHECK-DAG: %[[CLONE:.+]] = bufferization.clone %[[GLOBAL]]
  // CHECK-DAG: %[[CLONE2:.+]] = bufferization.clone %[[UB]]
  // CHECK: return %[[CLONE]], %[[CLONE2]]
  %1 = executor.abi.send %0 to %arg1 : tensor<10xi32>
  %2 = executor.abi.send %0 to %arg2 : tensor<10xi32>
  return %1, %2 : tensor<10xi32>, tensor<10xi32>
}
