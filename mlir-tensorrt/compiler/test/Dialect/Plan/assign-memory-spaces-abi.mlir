// RUN: mlir-tensorrt-opt %s -split-input-file -plan-assign-memory-spaces -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @abi_recv_host_memory_space
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32, #plan.memory_space<host>>>},
// CHECK-SAME:  %[[arg1:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32, #plan.memory_space<device>>>}) -> tensor<2xf32, #plan.memory_space<device>>
func.func @abi_recv_host_memory_space(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32>>}) -> tensor<2xf32>
      attributes {
        executor.func_abi = (tensor<2xf32>) -> (tensor<2xf32>)
      } {
  // CHECK: %[[recv:.+]] = executor.abi.recv %[[arg0]] {memory_space = #plan.memory_space<host>} : tensor<2xf32, #plan.memory_space<host>>
  // CHECK: %[[transfer:.+]] = plan.transfer %[[recv]] : tensor<2xf32, #plan.memory_space<host>> to tensor<2xf32, #plan.memory_space<device>>
  // CHECK: %[[c0:.+]] = arith.constant 0 : index
  // CHECK: %[[extract:.+]] = tensor.extract {{.*}}[%[[c0]]] : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: %[[from_elements:.+]] = tensor.from_elements %[[extract]], %[[extract]] : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: %[[send:.+]] = executor.abi.send {{.*}} to %[[arg1]] : tensor<2xf32, #plan.memory_space<device>>
  %0 = executor.abi.recv %arg0 {memory_space = #plan.memory_space<host>} : tensor<2xf32>
  %c0 = arith.constant 0 : index
  %1 = tensor.extract %0[%c0] : tensor<2xf32>
  %2 = tensor.from_elements %1, %1 : tensor<2xf32>
  %3 = executor.abi.send %2 to %arg1 : tensor<2xf32>
  return %3 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @abi_recv_device_memory_space
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32, #plan.memory_space<device>>>},
// CHECK-SAME:  %[[arg1:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32, #plan.memory_space<device>>>}) -> tensor<2xf32, #plan.memory_space<device>>
func.func @abi_recv_device_memory_space(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32>>}) -> tensor<2xf32>
      attributes {
        executor.func_abi = (tensor<2xf32>) -> (tensor<2xf32>)
      } {
  // CHECK: %[[recv:.+]] = executor.abi.recv %[[arg0]] {memory_space = #plan.memory_space<device>} : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: %[[c1:.+]] = arith.constant 1 : index
  // CHECK: tensor.extract {{.*}}[%[[c1]]] : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: tensor.from_elements {{.*}} : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: %[[send:.+]] = executor.abi.send {{.*}} to %[[arg1]] : tensor<2xf32, #plan.memory_space<device>>
  %0 = executor.abi.recv %arg0 {memory_space = #plan.memory_space<device>} : tensor<2xf32>
  %c1 = arith.constant 1 : index
  %1 = tensor.extract %0[%c1] : tensor<2xf32>
  %2 = tensor.from_elements %1, %1 : tensor<2xf32>
  %3 = executor.abi.send %2 to %arg1 : tensor<2xf32>
  return %3 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @abi_recv_host_pinned_memory_space
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<10xf32, #plan.memory_space<host_pinned>>>},
// CHECK-SAME:  %[[arg1:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<5xf32, #plan.memory_space<device>>>}) -> tensor<5xf32, #plan.memory_space<device>>
func.func @abi_recv_host_pinned_memory_space(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<10xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<5xf32>>}) -> tensor<5xf32>
      attributes {
        executor.func_abi = (tensor<10xf32>) -> (tensor<5xf32>)
      } {
  // CHECK: %[[recv:.+]] = executor.abi.recv %[[arg0]] {memory_space = #plan.memory_space<host_pinned>} : tensor<10xf32, #plan.memory_space<host_pinned>>
  // CHECK: %[[transfer:.+]] = plan.transfer %[[recv]] : tensor<10xf32, #plan.memory_space<host_pinned>> to tensor<10xf32, #plan.memory_space<device>>
  // CHECK: %[[c0:.+]] = arith.constant 0 : index
  // CHECK: %[[slice:.+]] = tensor.extract_slice {{.*}}[%[[c0]]] [5] [1] : tensor<10xf32, #plan.memory_space<device>> to tensor<5xf32, #plan.memory_space<device>>
  // CHECK: %[[send:.+]] = executor.abi.send {{.*}} to %[[arg1]] : tensor<5xf32, #plan.memory_space<device>>
  %0 = executor.abi.recv %arg0 {memory_space = #plan.memory_space<host_pinned>} : tensor<10xf32>
  %c0 = arith.constant 0 : index
  %1 = tensor.extract_slice %0[%c0][5][1] : tensor<10xf32> to tensor<5xf32>
  %2 = executor.abi.send %1 to %arg1 : tensor<5xf32>
  return %2 : tensor<5xf32>
}

// -----

// CHECK-LABEL: func.func @abi_recv_multiple_memory_spaces
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32, #plan.memory_space<host>>>},
// CHECK-SAME:  %[[arg1:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32, #plan.memory_space<device>>>},
// CHECK-SAME:  %[[arg2:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32, #plan.memory_space<device>>>}) -> tensor<2xf32, #plan.memory_space<device>>
func.func @abi_recv_multiple_memory_spaces(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32>>},
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32>>}) -> tensor<2xf32>
      attributes {
        executor.func_abi = (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>)
      } {
  // CHECK: executor.abi.recv %[[arg0]] {memory_space = #plan.memory_space<host>} : tensor<2xf32, #plan.memory_space<host>>
  // CHECK: plan.transfer {{.*}} : tensor<2xf32, #plan.memory_space<host>> to tensor<2xf32, #plan.memory_space<device>>
  // CHECK: executor.abi.recv %[[arg1]] {memory_space = #plan.memory_space<device>} : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: arith.addf {{.*}} : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: %[[send:.+]] = executor.abi.send {{.*}} to %[[arg2]] : tensor<2xf32, #plan.memory_space<device>>
  %0 = executor.abi.recv %arg0 {memory_space = #plan.memory_space<host>} : tensor<2xf32>
  %1 = executor.abi.recv %arg1 {memory_space = #plan.memory_space<device>} : tensor<2xf32>
  %2 = arith.addf %0, %1 : tensor<2xf32>
  %3 = executor.abi.send %2 to %arg2 : tensor<2xf32>
  return %3 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @abi_recv_with_operations
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<4xf32, #plan.memory_space<host>>>},
// CHECK-SAME:  %[[arg1:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32, #plan.memory_space<device>>>}) -> tensor<2xf32, #plan.memory_space<device>>
func.func @abi_recv_with_operations(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<4xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32>>}) -> tensor<2xf32>
      attributes {
        executor.func_abi = (tensor<4xf32>) -> (tensor<2xf32>)
      } {
  // CHECK: %[[recv:.+]] = executor.abi.recv %[[arg0]] {memory_space = #plan.memory_space<host>} : tensor<4xf32, #plan.memory_space<host>>
  // CHECK: %[[transfer:.+]] = plan.transfer %[[recv]] : tensor<4xf32, #plan.memory_space<host>> to tensor<4xf32, #plan.memory_space<device>>
  // CHECK: %[[c0:.+]] = arith.constant 0 : index
  // CHECK: %[[slice:.+]] = tensor.extract_slice {{.*}}[%[[c0]]] [2] [1] : tensor<4xf32, #plan.memory_space<device>> to tensor<2xf32, #plan.memory_space<device>>
  // CHECK: %[[cst:.+]] = arith.constant dense<1.000000e+00> : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: %[[add:.+]] = arith.addf {{.*}} : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: %[[send:.+]] = executor.abi.send {{.*}} to %[[arg1]] : tensor<2xf32, #plan.memory_space<device>>
  %0 = executor.abi.recv %arg0 {memory_space = #plan.memory_space<host>} : tensor<4xf32>
  %c0 = arith.constant 0 : index
  %1 = tensor.extract_slice %0[%c0][2][1] : tensor<4xf32> to tensor<2xf32>
  %cst = arith.constant dense<1.0> : tensor<2xf32>
  %2 = arith.addf %1, %cst : tensor<2xf32>
  %3 = executor.abi.send %2 to %arg1 : tensor<2xf32>
  return %3 : tensor<2xf32>
}

// -----

// Test with plan.memory_space constraint on argument - should override the recv memory_space attribute
// CHECK-LABEL: func.func @abi_recv_with_arg_constraint
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32, #plan.memory_space<device>>>, plan.memory_space = #plan.memory_space<device>},
// CHECK-SAME:  %[[arg1:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32, #plan.memory_space<device>>>, plan.memory_space = #plan.memory_space<device>}) -> tensor<2xf32, #plan.memory_space<device>>
func.func @abi_recv_with_arg_constraint(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32>>, plan.memory_space = #plan.memory_space<device>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32>>, plan.memory_space = #plan.memory_space<device>}) -> tensor<2xf32>
      attributes {
        executor.func_abi = (tensor<2xf32>) -> (tensor<2xf32>)
      } {
  // CHECK: executor.abi.recv %[[arg0]] {memory_space = #plan.memory_space<device>} : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: arith.constant dense<2.000000e+00> : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: arith.mulf {{.*}} : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: %[[send:.+]] = executor.abi.send {{.*}} to %[[arg1]] : tensor<2xf32, #plan.memory_space<device>>
  %0 = executor.abi.recv %arg0 {memory_space = #plan.memory_space<device>} : tensor<2xf32>
  %cst = arith.constant dense<2.0> : tensor<2xf32>
  %1 = arith.mulf %0, %cst : tensor<2xf32>
  %2 = executor.abi.send %1 to %arg1 : tensor<2xf32>
  return %2 : tensor<2xf32>
}

// -----

// Test with plan.memory_space constraint at function level - should apply to all arguments
// CHECK-LABEL: func.func @abi_recv_with_function_constraint
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32, #plan.memory_space<host>>>},
// CHECK-SAME:  %[[arg1:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32, #plan.memory_space<host>>>}) -> tensor<2xf32, #plan.memory_space<host>>
func.func @abi_recv_with_function_constraint(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32>>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32>>}) -> tensor<2xf32>
      attributes {
        executor.func_abi = (tensor<2xf32>) -> (tensor<2xf32>),
        plan.memory_space = #plan.memory_space<host>
      } {
  // CHECK: executor.abi.recv %[[arg0]] {memory_space = #plan.memory_space<host>} : tensor<2xf32, #plan.memory_space<host>>
  // CHECK: arith.constant dense<3.000000e+00> : tensor<2xf32, #plan.memory_space<host>>
  // CHECK: arith.divf {{.*}} : tensor<2xf32, #plan.memory_space<host>>
  // CHECK: %[[send:.+]] = executor.abi.send {{.*}} to %[[arg1]] : tensor<2xf32, #plan.memory_space<host>>
  %0 = executor.abi.recv %arg0 {memory_space = #plan.memory_space<host>} : tensor<2xf32>
  %cst = arith.constant dense<3.0> : tensor<2xf32>
  %1 = arith.divf %0, %cst : tensor<2xf32>
  %2 = executor.abi.send %1 to %arg1 : tensor<2xf32>
  return %2 : tensor<2xf32>
}

// -----

// Test with mixed constraints - argument constraint overrides function constraint
// CHECK-LABEL: func.func @abi_recv_with_mixed_constraints
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32, #plan.memory_space<device>>>, plan.memory_space = #plan.memory_space<device>},
// CHECK-SAME:  %[[arg1:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32, #plan.memory_space<host>>>},
// CHECK-SAME:  %[[arg2:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32, #plan.memory_space<host>>>}) -> tensor<2xf32, #plan.memory_space<host>>
func.func @abi_recv_with_mixed_constraints(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32>>, plan.memory_space = #plan.memory_space<device>},
    %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32>>},
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<2xf32>>}) -> tensor<2xf32>
      attributes {
        executor.func_abi = (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>),
        plan.memory_space = #plan.memory_space<host>
      } {
  // CHECK: executor.abi.recv %[[arg0]] {memory_space = #plan.memory_space<device>} : tensor<2xf32, #plan.memory_space<device>>
  // CHECK: plan.transfer {{.*}} : tensor<2xf32, #plan.memory_space<device>> to tensor<2xf32, #plan.memory_space<host>>
  // CHECK: executor.abi.recv %[[arg1]] {memory_space = #plan.memory_space<host>} : tensor<2xf32, #plan.memory_space<host>>
  // CHECK: arith.addf {{.*}} : tensor<2xf32, #plan.memory_space<host>>
  // CHECK: %[[send:.+]] = executor.abi.send {{.*}} to %[[arg2]] : tensor<2xf32, #plan.memory_space<host>>
  %0 = executor.abi.recv %arg0 {memory_space = #plan.memory_space<device>} : tensor<2xf32>
  %1 = executor.abi.recv %arg1 {memory_space = #plan.memory_space<host>} : tensor<2xf32>
  %2 = arith.addf %0, %1 : tensor<2xf32>
  %3 = executor.abi.send %2 to %arg2 : tensor<2xf32>
  return %3 : tensor<2xf32>
}

// -----

// Test abi.send without corresponding recv - output only function
// CHECK-LABEL: func.func @abi_send_only
// CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<4xf32, #plan.memory_space<device>>>}) -> tensor<4xf32, #plan.memory_space<device>>
func.func @abi_send_only(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<4xf32>>}) -> tensor<4xf32>
      attributes {
        executor.func_abi = () -> (tensor<4xf32>)
      } {
  // CHECK: arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32, #plan.memory_space<device>>
  // CHECK: %[[send:.+]] = executor.abi.send {{.*}} to %[[arg0]] : tensor<4xf32, #plan.memory_space<device>>
  %cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %0 = executor.abi.send %cst to %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Test conflicting memory_space attribute on abi.recv and plan.memory_space constraint on argument
func.func @abi_recv_conflict_error(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<2xf32>>, plan.memory_space = #plan.memory_space<host>})
      attributes {
        executor.func_abi = (tensor<2xf32>) -> ()
      } {
  // expected-error @below {{memory_space attribute #plan.memory_space<device> conflicts with plan.memory_space constraint #plan.memory_space<host> on argument 0}}
  %0 = executor.abi.recv %arg0 {memory_space = #plan.memory_space<device>} : tensor<2xf32>
  return
}
