// RUN: mlir-tensorrt-opt %s -split-input-file --plan-assign-memory-spaces -canonicalize | FileCheck %s


func.func private @cond() -> i1

// CHECK-LABEL: func.func @scf_while_loop_2
// CHECK: scf.while {{.*}}tensor<1xf32, #plan.memory_space<device>>) -> tensor<1xf32, #plan.memory_space<device>>
func.func @scf_while_loop_2(%arg0: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %1 = tensor.from_elements %arg0  : tensor<1xf32>
  %2 = scf.while (%arg1 = %1) : (tensor<1xf32>) -> tensor<1xf32> {
    %cond = func.call @cond() : () -> i1
    %e = tensor.extract %arg1[%c0] : tensor<1xf32>
    %f = arith.addf %e, %e : f32
    %3 = tensor.from_elements %f : tensor<1xf32>
    scf.condition(%cond) %3 : tensor<1xf32>
  } do {
  ^bb0(%arg1: tensor<1xf32>):
    %extract = tensor.extract %arg1[%c0] : tensor<1xf32>
    %3 = arith.addf %extract, %extract : f32
    %4 = tensor.from_elements %3 : tensor<1xf32>
    scf.yield %4 : tensor<1xf32>
  }
  %3 = tensor.extract %2[%c0] : tensor<1xf32>
  return %3 : f32
}

// -----

// CHECK-LABEL: func.func @arith_constant
// CHECK: arith.constant {{.*}} : tensor<2xf32, #plan.memory_space<device>>
// CHECK: arith.constant {{.*}} : tensor<2xf32, #plan.memory_space<device>>
func.func @arith_constant() -> (tensor<2xf32>, tensor<2xf32>) {
  %0 = arith.constant dense<[0.1, 0.2]> : tensor<2xf32>
  %1 = arith.constant dense_resource<__elided__> : tensor<2xf32>
  return %0, %1 : tensor<2xf32>, tensor<2xf32>
}

// -----

// CHECK-LABEL: module @nested_module
// CHECK-NOT: #plan.memory_space
module @outer {
module @nested_module {
  func.func @nested_func() -> tensor<2xf32> {
    %0 = arith.constant dense<[0.1, 0.2]> : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
}

// -----

// CHECK-LABEL: func.func @existing_constraint_1
// CHECK: tensor.extract {{.*}}<host>
func.func @existing_constraint_1(%arg0: tensor<2xf32, #plan.memory_space<host>>) -> f32 {
  %c0 = arith.constant 0 : index
  %0 = tensor.extract %arg0[%c0] : tensor<2xf32, #plan.memory_space<host>>
  return %0 : f32
}

// -----

// CHECK-LABEL: func.func @existing_constraint_2
// CHECK-NOT: tensor.cast
// CHECK: tensor.extract {{.*}}<host>
func.func @existing_constraint_2(%arg0: tensor<2xf32, #plan.memory_space<host>>) -> f32 {
  %c0 = arith.constant 0 : index
  %1 = tensor.cast %arg0 : tensor<2xf32, #plan.memory_space<host>> to tensor<2xf32>
  %0 = tensor.extract %1[%c0] : tensor<2xf32>
  return %0 : f32
}

