// RUN: mlir-tensorrt-opt %s -split-input-file -plan-confirm-argument-donation | FileCheck %s

trtrt.compiled_func @tensorrt_cluster_engine_data dense_resource<__elided__> : tensor<14020xi8>
func.func @donation_rejected(%arg0: memref<2x2xi32, #plan.memory_space<device>> {plan.aliasing_output = 0 : i32}) -> (memref<2x2xi32, #plan.memory_space<device>>) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<2x2xi32, #plan.memory_space<device>>
  return %alloc : memref<2x2xi32, #plan.memory_space<device>>
}

// CHECK-LABEL: @donation_rejected
//  CHECK-NOT: {plan.aliasing_output = 0 : i32}

// -----

func.func @donation_accepted(%arg0: memref<5x6xf32, #plan.memory_space<device>> {plan.aliasing_output = 0 : i32}, %arg1: memref<2x3xf32, #plan.memory_space<device>>) -> (memref<5x6xf32, #plan.memory_space<device>>) {
  return %arg0 : memref<5x6xf32, #plan.memory_space<device>>
}

// CHECK-LABEL: @donation_accepted
//  CHECK-SAME: {plan.aliasing_output = 0 : i32}

// -----

func.func @donation_uncertain(%arg0: memref<4xf32, #plan.memory_space<device>> {plan.aliasing_output = 0 : i32}, %arg1: memref<4xf32, #plan.memory_space<device>>, %cond: i1) -> (memref<4xf32, #plan.memory_space<device>>) {
  %result = arith.select %cond, %arg0, %arg1 : memref<4xf32, #plan.memory_space<device>>
  return %result : memref<4xf32, #plan.memory_space<device>>
}

// CHECK-LABEL: @donation_uncertain
// CHECK-NOT: {plan.aliasing_output = 0 : i32}
