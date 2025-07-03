// RUN: mlir-tensorrt-opt %s -plan-promote-host-tensors-to-host-pinned -split-input-file | FileCheck %s

func.func @test(%arg0: tensor<10xf32, #plan.memory_space<device>>) -> f32 {
  %c0 = arith.constant 0 : index
  %0 = tensor.cast %arg0 : tensor<10xf32, #plan.memory_space<device>> to tensor<10xf32, #plan.memory_space<host>>
  %1 = tensor.extract %0[%c0] : tensor<10xf32, #plan.memory_space<host>>
  return %1 : f32
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: %[[ARG0:.*]]: tensor<10xf32, #plan.memory_space<device>>
// CHECK: %[[CAST:.*]] = tensor.cast %[[ARG0]] : tensor<10xf32, #plan.memory_space<device>> to tensor<10xf32, #plan.memory_space<host_pinned>>
// CHECK: %[[EXTRACT:.*]] = tensor.extract %[[CAST]]{{.*}} : tensor<10xf32, #plan.memory_space<host_pinned>>
// CHECK: return %[[EXTRACT]] : f32

// -----

func.func @from_elements_case(%arg0 :f32) -> tensor<f32, #plan.memory_space<device>> {
  %0 = tensor.from_elements %arg0 : tensor<f32, #plan.memory_space<host>>
  %1 = tensor.cast %0 : tensor<f32, #plan.memory_space<host>> to tensor<f32, #plan.memory_space<device>>
  return %1 : tensor<f32, #plan.memory_space<device>>
}


// CHECK-LABEL: func.func @from_elements_case
//  CHECK-SAME: (%[[arg0:.+]]: f32) -> tensor<f32, #plan.memory_space<device>>
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[arg0]] : tensor<f32, #plan.memory_space<host_pinned>>
//       CHECK:     %[[cast:.+]] = tensor.cast %[[from_elements]]
//       CHECK:     return %[[cast]] :

