// RUN: mlir-tensorrt-opt %s -split-input-file -plan-create-shape-funcs="abi-version=1" -inline | FileCheck %s

// Test zero-rank tensor handling in shape function creation
// The fix ensures that when creating constant tensors with zero values,
// we create a rank-1 tensor with zeros instead of a zero-rank tensor.

func.func public @test_zero_rank_shape(%arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<f32>>},
                                %arg1: i32,
                                %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<?xf32>>},
                                %arg3: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<f32>>})
    attributes {executor.func_abi = (tensor<f32>, i32) -> (tensor<?xf32>, tensor<f32>)} {
  %0 = executor.abi.recv %arg0 : tensor<f32>
  %1 = stablehlo.exponential %0 : tensor<f32>
  %2 = plan.with_shape %1() : (tensor<f32>) -> tensor<f32>
  %size = tensor.from_elements %arg1 : tensor<1xi32>
  %3 = stablehlo.dynamic_broadcast_in_dim %2, %size, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  %4 = plan.with_shape %3(%arg1) : (tensor<?xf32>, i32) -> tensor<?xf32>
  executor.abi.send %4 to %arg2 : tensor<?xf32>
  executor.abi.send %2 to %arg3 : tensor<f32>
  return
}

// CHECK-LABEL: func.func public @test_zero_rank_shape(
// CHECK-LABEL: func.func public @test_zero_rank_shape_get_shapes(
//   CHECK: %[[cst:.+]] = arith.constant dense<0> : tensor<1xindex, #plan.memory_space<host>>
//   CHECK-DAG: %[[v0:.+]] = arith.index_cast %{{.+}} : i32 to index
//   CHECK-DAG: %[[v1:.+]] = tensor.from_elements %[[v0]] : tensor<1xindex, #plan.memory_space<host>>
//   CHECK: executor.abi.send %[[v1]] to %{{.+}}
//   CHECK: executor.abi.send %[[cst]] to %{{.+}}
//   CHECK: return
