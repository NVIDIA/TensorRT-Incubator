// RUN: executor-opt %s -executor-generate-abi-wrappers -split-input-file | FileCheck %s

// Simple function with scalar inputs and outputs
// CHECK-LABEL: func.func private @scalar_func_impl
// CHECK-SAME:    (%arg0: i32, %arg1: f32) -> i64
// CHECK-LABEL: func.func public @scalar_func
// CHECK-SAME:    (%arg0: i32, %arg1: f32, %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, i64>, executor.result_slot = 0 : i32})
// CHECK-SAME:    attributes {executor.func_abi = (i32, f32) -> i64}
// CHECK:         %[[CALL:.+]] = call @scalar_func_impl(%arg0, %arg1) : (i32, f32) -> i64
// CHECK:         executor.abi.send %[[CALL]] to %arg2 : i64
// CHECK:         return
func.func @scalar_func(%arg0: i32, %arg1: f32) -> i64 {
  %c0 = arith.constant 0 : i64
  return %c0 : i64
}

// -----

// Function with memref input and output
// CHECK-LABEL: func.func private @memref_func_impl
// CHECK-SAME:    (%arg0: memref<10xi32>) -> memref<5xf32>
// CHECK-LABEL: func.func public @memref_func
// CHECK-SAME:    (%arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
// CHECK-SAME:     %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<5xf32>>, executor.result_slot = 0 : i32})
// CHECK-SAME:    attributes {executor.func_abi = (memref<10xi32>) -> memref<5xf32>}
// CHECK:         %[[RECV:.+]] = executor.abi.recv %arg0 : memref<10xi32>
// CHECK:         %[[CALL:.+]] = call @memref_func_impl(%[[RECV]]) : (memref<10xi32>) -> memref<5xf32>
// CHECK:         executor.abi.send %[[CALL]] to %arg1 : memref<5xf32>
// CHECK:         return
func.func @memref_func(%arg0: memref<10xi32>) -> memref<5xf32> {
  %alloc = memref.alloc() : memref<5xf32>
  return %alloc : memref<5xf32>
}

// -----

// Function with mixed scalar and memref inputs/outputs
// CHECK-LABEL: func.func private @mixed_func_impl
// CHECK-SAME:    (%arg0: memref<10x20xf32>, %arg1: i32, %arg2: f64) -> (memref<5xi32>, index)
// CHECK-LABEL: func.func public @mixed_func
// CHECK-SAME:    (%arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10x20xf32>>},
// CHECK-SAME:     %arg1: i32,
// CHECK-SAME:     %arg2: f64,
// CHECK-SAME:     %arg3: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<5xi32>>, executor.result_slot = 0 : i32},
// CHECK-SAME:     %arg4: !executor.ptr<host> {executor.abi = #executor.arg<byref, index>, executor.result_slot = 1 : i32})
// CHECK-SAME:    attributes {executor.func_abi = (memref<10x20xf32>, i32, f64) -> (memref<5xi32>, index)}
// CHECK:         %[[RECV:.+]] = executor.abi.recv %arg0 : memref<10x20xf32>
// CHECK:         %[[CALL:.+]]:2 = call @mixed_func_impl(%[[RECV]], %arg1, %arg2) : (memref<10x20xf32>, i32, f64) -> (memref<5xi32>, index)
// CHECK:         executor.abi.send %[[CALL]]#0 to %arg3 : memref<5xi32>
// CHECK:         executor.abi.send %[[CALL]]#1 to %arg4 : index
// CHECK:         return
func.func @mixed_func(%arg0: memref<10x20xf32>, %arg1: i32, %arg2: f64) -> (memref<5xi32>, index) {
  %alloc = memref.alloc() : memref<5xi32>
  %c0 = arith.constant 0 : index
  return %alloc, %c0 : memref<5xi32>, index
}

// -----

// Function with multiple memref inputs and outputs
// CHECK-LABEL: func.func private @multi_memref_func_impl
// CHECK-SAME:    (%arg0: memref<10xi32>, %arg1: memref<20xf32>, %arg2: memref<?xi64>) -> (memref<5xi32>, memref<15xf64>)
// CHECK-LABEL: func.func public @multi_memref_func
// CHECK-SAME:    (%arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
// CHECK-SAME:     %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<20xf32>>},
// CHECK-SAME:     %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<?xi64>>},
// CHECK-SAME:     %arg3: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<5xi32>>, executor.result_slot = 0 : i32},
// CHECK-SAME:     %arg4: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<15xf64>>, executor.result_slot = 1 : i32})
// CHECK-SAME:    attributes {executor.func_abi = (memref<10xi32>, memref<20xf32>, memref<?xi64>) -> (memref<5xi32>, memref<15xf64>)}
// CHECK:         %[[RECV0:.+]] = executor.abi.recv %arg0 : memref<10xi32>
// CHECK:         %[[RECV1:.+]] = executor.abi.recv %arg1 : memref<20xf32>
// CHECK:         %[[RECV2:.+]] = executor.abi.recv %arg2 : memref<?xi64>
// CHECK:         %[[CALL:.+]]:2 = call @multi_memref_func_impl(%[[RECV0]], %[[RECV1]], %[[RECV2]]) : (memref<10xi32>, memref<20xf32>, memref<?xi64>) -> (memref<5xi32>, memref<15xf64>)
// CHECK:         executor.abi.send %[[CALL]]#0 to %arg3 : memref<5xi32>
// CHECK:         executor.abi.send %[[CALL]]#1 to %arg4 : memref<15xf64>
// CHECK:         return
func.func @multi_memref_func(%arg0: memref<10xi32>, %arg1: memref<20xf32>, %arg2: memref<?xi64>) -> (memref<5xi32>, memref<15xf64>) {
  %alloc1 = memref.alloc() : memref<5xi32>
  %alloc2 = memref.alloc() : memref<15xf64>
  return %alloc1, %alloc2 : memref<5xi32>, memref<15xf64>
}

// -----

// Function with tensor types
// CHECK-LABEL: func.func private @tensor_func_impl
// CHECK-SAME:    (%arg0: tensor<10xi32>) -> tensor<5xf32>
// CHECK-LABEL: func.func public @tensor_func
// CHECK-SAME:    (%arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, tensor<10xi32>>},
// CHECK-SAME:     %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, tensor<5xf32>>, executor.result_slot = 0 : i32})
// CHECK-SAME:    attributes {executor.func_abi = (tensor<10xi32>) -> tensor<5xf32>}
// CHECK:         %[[RECV:.+]] = executor.abi.recv %arg0 : tensor<10xi32>
// CHECK:         %[[CALL:.+]] = call @tensor_func_impl(%[[RECV]]) : (tensor<10xi32>) -> tensor<5xf32>
// CHECK:         executor.abi.send %[[CALL]] to %arg1 : tensor<5xf32>
// CHECK:         return
func.func @tensor_func(%arg0: tensor<10xi32>) -> tensor<5xf32> {
  %c0 = arith.constant dense<0.0> : tensor<5xf32>
  return %c0 : tensor<5xf32>
}

// -----

// Function with no outputs (only inputs)
// CHECK-LABEL: func.func private @no_output_func_impl
// CHECK-SAME:    (%arg0: memref<10xi32>, %arg1: i32)
// CHECK-LABEL: func.func public @no_output_func
// CHECK-SAME:    (%arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
// CHECK-SAME:     %arg1: i32)
// CHECK-SAME:    attributes {executor.func_abi = (memref<10xi32>, i32) -> ()}
// CHECK:         %[[RECV:.+]] = executor.abi.recv %arg0 : memref<10xi32>
// CHECK:         call @no_output_func_impl(%[[RECV]], %arg1) : (memref<10xi32>, i32) -> ()
// CHECK:         return
func.func @no_output_func(%arg0: memref<10xi32>, %arg1: i32) {
  return
}

// -----

// Private functions should not be transformed
// CHECK-LABEL: func.func private @private_func
// CHECK-SAME:    (%arg0: memref<10xi32>) -> memref<5xf32>
// CHECK-NOT:     executor.abi
func.func private @private_func(%arg0: memref<10xi32>) -> memref<5xf32> {
  %alloc = memref.alloc() : memref<5xf32>
  return %alloc : memref<5xf32>
}

// -----

// Two public functions that call each other
// CHECK-LABEL: func.func private @func_a_impl
// CHECK-SAME:    (%arg0: i32) -> i32
// CHECK:         %[[CALL:.+]] = call @func_b_impl(%arg0) : (i32) -> i32
// CHECK:         return %[[CALL]] : i32
// CHECK-LABEL: func.func public @func_a
// CHECK-SAME:    (%arg0: i32, %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, i32>, executor.result_slot = 0 : i32})
// CHECK-SAME:    attributes {executor.func_abi = (i32) -> i32}
// CHECK:         %[[CALL:.+]] = call @func_a_impl(%arg0) : (i32) -> i32
// CHECK:         executor.abi.send %[[CALL]] to %arg1 : i32
// CHECK-LABEL: func.func private @func_b_impl
// CHECK-SAME:    (%arg0: i32) -> i32
// CHECK:         %[[C1:.+]] = arith.constant 1 : i32
// CHECK:         %[[ADD:.+]] = arith.addi %arg0, %[[C1]] : i32
// CHECK:         %[[CALL:.+]] = call @func_a_impl(%[[ADD]]) : (i32) -> i32
// CHECK:         return %[[CALL]] : i32
// CHECK-LABEL: func.func public @func_b
// CHECK-SAME:    (%arg0: i32, %arg1: !executor.ptr<host> {executor.abi = #executor.arg<byref, i32>, executor.result_slot = 0 : i32})
// CHECK-SAME:    attributes {executor.func_abi = (i32) -> i32}
// CHECK:         %[[CALL:.+]] = call @func_b_impl(%arg0) : (i32) -> i32
// CHECK:         executor.abi.send %[[CALL]] to %arg1 : i32
func.func @func_a(%arg0: i32) -> i32 {
  %result = func.call @func_b(%arg0) : (i32) -> i32
  return %result : i32
}

func.func @func_b(%arg0: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %add = arith.addi %arg0, %c1 : i32
  %result = func.call @func_a(%add) : (i32) -> i32
  return %result : i32
}

// -----

// Function declarations should not be transformed
// CHECK-LABEL: func.func private @external_func
// CHECK-SAME:    (memref<10xi32>) -> memref<5xf32>
// CHECK-NOT:     executor.abi
func.func private @external_func(memref<10xi32>) -> memref<5xf32>
