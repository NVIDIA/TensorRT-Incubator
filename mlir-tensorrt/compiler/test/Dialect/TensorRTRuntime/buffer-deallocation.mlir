// RUN: mlir-tensorrt-opt %s -plan-ownership-based-buffer-deallocation -canonicalize -buffer-deallocation-simplification -split-input-file | FileCheck %s

trtrt.compiled_func @data dense_resource<__elided__> : tensor<19740xi8>


// CHECK-LABEL: func.func @test_return_allocation
func.func @test_return_allocation(%arg0: memref<512x1024x1024xf32, #plan.memory_space<device>>) 
    -> memref<512x1024x1024xf32, #plan.memory_space<device>> {
  %0 = trtrt.get_function @data : !trtrt.context
  %1 = cuda.get_global_stream 0
  // We should return the allocated buffer without copies or deallocations.
  // CHECK: %[[ALLOCATED_BUFFER:.*]] = trtrt.enqueue_alloc
  %2 = trtrt.enqueue_alloc %0 stream(%1) (%arg0) : (memref<512x1024x1024xf32, #plan.memory_space<device>>) -> memref<512x1024x1024xf32, #plan.memory_space<device>>
  // CHECK-NEXT: return %[[ALLOCATED_BUFFER]] 
  return %2 : memref<512x1024x1024xf32, #plan.memory_space<device>>
}

// -----

trtrt.compiled_func @data dense_resource<__elided__> : tensor<19740xi8>

func.func private @external(%arg0: memref<512x1024x1024xf32, #plan.memory_space<device>>)

// CHECK-LABEL: func.func @result_should_be_deallocated
func.func @result_should_be_deallocated(%arg0: memref<512x1024x1024xf32, #plan.memory_space<device>>) {
  %0 = trtrt.get_function @data : !trtrt.context
  %1 = cuda.get_global_stream 0
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[ALLOCATED_BUFFER:.*]] = trtrt.enqueue_alloc
  %2 = trtrt.enqueue_alloc %0 stream(%1) (%arg0) : (memref<512x1024x1024xf32, #plan.memory_space<device>>) -> memref<512x1024x1024xf32, #plan.memory_space<device>>
  // CHECK-NEXT: call @external(%[[ALLOCATED_BUFFER]]) 
  // CHECK-NEXT: bufferization.dealloc (%[[ALLOCATED_BUFFER]] : {{.*}}) if (%[[TRUE]])
  func.call @external(%2) : (memref<512x1024x1024xf32, #plan.memory_space<device>>) -> ()  
  return
}

// -----

trtrt.compiled_func @data dense_resource<__elided__> : tensor<19740xi8>

func.func private @external(%arg0: memref<512x1024x1024xf32, #plan.memory_space<device>>)

// Tests one result needing to be deallocated and one is returned to caller.

// CHECK-LABEL: func.func @test_mixed
func.func @test_mixed(%arg0: memref<512x1024x1024xf32, #plan.memory_space<device>>) -> memref<512x1024x1024xf32, #plan.memory_space<device>> {
  %0 = trtrt.get_function @data : !trtrt.context
  %1 = cuda.get_global_stream 0
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[ALLOC:.*]]:2 = trtrt.enqueue_alloc
  %2, %3 = trtrt.enqueue_alloc %0 stream(%1) (%arg0) : (memref<512x1024x1024xf32, #plan.memory_space<device>>) 
    -> (memref<512x1024x1024xf32, #plan.memory_space<device>>, memref<512x1024x1024xf32, #plan.memory_space<device>>)
  // CHECK-NEXT: call @external(%[[ALLOC]]#0)
  func.call @external(%2) : (memref<512x1024x1024xf32, #plan.memory_space<device>>) -> ()
  // CHECK-NEXT: bufferization.dealloc (%[[ALLOC]]#1 : {{.*}}) if (%[[TRUE]])
  // CHECK-NEXT: return %[[ALLOC]]#0
  return %2 : memref<512x1024x1024xf32, #plan.memory_space<device>>
}
