// RUN: mlir-tensorrt-opt %s -verify-diagnostics -split-input-file -empty-tensor-to-alloc-tensor -plan-module-bufferize | FileCheck %s

func.func @enqueue_simple(
    %ctx: !trtrt.context, %stream: !cuda.stream,
    %arg0: tensor<1x3x256x256xf32>, %arg1: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> {
  %0 = tensor.empty() : tensor<1x3x256x256xf32>
  %3 = trtrt.enqueue %ctx stream(%stream) (%arg0) outs(%arg1) : (tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
  return %3 : tensor<1x3x256x256xf32>
}

// CHECK-LABEL: @enqueue_simple
//  CHECK-SAME: (%[[arg0:.+]]: !trtrt.context, %[[arg1:.+]]: !cuda.stream, %[[arg2:.+]]: memref<1x3x256x256xf32, #plan.memory_space<device>>, %[[arg3:.+]]: memref<1x3x256x256xf32, #plan.memory_space<device>>)
//       CHECK:     trtrt.enqueue %[[arg0]] stream(%[[arg1]]) (%[[arg2]]) outs(%[[arg3]]) : (memref<1x3x256x256xf32, #plan.memory_space<device>>) -> memref<1x3x256x256xf32, #plan.memory_space<device>>
//       CHECK:     return

// -----

func.func @enqueue_alias(
    %ctx: !trtrt.context, %stream: !cuda.stream,
    %arg0: tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32> {
  %0 = tensor.empty() : tensor<1x3x256x256xf32>
  %3 = trtrt.enqueue %ctx stream(%stream) (%arg0) outs(%arg0) : (tensor<1x3x256x256xf32>) -> tensor<1x3x256x256xf32>
  return %3 : tensor<1x3x256x256xf32>
}

// CHECK-LABEL: func.func @enqueue_alias
//  CHECK-SAME: (%[[arg0:.+]]: !trtrt.context, %[[arg1:.+]]: !cuda.stream, %[[arg2:.+]]: memref<
//       CHECK:     %[[alloc:.+]] = memref.alloc() 
//       CHECK:     trtrt.enqueue %[[arg0]] stream(%[[arg1]]) (%[[arg2]]) outs(%[[alloc]]) 
//       CHECK:     return %[[alloc]] :

// -----

func.func @enqueue_host_tensors_space_check(
    %ctx: !trtrt.context, %stream: !cuda.stream,
    %arg0: tensor<4xi32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %host_tensor_alloc = bufferization.alloc_tensor() {
    memory_space = #plan.memory_space<host_pinned>
  } : tensor<4xi32>
  %host_tensor = bufferization.materialize_in_destination %arg0 in %host_tensor_alloc :
    (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %3 = trtrt.enqueue %ctx
    stream(%stream)
    host_tensor_args [0, 1, 3]
    (%host_tensor, %host_tensor, %arg0, %arg0)
    outs(%arg1) : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<128xf32>
  return %3 : tensor<128xf32>
}

// CHECK-LABEL: @enqueue_host_tensors_space_check
//  CHECK-SAME: (%[[arg0:.+]]: !trtrt.context, %[[arg1:.+]]: !cuda.stream, %[[arg2:.+]]: memref<4xi32, #plan.memory_space<device>>, %[[arg3:.+]]: memref<128xf32, #plan.memory_space<device>>)
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:     memref.copy %[[arg2]], %[[alloc]] : memref<4xi32, #plan.memory_space<device>> to memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:     %[[alloc_0:.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:     memref.copy %[[arg2]], %[[alloc_0]] : memref<4xi32, #plan.memory_space<device>> to memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:     trtrt.enqueue %[[arg0]] stream(%[[arg1]]) host_tensor_args [0, 1, 3] (%[[alloc]], %[[alloc]], %[[arg2]], %[[alloc_0]]) outs(%[[arg3]]) : (memref<4xi32, #plan.memory_space<host_pinned>>, memref<4xi32, #plan.memory_space<host_pinned>>, memref<4xi32, #plan.memory_space<device>>, memref<4xi32, #plan.memory_space<host_pinned>>) -> memref<128xf32, #plan.memory_space<device>>
//       CHECK:     return

// -----

func.func @enqueue_alloc_simple(
  %ctx: !trtrt.context, %stream: !cuda.stream,
  %arg0: tensor<1x3x256x256xf32>) -> tensor<?x?x?x?xf32> {
  %result = trtrt.enqueue_alloc %ctx stream(%stream) (%arg0) : (tensor<1x3x256x256xf32>) -> tensor<?x?x?x?xf32>
  return %result : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @enqueue_alloc_simple
//  CHECK-SAME: (%[[arg0:.+]]: !trtrt.context, %[[arg1:.+]]: !cuda.stream, %[[arg2:.+]]: memref<1x3x256x256xf32, #plan.memory_space<device>>) -> memref<?x?x?x?xf32, #plan.memory_space<device>>
//       CHECK: %[[result:.+]] = trtrt.enqueue_alloc %[[arg0]] stream(%[[arg1]]) (%[[arg2]]) : (memref<1x3x256x256xf32, #plan.memory_space<device>>) -> memref<?x?x?x?xf32, #plan.memory_space<device>>
//       CHECK:     return %[[result]] : memref<?x?x?x?xf32, #plan.memory_space<device>>

// -----

module {
  func.func @enqueue_alloc_multiple_returns(%arg0: !trtrt.context, %arg1: !cuda.stream, %arg2: memref<1x3x256x256xf32, #plan.memory_space<device>>) -> (memref<?x?x?x?xf32, #plan.memory_space<device>>, memref<?x?x?x?xf32, #plan.memory_space<device>>) {
    %0:2 = trtrt.enqueue_alloc %arg0 stream(%arg1) (%arg2) : (memref<1x3x256x256xf32, #plan.memory_space<device>>) -> (memref<?x?x?x?xf32, #plan.memory_space<device>>, memref<?x?x?x?xf32, #plan.memory_space<device>>)
    return %0#0, %0#1 : memref<?x?x?x?xf32, #plan.memory_space<device>>, memref<?x?x?x?xf32, #plan.memory_space<device>>
  }
}

// CHECK-LABEL: @enqueue_alloc_multiple_returns
//  CHECK-SAME: (%[[arg0:.+]]: !trtrt.context, %[[arg1:.+]]: !cuda.stream, %[[arg2:.+]]: memref<1x3x256x256xf32, #plan.memory_space<device>>) -> (memref<?x?x?x?xf32, #plan.memory_space<device>>, memref<?x?x?x?xf32, #plan.memory_space<device>>
//       CHECK: %[[result:.+]]:2 = trtrt.enqueue_alloc %[[arg0]] stream(%[[arg1]]) (%[[arg2]]) : (memref<1x3x256x256xf32, #plan.memory_space<device>>) -> (memref<?x?x?x?xf32, #plan.memory_space<device>>, memref<?x?x?x?xf32, #plan.memory_space<device>>)
//       CHECK:     return %[[result]]#0, %[[result]]#1 : memref<?x?x?x?xf32, #plan.memory_space<device>>, memref<?x?x?x?xf32, #plan.memory_space<device>>

// -----

func.func @enqueue_alloc_host_tensors_space_check(
    %ctx: !trtrt.context, %stream: !cuda.stream,
    %arg0: tensor<4xi32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %host_tensor_alloc = bufferization.alloc_tensor() {
    memory_space = #plan.memory_space<host_pinned>
  } : tensor<4xi32>
  %host_tensor = bufferization.materialize_in_destination %arg0 in %host_tensor_alloc :
    (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %3 = trtrt.enqueue_alloc %ctx
    stream(%stream)
    host_tensor_args [0, 1, 3]
    (%host_tensor, %host_tensor, %arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<128xf32>
  return %3 : tensor<128xf32>
}

// CHECK-LABEL: @enqueue_alloc_host_tensors_space_check
//  CHECK-SAME: (%[[arg0:.+]]: !trtrt.context, %[[arg1:.+]]: !cuda.stream, %[[arg2:.+]]: memref<4xi32, #plan.memory_space<device>>, %[[arg3:.+]]: memref<128xf32, #plan.memory_space<device>>) -> memref<128xf32, #plan.memory_space<device>>
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:     memref.copy %[[arg2]], %[[alloc]] : memref<4xi32, #plan.memory_space<device>> to memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:     %[[alloc_0:.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:     memref.copy %[[arg2]], %[[alloc_0]] : memref<4xi32, #plan.memory_space<device>> to memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:    %[[res:.+]] = trtrt.enqueue_alloc %[[arg0]] stream(%[[arg1]]) host_tensor_args [0, 1, 3] (%[[alloc]], %[[alloc]], %[[arg2]], %[[alloc_0]]) : (memref<4xi32, #plan.memory_space<host_pinned>>, memref<4xi32, #plan.memory_space<host_pinned>>, memref<4xi32, #plan.memory_space<device>>, memref<4xi32, #plan.memory_space<host_pinned>>) -> memref<128xf32, #plan.memory_space<device>>
//       CHECK:     return %[[res]] : memref<128xf32, #plan.memory_space<device>>