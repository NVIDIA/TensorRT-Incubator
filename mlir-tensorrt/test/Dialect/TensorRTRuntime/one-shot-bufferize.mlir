// RUN: mlir-tensorrt-opt %s -split-input-file -empty-tensor-to-alloc-tensor -plan-bufferize | FileCheck %s

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
//       CHECK:     return %[[arg3]]

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
//  CHECK-SAME: (%[[arg0:.+]]: !trtrt.context, %[[arg1:.+]]: !cuda.stream, %[[arg2:.+]]: memref<4xi32, #plan.memory_space<device>>, %[[arg3:.+]]: memref<128xf32, #plan.memory_space<device>>) -> memref<128xf32, #plan.memory_space<device>>
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:     memref.copy %[[arg2]], %[[alloc]] : memref<4xi32, #plan.memory_space<device>> to memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:     %[[alloc_0:.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:     memref.copy %[[arg2]], %[[alloc_0]] : memref<4xi32, #plan.memory_space<device>> to memref<4xi32, #plan.memory_space<host_pinned>>
//       CHECK:     trtrt.enqueue %[[arg0]] stream(%[[arg1]]) host_tensor_args [0, 1, 3] (%[[alloc]], %[[alloc]], %[[arg2]], %[[alloc_0]]) outs(%[[arg3]]) : (memref<4xi32, #plan.memory_space<host_pinned>>, memref<4xi32, #plan.memory_space<host_pinned>>, memref<4xi32, #plan.memory_space<device>>, memref<4xi32, #plan.memory_space<host_pinned>>) -> memref<128xf32, #plan.memory_space<device>>
//       CHECK:     return %[[arg3]]
