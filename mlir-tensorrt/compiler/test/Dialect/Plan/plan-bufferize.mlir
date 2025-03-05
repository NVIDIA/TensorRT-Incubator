// RUN: mlir-tensorrt-opt %s -split-input-file -empty-tensor-to-alloc-tensor -plan-module-bufferize | FileCheck %s

// tensorrt.module @trt_engines {
//   func.func @trt_while_loop_region(%arg0: tensor<1xi32>) -> tensor<i1> {
//     %cst_i32 = tensorrt.constant dense<10> : tensor<1xi32>
//     %0 = tensorrt.element_wise <kLESS>(%arg0, %cst_i32 : tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
//     %1 = tensorrt.collapse_rank %0 : tensor<1xi1> to tensor<i1>
//     return %1 : tensor<i1>
//   }
//   func.func @trt_while_loop_region_0(%arg0: tensor<10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32>) {
//     %cst_i32 = tensorrt.constant dense<1> : tensor<1xi32>
//     %0 = tensorrt.slice %arg0[%arg1: tensor<1xi32>][1][1] : tensor<10xf32> to tensor<1xf32>
//     %1 = tensorrt.element_wise <kSUM>(%cst_i32, %arg1 : tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
//     %2 = tensorrt.element_wise <kSUM>(%0, %arg2 : tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
//     return %1, %2 : tensor<1xi32>, tensor<1xf32>
//   }
// }

trtrt.compiled_func @trt_while_loop_region dense<0> : vector<1xi8>
trtrt.compiled_func @trt_while_loop_region_0 dense<0> : vector<1xi8>

func.func @main(%arg0: tensor<10xf32>) -> tensor<1xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
  %0 = cuda.stream.create : !cuda.stream
  %1 = tensor.empty() : tensor<i1>
  %2 = trtrt.get_function @trt_while_loop_region : !trtrt.context
  %3 = tensor.empty() : tensor<1xi32>
  %4 = tensor.empty() : tensor<1xf32>
  %5 = trtrt.get_function @trt_while_loop_region_0 : !trtrt.context
  %6:2 = scf.while (%arg1 = %cst, %arg2 = %cst_0) : (tensor<1xi32>, tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32>) {
    %7 = trtrt.enqueue %2 stream(%0) (%arg1) outs(%1) : (tensor<1xi32>) -> tensor<i1>
    %extracted = tensor.extract %7[] : tensor<i1>
    scf.condition(%extracted) %arg1, %arg2 : tensor<1xi32>, tensor<1xf32>
  } do {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<1xf32>):
    %7:2 = trtrt.enqueue %5 stream(%0) (%arg0, %arg1, %arg2) outs(%arg1, %arg2) : (tensor<10xf32>, tensor<1xi32>, tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32>)
    scf.yield %7#0, %7#1 : tensor<1xi32>, tensor<1xf32>
  }
  cuda.stream.sync %0 : !cuda.stream
  return %6#1 : tensor<1xf32>
}

//       CHECK:   memref.global "private" constant @__constant_1xf32
//       CHECK:   memref.global "private" constant @__constant_1xi32
// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: memref<10xf32, #plan.memory_space<device>>) -> memref<1xf32, #plan.memory_space<device>>
//       CHECK:     %[[v0:.+]] = memref.get_global @__constant_1xi32
//       CHECK:     %[[v1:.+]] = memref.get_global @__constant_1xf32
//       CHECK:     %[[v2:.+]] = cuda.stream.create : !cuda.stream
//       CHECK:     %[[alloc:.+]] = memref.alloc()
//       CHECK:     %[[v3:.+]] = trtrt.get_function @trt_while_loop_region
//       CHECK:     %[[v4:.+]] = trtrt.get_function @trt_while_loop_region_0
//       CHECK:     %[[alloc_0:.+]] = memref.alloc()
//       CHECK:     memref.copy %[[v0]], %[[alloc_0]]
//       CHECK:     %[[alloc_1:.+]] = memref.alloc()
//       CHECK:     memref.copy %[[v1]], %[[alloc_1]]
//       CHECK:     %[[v5:.+]]:2 = scf.while (%[[arg1:.+]] = %[[alloc_0]], %[[arg2:.+]] = %[[alloc_1]])
//       CHECK:       trtrt.enqueue %[[v3]] stream(%[[v2]]) (%[[arg1]]) outs(%[[alloc]])
//       CHECK:       %[[alloc_2:.+]] = memref.alloc() {{.*}} : memref<i1, #plan.memory_space<host_pinned>>
//       CHECK:       memref.copy %[[alloc]], %[[alloc_2]] : memref<i1, #plan.memory_space<device>> to memref<i1, #plan.memory_space<host_pinned>>
//       CHECK:       %[[v6:.+]] = memref.load %[[alloc_2]][] : memref<i1, #plan.memory_space<host_pinned>>
//       CHECK:       scf.condition(%[[v6]]) %[[arg1]], %[[arg2]]
//       CHECK:     } do {
//       CHECK:     ^bb0(%[[arg1:.+]]: memref<1xi32, #plan.memory_space<device>>, %[[arg2:.+]]: memref<1xf32, #plan.memory_space<device>>):
//       CHECK:       trtrt.enqueue %[[v4]] stream(%[[v2]]) (%[[arg0]], %[[arg1]], %[[arg2]]) outs(%[[arg1]], %[[arg2]])
//       CHECK:       scf.yield %[[arg1]], %[[arg2]]
//       CHECK:     cuda.stream.sync %[[v2]]
//       CHECK:     return %[[v5]]#1

// -----

func.func @copy_back_host_dynamic(%arg0: tensor<?xf32>, %arg1: index) -> f32 {
  %0 = tensor.extract %arg0[%arg1] : tensor<?xf32>
  return %0 : f32
}

// CHECK-LABEL: func.func @copy_back_host_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: memref<?xf32, #plan.memory_space<device>>, %[[arg1:.+]]: index) -> f32 {
//       CHECK:     %[[subview:.+]] = memref.subview %[[arg0]][%[[arg1]]] [1] [1] : memref<?xf32, #plan.memory_space<device>> to memref<1xf32, strided<[1], offset: ?>, #plan.memory_space<device>>
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 16 : i64} : memref<1xf32, #plan.memory_space<host_pinned>>
//       CHECK:     memref.copy %[[subview]], %[[alloc]] : memref<1xf32, strided<[1], offset: ?>, #plan.memory_space<device>> to memref<1xf32, #plan.memory_space<host_pinned>>
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[v0:.+]] = memref.load %[[alloc]][%[[c0]]] : memref<1xf32, #plan.memory_space<host_pinned>>
//       CHECK:     return %[[v0]] : f32


// -----

func.func @from_elements(%arg0: i32) -> tensor<4xi32, #plan.memory_space<host>> {
  %0 = tensor.from_elements %arg0, %arg0, %arg0, %arg0 : tensor<4xi32, #plan.memory_space<host>>
  return %0 : tensor<4xi32,#plan.memory_space<host>>
}

// CHECK-LABEL: @from_elements
//   CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[alloc:.+]] = memref.alloc() {alignment = 16 : i64} : memref<4xi32, #plan.memory_space<host>>
//  CHECK-NEXT:     memref.store %[[arg0]], %[[alloc]][%[[c0]]] : memref<4xi32, #plan.memory_space<host>>
//  CHECK-NEXT:     memref.store %[[arg0]], %[[alloc]][%[[c1]]] : memref<4xi32, #plan.memory_space<host>>
//  CHECK-NEXT:     memref.store %[[arg0]], %[[alloc]][%[[c2]]] : memref<4xi32, #plan.memory_space<host>>
//  CHECK-NEXT:     memref.store %[[arg0]], %[[alloc]][%[[c3]]] : memref<4xi32, #plan.memory_space<host>>
//  CHECK-NEXT:     return

// -----

func.func @copy_host_to_device(%arg0: tensor<4xi32, #plan.memory_space<host>>,
                               %arg1: tensor<4xi32, #plan.memory_space<device>>)
                               -> tensor<4xi32, #plan.memory_space<device>> {
  %0 = bufferization.materialize_in_destination %arg0 in %arg1
    : (tensor<4xi32, #plan.memory_space<host>>,
       tensor<4xi32, #plan.memory_space<device>>)
    -> tensor<4xi32, #plan.memory_space<device>>
  return %0 : tensor<4xi32,#plan.memory_space<device>>
}

// CHECK-LABEL: @copy_host_to_device
//  CHECK-SAME: (%[[arg0:.+]]: memref<4xi32, #plan.memory_space<host>>, %[[arg1:.+]]: memref<4xi32, #plan.memory_space<device>>)
//  CHECK-NEXT:     memref.copy %[[arg0]], %[[arg1]] : memref<4xi32, #plan.memory_space<host>> to memref<4xi32, #plan.memory_space<device>>
//  CHECK-NEXT:     return

// -----

func.func @copy_device_to_host(%arg0: tensor<4xi32, #plan.memory_space<device>>,
                               %arg1: tensor<4xi32, #plan.memory_space<host>>)
                               -> tensor<4xi32, #plan.memory_space<host>> {
  %0 = bufferization.materialize_in_destination %arg0 in %arg1
    : (tensor<4xi32, #plan.memory_space<device>>,
       tensor<4xi32, #plan.memory_space<host>>)
    -> tensor<4xi32, #plan.memory_space<host>>
  return %0 : tensor<4xi32,#plan.memory_space<host>>
}

// CHECK-LABEL: @copy_device_to_host
//  CHECK-SAME: (%[[arg0:.+]]: memref<4xi32, #plan.memory_space<device>>, %[[arg1:.+]]: memref<4xi32, #plan.memory_space<host>>)
//  CHECK-NEXT:     memref.copy %[[arg0]], %[[arg1]] : memref<4xi32, #plan.memory_space<device>> to memref<4xi32, #plan.memory_space<host>>
//  CHECK-NEXT:     return

// -----

func.func @copy_device_constant_to_host() -> (tensor<2xi32, #plan.memory_space<host>>) {
  %0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  // Create output tensor in host space.
  %1 = bufferization.alloc_tensor() {memory_space = #plan.memory_space<host>} : tensor<2xi32, #plan.memory_space<host>>
  %2 = bufferization.materialize_in_destination %0 in %1
    : (tensor<2xi32>,
       tensor<2xi32, #plan.memory_space<host>>)
    -> tensor<2xi32, #plan.memory_space<host>>
  return %2 : tensor<2xi32, #plan.memory_space<host>>
}

// CHECK-LABEL: @copy_device_constant_to_host()
//  CHECK-NEXT:     %0 = memref.get_global @__constant_2xi32 : memref<2xi32, #plan.memory_space<device>>
//  CHECK-NEXT:     %alloc = memref.alloc() {alignment = 16 : i64} : memref<2xi32, #plan.memory_space<host>>
//  CHECK-NEXT:     memref.copy %0, %alloc : memref<2xi32, #plan.memory_space<device>> to memref<2xi32, #plan.memory_space<host>>
//  CHECK-NEXT:     return