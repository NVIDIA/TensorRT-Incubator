// RUN: mlir-tensorrt-opt -split-input-file %s | mlir-tensorrt-opt -split-input-file | FileCheck %s

func.func @compile() -> !trtrt.context {
  %0 = trtrt.create_runtime : !trtrt.runtime
  %1 = trtrt.compile @trt_engines::@trt_engine1 : !trtrt.context
  return %1 : !trtrt.context
}

// CHECK-LABEL: @compile
//  CHECK-SAME: () -> !trtrt.context {
//       CHECK:     %[[v0:.+]] = trtrt.create_runtime : !trtrt.runtime
//       CHECK:     %[[v1:.+]] = trtrt.compile @trt_engines::@trt_engine1 : !trtrt.context
//       CHECK:     return %[[v1]] : !trtrt.context

// -----

func.func @enqueue(%arg0: memref<10xf32>, %arg1: memref<10xf32>, %ctx: !trtrt.context, %stream: !cuda.stream) {
  trtrt.enqueue %ctx stream(%stream) (%arg0) outs(%arg1) : (memref<10xf32>) -> (memref<10xf32>)
  return
}

// CHECK-LABEL: @enqueue
//  CHECK-SAME: (%[[arg0:.+]]: memref<10xf32>, %[[arg1:.+]]: memref<10xf32>, %[[arg2:.+]]: !trtrt.context, %[[arg3:.+]]: !cuda.stream) {
//       CHECK:     trtrt.enqueue %[[arg2]] stream(%[[arg3]]) (%[[arg0]]) outs(%[[arg1]]) : (memref<10xf32>) -> memref<10xf32>

// -----

func.func @enqueue_alloc(%arg0: memref<10xf32>, %ctx: !trtrt.context, %stream: !cuda.stream) -> memref<10xf32> {
  %result = trtrt.enqueue_alloc %ctx stream(%stream) (%arg0) : (memref<10xf32>) -> (memref<10xf32>)
  return %result : memref<10xf32>
}

// CHECK-LABEL: @enqueue_alloc
//  CHECK-SAME: (%[[arg0:.+]]: memref<10xf32>, %[[arg1:.+]]: !trtrt.context, %[[arg2:.+]]: !cuda.stream) -> memref<10xf32> {
//       CHECK: %[[v1:.+]] = trtrt.enqueue_alloc %[[arg1]] stream(%[[arg2]]) (%[[arg0]]) : (memref<10xf32>) -> memref<10xf32>
//       CHECK: return %[[v1]] : memref<10xf32>

// -----


func.func @enqueue_host_tensor(%arg0: !trtrt.context, %arg1: !cuda.stream,
                %arg2: tensor<1xf32>, %arg3: tensor<1xi32>, %arg4: tensor<1xf32>) -> tensor<1xf32> {
  %0 = trtrt.enqueue %arg0 stream(%arg1) host_tensor_args [1] (%arg2, %arg3) outs(%arg4)
    : (tensor<1xf32>, tensor<1xi32>) -> (tensor<1xf32>)
  return %0 : tensor<1xf32>
}

// CHECK-LABEL: @enqueue_host_tensor
//       CHECK:  trtrt.enqueue %{{.+}} stream(%{{.+}}) host_tensor_args [1] (%{{.+}}, %{{.+}}) outs(%{{.+}}) : (tensor<1xf32>, tensor<1xi32>) -> tensor<1xf32>