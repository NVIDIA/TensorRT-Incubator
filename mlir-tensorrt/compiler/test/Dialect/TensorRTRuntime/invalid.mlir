// RUN: mlir-tensorrt-opt %s -split-input-file -verify-diagnostics

executor.constant_resource @trt_func_engine_data dense<0> : vector<8xi8>
func.func @main(%arg0: memref<1x3x256x256xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<1x3x256x256xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg3: !trtrt.context) -> memref<1x3x256x256xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x256x256xf32>
  %0 = executor.load_constant_resource @trt_func_engine_data : !executor.ptr<host>
  %4 = cuda.stream.create : !cuda.stream
  // expected-error @below {{custom op 'trtrt.enqueue' 3 operands present, but expected 1}}
  trtrt.enqueue %arg3 stream(%4) (%arg3, %4, %arg0) outs(%alloc) : (memref<1x3x256x256xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x3x256x256xf32>
  cuda.stream.sync %4 : !cuda.stream
  return %alloc : memref<1x3x256x256xf32>
}

// -----

func.func @enqueue_host_tensor_oob(%arg0: !trtrt.context, %arg1: !cuda.stream,
                %arg2: tensor<1xf32>, %arg3: tensor<1xi32>, %arg4: tensor<1xf32>) -> tensor<1xf32> {
  // expected-error @below {{'trtrt.enqueue' op host_tensor_args value 2 is out of bounds}}
  %0 = trtrt.enqueue %arg0 stream(%arg1) host_tensor_args [2] (%arg2, %arg3) outs(%arg4)
    : (tensor<1xf32>, tensor<1xi32>) -> (tensor<1xf32>)
  return %0 : tensor<1xf32>
}

// -----

func.func @enqueue_host_tensor_el_type(%arg0: !trtrt.context, %arg1: !cuda.stream,
                %arg2: tensor<1xf32>, %arg3: tensor<1xi32>, %arg4: tensor<1xf32>) -> tensor<1xf32> {
  // expected-error @below {{'trtrt.enqueue' op host tensor arguments must have element type i32, but input arg 0 has type 'tensor<1xf32>'}}
  %0 = trtrt.enqueue %arg0 stream(%arg1) host_tensor_args [0] (%arg2, %arg3) outs(%arg4)
    : (tensor<1xf32>, tensor<1xi32>) -> (tensor<1xf32>)
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: @enqueue_alloc_no_results
func.func @enqueue_alloc_no_results(%ctx: !trtrt.context, %stream: !cuda.stream, %arg0: tensor<1x3x256x256xf32>) {
  // expected-error @+1 {{at least one result is required.}}
  trtrt.enqueue_alloc %ctx stream(%stream) (%arg0) : (tensor<1x3x256x256xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: @enqueue_alloc_unranked_tensor
func.func @enqueue_alloc_unranked_tensor(%ctx: !trtrt.context, %stream: !cuda.stream, %arg0: tensor<1x3x256x256xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{result must be either RankedTensorType or MemRefType}}
  %result = trtrt.enqueue_alloc %ctx stream(%stream) (%arg0) : (tensor<1x3x256x256xf32>) -> tensor<*xf32>
  return %result : tensor<*xf32>
}

// -----

// CHECK-LABEL: @enqueue_alloc_non_tensor_result
func.func @enqueue_alloc_non_tensor_result(%ctx: !trtrt.context, %stream: !cuda.stream, %arg0: tensor<1x3x256x256xf32>) -> f32 {
  // expected-error @+1 {{result #0 must be variadic of memref of any type values or tensor of any type values, but got 'f32}}
  %result = trtrt.enqueue_alloc %ctx stream(%stream) (%arg0) : (tensor<1x3x256x256xf32>) -> f32
  return %result : f32
}
