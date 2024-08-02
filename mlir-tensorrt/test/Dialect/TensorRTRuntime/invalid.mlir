// RUN: mlir-tensorrt-opt %s -split-input-file -verify-diagnostics

executor.constant_resource @trt_func_engine_data dense<0> : vector<8xi8>
func.func @main(%arg0: memref<1x3x256x256xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<1x3x256x256xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg3: !trtrt.context) -> memref<1x3x256x256xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x256x256xf32>
  %0 = executor.load_constant_resource @trt_func_engine_data : !executor.ptr<host>
  %1 = trtrt.create_runtime : !trtrt.runtime
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
