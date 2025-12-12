// RUN: kernel-opt -split-input-file %s -test-kernel-one-shot-bufferize -verify-diagnostics

// expected-error @below {{kernel one-shot-module-bufferize failed}}
gpu.module @invalid_global_alloc {
  func.func @func(%input: tensor<100xf32>, %output: tensor<100xf32>) -> tensor<100xf32> {
    // expected-error @below {{failed to bufferize op}}
    // expected-error @below {{bufferization attempted to allocate memory inside of device function with an unsupported address space attribute: #gpu.address_space<global>}}
    %global_alloc = bufferization.alloc_tensor() {memory_space = #gpu.address_space<global>} : tensor<100x100xf32>
    %gmem0 = tensor.insert_slice %input into %global_alloc[0, 0] [100, 1][1, 1] : tensor<100xf32> into tensor<100x100xf32>
    %gmem3 = tensor.extract_slice %global_alloc[0, 3] [100, 1][1, 1] : tensor<100x100xf32> to tensor<100xf32>
    %out0 = tensor.insert_slice %gmem3 into %output [0] [100][1] : tensor<100xf32> into tensor<100xf32>
    return %out0: tensor<100xf32>
  }
}
