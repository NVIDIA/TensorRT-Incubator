// RUN: mlir-tensorrt-opt %s -split-input-file -plan-module-bufferize | FileCheck %s

// CHECK-LABEL: @default_module
gpu.module @default_module attributes {
  kernel.gpu_module_kind = #kernel.gpu_module_kind.default
} {
  // CHECK: func.func @kernel(%arg0: memref<1024x1024xf32, strided<[?, ?], offset: ?>>
  func.func @kernel(%arg0: tensor<1024x1024xf32>,
   %arg1: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    return %arg1 : tensor<1024x1024xf32>
  }
}

