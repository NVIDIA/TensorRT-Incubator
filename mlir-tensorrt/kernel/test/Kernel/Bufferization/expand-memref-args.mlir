// RUN: kernel-opt %s -kernel-expand-memref-args -split-input-file | FileCheck %s

gpu.module @kernels attributes {
  kernel.gpu_module_kind = #kernel.gpu_module_kind.default<>
} {
  func.func @kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }

  func.func @kernel2(%arg0: memref<2x?xf32, strided<[?, ?], offset: ?>>,
                     %arg1: memref<1xf32, strided<[?], offset: 4>>,
                     %arg2: f32, %arg3: i32) {
    return
  }
}

// CHECK-LABEL: func.func @kernel
//  CHECK-SAME: (%[[arg0:.+]]: memref<f32>, %[[arg1:.+]]: memref<f32>)
// CHECK-LABEL: func.func @kernel2
//  CHECK-SAME: (%[[arg0:.+]]: memref<f32>, %[[arg1:.+]]: index, %[[arg2:.+]]: index, %[[arg3:.+]]: index, %[[arg4:.+]]: index, %[[arg5:.+]]: memref<f32>, %[[arg6:.+]]: index, %[[arg7:.+]]: f32, %[[arg8:.+]]: i32)
