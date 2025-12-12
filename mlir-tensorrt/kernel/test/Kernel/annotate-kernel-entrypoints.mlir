// RUN: kernel-opt %s -split-input-file -kernel-annotate-entrypoints -verify-diagnostics | FileCheck %s

// Test basic annotation: functions called from host should get gpu.kernel attribute
gpu.module @kernels {
  func.func @kernel1(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
    return
  }

  func.func @kernel2(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
    return
  }

  func.func @device_func(%arg0: memref<128xf32>) {
    return
  }
}

func.func @host_func(%arg0: memref<128xf32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  kernel.call @kernels::@kernel1 grid[%c1, %c1, %c1] block[%c32, %c1, %c1] (%arg0) outs(%arg0)
    : (memref<128xf32>, memref<128xf32>) -> ()
  kernel.call @kernels::@kernel2 grid[%c1, %c1, %c1] block[%c32, %c1, %c1] (%arg0) outs(%arg0)
    : (memref<128xf32>, memref<128xf32>) -> ()
  return
}

// CHECK-LABEL: gpu.module @kernels
// CHECK: func.func @kernel1
// CHECK-SAME: attributes {gpu.kernel}
// CHECK: func.func @kernel2
// CHECK-SAME: attributes {gpu.kernel}
// CHECK: func.func @device_func
// CHECK-NOT: gpu.kernel

// -----

// Test that functions only called from device code don't get annotated
gpu.module @kernels {
  func.func @kernel1(%arg0: memref<128xf32>) {
    return
  }

  func.func @kernel2(%arg0: memref<128xf32>) {
    func.call @kernel1(%arg0) : (memref<128xf32>) -> ()
    return
  }
}

// CHECK-LABEL: gpu.module @kernels
// CHECK: func.func @kernel1
// CHECK-NOT: gpu.kernel
// CHECK: func.func @kernel2
// CHECK-NOT: gpu.kernel

// -----

// Test multiple gpu.module operations
gpu.module @kernels1 {
  func.func @kernel1(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
    return
  }
}

gpu.module @kernels2 {
  func.func @kernel2(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
    return
  }

  func.func @device_func(%arg0: memref<128xf32>) {
    return
  }
}

func.func @host_func(%arg0: memref<128xf32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  kernel.call @kernels1::@kernel1 grid[%c1, %c1, %c1] block[%c32, %c1, %c1] (%arg0) outs(%arg0)
    : (memref<128xf32>, memref<128xf32>) -> ()
  kernel.call @kernels2::@kernel2 grid[%c1, %c1, %c1] block[%c32, %c1, %c1] (%arg0) outs(%arg0)
    : (memref<128xf32>, memref<128xf32>) -> ()
  return
}

// CHECK-LABEL: gpu.module @kernels1
// CHECK: func.func @kernel1
// CHECK-SAME: attributes {gpu.kernel}

// CHECK-LABEL: gpu.module @kernels2
// CHECK: func.func @kernel2
// CHECK-SAME: attributes {gpu.kernel}
// CHECK: func.func @device_func
// CHECK-NOT: gpu.kernel

// -----

// Test error: function called from both host and device should fail
gpu.module @kernels {
  // expected-error @below {{kernel function 'kernel1' cannot be called from both host and device code}}
  func.func @kernel1(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
    return
  }

  func.func @kernel2(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
    func.call @kernel1(%arg0, %arg1) : (memref<128xf32>, memref<128xf32>) -> ()
    return
  }
}

func.func @host_func(%arg0: memref<128xf32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  kernel.call @kernels::@kernel1 grid[%c1, %c1, %c1] block[%c32, %c1, %c1] (%arg0) outs(%arg0)
    : (memref<128xf32>, memref<128xf32>) -> ()
  return
}
