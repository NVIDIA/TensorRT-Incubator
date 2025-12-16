// RUN: kernel-opt -allow-unregistered-dialect --pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries use-encoding-for-memory-space},gpu.module(test-kernel-one-shot-bufferize),cse,canonicalize)" -split-input-file %s | FileCheck %s

// Note: more thorough testing of joint host/device bufferization is done under "compiler/test-internal"
// project through testing of 'plan-module-bufferize'.

// This test bufferizes the host program first and then the device program using the
// test pass 'test-kernel-one-shot-bufferize'. The main compiler uses 'plan-module-bufferize' to jointly
// bufferize the whole program in a single pass.

gpu.module @kernels attributes {
  kernel.gpu_module_kind = #kernel.gpu_module_kind.default
} {
  func.func @test_kernel(%arg0: tensor<1x32x4xi32>, %arg1: tensor<1x32x4xi32>) -> tensor<1x32x4xi32> {
    %1 = linalg.copy ins(%arg0: tensor<1x32x4xi32>) outs(%arg1: tensor<1x32x4xi32>) -> tensor<1x32x4xi32>
    return %1 : tensor<1x32x4xi32>
  }
}

#device_space = 1 : i64
#host_space = 2 : i64

func.func @main() -> index {
  %threads_per_block = arith.constant 32 : index
  %blocks = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c31 = arith.constant 31 : index

  %cFF = arith.constant dense<-1> : tensor<1x32x4xi32, #device_space>
  %cI0 = arith.constant dense<0> : tensor<1x32x4xi32, #device_space>
  %2 = kernel.call  @kernels::@test_kernel
    grid [%blocks, %c1, %c1] block [%threads_per_block, %c1, %c1]
    (%cFF) outs(%cI0) :
      (tensor<1x32x4xi32, #device_space>, tensor<1x32x4xi32, #device_space>) -> (tensor<1x32x4xi32, #device_space>)
  %host = bufferization.alloc_tensor() : tensor<1x32x4xi32, #host_space>
  %3 = bufferization.materialize_in_destination %2 in %host : (tensor<1x32x4xi32, #device_space>, tensor<1x32x4xi32, #host_space>) -> tensor<1x32x4xi32, #host_space>
  %val = tensor.extract %3[%c0, %c0, %c0] : tensor<1x32x4xi32, #host_space>
  "executor.print"(%val) {
    message = "\"host_tensor[0, 0, 0] = %d\""
  }: (i32) -> ()
  %val1 = tensor.extract %3[%c0, %c31, %c3] : tensor<1x32x4xi32, #host_space>
  "executor.print" (%val1) {
    message = "\"host_tensor[0, 31, 3] = %d\""
  } : (i32) -> ()
  return %c0 : index
}

// CHECK-LABEL: @main
//   CHECK-DAG:     %[[c32:.+]] = arith.constant 32 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:     %[[c31:.+]] = arith.constant 31 : index
//   CHECK-DAG:     %[[v0:.+]] = memref.get_global @__constant_1x32x4xi32 : memref<1x32x4xi32, 1>
//   CHECK-DAG:     %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x32x4xi32, 1>
//   CHECK-DAG:     %[[casted_v0:.+]] = memref.cast %[[v0]]
//   CHECK-DAG:     %[[casted_alloc:.+]] = memref.cast %[[alloc]]
//       CHECK:     kernel.call @kernels::@test_kernel grid[%[[c1]], %[[c1]], %[[c1]]] block[%[[c32]], %[[c1]], %[[c1]]] (%[[casted_v0]]) outs(%[[casted_alloc]])
//       CHECK:     %[[alloc_0:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x32x4xi32, 2>
//       CHECK:     memref.copy %[[alloc]], %[[alloc_0]] : memref<1x32x4xi32, 1> to memref<1x32x4xi32, 2>
//       CHECK:     %[[v1:.+]] = memref.load %[[alloc_0]][%[[c0]], %[[c0]], %[[c0]]] : memref<1x32x4xi32, 2>
//       CHECK:     %[[v2:.+]] = memref.load %[[alloc_0]][%[[c0]], %[[c31]], %[[c3]]] : memref<1x32x4xi32, 2>

// -----

// CHECK-LABEL: @kernel_func_read_write_outs_arg
builtin.module @kernel_func_read_write_outs_arg {
  gpu.module @kernels attributes {
    kernel.gpu_module_kind = #kernel.gpu_module_kind.default
  } {
    // Both kernels have read-write access on their 'outs' operands.
    func.func @kernel(%input: tensor<100xf32>, %output: tensor<f32>) -> tensor<f32> {
      %0 = linalg.reduce {arith.addf}
        ins(%input: tensor<100xf32>) outs(%output: tensor<f32>) dimensions = [0]
      return %0 : tensor<f32>
    }
    func.func @kernel1(%input: tensor<f32>, %output: tensor<f32>) -> tensor<f32> {
      %a = tensor.extract %input[] : tensor<f32>
      %b = tensor.extract %output[] : tensor<f32>
      %c = arith.addf %a, %b : f32
      %d = tensor.insert %c into %output[] : tensor<f32>
      return %d : tensor<f32>
    }
  }

  // CHECK: func.func @main
  func.func @main(%input: tensor<100xf32>) -> tensor<f32> {
    %c1 = arith.constant 1 : index
    %out = tensor.empty() : tensor<f32>
    %cst = arith.constant 0.0 : f32
    // CHECK: %[[B0:.+]] = memref.alloc() {{.*}} : memref<f32>
    %filled = linalg.fill ins(%cst: f32) outs(%out: tensor<f32>) -> tensor<f32>
    // CHECK: linalg.fill {{.*}} outs(%[[B0]] :
    // CHECK: %[[B1:.+]] = memref.alloc
    // CHECK: memref.copy %[[B0]], %[[B1]] :
    // CHECK: %[[B1_casted:.+]] = memref.cast %[[B1]]
    // CHECK: kernel.call @kernels::@kernel {{.*}} outs(%[[B1_casted]])
    %0 = kernel.call @kernels::@kernel
      grid [%c1, %c1, %c1] block[%c1, %c1, %c1] (%input) outs(%filled) : (tensor<100xf32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[B0_casted:.+]] = memref.cast %[[B0]]
    // CHECK: kernel.call @kernels::@kernel1 {{.*}} outs(%[[B0_casted]])
    %1 = kernel.call @kernels::@kernel1
      grid [%c1, %c1, %c1] block[%c1, %c1, %c1] (%0) outs(%filled) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %1  : tensor<f32>
  }
}
