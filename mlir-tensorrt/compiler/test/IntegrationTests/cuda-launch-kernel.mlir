// REQUIRES: host-has-at-least-1-gpus
// RUN: mlir-tensorrt-opt -pass-pipeline="builtin.module(kernel-set-gpu-target{infer-target-from-host},kernel-annotate-entrypoints,gpu.module(kernel-lower-to-nvvm,reconcile-unrealized-casts,translate-nvvm-to-ptx))" %s | \
// RUN: mlir-tensorrt-opt -pass-pipeline="builtin.module(convert-kernel-to-cuda,drop-nested-modules,convert-memref-to-cuda,convert-plan-to-executor,convert-cuda-to-executor,executor-lowering-pipeline)" | \
// RUN: mlir-tensorrt-translate -mlir-to-runtime-executable | \
// RUN: mlir-tensorrt-runner -input-type=rtexe -features=core,cuda

gpu.module @kernels attributes {
  kernel.gpu_module_kind = #kernel.gpu_module_kind.default
} {
  func.func @binary_shift_right(%arg0: memref<1x32x4xi32>, %arg2: i32, %arg3: f32, %arg1: memref<1x32x4xi32>) {

    %id = gpu.thread_id x
    %block_id = gpu.block_id x
    %c0 = arith.constant 0 : index

    // Vectorized load
    %0 = vector.load %arg0[%block_id, %id, %c0] : memref<1x32x4xi32>, vector<4xi32>

    // Extract each element.
    %1 = vector.extract %0[0] : i32 from vector<4xi32>
    %2 = vector.extract %0[0] : i32 from vector<4xi32>
    %3 = vector.extract %0[0] : i32 from vector<4xi32>
    %4 = vector.extract %0[0] : i32 from vector<4xi32>

    // Perform a logical right shift by 16 for each element.
    %arg3_i32 = arith.bitcast %arg3: f32 to i32
    %shift = arith.addi %arg2, %arg3_i32 : i32
    %5 = arith.shrui %1, %shift : i32
    %6 = arith.shrui %2, %shift : i32
    %7 = arith.shrui %3, %shift : i32
    %8 = arith.shrui %4, %shift : i32

    // Construct the result vector
    %9 = arith.constant dense<0> : vector<4xi32>
    %10 = vector.insert %5, %9[0] : i32 into vector<4xi32>
    %11 = vector.insert %6, %10[1] : i32 into vector<4xi32>
    %12 = vector.insert %7, %11[2] : i32 into vector<4xi32>
    %13 = vector.insert %8, %12[3] : i32 into vector<4xi32>

    // Vectorized store to the output operand.
    vector.store %13, %arg1[%block_id, %id, %c0] : memref<1x32x4xi32>, vector<4xi32>

    return
  }
}

!hostMemRef = memref<1x32x4xi32, #plan.memory_space<host>>
!devMemRef = memref<1x32x4xi32, #plan.memory_space<device>>

memref.global @buffer1 : !devMemRef = dense<-1>
memref.global @buffer2 : !devMemRef = dense<0>

func.func @main() -> index {
  %threads_per_block = arith.constant 32 : index
  %blocks = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c31 = arith.constant 31 : index
  %c7 = arith.constant 7 : i32
  %c1_i32 = arith.constant 1 : i32
  // To test the passing of float scalars, cast this 1 to
  // f32. We will cast back  to i32 and sum with c7 in the kernel
  // to get shift size of 8.
  %c1f = arith.bitcast %c1_i32 : i32 to f32

  %0 = memref.get_global @buffer1 : !devMemRef
  %1 = memref.get_global @buffer2 : !devMemRef
  %2 = memref.alloc() : !hostMemRef

  kernel.call  @kernels::@binary_shift_right
    grid [%blocks, %c1, %c1] block [%threads_per_block, %c1, %c1]
    (%0, %c7, %c1f) outs(%1) :
      (!devMemRef, i32, f32, !devMemRef) -> ()

  memref.copy %1, %2 : !devMemRef to !hostMemRef

  %val = memref.load %2[%c0, %c0, %c0] : !hostMemRef
  executor.print "host_memref[0, 0, 0] = %d"(%val: i32)
  %val1 = memref.load %2[%c0, %c31, %c3] : !hostMemRef
  executor.print "host_memref[0, 31, 3] = %d"(%val1: i32)

  kernel.call  @kernels::@binary_shift_right
    grid [%blocks, %c1, %c1] block [%threads_per_block, %c1, %c1]
    (%1, %c7, %c1f) outs(%0) :
      (!devMemRef, i32, f32, !devMemRef) -> ()

  memref.copy %0, %2 : !devMemRef to !hostMemRef

  %val2 = memref.load %2[%c0, %c0, %c0] : !hostMemRef
  executor.print "host_memref[0, 0, 0] = %d"(%val2: i32)
  %val3 = memref.load %2[%c0, %c31, %c3] : !hostMemRef
  executor.print "host_memref[0, 31, 3] = %d"(%val3: i32)

  return %c0 : index
}

// CHECK: host_memref[0, 0, 0] = 16777215
// CHECK: host_memref[0, 31, 3] = 16777215
// CHECK: host_memref[0, 0, 0] = 65535
// CHECK: host_memref[0, 31, 3] = 6553
