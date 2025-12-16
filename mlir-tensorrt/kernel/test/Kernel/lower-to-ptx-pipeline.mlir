// RUN: kernel-opt -split-input-file %s \
// RUN: -pass-pipeline="builtin.module(kernel-set-gpu-target{chip=sm_80},gpu.module(kernel-lower-to-ptx-pipeline))" | \
// RUN: FileCheck %s

gpu.module @kernels {
  func.func @kernel_with_non_vectorized_linalg_op(
      %arg0: memref<1x1x1xi32, strided<[?, ?, 1], offset: ?>>,
      %arg1: memref<50257x768xf32, strided<[?, 1], offset: ?>>,
      %arg2: memref<768xf32, strided<[1], offset: ?>>) {
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c768 = arith.constant 768 : index
    %c50256 = arith.constant 50256 : index
    %0 = gpu.block_id  x
    %1 = gpu.thread_id  x
    %2 = arith.remsi %1, %c256 : index
    %3 = arith.cmpi slt, %2, %c0 : index
    %4 = arith.addi %2, %c256 : index
    %5 = arith.select %3, %4, %2 : index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<768xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
    %6 = arith.muli %5, %c3 : index
    %7 = arith.addi %offset, %6 : index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%7], sizes: [3], strides: [1] : memref<f32> to memref<3xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%reinterpret_cast : memref<3xf32, strided<[1], offset: ?>>) {
    ^bb0(%out: f32):
      %8 = linalg.index 0 : index
      %9 = arith.muli %5, %c3 : index
      %10 = arith.addi %8, %9 : index
      %11 = arith.muli %0, %c768 : index
      %12 = arith.addi %10, %11 : index
      %13 = memref.load %arg0[%c0, %c0, %c0] : memref<1x1x1xi32, strided<[?, ?, 1], offset: ?>>
      %14 = arith.index_cast %13 : i32 to index
      %15 = arith.maxsi %14, %c0 : index
      %16 = arith.minsi %15, %c50256 : index
      %17 = memref.load %arg1[%16, %12] : memref<50257x768xf32, strided<[?, 1], offset: ?>>
      linalg.yield %17 : f32
    }
    return
  }
}

// CHECK-LABEL: llvm.func @kernel_with_non_vectorized_linalg_op
//       CHECK:   gpu.module.kernels.ptx_data
