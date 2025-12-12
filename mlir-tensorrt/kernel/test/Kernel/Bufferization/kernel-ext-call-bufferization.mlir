// RUN: kernel-opt %s -one-shot-bufferize="bufferize-function-boundaries use-encoding-for-memory-space function-boundary-type-conversion=identity-layout-map" | FileCheck %s

// This file tests diverse edge cases for kernel.ext_call bufferization.
// Each test demonstrates a specific aliasing and effects pattern.

module attributes {gpu.container_module} {
  gpu.module @kernels attributes {kernel.gpu_module_kind = #kernel.gpu_module_kind.default} {
    // Kernel signatures - implementation not needed for bufferization tests
    func.func @inplace_add(%data: memref<?xi32, strided<[?], offset: ?>>, %scalar: i32) attributes {gpu.kernel} { return }
    func.func @copy_kernel(%src: memref<?xi32, strided<[?], offset: ?>>, %dst: memref<?xi32, strided<[?], offset: ?>>) attributes {gpu.kernel} { return }
    func.func @reduce(%in: memref<?xi32, strided<[?], offset: ?>>, %out: memref<i32, strided<[], offset: ?>>) attributes {gpu.kernel} { return }
    func.func @multi_out(%in: memref<?xi32, strided<[?], offset: ?>>, %out1: memref<?xi32, strided<[?], offset: ?>>, %out2: memref<?xi32, strided<[?], offset: ?>>) attributes {gpu.kernel} { return }
    func.func @passthrough(%data: memref<?xi32, strided<[?], offset: ?>>) attributes {gpu.kernel} { return }
    func.func @read_only(%data: memref<?xi32, strided<[?], offset: ?>>, %scalar: i32) attributes {gpu.kernel} { return }
  }

  // Basic in-place update (read-write)
  // Tests: The most common pattern - kernel modifies a buffer in-place
  // Expects: Single allocation, kernel modifies it in-place
  // CHECK-LABEL: func.func @test_inplace_rw(
  //  CHECK-SAME: %[[arg0:.*]]: memref<?xi32>
  func.func @test_inplace_rw(%data: tensor<?xi32>) -> tensor<?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c42 = arith.constant 42 : i32

    // CHECK: %[[CAST:.*]] = memref.cast %[[arg0]]
    // CHECK: kernel.ext_call @kernels::@inplace_add {{.*}} args(%[[CAST]], %{{.*}})
    %result = kernel.ext_call @kernels::@inplace_add
      grid[%c1] block[%c100] args(%data, %c42 : tensor<?xi32>, i32)
      result_aliases = [0], effects = ["rw", "-"] : tensor<?xi32>
    // CHECK: return %[[arg0]]
    return %result : tensor<?xi32>
  }

  // CHECK-LABEL: func.func @type_cast(
  // CHECK-SAME: %[[data:.*]]: memref
  func.func @type_cast(%data: tensor<12xi32>) -> tensor<12xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c42 = arith.constant 42 : i32
    // CHECK: %[[CAST:.*]] = memref.cast %[[data]]
    // CHECK: kernel.ext_call @kernels::@inplace_add {{.*}} args(%[[CAST]], %{{.*}})
    %result = kernel.ext_call @kernels::@inplace_add
      grid[%c1] block[%c100] args(%data, %c42 : tensor<12xi32>, i32)
      result_aliases = [0], effects = ["rw", "-"] : tensor<12xi32>
    return %result : tensor<12xi32>
  }

  // Copy operation - output aliases a different buffer
  // Tests: Result aliases a different argument than the one being read
  // Expects: Two allocations, kernel copies src to dst
  // CHECK-LABEL: func.func @test_copy_alias
  //  CHECK-SAME: %[[src:.*]]: memref<?xi32>, %[[dst:.*]]: memref<?xi32>
  func.func @test_copy_alias(%src: tensor<?xi32>, %dst: tensor<?xi32>) -> tensor<?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c50 = arith.constant 50 : index

    // CHECK: %[[CAST_SRC:.*]] = memref.cast %[[src]]
    // CHECK: %[[CAST_DST:.*]] = memref.cast %[[dst]]
    // CHECK: kernel.ext_call @kernels::@copy_kernel
    // CHECK-SAME: args(%[[CAST_SRC]], %[[CAST_DST]] :
    %result = kernel.ext_call @kernels::@copy_kernel
      grid[%c1] block[%c50] args(%src, %dst : tensor<?xi32>, tensor<?xi32>)
      result_aliases = [1], effects = ["r", "w"] : tensor<?xi32>  // result0 aliases arg1 (dst)
    // CHECK: return %[[dst]]
    return %result : tensor<?xi32>
  }

  // Multiple outputs aliasing different inputs
  // Tests: Multiple results, each aliasing a different argument
  // Expects: Three allocations, kernel writes to out1 and out2
  // CHECK-LABEL: func.func @test_multi_alias(
  // CHECK-SAME: %[[in:.*]]: memref<?xi32>, %[[out1:.*]]: memref<?xi32>, %[[out2:.*]]: memref<?xi32>)
  func.func @test_multi_alias(%in: tensor<?xi32>, %out1: tensor<?xi32>, %out2: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    // CHECK: %[[CAST_IN:.*]] = memref.cast %[[in]]
    // CHECK: %[[CAST_OUT1:.*]] = memref.cast %[[out1]]
    // CHECK: %[[CAST_OUT2:.*]] = memref.cast %[[out2]]
    // CHECK: kernel.ext_call @kernels::@multi_out {{.*}} args(%[[CAST_IN]], %[[CAST_OUT1]], %[[CAST_OUT2]] :
    %res1, %res2 = kernel.ext_call @kernels::@multi_out
      grid[%c1] block[%c64] args(%in, %out1, %out2 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>)
      result_aliases = [2, 1], effects = ["r", "w", "w"] : tensor<?xi32>, tensor<?xi32>

    // CHECK: return %[[out2]], %[[out1]]
    return %res1, %res2 : tensor<?xi32>, tensor<?xi32>
  }

  // Read-only operation with pre-allocated output buffer
  // Tests: Read-only input with separate output buffer for reduction
  // Expects: Input and output allocated
  // CHECK-LABEL: func.func @test_read_only(
  // CHECK-SAME: %[[data:.*]]: memref<?xi32>, %[[out:.*]]: memref<i32>)
  func.func @test_read_only(%data: tensor<?xi32>, %out: tensor<i32>) -> tensor<i32> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    // CHECK-DAG: %[[CAST_DATA:.*]] = memref.cast %[[data]]
    // CHECK-DAG: %[[CAST_OUT:.*]] = memref.cast %[[out]]
    // CHECK: kernel.ext_call @kernels::@reduce
    // CHECK-SAME: (%[[CAST_DATA]], %[[CAST_OUT]] :
    %result = kernel.ext_call @kernels::@reduce
      grid[%c1] block[%c1] args(%data, %out : tensor<?xi32>, tensor<i32>)
      result_aliases = [1], effects = ["r", "w"] : tensor<i32>  // result0 aliases arg1 (out)

    // CHECK: return
    return %result : tensor<i32>
  }

  // Write-only operation
  // Tests: Kernel only writes to buffer, doesn't read (e.g., memset-like)
  // Expects: Single allocation, kernel writes without reading
  // CHECK-LABEL: func.func @test_write_only(
  // CHECK-SAME: %[[uninit:.*]]: memref<?xi32>)
  func.func @test_write_only(%uninit: tensor<?xi32>) -> tensor<?xi32> {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c1_i32 = arith.constant 1 : i32

    // CHECK: %[[CAST:.*]] = memref.cast %[[uninit]]
    // CHECK: kernel.ext_call @kernels::@inplace_add
    // CHECK-SAME: (%[[CAST]], %{{.*}} :
    %result = kernel.ext_call @kernels::@inplace_add
      grid[%c1] block[%c128] args(%uninit, %c1_i32 : tensor<?xi32>, i32)
      result_aliases = [0], effects = ["w", "-"] : tensor<?xi32>

    // CHECK: return
    return %result : tensor<?xi32>
  }

  // CHECK-LABEL: func.func @write_only(
  // CHECK-SAME: %[[data:.*]]: memref<?xi32>
  func.func @write_only(%data: tensor<?xi32>) -> tensor<?xi32> {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    // CHECK: %[[CAST:.*]] = memref.cast %[[data]]
    // CHECK: kernel.ext_call @kernels::@passthrough
    // CHECK-SAME: args(%[[CAST]] :
    %result = kernel.ext_call @kernels::@passthrough
      grid[%c1] block[%c1] args(%data : tensor<?xi32>)
      result_aliases = [0], effects = ["w"] : tensor<?xi32>  // write-only identity

    // CHECK: return
    return %result : tensor<?xi32>
  }

  // Chain of operations sharing buffers
  // Tests: Multiple kernels operating on same buffer sequentially
  // Expects: Single buffer used by both kernels
  // CHECK-LABEL: func.func @test_chain(
  // CHECK-SAME: %[[data:.*]]: memref<?xi32>
  func.func @test_chain(%data: tensor<?xi32>) -> tensor<?xi32> {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32

    // First op: in-place add
    // CHECK: %[[CAST:.*]] = memref.cast %[[data]]
    // CHECK: kernel.ext_call @kernels::@inplace_add
    // CHECK-SAME: args(%[[CAST]], %{{.*}} :
    %step1 = kernel.ext_call @kernels::@inplace_add
      grid[%c1] block[%c256] args(%data, %c5 : tensor<?xi32>, i32)
      result_aliases = [0], effects = ["rw", "-"] : tensor<?xi32>

    // Second op: another in-place add on same buffer
    // CHECK: %[[CAST2:.*]] = memref.cast %[[data]]
    // CHECK: kernel.ext_call @kernels::@inplace_add
    // CHECK-SAME: args(%[[CAST2]], %{{.*}} :
    %step2 = kernel.ext_call @kernels::@inplace_add
      grid[%c1] block[%c256] args(%step1, %c10 : tensor<?xi32>, i32)
      result_aliases = [0], effects = ["rw", "-"] : tensor<?xi32>

    // CHECK: return %[[data]]
    return %step2 : tensor<?xi32>
  }

  // Read-After-Write conflict, but does not require a copy
  // Tests: First kernel writes to input, second kernel reads original value
  // Expects: Bufferization allocates new buffer for write to preserve original
  // CHECK-LABEL: func.func @test_raw_conflict(
  // CHECK-SAME: %[[arg0:.*]]: memref<?xi32>, %[[out:.*]]: memref<i32>
  func.func @test_raw_conflict(%arg0: tensor<?xi32>, %out: tensor<i32>) -> tensor<i32> {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c42 = arith.constant 42 : i32

    // First kernel: WRITES to arg0 (modifies it in-place)
    // CHECK: %[[ALLOC:.*]] = memref.alloc
    // CHECK-NOT: memref.copy
    // CHECK: %[[CAST_ALLOC:.*]] = memref.cast %[[ALLOC]]
    // CHECK: kernel.ext_call @kernels::@inplace_add
    // CHECK-SAME: args(%[[CAST_ALLOC]], %{{.*}} :
    %modified = kernel.ext_call @kernels::@inplace_add
      grid[%c1] block[%c128] args(%arg0, %c42 : tensor<?xi32>, i32)
      result_aliases = [0], effects = ["w", "-"] : tensor<?xi32>

    // Second kernel: READS the ORIGINAL arg0 value (not modified)
    // This creates a RAW conflict - first kernel wrote to arg0,
    // but second kernel needs the original value
    // CHECK: %[[CAST_ORIG:.*]] = memref.cast %[[arg0]]
    // CHECK: %[[CAST_OUT:.*]] = memref.cast %[[out]]
    // CHECK: kernel.ext_call @kernels::@reduce
    // CHECK-SAME: args(%[[CAST_ORIG]], %[[CAST_OUT]] :
    %sum = kernel.ext_call @kernels::@reduce
      grid[%c1] block[%c1] args(%arg0, %out : tensor<?xi32>, tensor<i32>)
    result_aliases = [1], effects = ["r", "w"] : tensor<i32>

    // CHECK: return %[[out]]
    return %sum : tensor<i32>
  }


}
