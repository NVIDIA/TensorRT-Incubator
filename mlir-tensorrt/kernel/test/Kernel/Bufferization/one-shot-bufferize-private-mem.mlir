// RUN: kernel-opt --pass-pipeline="builtin.module(gpu.module(test-kernel-one-shot-bufferize))" -split-input-file %s | FileCheck %s --check-prefix=INITIAL
// RUN: kernel-opt --pass-pipeline="builtin.module(kernel-module-bufferization-pipeline)" -split-input-file %s | FileCheck %s --check-prefix=PIPELINE

// RUN: rm -rf %t || true
// RUN: mkdir -p %t
// RUN: kernel-opt --pass-pipeline="builtin.module(kernel-set-gpu-target{chip=sm_80},\
// RUN:   kernel-module-bufferization-pipeline, \
// RUN:   gpu.module(kernel-lower-to-ptx-pipeline{dump-ptx=%t}))" \
// RUN:                     -split-input-file %s
// RUN: cat %t/no-symbol-name_test_alloc_to_alloca.ptx %t/no-symbol-name_test_private_alloc.ptx \
// RUN:  %t/no-symbol-name_test_promotion_negative.ptx| FileCheck %s --check-prefix=PTX

gpu.module @test_alloc_to_alloca {
  func.func @alloc_to_alloca(%arg1: tensor<10xf32>) -> tensor<10xf32> attributes {gpu.kernel} {
    %0 = tensor.empty() : tensor<1xf32>
    %cst = arith.constant 1.0 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = gpu.thread_id x
    %3 = tensor.insert_slice %1 into %arg1[%2][1][1] : tensor<1xf32> into tensor<10xf32>
    return %3 : tensor<10xf32>
  }
}

// INITIAL-LABEL: func.func @alloc_to_alloca
//    INITIAL-DAG:       %[[alloc:.+]] = memref.alloc()
//    INITIAL-DAG:       %[[cst:.+]] = arith.constant 1.000000e+00 : f32
//    INITIAL-DAG:       linalg.fill ins(%[[cst]] : f32) outs(%[[alloc]] : memref<1xf32>)

// PIPELINE-LABEL: func.func @alloc_to_alloca
//   PIPELINE-NOT:       memref.alloc()
//       PIPELINE:       memref.alloca
//       PIPELINE:       memref.copy


// Note: if you inspect the PTX file, you may see 4 i8 store instead of
// one f32 store due to lower of memref.copy to LLVM's memcpy intrinsic.
// We could further optimize such copies before lowering to LLVM by
// promoting to registers, but use of private memory is already a
// "fallback" path that is sub-optimal, so likely not worth the effort.

// PTX-LABEL: .entry alloc_to_alloca
//   PTX-NOT: malloc

// -----

gpu.module @test_private_alloc {
  func.func @explicit_private_annotation(%arg1: tensor<10xf32>) -> tensor<10xf32> attributes {gpu.kernel} {
    %0 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<private>} : tensor<1xf32>
    %cst = arith.constant 1.0 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = gpu.thread_id x
    %3 = tensor.insert_slice %1 into %arg1[%2][1][1] : tensor<1xf32> into tensor<10xf32>
    return %3 : tensor<10xf32>
  }
}


// PTX-LABEL: .entry explicit_private_annotation
//   PTX-NOT: malloc

// -----

gpu.module @test_promotion_negative {
  func.func @dont_promote_shared_alloc(%arg1: tensor<5xf32>) -> tensor<5xf32> attributes {gpu.kernel} {
    %0 = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<5xf32>
    %cst = arith.constant 1.0 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
    %2 = gpu.thread_id x
    %3 = tensor.extract_slice %0 [%2][1][1] : tensor<5xf32> to tensor<1xf32>
    %4 = tensor.insert_slice %3 into %arg1[%2][1][1] : tensor<1xf32> into tensor<5xf32>
    return %4 : tensor<5xf32>
  }
}

// PTX-LABEL: .entry dont_promote_shared_alloc
//   PTX: .shared {{.*}} __shared_memory__
