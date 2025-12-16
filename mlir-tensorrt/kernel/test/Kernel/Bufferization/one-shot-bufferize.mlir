// RUN: kernel-opt -allow-unregistered-dialect --pass-pipeline="builtin.module(gpu.module(test-kernel-one-shot-bufferize),cse,canonicalize)" -split-input-file %s | FileCheck %s


#map2 = affine_map<()[s0] -> (s0 * 512)>
#map3 = affine_map<()[s0] -> ((s0 mod 128) floordiv 32)>
#map4 = affine_map<()[s0] -> (s0 * 128)>
#map5 = affine_map<()[s0] -> (s0 mod 32)>
#map6 = affine_map<()[s0] -> (s0 * 4)>

gpu.module @kernels {
  func.func @codegen_cluster_35_kernel(%arg0: tensor<65536xi32>, %arg1: tensor<65536xi32>, %arg2: tensor<65536xi32>, %arg3: tensor<65536xi32>, %arg4: tensor<65536xi32>) -> tensor<65536xi32> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<32> : vector<4xi32>
    %cst_0 = arith.constant dense<0> : vector<4xi32>
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %0 = gpu.block_id  x
    %1 = affine.apply #map2()[%0]
    %extracted_slice = tensor.extract_slice %arg4[%1] [512] [1] : tensor<65536xi32> to tensor<512xi32>
    %2 = gpu.thread_id  x
    %3 = affine.apply #map3()[%2]
    %4 = affine.delinearize_index %3 into (%c4) : index
    %5 = affine.apply #map4()[%4]
    %extracted_slice_1 = tensor.extract_slice %extracted_slice[%5] [128] [1] : tensor<512xi32> to tensor<128xi32>
    %6 = affine.apply #map5()[%2]
    %7 = affine.delinearize_index %6 into (%c32) : index
    %8 = affine.apply #map6()[%7]
    %9 = arith.addi %8, %5 : index
    %10 = arith.addi %9, %1 : index
    %11 = vector.transfer_read %arg0[%10], %c0_i32 {in_bounds = [true]} : tensor<65536xi32>, vector<4xi32>
    %12 = vector.transfer_read %arg1[%10], %c0_i32 {in_bounds = [true]} : tensor<65536xi32>, vector<4xi32>
    %13 = vector.transfer_read %arg2[%10], %c0_i32 {in_bounds = [true]} : tensor<65536xi32>, vector<4xi32>
    %14 = vector.transfer_read %arg3[%10], %c0_i32 {in_bounds = [true]} : tensor<65536xi32>, vector<4xi32>
    %15 = arith.shrui %12, %14 : vector<4xi32>
    %16 = arith.cmpi ult, %14, %cst : vector<4xi32>
    %17 = arith.select %16, %15, %cst_0 : vector<4xi1>, vector<4xi32>
    %18 = arith.shli %12, %13 : vector<4xi32>
    %19 = arith.cmpi ult, %13, %cst : vector<4xi32>
    %20 = arith.select %19, %18, %cst_0 : vector<4xi1>, vector<4xi32>
    %21 = arith.ori %20, %17 : vector<4xi32>
    %22 = arith.xori %11, %21 : vector<4xi32>
    %23 = vector.transfer_write %22, %extracted_slice_1[%8] {in_bounds = [true]} : vector<4xi32>, tensor<128xi32>
    %inserted_slice = tensor.insert_slice %23 into %extracted_slice[%5] [128] [1] : tensor<128xi32> into tensor<512xi32>
    %inserted_slice_2 = tensor.insert_slice %inserted_slice into %arg4[%1] [512] [1] : tensor<512xi32> into tensor<65536xi32>
    return %inserted_slice_2 : tensor<65536xi32>
  }
}


// CHECK-LABEL: func.func @codegen_cluster_35_kernel
//  CHECK-SAME: (%[[arg0:.+]]: memref<65536xi32, strided<[?], offset: ?>>, %[[arg1:.+]]: memref<65536xi32, strided<[?], offset: ?>>, %[[arg2:.+]]: memref<65536xi32, strided<[?], offset: ?>>, %[[arg3:.+]]: memref<65536xi32, strided<[?], offset: ?>>, %[[arg4:.+]]: memref<65536xi32, strided<[?], offset: ?>>)
//       CHECK:       vector.transfer_read %[[arg0]]
//       CHECK:       vector.transfer_read %[[arg1]]
//       CHECK:       vector.transfer_read %[[arg2]]
//       CHECK:       vector.transfer_read %[[arg3]]
//       CHECK:       vector.transfer_write
//   CHECK-NOT:       memref.copy
//   CHECK-NOT:       return %{{.*}}


// -----

// CHECK-LABEL: @test_smem_alloc_ok
gpu.module @test_smem_alloc_ok {
  func.func @func(%input: tensor<100xf32>, %output: tensor<100xf32>) -> tensor<100xf32> {
    // CHECK: memref.alloc()
    %smem_alloc = bufferization.alloc_tensor() { memory_space = #gpu.address_space<workgroup>} : tensor<100x100xf32>
    %smem0 = tensor.insert_slice %input into %smem_alloc[0, 0] [100, 1][1, 1] : tensor<100xf32> into tensor<100x100xf32>
    %smem3 = tensor.extract_slice %smem_alloc[0, 3] [100, 1][1, 1] : tensor<100x100xf32> to tensor<100xf32>
    %out0 = tensor.insert_slice %smem3 into %output [0] [100][1] : tensor<100xf32> into tensor<100xf32>
    return %out0: tensor<100xf32>
  }
}
