// RUN: kernel-opt %s -split-input-file -test-outlining | FileCheck %s


!type = tensor<2 x 32 x f32>
!type1d = tensor<32 x f32>

func.func @outline_using_block_mapping(%x: !type, %y: !type, %z: !type, %t: !type1d, %alpha : f32) -> !type {
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : index
  %r = scf.forall (%i, %j) in (%c7, %c9) shared_outs(%out = %z) -> !type {
    %0 = tensor.extract_slice %x[%i, %j][1, 1][1, 1] : !type to  tensor<1x1xf32>
    %1 = tensor.extract_slice %y[%i, %j][1, 1][1, 1] : !type to tensor<1x1xf32>
    %4 = tensor.extract %0[%c0, %c0] : tensor<1x1xf32>
    %5 = tensor.extract %1[%c0, %c0] : tensor<1x1xf32>
    %6 = math.fma %alpha, %4, %5 : f32
    %7 = tensor.extract_slice %out[%i, %j][1, 1][1, 1] : !type to tensor<1x1xf32>
    %8 = tensor.insert %6 into %7[%c0, %c0] : tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %out[%i, %j][1, 1][1, 1] : tensor<1x1xf32> into !type
    }
  }  { mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %r : !type
}

// CHECK-LABEL: func.func @outline_using_block_mapping
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x32xf32>, %[[arg1:.+]]: tensor<2x32xf32>, %[[arg2:.+]]: tensor<2x32xf32>, %[[arg3:.+]]: tensor<32xf32>, %[[arg4:.+]]: f32)
//   CHECK-DAG:     %[[c9:.+]] = arith.constant 9 : index
//   CHECK-DAG:     %[[c7:.+]] = arith.constant 7 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c1_0:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[v0:.+]] = kernel.call @kernels::@forall grid[%[[c7]], %[[c9]]] block[%[[c1_0]]] (%[[arg0]], %[[arg1]], %[[arg4]]) outs(%[[arg2]])
//   CHECK-DAG:     return %[[v0]] : tensor<2x32xf32>

//       CHECK:   gpu.module @kernels
// CHECK-LABEL: func.func @forall
//  CHECK-SAME: (%[[arg0:.+]]: memref<2x32xf32, strided<[?, ?], offset: ?>>, %[[arg1:.+]]: memref<{{.*}}>, %[[arg3:.+]]: f32, %[[arg4:.+]]: memref<{{.*}}>)
//   CHECK-DAG:   %[[arg0_tensor:.+]] = bufferization.to_tensor %[[arg0]] restrict
//   CHECK-DAG:   %[[arg1_tensor:.+]] = bufferization.to_tensor %[[arg1]] restrict
//   CHECK-DAG:   %[[arg4_tensor:.+]] = bufferization.to_tensor %[[arg4]] restrict writable
//   CHECK-DAG:   %[[block_id_y:.+]] = gpu.block_id  y
//   CHECK-DAG:   %[[block_id_x:.+]] = gpu.block_id  x
//   CHECK-DAG:   %[[extracted_slice:.+]] = tensor.extract_slice %[[arg0_tensor]][%[[block_id_y]], %[[block_id_x]]] [1, 1] [1, 1] : tensor<2x32xf32> to tensor<1x1xf32>
//   CHECK-DAG:   %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg1_tensor]][%[[block_id_y]], %[[block_id_x]]] [1, 1] [1, 1] : tensor<2x32xf32> to tensor<1x1xf32>

// -----

!type = tensor<2 x 32 x f32>
!type1d = tensor<32 x f32>

func.func @outline_using_linear_block_mapping(%x: !type, %y: !type, %z: !type, %t: !type1d, %alpha : f32) -> !type {
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : index
  %r = scf.forall (%i, %j) in (%c7, %c9) shared_outs(%out = %z) -> !type {
    %0 = tensor.extract_slice %x[%i, %j][1, 1][1, 1] : !type to  tensor<1x1xf32>
    %1 = tensor.extract_slice %y[%i, %j][1, 1][1, 1] : !type to tensor<1x1xf32>
    %4 = tensor.extract %0[%c0, %c0] : tensor<1x1xf32>
    %5 = tensor.extract %1[%c0, %c0] : tensor<1x1xf32>
    %6 = math.fma %alpha, %4, %5 : f32
    %7 = tensor.extract_slice %out[%i, %j][1, 1][1, 1] : !type to tensor<1x1xf32>
    %8 = tensor.insert %6 into %7[%c0, %c0] : tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %out[%i, %j][1, 1][1, 1] : tensor<1x1xf32> into !type
    }
  } { mapping = [#gpu.block<linear_dim_0>, #gpu.block<linear_dim_1>]}
  return %r : !type
}

//       CHECK: #[[$map:.+]] = affine_map<()[s0, s1, s2, s3, s4] -> (s3 * s1 + s4 + s2 * (s1 * s0))>
//       CHECK: module {
// CHECK-LABEL: func.func @outline_using_linear_block_mapping
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x32xf32>, %[[arg1:.+]]: tensor<2x32xf32>, %[[arg2:.+]]: tensor<2x32xf32>, %[[arg3:.+]]: tensor<32xf32>, %[[arg4:.+]]: f32)
//   CHECK-DAG:     %[[c9:.+]] = arith.constant 9 : index
//   CHECK-DAG:     %[[c7:.+]] = arith.constant 7 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c1_0:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[v0:.+]] = kernel.call @kernels::@forall grid[%[[c7]], %[[c9]]] block[%[[c1_0]]] (%[[arg0]], %[[arg1]], %[[arg4]]) outs(%[[arg2]])
//       CHECK:     return %[[v0]] : tensor<2x32xf32>

//       CHECK:   gpu.module @kernels
// CHECK-LABEL: func.func @forall
//  CHECK-SAME: (%[[arg0:.+]]: memref<2x32xf32, strided<[?, ?], offset: ?>>, %[[arg1:.+]]: memref<{{.*}}>, %[[arg3:.+]]: f32, %[[arg4:.+]]: memref<{{.*}}>)
//   CHECK-DAG:     %[[arg0_tensor:.+]] = bufferization.to_tensor %[[arg0]] restrict
//   CHECK-DAG:     %[[arg1_tensor:.+]] = bufferization.to_tensor %[[arg1]] restrict
//   CHECK-DAG:     %[[arg4_tensor:.+]] = bufferization.to_tensor %[[arg4]] restrict writable
//   CHECK-DAG:     %[[block_id_x:.+]] = gpu.block_id  x
//   CHECK-DAG:     %[[block_id_y:.+]] = gpu.block_id  y
//   CHECK-DAG:     %[[block_id_z:.+]] = gpu.block_id  z
//   CHECK-DAG:     %[[block_dim_x:.+]] = gpu.block_dim  x
//   CHECK-DAG:     %[[block_dim_y:.+]] = gpu.block_dim  y
//   CHECK-DAG:     %[[block_dim_z:.+]] = gpu.block_dim  z
//   CHECK-DAG:     %[[c9:.+]] = arith.constant 9 : index
//   CHECK-DAG:     %[[c7:.+]] = arith.constant 7 : index
//   CHECK-DAG:     %[[v0:.+]] = affine.apply #[[$map]]()[%[[block_dim_y]], %[[block_dim_z]], %[[block_id_x]], %[[block_id_y]], %[[block_id_z]]]
//   CHECK-DAG:     %[[v1:.+]]:2 = affine.delinearize_index %[[v0]] into (%[[c7]], %[[c9]]) : index, index
//   CHECK-DAG:     tensor.extract_slice %[[arg0_tensor]][%[[v1]]#0, %[[v1]]#1] [1, 1] [1, 1] : tensor<2x32xf32> to tensor<1x1xf32>
//   CHECK-DAG:     tensor.extract_slice %[[arg1_tensor]][%[[v1]]#0, %[[v1]]#1] [1, 1] [1, 1] : tensor<2x32xf32> to tensor<1x1xf32>

// -----

!type = tensor<2 x 32 x f32>
!type1d = tensor<32 x f32>

func.func @fallback_method(%x: !type, %y: !type, %z: !type, %t: !type1d, %alpha : f32) -> !type {
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : index
  %r = scf.forall (%i, %j) in (%c7, %c9) shared_outs(%out = %z) -> !type {
    %0 = tensor.extract_slice %x[%i, %j][1, 1][1, 1] : !type to  tensor<1x1xf32>
    %1 = tensor.extract_slice %y[%i, %j][1, 1][1, 1] : !type to tensor<1x1xf32>
    %4 = tensor.extract %0[%c0, %c0] : tensor<1x1xf32>
    %5 = tensor.extract %1[%c0, %c0] : tensor<1x1xf32>
    %6 = math.fma %alpha, %4, %5 : f32
    %7 = tensor.extract_slice %out[%i, %j][1, 1][1, 1] : !type to tensor<1x1xf32>
    %8 = tensor.insert %6 into %7[%c0, %c0] : tensor<1x1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %out[%i, %j][1, 1][1, 1] : tensor<1x1xf32> into !type
    }
  }
  return %r : !type
}

// CHECK-LABEL: func.func @fallback_method
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x32xf32>, %[[arg1:.+]]: tensor<2x32xf32>, %[[arg2:.+]]: tensor<2x32xf32>, %[[arg3:.+]]: tensor<32xf32>, %[[arg4:.+]]: f32)
//   CHECK-DAG:     %[[c9:.+]] = arith.constant 9 : index
//   CHECK-DAG:     %[[c7:.+]] = arith.constant 7 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c1_0:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[v0:.+]] = kernel.call @kernels::@forall grid[%[[c7]], %[[c9]]] block[%[[c1_0]]] (%[[arg0]], %[[arg1]], %[[arg4]]) outs(%[[arg2]])
//       CHECK:     return %[[v0]] : tensor<2x32xf32>

// CHECK-LABEL: func.func @forall
//  CHECK-SAME: (%[[arg0:.+]]: memref<2x32xf32, strided<[?, ?], offset: ?>>, %[[arg1:.+]]: memref<{{.*}}>, %[[arg3:.+]]: f32, %[[arg4:.+]]: memref<{{.*}}>)
//   CHECK-DAG:     %[[arg0_tensor:.+]] = bufferization.to_tensor %[[arg0]] restrict
//   CHECK-DAG:     %[[arg1_tensor:.+]] = bufferization.to_tensor %[[arg1]] restrict
//   CHECK-DAG:     %[[arg4_tensor:.+]] = bufferization.to_tensor %[[arg4]] restrict writable
//   CHECK-DAG:     %[[block_id_x:.+]] = gpu.block_id  x
//   CHECK-DAG:     %[[block_id_y:.+]] = gpu.block_id  y
//   CHECK-DAG:     tensor.extract_slice %[[arg0_tensor]][%[[block_id_x]], %[[block_id_y]]] [1, 1] [1, 1] : tensor<2x32xf32> to tensor<1x1xf32>
//   CHECK-DAG:     tensor.extract_slice %[[arg1_tensor]][%[[block_id_x]], %[[block_id_y]]] [1, 1] [1, 1] : tensor<2x32xf32> to tensor<1x1xf32>
