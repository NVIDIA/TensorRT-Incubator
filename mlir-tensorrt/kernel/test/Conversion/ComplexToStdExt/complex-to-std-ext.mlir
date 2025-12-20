// RUN: kernel-opt %s -convert-complex-to-std-ext="convert-op-generically=kernel.ext_call" --split-input-file | FileCheck %s

gpu.module @kernel_module {
  func.func @complex_args(%arg0: memref<1024xcomplex<f32>>,
                          %arg1: memref<1024xcomplex<f32>>,
                          %arg2: memref<1024xcomplex<f32>>) {
    return
  }
}

func.func @host_func(%arg0: complex<f32>,
                     %arg1: memref<complex<f32>>) -> complex<f32> {
  return %arg0 : complex<f32>
}

//       CHECK: gpu.module @kernel_module
// CHECK-LABEL: func.func @complex_args
//  CHECK-SAME: (%[[arg0:.+]]: memref<1024xi64>, %[[arg1:.+]]: memref<1024xi64>, %[[arg2:.+]]: memref<1024xi64>)

// CHECK-LABEL: func.func @host_func
//  CHECK-SAME: (%[[arg0:.+]]: complex<f32>, %[[arg1:.+]]: memref<complex<f32>>) -> complex<f32>

// -----

#map = affine_map<(d0) -> (d0)>

gpu.module @kernels {
  func.func @misc_operations(%arg0: memref<1024xcomplex<f32>>,
                            %arg1: memref<1024xf32>,
                            %offset: index) {

    %c64 = arith.constant 64 : index
    %arg0_tensor = bufferization.to_tensor %arg0 : memref<1024xcomplex<f32>> to tensor<1024xcomplex<f32>>
    %arg1_tensor = bufferization.to_tensor %arg1 : memref<1024xf32> to tensor<1024xf32>

    %slice_a = tensor.extract_slice %arg0_tensor[%offset] [64] [1]
                : tensor<1024xcomplex<f32>> to tensor<64xcomplex<f32>>
    %slice_b = tensor.extract_slice %arg1_tensor[%offset] [64] [1]
                : tensor<1024xf32> to tensor<64xf32>

    %result_slice = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]
    } ins(%slice_a : tensor<64xcomplex<f32>>)
      outs(%slice_b : tensor<64xf32>) {
    ^bb0(%in: complex<f32>, %out: f32):
      %abs = complex.abs %in : complex<f32>
      linalg.yield %abs : f32
    } -> tensor<64xf32>


    %result_slice_memref = bufferization.to_buffer %result_slice : tensor<64xf32> to memref<64xf32, strided<[1], offset: ?>>
    %out_view = memref.subview %arg1[%offset] [64] [1] : memref<1024xf32> to memref<64xf32, strided<[1], offset: ?>>
    memref.copy %result_slice_memref, %out_view : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
    return
  }
}

func.func @caller(%arg0: memref<1024xcomplex<f32>>, %arg1: memref<1024xf32>, %offset: index) {
  %c1 = arith.constant 1 : index
  kernel.ext_call @kernels::@misc_operations
    grid[%c1] block[%c1]
    args(%arg0, %arg1, %offset : memref<1024xcomplex<f32>>, memref<1024xf32>, index)
    result_aliases = [0],
    effects = ["rw", "rw", "-"]
  return
}

//       CHECK: gpu.module @kernels
// CHECK-LABEL: func.func @misc_operations
//  CHECK-SAME: (%[[arg0:.+]]: memref<1024xi64>, %[[arg1:.+]]: memref<1024xf32>, %[[arg2:.+]]: index)
//   CHECK-DAG:       %[[c64:.+]] = arith.constant 64 : index
//   CHECK-DAG:       %[[v0:.+]] = bufferization.to_tensor %[[arg0]] : memref<1024xi64> to tensor<1024xi64>
//   CHECK-DAG:       %[[v1:.+]] = bufferization.to_tensor %[[arg1]] : memref<1024xf32> to tensor<1024xf32>
//   CHECK-DAG:       %[[extracted_slice:.+]] = tensor.extract_slice %[[v0]][%[[arg2]]] [64] [1] : tensor<1024xi64> to tensor<64xi64>
//   CHECK-DAG:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[v1]][%[[arg2]]] [64] [1] : tensor<1024xf32> to tensor<64xf32>
//   CHECK-DAG:       %[[v2:.+]] = linalg.generic {{.*}} ins(%[[extracted_slice]] : tensor<64xi64>) outs(%[[extracted_slice_0]] : tensor<64xf32>)
//       CHECK:       ^bb0(%[[in:.+]]: i64, %[[out:.+]]: f32):
//   CHECK-DAG:         %[[v4:.+]] = arith.trunci %[[in]] : i64 to i32
//   CHECK-DAG:         %[[v5:.+]] = arith.bitcast %[[v4]] : i32 to f32
//   CHECK-DAG:         %[[v6:.+]] = arith.trunci %[[in]] : i64 to i32
//   CHECK-DAG:         %[[c32_i64:.+]] = arith.constant 32 : i64
//   CHECK-DAG:         %[[v7:.+]] = arith.shrui %[[in]], %[[c32_i64]] : i64
//   CHECK-DAG:         %[[v8:.+]] = arith.trunci %[[v7]] : i64 to i32
//   CHECK-DAG:         %[[v9:.+]] = arith.bitcast %[[v8]] : i32 to f32
//   CHECK-DAG:         %[[cst:.+]] = arith.constant 1.000000e+00 : f32
//   CHECK-DAG:         %[[v10:.+]] = math.absf %[[v5]] : f32
//   CHECK-DAG:         %[[v11:.+]] = math.absf %[[v9]] : f32
//   CHECK-DAG:         %[[v12:.+]] = arith.maximumf %[[v10]], %[[v11]] : f32
//   CHECK-DAG:         %[[v13:.+]] = arith.minimumf %[[v10]], %[[v11]] : f32
//   CHECK-DAG:         %[[v14:.+]] = arith.divf %[[v13]], %[[v12]] : f32
//   CHECK-DAG:         %[[v15:.+]] = arith.mulf %[[v14]], %[[v14]] : f32
//   CHECK-DAG:         %[[v16:.+]] = arith.addf %[[v15]], %[[cst]] : f32
//   CHECK-DAG:         %[[v17:.+]] = math.sqrt %[[v16]] : f32
//   CHECK-DAG:         %[[v18:.+]] = arith.mulf %[[v12]], %[[v17]] : f32
//   CHECK-DAG:         %[[v19:.+]] = arith.cmpf uno, %[[v18]], %[[v18]] : f32
//   CHECK-DAG:         %[[v20:.+]] = arith.select %[[v19]], %[[v13]], %[[v18]] : f32
//       CHECK:         linalg.yield %[[v20]] : f32
//   CHECK-DAG:       %[[v3:.+]] = bufferization.to_buffer %[[v2]] : tensor<64xf32> to memref<64xf32, strided<[1], offset: ?>>
//   CHECK-DAG:       %[[subview:.+]] = memref.subview %[[arg1]][%[[arg2]]] [64] [1] : memref<1024xf32> to memref<64xf32, strided<[1], offset: ?>>
//   CHECK-DAG:       memref.copy %[[v3]], %[[subview]] : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
//       CHECK:       return

// CHECK-LABEL: func.func @caller
//  CHECK-SAME: (%[[arg0:.+]]: memref<1024xcomplex<f32>>, %[[arg1:.+]]: memref<1024xf32>, %[[arg2:.+]]: index)
//       CHECK:     %[[v0:.+]] = executor.buffer_bitcast %[[arg0]] : memref<1024xcomplex<f32>> to memref<1024xi64>
//       CHECK:     kernel.ext_call @kernels::@misc_operations {{.*}} args(%[[v0]], %[[arg1]], %[[arg2]] :

// -----

gpu.module @complex_arith {
  func.func @complex_arith_select(%arg0: i1, %arg1: complex<f32>, %arg2: complex<f32>) -> complex<f32> {
    %0 = arith.select %arg0, %arg1, %arg2 : complex<f32>
    return %0 : complex<f32>
  }
}

// CHECK-LABEL: func.func @complex_arith_select
//  CHECK-SAME: (%[[arg0:.+]]: i1, %[[arg1:.+]]: i64, %[[arg2:.+]]: i64)
//       CHECK:     %[[v0:.+]] = arith.select %[[arg0]], %[[arg1]], %[[arg2]] : i64
//       CHECK:     return %[[v0]] : i64
