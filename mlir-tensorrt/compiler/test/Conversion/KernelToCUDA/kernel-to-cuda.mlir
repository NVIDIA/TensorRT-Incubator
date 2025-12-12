// RUN: mlir-tensorrt-opt %s -pass-pipeline="builtin.module(convert-kernel-to-cuda)" -split-input-file -verify-diagnostics | FileCheck %s

gpu.module @kernels attributes {
  kernel.ptx_data = dense<"0xFF"> : vector<1xi8>
} {
  func.func @kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }

  func.func @kernel2(%arg0: memref<2x?xf32, strided<[?, ?], offset: ?>>,
                    %arg1: memref<1xf32, strided<[?], offset: 4>>) {
    return
  }
  func.func @kernel3(%arg0: memref<2x?xf32, strided<[?, ?], offset: ?>>,
                    %arg1: memref<1xf32, strided<[?], offset: 4>>) {
    return
  }
}

func.func @caller(%arg0: memref<1xf32, #executor.memory_type<device>>,
    %arg1: memref<1xf32, #executor.memory_type<device>>,
    %arg2: index,
    %arg3: index,
    %arg4: memref<2x?xf32, strided<[?, ?], offset: ?>, #executor.memory_type<device>>,
    %arg5: memref<1xf32, strided<[?], offset: 4>, #executor.memory_type<device>>) {
  kernel.call @kernels::@kernel grid [%arg2] block[%arg3] (%arg0) outs(%arg1) : (memref<1xf32, #executor.memory_type<device>>, memref<1xf32, #executor.memory_type<device>>) -> ()

  kernel.call @kernels::@kernel2 grid [%arg2] block[%arg3] (%arg4) outs(%arg5)
    : (memref<2x?xf32, strided<[?, ?], offset: ?>, #executor.memory_type<device>>,
       memref<1xf32, strided<[?], offset: 4>, #executor.memory_type<device>>) -> ()

  kernel.ext_call @kernels::@kernel3 grid [%arg2] block[%arg3]
    args(%arg4, %arg5 : memref<2x?xf32, strided<[?, ?], offset: ?>, #executor.memory_type<device>>,
                        memref<1xf32, strided<[?], offset: 4>, #executor.memory_type<device>>)
    result_aliases = [1], effects = ["r", "w"]
  return
}

// CHECK-LABEL: cuda.compiled_module @kernels_cuModule_0 dense<
// CHECK-LABEL: func.func @caller
//  CHECK-SAME: (%[[arg0:.+]]: memref<1xf32, #executor.memory_type<device>>, %[[arg1:.+]]: memref<1xf32, #executor.memory_type<device>>, %[[arg2:.+]]: index, %[[arg3:.+]]: index, %[[arg4:.+]]: memref<2x?xf32, strided<[?, ?], offset: ?>, #executor.memory_type<device>>, %[[arg5:.+]]: memref<1xf32,
//   CHECK-DAG:     %[[v0:.+]] = cuda.get_function "kernel" from @kernels_cuModule_0
//   CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[base_buffer:.+]], %[[offset:.+]], %[[sizes:.+]], %[[strides:.+]] = memref.extract_strided_metadata %[[arg0]] :
//   CHECK-DAG:     %[[base_buffer_0:.+]], %[[offset_1:.+]], %[[sizes_2:.+]], %[[strides_3:.+]] = memref.extract_strided_metadata %[[arg1]]
//   CHECK-DAG:     %[[v1:.+]] = arith.index_cast %[[arg2]] : index to i32
//   CHECK-DAG:     %[[v2:.+]] = arith.index_cast %[[arg3]] : index to i32
//   CHECK-DAG:     %[[v3:.+]] = cuda.get_global_stream
//   CHECK-DAG:     cuda.launch %[[v0]](%[[base_buffer]], %[[base_buffer_0]] : {{.*}}) with
//   CHECK-DAG:      grid(%[[v1]], %[[c1_i32]], %[[c1_i32]])
//   CHECK-DAG:      block(%[[v2]], %[[c1_i32]], %[[c1_i32]])
//   CHECK-DAG:      smem(%[[c0_i32]]) stream(%[[v3]])

//       CHECK:     %[[v4:.+]] = cuda.get_function "kernel2" from @kernels_cuModule_0
//   CHECK-DAG:     %[[c1_i32_4:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c0_i32_5:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[base_buffer_6:.+]], %[[offset_7:.+]], %[[sizes_8:.+]]:2, %[[strides_9:.+]]:2 = memref.extract_strided_metadata %[[arg4]] :
//   CHECK-DAG:     %[[base_buffer_10:.+]], %[[offset_11:.+]], %[[sizes_12:.+]], %[[strides_13:.+]] = memref.extract_strided_metadata %[[arg5]] :
//   CHECK-DAG:     %[[v5:.+]] = arith.index_cast %[[arg2]] : index to i32
//   CHECK-DAG:     %[[v6:.+]] = arith.index_cast %[[arg3]] : index to i32
//   CHECK-DAG:     %[[v7:.+]] = cuda.get_global_stream
//   CHECK-DAG:     cuda.launch %[[v4]](%[[base_buffer_6]], %[[offset_7]], %[[sizes_8]]#1, %[[strides_9]]#0, %[[strides_9]]#1, %[[base_buffer_10]], %[[strides_13]]
//   CHECK-DAG:      grid(%[[v5]], %[[c1_i32_4]], %[[c1_i32_4]])
//   CHECK-DAG:      block(%[[v6]], %[[c1_i32_4]], %[[c1_i32_4]])
//   CHECK-DAG:      smem(%[[c0_i32_5]]) stream(%[[v7]])

//       CHECK:     %[[v10:.+]] = cuda.get_function "kernel3" from @kernels_cuModule_0
//   CHECK-DAG:     %[[c1_i32_14:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c0_i32_15:.+]] = arith.constant 0 : i32
//       CHECK:     %[[base_buffer_16:.+]], %[[offset_17:.+]], %[[sizes_18:.+]]:2, %[[strides_19:.+]]:2 = memref.extract_strided_metadata %[[arg4]]
//       CHECK:     %[[base_buffer_20:.+]], %[[offset_21:.+]], %[[sizes_22:.+]], %[[strides_23:.+]] = memref.extract_strided_metadata %[[arg5]]
//   CHECK-DAG:     %[[v11:.+]] = arith.index_cast %[[arg2]]
//   CHECK-DAG:     %[[v12:.+]] = arith.index_cast %[[arg3]]
//   CHECK-DAG:     %[[v13:.+]] = cuda.get_active_device
//   CHECK-DAG:     %[[v14:.+]] = cuda.get_global_stream device(%[[v13]]) [0]
//       CHECK:     cuda.launch %[[v10]](%[[base_buffer_16]], %[[offset_17]], %[[sizes_18]]#1, %[[strides_19]]#0, %[[strides_19]]#1, %[[base_buffer_20]], %[[strides_23]]
//       CHECK:      grid(%[[v11]], %[[c1_i32_14]], %[[c1_i32_14]])
//       CHECK:      block(%[[v12]], %[[c1_i32_14]], %[[c1_i32_14]])
//       CHECK:      smem(%[[c0_i32_15]]) stream(%[[v14]])

// -----

gpu.module @dedup_globals attributes {
  kernel.ptx_data = dense<"0xFF"> : vector<1xi8>
} {
  func.func @kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
  func.func @kernel2(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
}

func.func @dedup(%arg0: memref<1xf32, #executor.memory_type<device>>, %arg1: memref<1xf32, #executor.memory_type<device>>, %arg2: index, %arg3: index) {
  kernel.call @dedup_globals::@kernel grid [%arg2] block[%arg3] (%arg0) outs(%arg1) : (memref<1xf32, #executor.memory_type<device>>, memref<1xf32, #executor.memory_type<device>>) -> ()
  kernel.call @dedup_globals::@kernel grid [%arg2] block[%arg3] (%arg0) outs(%arg1) : (memref<1xf32, #executor.memory_type<device>>, memref<1xf32, #executor.memory_type<device>>) -> ()
  kernel.call @dedup_globals::@kernel2 grid [%arg2] block[%arg3] (%arg0) outs(%arg1) : (memref<1xf32, #executor.memory_type<device>>, memref<1xf32, #executor.memory_type<device>>) -> ()
  return
}

// CHECK-LABEL: cuda.compiled_module @dedup_globals_cuModule_0 dense<-1> : vector<1xi8>
// CHECK-LABEL: func.func @dedup
//  CHECK-SAME: (%[[arg0:.+]]: memref<1xf32, #executor.memory_type<device>>, %[[arg1:.+]]: memref<1xf32, #executor.memory_type<device>>, %[[arg2:.+]]: index, %[[arg3:.+]]: index) {
//   CHECK-DAG:     %[[v0:.+]] = cuda.get_function "kernel" from @dedup_globals_cuModule_0
//   CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[v1:.+]] = arith.index_cast %[[arg2]] : index to i32
//   CHECK-DAG:     %[[v2:.+]] = arith.index_cast %[[arg3]] : index to i32
//   CHECK-DAG:     %[[v3:.+]] = cuda.get_global_stream
//       CHECK:     cuda.launch
//       CHECK:      grid(%[[v1]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      block(%[[v2]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      smem(%[[c0_i32]]) stream(%[[v3]])
//   CHECK-DAG:     %[[v4:.+]] = cuda.get_function "kernel" from @dedup_globals_cuModule_0
//   CHECK-DAG:     %[[c1_i32_0:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c0_i32_1:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[v5:.+]] = arith.index_cast %[[arg2]] : index to i32
//   CHECK-DAG:     %[[v6:.+]] = arith.index_cast %[[arg3]] : index to i32
//   CHECK-DAG:     %[[v7:.+]] = cuda.get_global_stream
//       CHECK:     cuda.launch %[[v4]](
//       CHECK:      grid(%[[v5]], %[[c1_i32_0]], %[[c1_i32_0]])
//       CHECK:      block(%[[v6]], %[[c1_i32_0]], %[[c1_i32_0]])
//       CHECK:      smem(%[[c0_i32_1]]) stream(%[[v7]])
//   CHECK-DAG:     %[[v8:.+]] = cuda.get_function "kernel2" from @dedup_globals_cuModule_0
//   CHECK-DAG:     %[[c1_i32_2:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c0_i32_3:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[v9:.+]] = arith.index_cast %[[arg2]] : index to i32
//   CHECK-DAG:     %[[v10:.+]] = arith.index_cast %[[arg3]] : index to i32
//   CHECK-DAG:     %[[v11:.+]] = cuda.get_global_stream
//   CHECK-DAG:     cuda.launch %[[v8]](
//   CHECK-DAG:      grid(%[[v9]], %[[c1_i32_2]], %[[c1_i32_2]])
//   CHECK-DAG:      block(%[[v10]], %[[c1_i32_2]], %[[c1_i32_2]])
//   CHECK-DAG:      smem(%[[c0_i32_3]]) stream(%[[v11]])

// -----

// expected-error @below {{gpu.module "missing_ptx_data" is missing serialized PTX IR}}
gpu.module @missing_ptx_data {
  func.func @kernel(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    return
  }
}

func.func @caller(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: index, %arg3: index) {
  kernel.call @missing_ptx_data::@kernel grid [%arg2] block[%arg3] (%arg0) outs(%arg1) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}
