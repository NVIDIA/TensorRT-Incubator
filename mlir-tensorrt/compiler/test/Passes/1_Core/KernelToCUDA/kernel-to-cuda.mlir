// RUN: mlir-tensorrt-opt %s -pass-pipeline="builtin.module(convert-kernel-to-cuda,cse)" -split-input-file -verify-diagnostics | FileCheck %s

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

// CHECK-LABEL: cuda.compiled_module @kernels_cuModule_0 dense<-1> : vector<1xi8>
// CHECK-LABEL: gpu.module @kernels attributes {kernel.ptx_data = dense<-1> : vector<1xi8>}
// CHECK-LABEL: func.func @caller
//  CHECK-SAME: (%[[arg0:.+]]: memref<1xf32, #executor.memory_type<device>>, %[[arg1:.+]]: memref<1xf32, #executor.memory_type<device>>, %[[arg2:.+]]: index, %[[arg3:.+]]: index, %[[arg4:.+]]: memref<2x?xf32, strided<[?, ?], offset: ?>, #executor.memory_type<device>>, %[[arg5:.+]]: memref<1xf32, strided<[?], offset: 4>, #executor.memory_type<device>>) {
//   CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[dev:.+]] = cuda.get_program_device %[[c0_i32]] : i32
//   CHECK-DAG:     %[[stream:.+]] = cuda.get_global_stream device(%[[dev]]) [0]
//   CHECK-DAG:     %[[func0:.+]] = cuda.get_function "kernel" from @kernels_cuModule_0
//   CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[base_buffer:.+]], %[[offset:.+]], %[[sizes:.+]], %[[strides:.+]] = memref.extract_strided_metadata %[[arg0]]
//   CHECK-DAG:     %[[base_buffer_0:.+]], %[[offset_1:.+]], %[[sizes_2:.+]], %[[strides_3:.+]] = memref.extract_strided_metadata %[[arg1]]
//   CHECK-DAG:     %[[grid_x:.+]] = arith.index_cast %[[arg2]] : index to i32
//   CHECK-DAG:     %[[block_x:.+]] = arith.index_cast %[[arg3]] : index to i32
//       CHECK:     cuda.launch %[[func0]](%[[base_buffer]], %[[base_buffer_0]] : memref<f32, #executor.memory_type<device>>, memref<f32, #executor.memory_type<device>>) with
//       CHECK:      grid(%[[grid_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      block(%[[block_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      smem(%[[c0_i32]]) stream(%[[stream]])
//   CHECK-DAG:     %[[func1:.+]] = cuda.get_function "kernel2" from @kernels_cuModule_0
//   CHECK-DAG:     %[[base_buffer_4:.+]], %[[offset_5:.+]], %[[sizes_6:.+]]:2, %[[strides_7:.+]]:2 = memref.extract_strided_metadata %[[arg4]]
//   CHECK-DAG:     %[[base_buffer_8:.+]], %[[offset_9:.+]], %[[sizes_10:.+]], %[[strides_11:.+]] = memref.extract_strided_metadata %[[arg5]]
//       CHECK:     cuda.launch %[[func1]](%[[base_buffer_4]], %[[offset_5]], %[[sizes_6]]#1, %[[strides_7]]#0, %[[strides_7]]#1, %[[base_buffer_8]], %[[strides_11]] : memref<f32, #executor.memory_type<device>>, index, index, index, index, memref<f32, #executor.memory_type<device>>, index) with
//       CHECK:      grid(%[[grid_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      block(%[[block_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      smem(%[[c0_i32]]) stream(%[[stream]])
//   CHECK-DAG:     %[[func2:.+]] = cuda.get_function "kernel3" from @kernels_cuModule_0
//       CHECK:     cuda.launch %[[func2]](%[[base_buffer_4]], %[[offset_5]], %[[sizes_6]]#1, %[[strides_7]]#0, %[[strides_7]]#1, %[[base_buffer_8]], %[[strides_11]] : memref<f32, #executor.memory_type<device>>, index, index, index, index, memref<f32, #executor.memory_type<device>>, index) with
//       CHECK:      grid(%[[grid_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      block(%[[block_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      smem(%[[c0_i32]]) stream(%[[stream]])

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

// CHECK-LABEL: cuda.compiled_module @dedup_globals_cuModule_0 dense<-1> : vector<1xi8>
// CHECK-LABEL: gpu.module @dedup_globals attributes {kernel.ptx_data = dense<-1> : vector<1xi8>}
// CHECK-LABEL: func.func @dedup
//  CHECK-SAME: (%[[arg0:.+]]: memref<1xf32, #executor.memory_type<device>>, %[[arg1:.+]]: memref<1xf32, #executor.memory_type<device>>, %[[arg2:.+]]: index, %[[arg3:.+]]: index) {
//   CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[dev:.+]] = cuda.get_program_device %[[c0_i32]] : i32
//   CHECK-DAG:     %[[stream:.+]] = cuda.get_global_stream device(%[[dev]]) [0]
//   CHECK-DAG:     %[[func0:.+]] = cuda.get_function "kernel" from @dedup_globals_cuModule_0
//   CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[base_buffer:.+]], %[[offset:.+]], %[[sizes:.+]], %[[strides:.+]] = memref.extract_strided_metadata %[[arg0]]
//   CHECK-DAG:     %[[base_buffer_0:.+]], %[[offset_1:.+]], %[[sizes_2:.+]], %[[strides_3:.+]] = memref.extract_strided_metadata %[[arg1]]
//   CHECK-DAG:     %[[grid_x:.+]] = arith.index_cast %[[arg2]] : index to i32
//   CHECK-DAG:     %[[block_x:.+]] = arith.index_cast %[[arg3]] : index to i32
//       CHECK:     cuda.launch %[[func0]](%[[base_buffer]], %[[base_buffer_0]] : memref<f32, #executor.memory_type<device>>, memref<f32, #executor.memory_type<device>>) with
//       CHECK:      grid(%[[grid_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      block(%[[block_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      smem(%[[c0_i32]]) stream(%[[stream]])
//       CHECK:     cuda.launch %[[func0]](%[[base_buffer]], %[[base_buffer_0]] : memref<f32, #executor.memory_type<device>>, memref<f32, #executor.memory_type<device>>) with
//       CHECK:      grid(%[[grid_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      block(%[[block_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      smem(%[[c0_i32]]) stream(%[[stream]])
//   CHECK-DAG:     %[[func1:.+]] = cuda.get_function "kernel2" from @dedup_globals_cuModule_0
//       CHECK:     cuda.launch %[[func1]](%[[base_buffer]], %[[base_buffer_0]] : memref<f32, #executor.memory_type<device>>, memref<f32, #executor.memory_type<device>>) with
//       CHECK:      grid(%[[grid_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      block(%[[block_x]], %[[c1_i32]], %[[c1_i32]])
//       CHECK:      smem(%[[c0_i32]]) stream(%[[stream]])

func.func @dedup(%arg0: memref<1xf32, #executor.memory_type<device>>, %arg1: memref<1xf32, #executor.memory_type<device>>, %arg2: index, %arg3: index) {
  kernel.call @dedup_globals::@kernel grid [%arg2] block[%arg3] (%arg0) outs(%arg1) : (memref<1xf32, #executor.memory_type<device>>, memref<1xf32, #executor.memory_type<device>>) -> ()
  kernel.call @dedup_globals::@kernel grid [%arg2] block[%arg3] (%arg0) outs(%arg1) : (memref<1xf32, #executor.memory_type<device>>, memref<1xf32, #executor.memory_type<device>>) -> ()
  kernel.call @dedup_globals::@kernel2 grid [%arg2] block[%arg3] (%arg0) outs(%arg1) : (memref<1xf32, #executor.memory_type<device>>, memref<1xf32, #executor.memory_type<device>>) -> ()
  return
}


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
