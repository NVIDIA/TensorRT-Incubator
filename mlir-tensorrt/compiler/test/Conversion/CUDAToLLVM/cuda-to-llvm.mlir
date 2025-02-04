// RUN: mlir-tensorrt-opt %s -convert-cuda-to-llvm -split-input-file | FileCheck %s

// RUN: rm -rf %t.artifacts
// RUN: mkdir -p %t.artifacts
// RUN: mlir-tensorrt-opt -split-input-file -convert-cuda-to-llvm="artifacts-dir=%t.artifacts" %s | FileCheck %s --check-prefix=FILE
// RUN: file %t.artifacts/kernels.ptx

cuda.compiled_module @kernels dense<[0xFF,0x00]> : vector<2xi8>

func.func @test_get_func() -> !cuda.function {
  %func = cuda.get_function "kernel"from @kernels
  return %func: !cuda.function
}

// CHECK-LABEL: llvm.mlir.global private constant @kernel_name("kernel\00") {addr_space = 0 : i32}
// CHECK-LABEL: func.func @test_get_func
//   CHECK-DAG:     %[[v0:.+]] = llvm.mlir.addressof @[[kernel_name:.+]] : !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[v1]] : !llvm.ptr to !cuda.function
//   CHECK-DAG:     return %[[v2]] : !cuda.function

// CHECK-LABEL: llvm.func @kernels_init
//   CHECK-DAG:     %[[v0:.+]] = llvm.mlir.addressof @kernels_ptx : !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.mlir.constant(2 : i64) : i64
//   CHECK-DAG:     %[[v2:.+]] = llvm.call @mtrt_cuda_module_load_from_ptx(%[[v0]], %[[v1]])
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.addressof @kernels_0 : !llvm.ptr
//       CHECK:     llvm.store %[[v2]], %[[v3]] : !llvm.ptr, !llvm.ptr
//       CHECK:     llvm.return
//       CHECK:   llvm.mlir.global_ctors {ctors = [@kernels_init], priorities = [0 : i32]}

// CHECK-LABEL: llvm.func @kernels_deinit
//       CHECK:     %[[v0:.+]] = llvm.mlir.addressof @kernels_0 : !llvm.ptr
//       CHECK:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//       CHECK:     llvm.call @mtrt_cuda_module_unload(%[[v1]]) : (!llvm.ptr) -> ()
//       CHECK:     llvm.return
//       CHECK:   llvm.mlir.global_dtors {dtors = [@kernels_deinit], priorities = [0 : i32]}

// CHECK-LABEL: llvm.func @kernels_0_kernel_init
//   CHECK-DAG:     %[[v0:.+]] = llvm.mlir.addressof @kernel_name : !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.mlir.constant(6 : i64) : i64
//   CHECK-DAG:     %[[v2:.+]] = llvm.mlir.addressof @kernels_0 : !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = llvm.load %[[v2]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     %[[v4:.+]] = llvm.call @mtrt_cuda_module_get_function(%[[v3]], %[[v0]], %[[v1]])
//   CHECK-DAG:     %[[v5:.+]] = llvm.mlir.addressof @kernels_0_kernel : !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v4]], %[[v5]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     llvm.return
//       CHECK:   llvm.mlir.global_ctors {ctors = [@kernels_0_kernel_init], priorities = [1 : i32]}

// FILE-LABEL: @test_get_func
// FILE-LABEL: llvm.func @kernels_init
//       FILE:     %[[v0:.+]] = llvm.mlir.addressof @kernels_filename : !llvm.ptr
//       FILE:     %[[v1:.+]] = llvm.mlir.constant(11 : i64) : i64
//       FILE:     %[[v2:.+]] = llvm.call @mtrt_cuda_module_load_from_ptx_file(%[[v0]], %[[v1]]) :
//       FILE:     %[[v3:.+]] = llvm.mlir.addressof @kernels_0 : !llvm.ptr
//       FILE:     llvm.store %[[v2]], %[[v3]] : !llvm.ptr, !llvm.ptr
// -----

!memref_4xi80 = memref<4xf32>
!memref_4xi81 = memref<4xf32>

func.func @test_cuda_launch(
    %func: !cuda.function,
    %stream: !cuda.stream,
    %arg0: !memref_4xi80,
    %arg1: !memref_4xi81,
    %arg2: index, %arg3: index) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %1 = arith.index_cast %arg2 : index to i32
    %2 = arith.index_cast %arg3 : index to i32
    cuda.launch %func(%arg0, %arg1 : !memref_4xi80, !memref_4xi81) with
      grid(%1, %c1_i32, %c1_i32)
      block(%2, %c1_i32, %c1_i32)
      smem(%c0_i32) stream(%stream)    // --> llvm.call @cudaLaunchKernelExC
  return
}

// FILE-LABEL: @test_cuda_launch

// CHECK-LABEL: func.func @test_cuda_launch
//  CHECK-SAME: (%[[arg0:.+]]: !cuda.function, %[[arg1:.+]]: !cuda.stream, %[[arg2:.+]]: memref<4xf32>, %[[arg3:.+]]: memref<4xf32>, %[[arg4:.+]]: index, %[[arg5:.+]]: index) {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg3]] : memref<4xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : memref<4xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : !cuda.function to !llvm.ptr
//   CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[v4:.+]] = arith.index_cast %[[arg4]] : index to i32
//   CHECK-DAG:     %[[v5:.+]] = arith.index_cast %[[arg5]] : index to i32
//   CHECK-DAG:     %[[v6:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:     %[[v7:.+]] = llvm.extractvalue %[[v1]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v8:.+]] = llvm.extractvalue %[[v1]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v9:.+]] = llvm.extractvalue %[[v1]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v10:.+]] = llvm.extractvalue %[[v1]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v11:.+]] = llvm.extractvalue %[[v1]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v12:.+]] = llvm.extractvalue %[[v0]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v13:.+]] = llvm.extractvalue %[[v0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v14:.+]] = llvm.extractvalue %[[v0]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v15:.+]] = llvm.extractvalue %[[v0]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v16:.+]] = llvm.extractvalue %[[v0]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v17:.+]] = llvm.alloca %[[v6]] x !llvm.ptr : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v7]], %[[v17]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v18:.+]] = llvm.alloca %[[v6]] x !llvm.ptr : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v8]], %[[v18]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v19:.+]] = llvm.alloca %[[v6]] x i64 : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v9]], %[[v19]] : i64, !llvm.ptr
//   CHECK-DAG:     %[[v20:.+]] = llvm.alloca %[[v6]] x i64 : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v10]], %[[v20]] : i64, !llvm.ptr
//   CHECK-DAG:     %[[v21:.+]] = llvm.alloca %[[v6]] x i64 : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v11]], %[[v21]] : i64, !llvm.ptr
//   CHECK-DAG:     %[[v22:.+]] = llvm.alloca %[[v6]] x !llvm.ptr : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v12]], %[[v22]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v23:.+]] = llvm.alloca %[[v6]] x !llvm.ptr : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v13]], %[[v23]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v24:.+]] = llvm.alloca %[[v6]] x i64 : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v14]], %[[v24]] : i64, !llvm.ptr
//   CHECK-DAG:     %[[v25:.+]] = llvm.alloca %[[v6]] x i64 : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v15]], %[[v25]] : i64, !llvm.ptr
//   CHECK-DAG:     %[[v26:.+]] = llvm.alloca %[[v6]] x i64 : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v16]], %[[v26]] : i64, !llvm.ptr
//   CHECK-DAG:     %[[v27:.+]] = llvm.alloca %[[v6]] x !llvm.array<10 x ptr> : (i64) -> !llvm.ptr
//   CHECK-DAG:     %[[v28:.+]] = llvm.getelementptr %[[v27]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x ptr>
//   CHECK-DAG:     llvm.store %[[v17]], %[[v28]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v29:.+]] = llvm.getelementptr %[[v27]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x ptr>
//   CHECK-DAG:     llvm.store %[[v18]], %[[v29]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v30:.+]] = llvm.getelementptr %[[v27]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x ptr>
//   CHECK-DAG:     llvm.store %[[v19]], %[[v30]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v31:.+]] = llvm.getelementptr %[[v27]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x ptr>
//   CHECK-DAG:     llvm.store %[[v20]], %[[v31]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v32:.+]] = llvm.getelementptr %[[v27]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x ptr>
//   CHECK-DAG:     llvm.store %[[v21]], %[[v32]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v33:.+]] = llvm.getelementptr %[[v27]][0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x ptr>
//   CHECK-DAG:     llvm.store %[[v22]], %[[v33]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v34:.+]] = llvm.getelementptr %[[v27]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x ptr>
//   CHECK-DAG:     llvm.store %[[v23]], %[[v34]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v35:.+]] = llvm.getelementptr %[[v27]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x ptr>
//   CHECK-DAG:     llvm.store %[[v24]], %[[v35]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v36:.+]] = llvm.getelementptr %[[v27]][0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x ptr>
//   CHECK-DAG:     llvm.store %[[v25]], %[[v36]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v37:.+]] = llvm.getelementptr %[[v27]][0, 9] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x ptr>
//   CHECK-DAG:     llvm.store %[[v26]], %[[v37]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     llvm.call @mtrt_cuda_launch_kernel(%[[v3]], %[[v4]], %[[c1_i32]], %[[c1_i32]], %[[v5]], %[[c1_i32]], %[[c1_i32]], %[[c0_i32]], %[[v2]], %[[v27]]) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> ()
//       CHECK:     return


// -----

func.func @cuda_global_stream() -> (!cuda.stream, !cuda.stream, !cuda.stream) {
  %0 = cuda.get_global_stream 0
  %1 = cuda.get_global_stream 1
  %2 = cuda.get_global_stream 0
  return %0, %1, %2 : !cuda.stream, !cuda.stream, !cuda.stream
}

// CHECK-LABEL:   llvm.mlir.global internal @stream_1() {addr_space = 0 : i32} : !llvm.ptr
// CHECK-LABEL:   llvm.mlir.global internal @stream_0() {addr_space = 0 : i32} : !llvm.ptr

// CHECK-LABEL: func.func @cuda_global_stream
//   CHECK-DAG:     %[[v0:.+]] = llvm.mlir.addressof @stream_0 : !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[v1]] : !llvm.ptr to !cuda.stream
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.addressof @stream_1 : !llvm.ptr
//   CHECK-DAG:     %[[v4:.+]] = llvm.load %[[v3]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     %[[v5:.+]] = builtin.unrealized_conversion_cast %[[v4]] : !llvm.ptr to !cuda.stream
//   CHECK-DAG:     %[[v6:.+]] = llvm.mlir.addressof @stream_0 : !llvm.ptr
//   CHECK-DAG:     %[[v7:.+]] = llvm.load %[[v6]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     %[[v8:.+]] = builtin.unrealized_conversion_cast %[[v7]] : !llvm.ptr to !cuda.stream
//   CHECK-DAG:     return %[[v2]], %[[v5]], %[[v8]] : !cuda.stream, !cuda.stream, !cuda.stream


// CHECK-LABEL: llvm.func @stream_0_init
//       CHECK:     %[[v0:.+]] = llvm.call @mtrt_cuda_stream_create() : () -> !llvm.ptr
//       CHECK:     %[[v1:.+]] = llvm.mlir.addressof @stream_0 : !llvm.ptr
//       CHECK:     llvm.store %[[v0]], %[[v1]] : !llvm.ptr, !llvm.ptr
//       CHECK:     llvm.return
//       CHECK:   llvm.mlir.global_ctors {ctors = [@stream_0_init], priorities = [0 : i32]}

// CHECK-LABEL: llvm.func @stream_0_deinit
//       CHECK:     %[[v0:.+]] = llvm.mlir.addressof @stream_0 : !llvm.ptr
//       CHECK:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//       CHECK:     llvm.call @mtrt_cuda_stream_destroy(%[[v1]]) : (!llvm.ptr) -> ()
//       CHECK:     llvm.return
//       CHECK:   llvm.mlir.global_dtors {dtors = [@stream_0_deinit], priorities = [0 : i32]}

// CHECK-LABEL: llvm.func @stream_1_init
//       CHECK:     %[[v0:.+]] = llvm.call @mtrt_cuda_stream_create() : () -> !llvm.ptr
//       CHECK:     %[[v1:.+]] = llvm.mlir.addressof @stream_1 : !llvm.ptr
//       CHECK:     llvm.store %[[v0]], %[[v1]] : !llvm.ptr, !llvm.ptr
//       CHECK:     llvm.return
//       CHECK:   llvm.mlir.global_ctors {ctors = [@stream_1_init], priorities = [0 : i32]}

// CHECK-LABEL: llvm.func @stream_1_deinit
//       CHECK:     %[[v0:.+]] = llvm.mlir.addressof @stream_1 : !llvm.ptr
//       CHECK:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//       CHECK:     llvm.call @mtrt_cuda_stream_destroy(%[[v1]]) : (!llvm.ptr) -> ()
//       CHECK:     llvm.return
//       CHECK:   llvm.mlir.global_dtors {dtors = [@stream_1_deinit], priorities = [0 : i32]}

// -----

!memref_4xi8 = memref<?x2x?xf32, #plan.memory_space<device>>

func.func @device_alloc(%arg0: index, %arg1: index, %stream: !cuda.stream, %device: i32) -> !memref_4xi8 {
  %0 = cuda.alloc(%arg0, %arg1) stream(%stream) device(%device) align 8 : !memref_4xi8
  return %0 : !memref_4xi8
}

// CHECK-LABEL: func.func @device_alloc
//  CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index, %[[arg2:.+]]: !cuda.stream, %[[arg3:.+]]: i32)
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : index to i64
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : index to i64
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(0 : i8) : i8
//   CHECK-DAG:     %[[v4:.+]] = llvm.mlir.constant(0 : i8) : i8
//   CHECK-DAG:     %[[v5:.+]] = llvm.mlir.constant(2 : index) : i64
//   CHECK-DAG:     %[[v6:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:     %[[v7:.+]] = llvm.mul %[[v0]], %[[v5]] : i64
//   CHECK-DAG:     %[[v8:.+]] = llvm.mul %[[v7]], %[[v1]] : i64
//   CHECK-DAG:     %[[v9:.+]] = llvm.mlir.zero : !llvm.ptr
//   CHECK-DAG:     %[[v10:.+]] = llvm.getelementptr %[[v9]][%[[v8]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK-DAG:     %[[v11:.+]] = llvm.ptrtoint %[[v10]] : !llvm.ptr to i64
//   CHECK-DAG:     %[[v12:.+]] = llvm.mlir.constant(8 : i32) : i32
//   CHECK-DAG:     %[[v13:.+]] = llvm.call @mtrt_cuda_alloc_async(%[[v2]], %[[arg3]], %[[v11]], %[[v12]], %[[v4]], %[[v3]]) :
//   CHECK-DAG:     %[[v14:.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v15:.+]] = llvm.insertvalue %[[v13]], %[[v14]][0] :
//   CHECK-DAG:     %[[v16:.+]] = llvm.insertvalue %[[v13]], %[[v15]][1] :
//   CHECK-DAG:     %[[v17:.+]] = llvm.mlir.constant(0 : index) : i64
//   CHECK-DAG:     %[[v18:.+]] = llvm.insertvalue %[[v17]], %[[v16]][2] :
//   CHECK-DAG:     %[[v19:.+]] = llvm.insertvalue %[[v1]], %[[v18]][3, 0]
//   CHECK-DAG:     %[[v20:.+]] = llvm.insertvalue %[[v5]], %[[v19]][3, 1]
//   CHECK-DAG:     %[[v21:.+]] = llvm.insertvalue %[[v0]], %[[v20]][3, 2]
//   CHECK-DAG:     %[[v22:.+]] = llvm.insertvalue %[[v7]], %[[v21]][4, 0]
//   CHECK-DAG:     %[[v23:.+]] = llvm.insertvalue %[[v0]], %[[v22]][4, 1]
//   CHECK-DAG:     %[[v24:.+]] = llvm.insertvalue %[[v6]], %[[v23]][4, 2]
//   CHECK-DAG:     %[[v25:.+]] = builtin.unrealized_conversion_cast %[[v24]]
//   CHECK-DAG:     return %[[v25]] : memref<?x2x?xf32, #plan.memory_space<device>>

// -----

func.func @memref_device_alloc_i1(%arg0: !cuda.stream, %device: i32) -> memref<1500x1500xi1, #plan.memory_space<device>> {
  %0 = cuda.alloc () stream(%arg0) device(%device) : memref<1500x1500xi1, #plan.memory_space<device>>
  return %0 : memref<1500x1500xi1, #plan.memory_space<device>>
}

// CHECK-LABEL: func.func @memref_device_alloc_i1
//  CHECK-SAME: (%[[arg0:.+]]: !cuda.stream, %[[arg1:.+]]: i32) -> memref<1500x1500xi1, #plan.memory_space<device>> {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.mlir.constant(0 : i8) : i8
//   CHECK-DAG:     %[[v2:.+]] = llvm.mlir.constant(0 : i8) : i8
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(1500 : index) : i64
//   CHECK-DAG:     %[[v4:.+]] = llvm.mlir.constant(1500 : index) : i64
//   CHECK-DAG:     %[[v5:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:     %[[v6:.+]] = llvm.mlir.constant(2250000 : index) : i64
//   CHECK-DAG:     %[[v7:.+]] = llvm.mlir.zero : !llvm.ptr
//   CHECK-DAG:     %[[v8:.+]] = llvm.getelementptr %[[v7]][%[[v6]]] : (!llvm.ptr, i64) -> !llvm.ptr, i1
//   CHECK-DAG:     %[[v9:.+]] = llvm.ptrtoint %[[v8]] : !llvm.ptr to i64
//   CHECK-DAG:     %[[v10:.+]] = llvm.mlir.constant(16 : i32) : i32
//   CHECK-DAG:     %[[v11:.+]] = llvm.call @mtrt_cuda_alloc_async(%[[v0]], %[[arg1]], %[[v9]], %[[v10]], %[[v2]], %[[v1]]) : (!llvm.ptr, i32, i64, i32, i8, i8) -> !llvm.ptr
//   CHECK-DAG:     %[[v12:.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v13:.+]] = llvm.insertvalue %[[v11]], %[[v12]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v14:.+]] = llvm.insertvalue %[[v11]], %[[v13]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v15:.+]] = llvm.mlir.constant(0 : index) : i64
//   CHECK-DAG:     %[[v16:.+]] = llvm.insertvalue %[[v15]], %[[v14]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v17:.+]] = llvm.insertvalue %[[v3]], %[[v16]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v18:.+]] = llvm.insertvalue %[[v4]], %[[v17]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v19:.+]] = llvm.insertvalue %[[v4]], %[[v18]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v20:.+]] = llvm.insertvalue %[[v5]], %[[v19]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v21:.+]] = builtin.unrealized_conversion_cast %[[v20]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1500x1500xi1, #plan.memory_space<device>>
//   CHECK-DAG:     return %[[v21]] : memref<1500x1500xi1, #plan.memory_space<device>>

// -----

!memref_4xi8 = memref<?x2x?xf32, #plan.memory_space<host_pinned>>

func.func @pinned_alloc(%arg0: index, %arg1: index, %stream: !cuda.stream, %device: i32) -> !memref_4xi8 {
  %0 = cuda.alloc(%arg0, %arg1) align 8 : !memref_4xi8
  return %0 : !memref_4xi8
}

// CHECK-LABEL: func.func @pinned_alloc
//  CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index, %[[arg2:.+]]: !cuda.stream, %[[arg3:.+]]: i32) -> memref<?x2x?xf32, #plan.memory_space<host_pinned>> {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : index to i64
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : index to i64
//   CHECK-DAG:     %[[v2:.+]] = llvm.mlir.constant(0 : i8) : i8
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(1 : i8) : i8
//   CHECK-DAG:     %[[v4:.+]] = llvm.mlir.constant(2 : index) : i64
//   CHECK-DAG:     %[[v5:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:     %[[v6:.+]] = llvm.mul %[[v0]], %[[v4]] : i64
//   CHECK-DAG:     %[[v7:.+]] = llvm.mul %[[v6]], %[[v1]] : i64
//   CHECK-DAG:     %[[v8:.+]] = llvm.mlir.zero : !llvm.ptr
//   CHECK-DAG:     %[[v9:.+]] = llvm.getelementptr %[[v8]][%[[v7]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK-DAG:     %[[v10:.+]] = llvm.ptrtoint %[[v9]] : !llvm.ptr to i64
//   CHECK-DAG:     %[[v11:.+]] = llvm.mlir.zero : !llvm.ptr
//   CHECK-DAG:     %[[v12:.+]] = llvm.mlir.constant(8 : i32) : i32
//   CHECK-DAG:     %[[v13:.+]] = llvm.mlir.constant(-1 : i32) : i32
//   CHECK-DAG:     %[[v14:.+]] = llvm.call @mtrt_cuda_alloc_async(%[[v11]], %[[v13]], %[[v10]], %[[v12]], %[[v3]], %[[v2]]) :
//   CHECK-DAG:     %[[v15:.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v16:.+]] = llvm.insertvalue %[[v14]], %[[v15]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v17:.+]] = llvm.insertvalue %[[v14]], %[[v16]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v18:.+]] = llvm.mlir.constant(0 : index) : i64
//   CHECK-DAG:     %[[v19:.+]] = llvm.insertvalue %[[v18]], %[[v17]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v20:.+]] = llvm.insertvalue %[[v1]], %[[v19]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v21:.+]] = llvm.insertvalue %[[v4]], %[[v20]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v22:.+]] = llvm.insertvalue %[[v0]], %[[v21]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v23:.+]] = llvm.insertvalue %[[v6]], %[[v22]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v24:.+]] = llvm.insertvalue %[[v0]], %[[v23]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v25:.+]] = llvm.insertvalue %[[v5]], %[[v24]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v26:.+]] = builtin.unrealized_conversion_cast %[[v25]] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<?x2x?xf32, #plan.memory_space<host_pinned>>
//   CHECK-DAG:     return %[[v26]] : memref<?x2x?xf32, #plan.memory_space<host_pinned>>

// -----

!memref_4xi8 = memref<?x2x?xf32, #plan.memory_space<device>>
func.func @device_free(%arg0: !cuda.stream, %arg1: !memref_4xi8) {
  cuda.dealloc stream(%arg0) %arg1  : !memref_4xi8
  return
}

// CHECK-LABEL: func.func @device_free
//  CHECK-SAME: (%[[arg0:.+]]: !cuda.stream, %[[arg1:.+]]: memref<?x2x?xf32, #plan.memory_space<device>>) {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<?x2x?xf32, #plan.memory_space<device>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v2:.+]] = llvm.mlir.constant(0 : i8) : i8
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(0 : i8) : i8
//   CHECK-DAG:     %[[v4:.+]] = llvm.extractvalue %[[v0]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     llvm.call @mtrt_cuda_free(%[[v1]], %[[v4]], %[[v2]], %[[v3]])

// -----

!memref_4xi8 = memref<?x2x?xf32, #plan.memory_space<host_pinned>>
func.func @free_host_pinned(%arg0: !cuda.stream, %arg1: !memref_4xi8) {
  cuda.dealloc stream(%arg0) %arg1  : !memref_4xi8
  return
}

// CHECK-LABEL: func.func @free_host_pinned
//  CHECK-SAME: (%[[arg0:.+]]: !cuda.stream, %[[arg1:.+]]: memref<?x2x?xf32, #plan.memory_space<host_pinned>>) {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg1]]
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg0]]
//   CHECK-DAG:     %[[v2:.+]] = llvm.mlir.constant(1 : i8) : i8
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(0 : i8) : i8
//   CHECK-DAG:     %[[v4:.+]] = llvm.extractvalue %[[v0]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     llvm.call @mtrt_cuda_free(%[[v1]], %[[v4]], %[[v2]], %[[v3]])

// -----

!memref_4xi8 = memref<?x2x?xf32, #plan.memory_space<unified>>
func.func @free_unified(%arg0: !cuda.stream, %arg1: !memref_4xi8) {
  cuda.dealloc stream(%arg0) %arg1  : !memref_4xi8
  return
}

// CHECK-LABEL: func.func @free_unified
//  CHECK-SAME: (%[[arg0:.+]]: !cuda.stream, %[[arg1:.+]]: memref<?x2x?xf32, #plan.memory_space<unified>>) {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg1]]
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg0]]
//   CHECK-DAG:     %[[v2:.+]] = llvm.mlir.constant(0 : i8) : i8
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(1 : i8) : i8
//   CHECK-DAG:     %[[v4:.+]] = llvm.extractvalue %[[v0]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     llvm.call @mtrt_cuda_free(%[[v1]], %[[v4]], %[[v2]], %[[v3]])

// -----

#device_space = #plan.memory_space<device>
#host_space = #plan.memory_space<host>
!src_memref_type = memref<?x2x?xf32, #device_space>
!dst_memref_type = memref<?x2x?xf32, #device_space>

func.func @copy_d2d(%arg0: !src_memref_type, %arg1: !dst_memref_type, %stream: !cuda.stream) {
  cuda.copy_d2d stream(%stream) %arg0, %arg1 : !src_memref_type to !dst_memref_type
  return
}

// CHECK-LABEL: func.func @copy_d2d
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x2x?xf32, #plan.memory_space<device>>, %[[arg1:.+]]: memref<?x2x?xf32, #plan.memory_space<device>>, %[[arg2:.+]]: !cuda.stream) {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<?x2x?xf32, #plan.memory_space<device>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<?x2x?xf32, #plan.memory_space<device>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = llvm.extractvalue %[[v1]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v4:.+]] = llvm.extractvalue %[[v1]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v5:.+]] = llvm.mul %[[v3]], %[[v4]] : i64
//   CHECK-DAG:     %[[v6:.+]] = llvm.mlir.zero : !llvm.ptr
//   CHECK-DAG:     %[[v7:.+]] = llvm.getelementptr %[[v6]][%[[v5]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK-DAG:     %[[v8:.+]] = llvm.ptrtoint %[[v7]] : !llvm.ptr to i64
//   CHECK-DAG:     %[[v9:.+]] = llvm.extractvalue %[[v1]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v10:.+]] = llvm.extractvalue %[[v0]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     llvm.call @mtrt_cuda_memcpy_async(%[[v2]], %[[v9]], %[[v10]], %[[v8]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
//   CHECK-DAG:     return

// -----

func.func @copy_d2h_offset(%arg0: memref<128x16xf32, strided<[16, 1], offset: 16>, #plan.memory_space<device>>,
                           %arg1: memref<128x16xf32, strided<[16, 1], offset: 8>, #plan.memory_space<host>>,
                           %arg2: !cuda.stream) {
  cuda.copy_d2h stream(%arg2) %arg0, %arg1 :
    memref<128x16xf32, strided<[16, 1], offset: 16>, #plan.memory_space<device>>
    to memref<128x16xf32, strided<[16, 1], offset: 8>, #plan.memory_space<host>>
  return
}

// CHECK-LABEL: func.func @copy_d2h_offset
//  CHECK-SAME: (%[[arg0:.+]]: memref<128x16xf32, strided<[16, 1], offset: 16>, #plan.memory_space<device>>, %[[arg1:.+]]: memref<128x16xf32, strided<[16, 1], offset: 8>, #plan.memory_space<host>>, %[[arg2:.+]]: !cuda.stream) {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<128x16xf32, strided<[16, 1], offset: 8>, #plan.memory_space<host>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<128x16xf32, strided<[16, 1], offset: 16>, #plan.memory_space<device>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(2048 : index) : i64
//   CHECK-DAG:     %[[v4:.+]] = llvm.mlir.zero : !llvm.ptr
//   CHECK-DAG:     %[[v5:.+]] = llvm.getelementptr %[[v4]][%[[v3]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK-DAG:     %[[v6:.+]] = llvm.ptrtoint %[[v5]] : !llvm.ptr to i64
//   CHECK-DAG:     %[[v7:.+]] = llvm.extractvalue %[[v1]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v8:.+]] = llvm.mlir.constant(16 : index) : i64
//   CHECK-DAG:     %[[v9:.+]] = llvm.getelementptr %[[v7]][%[[v8]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK-DAG:     %[[v10:.+]] = llvm.extractvalue %[[v0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v11:.+]] = llvm.mlir.constant(8 : index) : i64
//   CHECK-DAG:     %[[v12:.+]] = llvm.getelementptr %[[v10]][%[[v11]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK-DAG:     llvm.call @mtrt_cuda_memcpy_async(%[[v2]], %[[v9]], %[[v12]], %[[v6]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
//   CHECK-DAG:     return

// -----

!srcType = memref<6xf32, strided<[2], offset: 2>, #plan.memory_space<device>>
!dstType = memref<6xf32, strided<[2], offset: 4>, #plan.memory_space<host>>

func.func @copy_d2h_strided(%arg0: !srcType,
                           %arg1: !dstType, %arg2: !cuda.stream) {
  cuda.copy_d2h stream(%arg2) %arg0, %arg1 : !srcType to !dstType
  return
}

// CHECK-LABEL: func.func @copy_d2h_strided
//  CHECK-SAME: (%[[arg0:.+]]: memref<6xf32, strided<[2], offset: 2>, #plan.memory_space<device>>, %[[arg1:.+]]: memref<6xf32, strided<[2], offset: 4>, #plan.memory_space<host>>, %[[arg2:.+]]: !cuda.stream) {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<6xf32, strided<[2], offset: 4>, #plan.memory_space<host>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<6xf32, strided<[2], offset: 2>, #plan.memory_space<device>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(1 : i64) : i64
//   CHECK-DAG:     %[[v4:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:     %[[v5:.+]] = llvm.alloca %[[v4]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v1]], %[[v5]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
//   CHECK-DAG:     %[[v6:.+]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v7:.+]] = llvm.insertvalue %[[v3]], %[[v6]][0] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v8:.+]] = llvm.insertvalue %[[v5]], %[[v7]][1] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v9:.+]] = llvm.mlir.constant(1 : i64) : i64
//   CHECK-DAG:     %[[v10:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:     %[[v11:.+]] = llvm.alloca %[[v10]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v0]], %[[v11]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
//   CHECK-DAG:     %[[v12:.+]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v13:.+]] = llvm.insertvalue %[[v9]], %[[v12]][0] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v14:.+]] = llvm.insertvalue %[[v11]], %[[v13]][1] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v15:.+]] = llvm.extractvalue %[[v8]][0] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v16:.+]] = llvm.extractvalue %[[v8]][1] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v17:.+]] = llvm.extractvalue %[[v14]][0] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v18:.+]] = llvm.extractvalue %[[v14]][1] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     llvm.call @mtrt_cuda_memcpy_strided_async(%[[v2]], %[[v15]], %[[v16]], %[[v17]], %[[v18]]) : (!llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr) -> ()

// -----

!srcType = memref<1x8x4xf32, strided<[256, 4, 1], offset: ?>, #plan.memory_space<device>>
!dstType = memref<1x8x4xf32, strided<[128, 4, 1], offset: ?>, #plan.memory_space<host>>

func.func @memref_copy_contiguous_non_identity(%arg0: !srcType, %arg1: !dstType,
    %arg2: !cuda.stream) {
  cuda.copy_d2h stream(%arg2) %arg0, %arg1 : !srcType to !dstType
  return
}

// CHECK-LABEL: func.func @memref_copy_contiguous_non_identity
//  CHECK-SAME: (%[[arg0:.+]]: memref<1x8x4xf32, strided<[256, 4, 1], offset: ?>, #plan.memory_space<device>>, %[[arg1:.+]]: memref<1x8x4xf32, strided<[128, 4, 1], offset: ?>, #plan.memory_space<host>>, %[[arg2:.+]]: !cuda.stream) {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<1x8x4xf32, strided<[128, 4, 1], offset: ?>, #plan.memory_space<host>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<1x8x4xf32, strided<[256, 4, 1], offset: ?>, #plan.memory_space<device>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(32 : index) : i64
//   CHECK-DAG:     %[[v4:.+]] = llvm.mlir.zero : !llvm.ptr
//   CHECK-DAG:     %[[v5:.+]] = llvm.getelementptr %[[v4]][%[[v3]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK-DAG:     %[[v6:.+]] = llvm.ptrtoint %[[v5]] : !llvm.ptr to i64
//   CHECK-DAG:     %[[v7:.+]] = llvm.extractvalue %[[v1]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v8:.+]] = llvm.extractvalue %[[v1]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v9:.+]] = llvm.getelementptr %[[v7]][%[[v8]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK-DAG:     %[[v10:.+]] = llvm.extractvalue %[[v0]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v11:.+]] = llvm.extractvalue %[[v0]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v12:.+]] = llvm.getelementptr %[[v10]][%[[v11]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK-DAG:     llvm.call @mtrt_cuda_memcpy_async(%[[v2]], %[[v9]], %[[v12]], %[[v6]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
//   CHECK-DAG:     return

// -----

#device_space = #plan.memory_space<device>
#host_space = #plan.memory_space<host>

!src_memref_type = memref<?x2x?xf32, #device_space>
!dst_memref_type = memref<?x2x?xf32, #host_space>

func.func @copy_d2h(%arg0: !src_memref_type, %arg1: !dst_memref_type,  %stream: !cuda.stream) {
  cuda.copy_d2h stream(%stream) %arg0, %arg1 : !src_memref_type to !dst_memref_type
  return
}

// CHECK-LABEL: func.func @copy_d2h
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x2x?xf32, #plan.memory_space<device>>, %[[arg1:.+]]: memref<?x2x?xf32, #plan.memory_space<host>>, %[[arg2:.+]]: !cuda.stream)
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<?x2x?xf32, #plan.memory_space<host>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<?x2x?xf32, #plan.memory_space<device>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = llvm.extractvalue %[[v1]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v4:.+]] = llvm.extractvalue %[[v1]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v5:.+]] = llvm.mul %[[v3]], %[[v4]] : i64
//   CHECK-DAG:     %[[v6:.+]] = llvm.mlir.zero : !llvm.ptr
//   CHECK-DAG:     %[[v7:.+]] = llvm.getelementptr %[[v6]][%[[v5]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK-DAG:     %[[v8:.+]] = llvm.ptrtoint %[[v7]] : !llvm.ptr to i64
//   CHECK-DAG:     %[[v9:.+]] = llvm.extractvalue %[[v1]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v10:.+]] = llvm.extractvalue %[[v0]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     llvm.call @mtrt_cuda_memcpy_async(%[[v2]], %[[v9]], %[[v10]], %[[v8]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> ()
//   CHECK-DAG:     return

// -----

#device_space = #plan.memory_space<host>
#host_space = #plan.memory_space<device>
!src_memref_type = memref<?x2x?xf32, #device_space>
!dst_memref_type = memref<?x2x?xf32, #device_space>

func.func @copy_h2d(%arg0: !src_memref_type, %arg1: !dst_memref_type,  %stream: !cuda.stream) {
  cuda.copy_h2d stream(%stream) %arg0, %arg1 : !src_memref_type to !dst_memref_type
  return
}

// CHECK-LABEL: func.func @copy_h2d
//  CHECK-SAME: (%[[arg0:.+]]: memref<?x2x?xf32, #plan.memory_space<host>>, %[[arg1:.+]]: memref<?x2x?xf32, #plan.memory_space<host>>, %[[arg2:.+]]: !cuda.stream) {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<?x2x?xf32, #plan.memory_space<host>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<?x2x?xf32, #plan.memory_space<host>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = llvm.extractvalue %[[v1]][4, 0]
//   CHECK-DAG:     %[[v4:.+]] = llvm.extractvalue %[[v1]][3, 0]
//   CHECK-DAG:     %[[v5:.+]] = llvm.mul %[[v3]], %[[v4]] : i64
//   CHECK-DAG:     %[[v6:.+]] = llvm.mlir.zero : !llvm.ptr
//   CHECK-DAG:     %[[v7:.+]] = llvm.getelementptr %[[v6]][%[[v5]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK-DAG:     %[[v8:.+]] = llvm.ptrtoint %[[v7]] : !llvm.ptr to i64
//   CHECK-DAG:     %[[v9:.+]] = llvm.extractvalue %[[v1]][1]
//   CHECK-DAG:     %[[v10:.+]] = llvm.extractvalue %[[v0]][1]
//   CHECK-DAG:     llvm.call @mtrt_cuda_memcpy_async(%[[v2]], %[[v9]], %[[v10]], %[[v8]])

// -----

func.func @cuda_get_current_device() -> i32 {
  %0 = cuda.get_current_device
  return %0 : i32
}

// CHECK-LABEL: func.func @cuda_get_current_device
//   CHECK-DAG:   %[[v0:.+]] = llvm.call @mtrt_cuda_get_current_device() : () -> i32
//   CHECK-DAG:   return %[[v0]] : i32