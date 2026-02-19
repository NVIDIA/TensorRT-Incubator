// RUN: mlir-tensorrt-opt -split-input-file -convert-tensorrt-runtime-to-llvm %s | FileCheck %s

func.func @test_enqueue() -> (!trtrt.context, !trtrt.context, !trtrt.context) {
  %1 = trtrt.get_function @foo : !trtrt.context
  %2 = trtrt.get_function @bar : !trtrt.context
  %3 = trtrt.get_function @foo : !trtrt.context
  return %1, %2, %3 : !trtrt.context, !trtrt.context, !trtrt.context
}

trtrt.compiled_func @foo dense<[0,1,2,3,4,5,6,7]> : vector<8xi8>
trtrt.compiled_func @bar dense<[0,1,2,3,4,5,6,7]> : vector<8xi8>

// CHECK-LABEL: func.func @test_enqueue
//   CHECK-DAG:     %[[v0:.+]] = llvm.mlir.addressof @foo.context : !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[v1]] : !llvm.ptr to !trtrt.context
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.addressof @bar.context : !llvm.ptr
//   CHECK-DAG:     %[[v4:.+]] = llvm.load %[[v3]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     %[[v5:.+]] = builtin.unrealized_conversion_cast %[[v4]] : !llvm.ptr to !trtrt.context
//   CHECK-DAG:     %[[v6:.+]] = llvm.mlir.addressof @foo.context : !llvm.ptr
//   CHECK-DAG:     %[[v7:.+]] = llvm.load %[[v6]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     %[[v8:.+]] = builtin.unrealized_conversion_cast %[[v7]] : !llvm.ptr to !trtrt.context
//   CHECK-DAG:     return %[[v2]], %[[v5]], %[[v8]] : !trtrt.context, !trtrt.context, !trtrt.context

// CHECK-LABEL: llvm.func @tensorrt_runtime_init
//   CHECK-DAG:     %[[v0:.+]] = llvm.call @mtrt_tensorrt_runtime_create() : () -> !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.mlir.addressof @tensorrt_runtime : !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v0]], %[[v1]] : !llvm.ptr, !llvm.ptr
//       CHECK: llvm.mlir.global_ctors ctors = [@tensorrt_runtime_init], priorities = [10 : i32]

// CHECK-LABEL: llvm.func @mtrt_tensorrt_runtime_create

// CHECK-LABEL: llvm.func @tensorrt_runtime_deinit
//   CHECK-DAG:     %[[v0:.+]] = llvm.mlir.addressof @tensorrt_runtime : !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     llvm.call @mtrt_tensorrt_runtime_destroy(%[[v1]]) : (!llvm.ptr) -> ()
//   CHECK-DAG:   llvm.mlir.global_dtors dtors = [@tensorrt_runtime_deinit], priorities = [0 : i32]

// CHECK-LABEL: llvm.func @foo_context_init
//   CHECK-DAG:     %[[v0:.+]] = llvm.mlir.addressof @tensorrt_runtime : !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     %[[v2:.+]] = llvm.mlir.addressof @foo_filename : !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(31 : i64) : i64
//   CHECK-DAG:     %[[v4:.+]] = llvm.call @mtrt_load_tensorrt_engine_from_file(%[[v1]], %[[v2]], %[[v3]]) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
//   CHECK-DAG:     %[[v5:.+]] = llvm.mlir.addressof @foo.context : !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v4]], %[[v5]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:   llvm.mlir.global_ctors ctors = [@foo_context_init], priorities = [9 : i32]

// CHECK-LABEL: llvm.func @foo_context_deinit
//   CHECK-DAG:     %[[v0:.+]] = llvm.mlir.addressof @foo.context : !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     llvm.call @mtrt_tensorrt_execution_context_destroy(%[[v1]]) : (!llvm.ptr) -> ()
//   CHECK-DAG:   llvm.mlir.global_ctors ctors = [@foo_context_deinit], priorities = [1 : i32]

// CHECK-LABEL: llvm.func @bar_context_init
//   CHECK-DAG:     %[[v0:.+]] = llvm.mlir.addressof @tensorrt_runtime : !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     %[[v2:.+]] = llvm.mlir.addressof @bar_filename : !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(31 : i64) : i64
//   CHECK-DAG:     %[[v4:.+]] = llvm.call @mtrt_load_tensorrt_engine_from_file(%[[v1]], %[[v2]], %[[v3]]) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
//   CHECK-DAG:     %[[v5:.+]] = llvm.mlir.addressof @bar.context : !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v4]], %[[v5]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:   llvm.mlir.global_ctors ctors = [@bar_context_init], priorities = [9 : i32]

// CHECK-LABEL: llvm.func @bar_context_deinit
//   CHECK-DAG:     %[[v0:.+]] = llvm.mlir.addressof @bar.context : !llvm.ptr
//   CHECK-DAG:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//   CHECK-DAG:     llvm.call @mtrt_tensorrt_execution_context_destroy(%[[v1]]) : (!llvm.ptr) -> ()
//   CHECK-DAG:     llvm.return
//   CHECK-DAG:   llvm.mlir.global_ctors ctors = [@bar_context_deinit], priorities = [1 : i32]

//  FILE-LABEL: func.func @test_enqueue
//  FILE-LABEL: llvm.func @foo_context_init
//    FILE-DAG:     %[[v0:.+]] = llvm.mlir.addressof @tensorrt_runtime : !llvm.ptr
//    FILE-DAG:     %[[v1:.+]] = llvm.load %[[v0]] : !llvm.ptr -> !llvm.ptr
//    FILE-DAG:     %[[v2:.+]] = llvm.mlir.addressof @foo_filename : !llvm.ptr
//    FILE-DAG:     %[[v3:.+]] = llvm.mlir.constant(16 : i64) : i64
//    FILE-DAG:     %[[v4:.+]] = llvm.call @mtrt_load_tensorrt_engine_from_file(%[[v1]], %[[v2]], %[[v3]])


// -----

func.func @enqueue(
    %arg0: !trtrt.context,
    %arg1: memref<?x?xf32>,
    %arg2: memref<?x?xf32>,
    %arg3: !cuda.stream) {
  trtrt.enqueue %arg0 stream(%arg3) (%arg1) outs(%arg2) : (memref<?x?xf32>) -> memref<?x?xf32>
  return
}

// CHECK-LABEL: func.func @enqueue
//  CHECK-SAME: (%[[arg0:.+]]: !trtrt.context, %[[arg1:.+]]: memref<?x?xf32>, %[[arg2:.+]]: memref<?x?xf32>, %[[arg3:.+]]: !cuda.stream)
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : memref<?x?xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<?x?xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg3]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : !trtrt.context to !llvm.ptr
//   CHECK-DAG:     %[[v4:.+]] = llvm.mlir.constant(1 : index) : i32
//   CHECK-DAG:     %[[v5:.+]] = llvm.mlir.constant(1 : index) : i32
//   CHECK-DAG:     %[[v6:.+]] = llvm.mlir.constant(2 : i64) : i64
//   CHECK-DAG:     %[[v7:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:     %[[v8:.+]] = llvm.alloca %[[v7]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v1]], %[[v8]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
//   CHECK-DAG:     %[[v9:.+]] = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v10:.+]] = llvm.insertvalue %[[v6]], %[[v9]][0] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v11:.+]] = llvm.insertvalue %[[v8]], %[[v10]][1] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v12:.+]] = llvm.mlir.constant(2 : i64) : i64
//   CHECK-DAG:     %[[v13:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:     %[[v14:.+]] = llvm.alloca %[[v13]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v0]], %[[v14]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
//   CHECK-DAG:     %[[v15:.+]] = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v16:.+]] = llvm.insertvalue %[[v12]], %[[v15]][0] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v17:.+]] = llvm.insertvalue %[[v14]], %[[v16]][1] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v18:.+]] = llvm.mlir.constant(1 : i32) : i32
//   CHECK-DAG:     %[[v19:.+]] = llvm.alloca %[[v18]] x !llvm.struct<(struct<(i64, ptr)>)> : (i32) -> !llvm.ptr
//   CHECK-DAG:     %[[v20:.+]] = llvm.mlir.constant(1 : i32) : i32
//   CHECK-DAG:     %[[v21:.+]] = llvm.alloca %[[v20]] x !llvm.ptr : (i32) -> !llvm.ptr
//   CHECK-DAG:     %[[v22:.+]] = llvm.getelementptr %[[v19]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i64, ptr)>)>
//   CHECK-DAG:     llvm.store %[[v11]], %[[v22]] : !llvm.struct<(i64, ptr)>, !llvm.ptr
//   CHECK-DAG:     %[[v23:.+]] = llvm.getelementptr %[[v21]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v22]], %[[v23]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v24:.+]] = llvm.mlir.constant(1 : i32) : i32
//   CHECK-DAG:     %[[v25:.+]] = llvm.alloca %[[v24]] x !llvm.struct<(struct<(i64, ptr)>)> : (i32) -> !llvm.ptr
//   CHECK-DAG:     %[[v26:.+]] = llvm.mlir.constant(1 : i32) : i32
//   CHECK-DAG:     %[[v27:.+]] = llvm.alloca %[[v26]] x !llvm.ptr : (i32) -> !llvm.ptr
//   CHECK-DAG:     %[[v28:.+]] = llvm.getelementptr %[[v25]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i64, ptr)>)>
//   CHECK-DAG:     llvm.store %[[v17]], %[[v28]] : !llvm.struct<(i64, ptr)>, !llvm.ptr
//   CHECK-DAG:     %[[v29:.+]] = llvm.getelementptr %[[v27]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v28]], %[[v29]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     llvm.call @mtrt_tensorrt_enqueue(%[[v3]], %[[v2]], %[[v4]], %[[v21]], %[[v5]], %[[v27]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr) -> ()

//   FILE-LABEL: func.func @enqueue

// -----

func.func @convert_enqueue_alloc(
  %context: !trtrt.context, %stream: !cuda.stream,
  %arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) -> (memref<?x?xf32>) {
  %0 = trtrt.enqueue_alloc %context stream(%stream) (%arg0)
    : (memref<?x?xf32>) -> memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// CHECK-LABEL: func.func @convert_enqueue_alloc
//  CHECK-SAME: (%[[arg0:.+]]: !trtrt.context, %[[arg1:.+]]: !cuda.stream, %[[arg2:.+]]: memref<?x?xf32>, %[[arg3:.+]]: memref<?x?xf32>) -> memref<?x?xf32> {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : memref<?x?xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : !cuda.stream to !llvm.ptr
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : !trtrt.context to !llvm.ptr
//   CHECK-DAG:     %[[v3:.+]] = llvm.mlir.constant(1 : index) : i32
//   CHECK-DAG:     %[[v4:.+]] = llvm.mlir.constant(1 : index) : i32
//   CHECK-DAG:     %[[v5:.+]] = llvm.mlir.constant(2 : i64) : i64
//   CHECK-DAG:     %[[v6:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:     %[[v7:.+]] = llvm.alloca %[[v6]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v0]], %[[v7]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
//   CHECK-DAG:     %[[v8:.+]] = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v9:.+]] = llvm.insertvalue %[[v5]], %[[v8]][0] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v10:.+]] = llvm.insertvalue %[[v7]], %[[v9]][1] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v11:.+]] = llvm.mlir.constant(1 : i32) : i32
//   CHECK-DAG:     %[[v12:.+]] = llvm.alloca %[[v11]] x !llvm.struct<(struct<(i64, ptr)>)> : (i32) -> !llvm.ptr
//   CHECK-DAG:     %[[v13:.+]] = llvm.mlir.constant(1 : i32) : i32
//   CHECK-DAG:     %[[v14:.+]] = llvm.alloca %[[v13]] x !llvm.ptr : (i32) -> !llvm.ptr
//   CHECK-DAG:     %[[v15:.+]] = llvm.getelementptr %[[v12]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i64, ptr)>)>
//   CHECK-DAG:     llvm.store %[[v10]], %[[v15]] : !llvm.struct<(i64, ptr)>, !llvm.ptr
//   CHECK-DAG:     %[[v16:.+]] = llvm.getelementptr %[[v14]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v15]], %[[v16]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     %[[v17:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v18:.+]] = llvm.mlir.constant(2 : i64) : i64
//   CHECK-DAG:     %[[v19:.+]] = llvm.mlir.constant(1 : index) : i64
//   CHECK-DAG:     %[[v20:.+]] = llvm.alloca %[[v19]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v17]], %[[v20]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
//   CHECK-DAG:     %[[v21:.+]] = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v22:.+]] = llvm.insertvalue %[[v18]], %[[v21]][0] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v23:.+]] = llvm.insertvalue %[[v20]], %[[v22]][1] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v24:.+]] = llvm.mlir.constant(1 : i32) : i32
//   CHECK-DAG:     %[[v25:.+]] = llvm.alloca %[[v24]] x !llvm.struct<(struct<(i64, ptr)>)> : (i32) -> !llvm.ptr
//   CHECK-DAG:     %[[v26:.+]] = llvm.mlir.constant(1 : i32) : i32
//   CHECK-DAG:     %[[v27:.+]] = llvm.alloca %[[v26]] x !llvm.ptr : (i32) -> !llvm.ptr
//   CHECK-DAG:     %[[v28:.+]] = llvm.getelementptr %[[v25]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i64, ptr)>)>
//   CHECK-DAG:     llvm.store %[[v23]], %[[v28]] : !llvm.struct<(i64, ptr)>, !llvm.ptr
//   CHECK-DAG:     %[[v29:.+]] = llvm.getelementptr %[[v27]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     llvm.store %[[v28]], %[[v29]] : !llvm.ptr, !llvm.ptr
//   CHECK-DAG:     llvm.call @mtrt_tensorrt_enqueue_alloc(%[[v2]], %[[v1]], %[[v3]], %[[v14]], %[[v4]], %[[v27]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr) -> ()
//   CHECK-DAG:     %[[v30:.+]] = llvm.extractvalue %[[v23]][1] : !llvm.struct<(i64, ptr)>
//   CHECK-DAG:     %[[v31:.+]] = llvm.load %[[v30]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//   CHECK-DAG:     %[[v32:.+]] = builtin.unrealized_conversion_cast %[[v31]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<?x?xf32>
//   CHECK-DAG:     return %[[v32]] : memref<?x?xf32>
