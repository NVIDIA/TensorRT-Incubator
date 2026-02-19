// RUN: mlir-tensorrt-opt -split-input-file -convert-tensorrt-runtime-to-executor -cse -canonicalize %s | FileCheck %s

func.func @test_enqueue(
    %stream: !cuda.stream,
    %arg0: memref<1x3x256x256xf32>, %arg1: memref<1x3x256x256xf32>,
    %arg2: memref<1x3x256x256xf32>) {
  %1 = trtrt.get_function @trt_func_engine : !trtrt.context
  trtrt.enqueue %1 stream(%stream) (%arg0) outs(%arg2) : (memref<1x3x256x256xf32>) -> memref<1x3x256x256xf32>
  return
}
trtrt.compiled_func @trt_func_engine dense<0> : vector<8xi8>

//   CHECK-DAG:   executor.func private @_trtrt_enqueue(!executor.ptr<host>, !executor.ptr<host>, ...)
//   CHECK-DAG:   executor.func private @_trtrt_create_context(!executor.ptr<host>) -> !executor.ptr<host>
//   CHECK-DAG:   executor.func private @_trtrt_load(!executor.ptr<host>, !executor.ptr<host>, i64) -> !executor.ptr<host>
//   CHECK-DAG:   executor.func private @_trtrt_create_runtime() -> !executor.ptr<host>
// CHECK-LABEL:   executor.global @tensorrt_runtime : !executor.ptr<host> {
//   CHECK-DAG:     %[[v0:.+]] = executor.call @_trtrt_create_runtime() : () -> !executor.ptr<host>
//   CHECK-DAG:     executor.return %[[v0]] : !executor.ptr<host>
//       CHECK:   executor.data_segment @trt_func_engine_0
// CHECK-LABEL:   executor.global @trt_func_engine_exec_ctx constant : !executor.ptr<host> {
//   CHECK-DAG:     %[[v0:.+]] = executor.load_data_segment @trt_func_engine_0 : !executor.ptr<host>
//   CHECK-DAG:     %[[size:.+]] = executor.getoffset[8] : () -> i64, i8
//   CHECK-DAG:     %[[v1:.+]] = executor.get_global @tensorrt_runtime : !executor.ptr<host>
//   CHECK-DAG:     %[[v2:.+]] = executor.call @_trtrt_load(%[[v1]], %[[v0]], %[[size]]) : (!executor.ptr<host>, !executor.ptr<host>, i64) -> !executor.ptr<host>
//   CHECK-DAG:     %[[v3:.+]] = executor.call @_trtrt_create_context(%[[v2]]) : (!executor.ptr<host>) -> !executor.ptr<host>
//   CHECK-DAG:     executor.return %[[v3]] : !executor.ptr<host>
// CHECK-LABEL: @test_enqueue
//  CHECK-SAME: (%[[arg0:.+]]: !cuda.stream, %[[arg1:.+]]: memref<1x3x256x256xf32>, %[[arg2:.+]]: memref<1x3x256x256xf32>, %[[arg3:.+]]: memref<1x3x256x256xf32>) {
//   CHECK-DAG:     %[[c0_i64:.+]] = executor.constant 0 : i64
//   CHECK-DAG:     %[[c256_i64:.+]] = executor.constant 256 : i64
//   CHECK-DAG:     %[[c3_i64:.+]] = executor.constant 3 : i64
//   CHECK-DAG:     %[[c1_i64:.+]] = executor.constant 1 : i64
//   CHECK-DAG:     %[[c4_i64:.+]] = executor.constant 4 : i64
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg3]] : memref<1x3x256x256xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<1x3x256x256xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//   CHECK-DAG:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : !cuda.stream to !executor.ptr<host>
//   CHECK-DAG:     %[[v3:.+]] = executor.get_global @trt_func_engine_exec_ctx : !executor.ptr<host>
//   CHECK-DAG:     %[[v4:.+]] = executor.table.get %[[v1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//   CHECK-DAG:     %[[v5:.+]] = executor.table.get %[[v0]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//   CHECK-DAG:     %[[v6:.+]] = executor.table.create(%[[v4]], %[[c0_i64]], %[[c4_i64]], %[[c1_i64]], %[[c3_i64]], %[[c256_i64]], %[[c256_i64]], %[[v5]], %[[c0_i64]], %[[c4_i64]], %[[c1_i64]], %[[c3_i64]], %[[c256_i64]], %[[c256_i64]] : !executor.ptr<host>, i64, i64, i64, i64, i64, i64, !executor.ptr<host>, i64, i64, i64, i64, i64, i64) : <!executor.ptr<host>, i64, i64, i64, i64, i64, i64, !executor.ptr<host>, i64, i64, i64, i64, i64, i64>
//   CHECK-DAG:     executor.call @_trtrt_enqueue(%[[v3]], %[[v2]], %[[v6]]) :


// -----

#device = #executor.memory_type<device>
#host = #executor.memory_type<host>

func.func @convert_enqueue_alloc(%arg0: memref<?xf32, #device>,
                %arg1: memref<?x?xi32, #device>, %context: !trtrt.context, %stream: !cuda.stream)
                -> (memref<?xf32, #device>, memref<?x?xf32, #host>) {
  %2:2 = trtrt.enqueue_alloc %context stream(%stream) (%arg1, %arg0)
    : (memref<?x?xi32, #device>, memref<?xf32, #device>) ->
       (memref<?xf32, #device>, memref<?x?xf32, #host>)
  return %2#0, %2#1 : memref<?xf32, #device>, memref<?x?xf32, #host>
}

// CHECK-LABEL: func.func @convert_enqueue_alloc
//  CHECK-SAME: (%[[arg0:.+]]: memref<?xf32, #executor.memory_type<device>>, %[[arg1:.+]]: memref<?x?xi32, #executor.memory_type<device>>, %[[arg2:.+]]: !trtrt.context, %[[arg3:.+]]: !cuda.stream) -> (memref<?xf32, #executor.memory_type<device>>, memref<?x?xf32, #executor.memory_type<host>>) {
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[c2_i64:.+]] = executor.constant 2 : i64
//       CHECK:     %[[c1_i64:.+]] = executor.constant 1 : i64
//       CHECK:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<?xf32, #executor.memory_type<device>> to !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//       CHECK:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<?x?xi32, #executor.memory_type<device>> to !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>
//       CHECK:     %[[v2:.+]] = builtin.unrealized_conversion_cast %[[arg3]] : !cuda.stream to !executor.ptr<host>
//       CHECK:     %[[v3:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : !trtrt.context to !executor.ptr<host>
//       CHECK:     %[[v4:.+]] = executor.alloca %[[c1_i64]] x !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64> : (i64) -> !executor.ptr<host>
//       CHECK:     %[[v5:.+]] = executor.getoffset[0, 0] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     executor.store %[[c2_i64]] to %[[v4]] + %[[v5]] : i64, !executor.ptr<host>, i64
//       CHECK:     %[[v6:.+]] = executor.getoffset[0, 1] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     executor.store %[[c1_i64]] to %[[v4]] + %[[v6]] : i64, !executor.ptr<host>, i64
//       CHECK:     %[[v7:.+]] = executor.getoffset[0, 5] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     executor.store %[[c2_i64]] to %[[v4]] + %[[v7]] : i64, !executor.ptr<host>, i64
//       CHECK:     %[[v8:.+]] = executor.table.get %[[v1]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>
//       CHECK:     %[[v9:.+]] = executor.table.get %[[v1]][3] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>
//       CHECK:     %[[v10:.+]] = executor.table.get %[[v1]][4] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64>
//       CHECK:     %[[v11:.+]] = executor.table.get %[[v0]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//       CHECK:     %[[v12:.+]] = executor.table.get %[[v0]][3] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//       CHECK:     %[[v13:.+]] = executor.table.create(%[[v8]], %[[c0_i64]], %[[c2_i64]], %[[v9]], %[[v10]], %[[v11]], %[[c0_i64]], %[[c1_i64]], %[[v12]] : !executor.ptr<device>, i64, i64, i64, i64, !executor.ptr<device>, i64, i64, i64) : <!executor.ptr<device>, i64, i64, i64, i64, !executor.ptr<device>, i64, i64, i64>
//       CHECK:     executor.call @_trtrt_enqueue_alloc(%[[v3]], %[[v2]], %[[v4]], %[[v13]]) : (!executor.ptr<host>, !executor.ptr<host>, !executor.ptr<host>, !executor.table<!executor.ptr<device>, i64, i64, i64, i64, !executor.ptr<device>, i64, i64, i64>) -> ()
//       CHECK:     %[[v14:.+]] = executor.getoffset[0, 2] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v15:.+]] = executor.load %[[v4]] + %[[v14]] : (!executor.ptr<host>, i64) -> i64
//       CHECK:     %[[v16:.+]] = executor.inttoptr %[[v15]] : (i64) -> !executor.ptr<device>
//       CHECK:     %[[v18:.+]] = executor.getoffset[0, 3] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v19:.+]] = executor.load %[[v4]] + %[[v18]] : (!executor.ptr<host>, i64) -> i64
//       CHECK:     %[[v20:.+]] = executor.getoffset[0, 4] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v21:.+]] = executor.load %[[v4]] + %[[v20]] : (!executor.ptr<host>, i64) -> i64
//       CHECK:     %[[v22:.+]] = executor.table.create(%[[v16]], %[[v16]], %[[c0_i64]], %[[v19]], %[[v21]] : !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64) : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//       CHECK:     %[[v23:.+]] = builtin.unrealized_conversion_cast %[[v22]] : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64> to memref<?xf32, #executor.memory_type<device>>
//       CHECK:     %[[v24:.+]] = executor.getoffset[0, 6] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v25:.+]] = executor.load %[[v4]] + %[[v24]] : (!executor.ptr<host>, i64) -> i64
//       CHECK:     %[[v26:.+]] = executor.inttoptr %[[v25]] : (i64) -> !executor.ptr<host>
//       CHECK:     %[[v28:.+]] = executor.getoffset[0, 7] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v29:.+]] = executor.load %[[v4]] + %[[v28]] : (!executor.ptr<host>, i64) -> i64
//       CHECK:     %[[v30:.+]] = executor.getoffset[0, 8] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v31:.+]] = executor.load %[[v4]] + %[[v30]] : (!executor.ptr<host>, i64) -> i64
//       CHECK:     %[[v32:.+]] = executor.getoffset[0, 9] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v33:.+]] = executor.load %[[v4]] + %[[v32]] : (!executor.ptr<host>, i64) -> i64
//       CHECK:     %[[v34:.+]] = executor.getoffset[0, 10] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//       CHECK:     %[[v35:.+]] = executor.load %[[v4]] + %[[v34]] : (!executor.ptr<host>, i64) -> i64
//       CHECK:     %[[v36:.+]] = executor.table.create(%[[v26]], %[[v26]], %[[c0_i64]], %[[v29]], %[[v31]], %[[v33]], %[[v35]] : !executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64) : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64>
//       CHECK:     %[[v37:.+]] = builtin.unrealized_conversion_cast %[[v36]] : !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64> to memref<?x?xf32, #executor.memory_type<host>>
//       CHECK:     return %[[v23]], %[[v37]] : memref<?xf32, #executor.memory_type<device>>, memref<?x?xf32, #executor.memory_type<host>>

// -----

#device = #executor.memory_type<device>

func.func @convert_get_function(%stream: !cuda.stream,
                %arg0: memref<4xf32, #device>, %arg1: memref<4xf32, #device>) -> !trtrt.context {
  %0 = trtrt.get_function @get_func_engine : !trtrt.context
  return %0 : !trtrt.context
}
trtrt.compiled_func @get_func_engine dense<0> : vector<4xi8>

// CHECK-LABEL:   executor.global @tensorrt_runtime : !executor.ptr<host> {
// CHECK:           executor.return %{{.+}} : !executor.ptr<host>
// CHECK-LABEL:   executor.global @get_func_engine_exec_ctx constant : !executor.ptr<host> {
// CHECK-DAG:       %[[runtime:.+]] = executor.get_global @tensorrt_runtime : !executor.ptr<host>
// CHECK-DAG:       %[[data:.+]] = executor.load_data_segment @get_func_engine_0 : !executor.ptr<host>
// CHECK-DAG:       %[[size:.+]] = executor.getoffset[4] : () -> i64, i8
// CHECK-DAG:       %[[engine:.+]] = executor.call @_trtrt_load(%[[runtime]], %[[data]], %[[size]]) : (!executor.ptr<host>, !executor.ptr<host>, i64) -> !executor.ptr<host>
// CHECK-DAG:       %[[ctx:.+]] = executor.call @_trtrt_create_context(%[[engine]]) : (!executor.ptr<host>) -> !executor.ptr<host>
// CHECK:           executor.return %[[ctx]] : !executor.ptr<host>
// CHECK-LABEL:   func.func @convert_get_function
// CHECK:           %[[ctx:.+]] = executor.get_global @get_func_engine_exec_ctx : !executor.ptr<host>
// CHECK:           %[[result:.+]] = builtin.unrealized_conversion_cast %[[ctx]] : !executor.ptr<host> to !trtrt.context
// CHECK:           return %[[result]] : !trtrt.context
