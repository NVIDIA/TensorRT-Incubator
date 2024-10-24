// RUN: mlir-tensorrt-opt -split-input-file -convert-tensorrt-runtime-to-executor -cse -canonicalize %s | FileCheck %s

tensorrt.module @trt_funcs {
  func.func @my_func() attributes {tensorrt.engine = dense<0> : vector<10xi8>} {
    return
  }
}

func.func @main(%arg0: memref<1x3x256x256xf32>, %arg1: memref<1x3x256x256xf32>) -> memref<1x3x256x256xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x256x256xf32>
  %0 = trtrt.compile @trt_funcs::@my_func : !trtrt.context
  %1 = cuda.stream.create : !cuda.stream
  trtrt.enqueue %0 stream(%1) (%arg0) outs(%alloc) : (memref<1x3x256x256xf32>) -> memref<1x3x256x256xf32>
  cuda.stream.sync %1 : !cuda.stream
  return %alloc : memref<1x3x256x256xf32>
}

//   CHECK-DAG:   executor.func private @_trtrt_enqueue(!executor.opaque<"trtrt_context">, !executor.ptr<host>, ...)
//   CHECK-DAG:   executor.func private @_trtrt_create_context(!executor.opaque<"trtrt_engine">) -> !executor.opaque<"trtrt_context">
//   CHECK-DAG:   executor.func private @_trtrt_load(!executor.opaque<"trtrt_runtime">, !executor.ptr<host>, i64) -> !executor.opaque<"trtrt_engine">
//   CHECK-DAG:   executor.func private @_trtrt_create_runtime() -> !executor.opaque<"trtrt_runtime">
// CHECK-LABEL:   executor.global @tensorrt_runtime : !executor.opaque<"trtrt_runtime"> {
//   CHECK-DAG:     %[[v0:.+]] = executor.call @_trtrt_create_runtime() : () -> !executor.opaque<"trtrt_runtime">
//   CHECK-DAG:     executor.return %[[v0]] : !executor.opaque<"trtrt_runtime">
//       CHECK:   executor.constant_resource @my_func_engine_data dense<0> : vector<10xi8>
// CHECK-LABEL:   executor.global @my_func_exec_ctx constant : !executor.opaque<"trtrt_context"> {
//   CHECK-DAG:     %[[v0:.+]] = executor.load_constant_resource @my_func_engine_data : !executor.ptr<host>
//   CHECK-DAG:     %[[size:.+]] = executor.getoffset[10] : () -> i64, i8
//   CHECK-DAG:     %[[v1:.+]] = executor.get_global @tensorrt_runtime : !executor.opaque<"trtrt_runtime">
//   CHECK-DAG:     %[[v2:.+]] = executor.call @_trtrt_load(%[[v1]], %[[v0]], %[[size]]) : (!executor.opaque<"trtrt_runtime">, !executor.ptr<host>, i64) -> !executor.opaque<"trtrt_engine">
//   CHECK-DAG:     %[[v3:.+]] = executor.call @_trtrt_create_context(%[[v2]]) : (!executor.opaque<"trtrt_engine">) -> !executor.opaque<"trtrt_context">
//   CHECK-DAG:     executor.return %[[v3]] : !executor.opaque<"trtrt_context">
// CHECK-LABEL: tensorrt.module @trt_funcs {
// CHECK-LABEL: @my_func
//  CHECK-SAME:    tensorrt.engine = dense<0> : vector<10xi8>
// CHECK-LABEL: @main
//  CHECK-SAME: (%[[arg0:.+]]: memref<1x3x256x256xf32>, %[[arg1:.+]]: memref<1x3x256x256xf32>) -> memref<1x3x256x256xf32> {
//   CHECK-DAG:     %[[c0_i64:.+]] = executor.constant 0 : i64
//   CHECK-DAG:     %[[c256_i64:.+]] = executor.constant 256 : i64
//   CHECK-DAG:     %[[c3_i64:.+]] = executor.constant 3 : i64
//   CHECK-DAG:     %[[c1_i64:.+]] = executor.constant 1 : i64
//   CHECK-DAG:     %[[c4_i64:.+]] = executor.constant 4 : i64
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<1x3x256x256xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//   CHECK-DAG:     %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x3x256x256xf32>
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[alloc]] : memref<1x3x256x256xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//   CHECK-DAG:     %[[v2:.+]] = executor.get_global @my_func_exec_ctx : !executor.opaque<"trtrt_context">
//   CHECK-DAG:     %[[v3:.+]] = cuda.stream.create : !cuda.stream
//   CHECK-DAG:     %[[v4:.+]] = builtin.unrealized_conversion_cast %[[v3]] : !cuda.stream to !executor.ptr<host>
//   CHECK-DAG:     %[[v5:.+]] = executor.table.get %[[v0]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//   CHECK-DAG:     %[[v6:.+]] = executor.table.get %[[v1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64, i64, i64, i64, i64, i64, i64>
//   CHECK-DAG:     %[[v7:.+]] = executor.table.create(%[[v5]], %[[c0_i64]], %[[c4_i64]], %[[c1_i64]], %[[c3_i64]], %[[c256_i64]], %[[c256_i64]], %[[v6]], %[[c0_i64]], %[[c4_i64]], %[[c1_i64]], %[[c3_i64]], %[[c256_i64]], %[[c256_i64]] : !executor.ptr<host>, i64, i64, i64, i64, i64, i64, !executor.ptr<host>, i64, i64, i64, i64, i64, i64) : <!executor.ptr<host>, i64, i64, i64, i64, i64, i64, !executor.ptr<host>, i64, i64, i64, i64, i64, i64>
//   CHECK-DAG:     executor.call @_trtrt_enqueue(%[[v2]], %[[v4]], %[[v7]]) : (!executor.opaque<"trtrt_context">, !executor.ptr<host>, !executor.table<!executor.ptr<host>, i64, i64, i64, i64, i64, i64, !executor.ptr<host>, i64, i64, i64, i64, i64, i64>) -> ()
//   CHECK-DAG:     cuda.stream.sync %[[v3]] : !cuda.stream
//   CHECK-DAG:     return %[[alloc]] : memref<1x3x256x256xf32>

// -----

tensorrt.module @trt_funcs {
  func.func @my_func() attributes {tensorrt.engine = dense<0> : vector<10xi8>} {
    return
  }
}

func.func @main(%arg0: memref<1x3x256x256xf32, #executor.memory_type<device>>) -> memref<?x?x?x?xf32, #executor.memory_type<device>> {
  %0 = trtrt.compile @trt_funcs::@my_func : !trtrt.context
  %1 = cuda.stream.create : !cuda.stream
  %2 = trtrt.enqueue_alloc %0 stream(%1) (%arg0) : (memref<1x3x256x256xf32, #executor.memory_type<device>>) -> memref<?x?x?x?xf32, #executor.memory_type<device>>
  cuda.stream.sync %1 : !cuda.stream
  return %2 : memref<?x?x?x?xf32, #executor.memory_type<device>>
}

// CHECK-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64 : i64>, #dlti.dl_entry<!executor.ptr<host>, 64 : i64>, #dlti.dl_entry<!executor.ptr<device>, 64 : i64>>} {
// CHECK-DAG: executor.func private @_trtrt_enqueue_alloc(!executor.opaque<"trtrt_context">, !executor.ptr<host>, !executor.ptr<host>, ...)
// CHECK-DAG: executor.func private @_trtrt_create_runtime() -> !executor.opaque<"trtrt_runtime">
// CHECK-DAG: executor.func private @_trtrt_create_context(!executor.opaque<"trtrt_engine">) -> !executor.opaque<"trtrt_context">
// CHECK-DAG: executor.func private @_trtrt_load(!executor.opaque<"trtrt_runtime">, !executor.ptr<host>, i64) -> !executor.opaque<"trtrt_engine">
// CHECK-LABEL: executor.global @tensorrt_runtime : !executor.opaque<"trtrt_runtime"> {
// CHECK: %[[v0:.*]] = executor.call @_trtrt_create_runtime() : () -> !executor.opaque<"trtrt_runtime">
// CHECK: executor.return %[[v0]] : !executor.opaque<"trtrt_runtime">
// CHECK: }
// CHECK: executor.constant_resource @my_func_engine_data dense<0> : vector<10xi8>
// CHECK-LABEL: executor.global @my_func_exec_ctx constant : !executor.opaque<"trtrt_context"> {
// CHECK-DAG: %[[v1:.*]] = executor.load_constant_resource @my_func_engine_data : !executor.ptr<host>
// CHECK-DAG: %[[v2:.*]] = executor.getoffset[10] : () -> i64, i8
// CHECK-DAG: %[[v3:.*]] = executor.get_global @tensorrt_runtime : !executor.opaque<"trtrt_runtime">
// CHECK: %[[v4:.*]] = executor.call @_trtrt_load(%[[v3]], %[[v1]], %[[v2]]) : (!executor.opaque<"trtrt_runtime">, !executor.ptr<host>, i64) -> !executor.opaque<"trtrt_engine">
// CHECK: %[[v5:.*]] = executor.call @_trtrt_create_context(%[[v4]]) : (!executor.opaque<"trtrt_engine">) -> !executor.opaque<"trtrt_context">
// CHECK: executor.return %[[v5]] : !executor.opaque<"trtrt_context">
// CHECK: }
// CHECK-LABEL: tensorrt.module @trt_funcs {
// CHECK: func.func @my_func() attributes {tensorrt.engine = dense<0> : vector<10xi8>}
// CHECK: }
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[arg0:.*]]: memref<1x3x256x256xf32, #executor.memory_type<device>>) -> memref<?x?x?x?xf32, #executor.memory_type<device>> {
// CHECK-DAG: %[[c256:.*]] = executor.constant 256 : i64
// CHECK-DAG: %[[c3:.*]] = executor.constant 3 : i64
// CHECK-DAG: %[[c4:.*]] = executor.constant 4 : i64
// CHECK-DAG: %[[c0:.*]] = executor.constant 0 : i64
// CHECK-DAG: %[[c1:.*]] = executor.constant 1 : i64
// CHECK: %[[v6:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<1x3x256x256xf32, #executor.memory_type<device>> to !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v7:.*]] = executor.get_global @my_func_exec_ctx : !executor.opaque<"trtrt_context">
// CHECK: %[[v8:.*]] = cuda.stream.create : !cuda.stream
// CHECK: %[[v9:.*]] = builtin.unrealized_conversion_cast %[[v8]] : !cuda.stream to !executor.ptr<host>
// CHECK: %[[v10:.*]] = executor.alloca %[[c1]] x !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64> : (i64) -> !executor.ptr<host>
// CHECK: %[[v11:.*]] = executor.getoffset[0, 0] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: executor.store %[[c1]] to %[[v10]] + %[[v11]] : i64, !executor.ptr<host>, i64
// CHECK: %[[v12:.*]] = executor.getoffset[0, 1] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: executor.store %[[c4]] to %[[v10]] + %[[v12]] : i64, !executor.ptr<host>, i64
// CHECK: %[[v13:.*]] = executor.table.get %[[v6]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v14:.*]] = executor.table.create(%[[v13]], %[[c0]], %[[c4]], %[[c1]], %[[c3]], %[[c256]], %[[c256]] : !executor.ptr<device>, i64, i64, i64, i64, i64, i64) : <!executor.ptr<device>, i64, i64, i64, i64, i64, i64>
// CHECK: executor.call @_trtrt_enqueue_alloc(%[[v7]], %[[v9]], %[[v10]], %[[v14]]) : (!executor.opaque<"trtrt_context">, !executor.ptr<host>, !executor.ptr<host>, !executor.table<!executor.ptr<device>, i64, i64, i64, i64, i64, i64>) -> ()
// CHECK: %[[v15:.*]] = executor.getoffset[0, 2] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v16:.*]] = executor.load %[[v10]] + %[[v12]] : (!executor.ptr<host>, i64) -> i64
// CHECK: %[[v17:.*]] = executor.load %[[v10]] + %[[v15]] : (!executor.ptr<host>, i64) -> i64
// CHECK: %[[v18:.*]] = executor.inttoptr %[[v17]] : (i64) -> !executor.ptr<device>
// CHECK: %[[v19:.*]] = executor.getoffset[0, 3] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v20:.*]] = executor.load %[[v10]] + %[[v19]] : (!executor.ptr<host>, i64) -> i64
// CHECK: %[[v21:.*]] = executor.getoffset[0, 4] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v22:.*]] = executor.load %[[v10]] + %[[v21]] : (!executor.ptr<host>, i64) -> i64
// CHECK: %[[v23:.*]] = executor.getoffset[0, 5] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v24:.*]] = executor.load %[[v10]] + %[[v23]] : (!executor.ptr<host>, i64) -> i64
// CHECK: %[[v25:.*]] = executor.getoffset[0, 6] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v26:.*]] = executor.load %[[v10]] + %[[v25]] : (!executor.ptr<host>, i64) -> i64
// CHECK: %[[v27:.*]] = executor.getoffset[0, 7] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v28:.*]] = executor.load %[[v10]] + %[[v27]] : (!executor.ptr<host>, i64) -> i64
// CHECK: %[[v29:.*]] = executor.getoffset[0, 8] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v30:.*]] = executor.load %[[v10]] + %[[v29]] : (!executor.ptr<host>, i64) -> i64
// CHECK: %[[v31:.*]] = executor.getoffset[0, 9] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v32:.*]] = executor.load %[[v10]] + %[[v31]] : (!executor.ptr<host>, i64) -> i64
// CHECK: %[[v33:.*]] = executor.getoffset[0, 10] : () -> i64, !executor.table<i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v34:.*]] = executor.load %[[v10]] + %[[v33]] : (!executor.ptr<host>, i64) -> i64
// CHECK: %[[v35:.*]] = executor.table.create(%[[v18]], %[[v18]], %[[c0]], %[[v20]], %[[v22]], %[[v24]], %[[v26]], %[[v28]], %[[v30]], %[[v32]], %[[v34]] : !executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64, i64, i64) : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64, i64, i64>
// CHECK: %[[v36:.*]] = builtin.unrealized_conversion_cast %[[v35]] : !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64, i64, i64, i64, i64, i64, i64> to memref<?x?x?x?xf32, #executor.memory_type<device>>
// CHECK: cuda.stream.sync %[[v8]] : !cuda.stream
// CHECK: return %[[v36]] : memref<?x?x?x?xf32, #executor.memory_type<device>>
// CHECK: }