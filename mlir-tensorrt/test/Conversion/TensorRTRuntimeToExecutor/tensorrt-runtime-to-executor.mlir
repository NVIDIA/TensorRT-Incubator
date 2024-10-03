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
