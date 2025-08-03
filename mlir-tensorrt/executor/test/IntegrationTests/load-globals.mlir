// REQUIRES: host-has-at-least-1-gpus
// RUN: executor-opt %s \
// RUN:    -executor-lower-to-runtime-builtins | \
// RUN: executor-translate -mlir-to-runtime-executable |\
// RUN: executor-runner -input-type=rtexe -features=core,cuda | FileCheck %s

executor.data_segment @dense_i32 constant dense<[32, 33]> : tensor<2xi32>
executor.data_segment @device_i32 constant address_space<device> dense<[99, 101]> : tensor<2xi32>
executor.data_segment @host_i32 uninitialized dense<0> : tensor<2xi32>

executor.func private @__cuda_memcpy_device2host(!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<host>, i64, i64) -> ()
executor.func private @__cuda_stream_create() -> (!executor.ptr<host>)
executor.func private @__cuda_stream_sync(!executor.ptr<host>) -> ()
executor.func private @__cuda_alloc_device(!executor.ptr<host>, i32, i64, i32) -> (!executor.ptr<device>)
executor.func private @__cuda_memset_32(!executor.ptr<device>, i64, i64, i32) -> ()
executor.func private @executor_alloc(i64, i32) -> (!executor.ptr<host>)




func.func @main() -> (i32) {
  %c0 = executor.constant 0 : i32
  %c10 = executor.constant 10 : i32
  %true = executor.constant 1 : i1
  %c4 = executor.constant 4 : i32
  %c8 = executor.constant 8 : i64
  %c0_i64 = executor.constant 0 : i64

  %data_segment_f32 = executor.load_data_segment @dense_i32 : !executor.ptr<host>
  %0 = executor.load %data_segment_f32 + %c0 : (!executor.ptr<host>, i32) -> i32
  %1 = executor.load %data_segment_f32 + %c4 : (!executor.ptr<host>, i32) -> i32
  executor.print "data_segment_f32[0] = %d"(%0 : i32)
  executor.print "data_segment_f32[1] = %d"(%1 : i32)

  %dev = executor.load_data_segment @device_i32 : !executor.ptr<device>
  %host = executor.load_data_segment @host_i32 : !executor.ptr<host>
  %stream = executor.call @__cuda_stream_create() : () -> (!executor.ptr<host>)
  executor.call @__cuda_memcpy_device2host(%stream, %dev, %c0_i64, %host, %c0_i64, %c8)
    : (!executor.ptr<host>, !executor.ptr<device>, i64, !executor.ptr<host>, i64, i64) -> ()
  executor.call @__cuda_stream_sync(%stream) : (!executor.ptr<host>) -> ()
  %h0 = executor.load %host + %c0 : (!executor.ptr<host>, i32) -> i32
  %h1 = executor.load %host + %c4 : (!executor.ptr<host>, i32) -> i32
  executor.print "host[0] = %d"(%h0 : i32)
  executor.print "host[1] = %d"(%h1 : i32)
  return %c0 : i32
}

//       CHECK: data_segment_f32[0] = 32
//       CHECK: data_segment_f32[1] = 33
//       CHECK: host[0] = 99
//       CHECK: host[1] = 101
