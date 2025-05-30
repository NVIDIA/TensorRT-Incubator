// RUN: executor-opt %s -executor-lowering-pipeline | \
// RUN: executor-translate -mlir-to-lua | \
// RUN: executor-runner -input-type=lua | FileCheck %s

func.func @host_ptr_to_int(%arg0: !executor.ptr<host>) -> i64
    attributes {no_inline} {
  %0 = executor.ptrtoint %arg0 : (!executor.ptr<host>) -> i64
  return %0 : i64
}

func.func @device_ptr_to_int(%arg0: !executor.ptr<device>) -> i64
    attributes {no_inline} {
  %0 = executor.ptrtoint %arg0 : (!executor.ptr<device>) -> i64
  return %0 : i64
}

func.func @main() -> i32 {
  %c0_i32 = executor.constant 0 : i32
  %c0 = executor.constant 0 : i64
  %c1 = executor.constant 1 : i64
  %1 = executor.inttoptr %c0 : (i64) -> !executor.ptr<host>
  %2 = func.call @host_ptr_to_int(%1) : (!executor.ptr<host>) -> i64
  executor.print "host pointer as i64 = %d"(%2 : i64)

  %3 = executor.inttoptr %c1 : (i64) -> !executor.ptr<device>
  %4 = func.call @device_ptr_to_int(%3) : (!executor.ptr<device>) -> i64
  executor.print "device pointer as i64 = %d"(%4 : i64)
  return %c0_i32 : i32
}

// CHECK: host pointer as i64 = 0
// CHECK: device pointer as i64 = 1
