// RUN: executor-opt %s -convert-executor-to-executor \
// RUN:   | executor-translate -mlir-to-lua \
// RUN:   | executor-runner -input-type=lua | FileCheck %s

func.func @main() -> i32 {
  %cst0 = executor.constant 123 : i32
  %ptr0 = executor.inttoptr %cst0 : (i32) -> !executor.ptr<host>
  %int0 = executor.ptrtoint %ptr0 : (!executor.ptr<host>) -> i32
  %cst1 = executor.constant 456 : i64
  %ptr1 = executor.inttoptr %cst1 : (i64) -> !executor.ptr<host>
  %int1 = executor.ptrtoint %ptr1 : (!executor.ptr<host>) -> i64
  executor.print "int0=%d, int1=%d"(%int0, %int1 : i32, i64)
  %c0 = executor.constant 0 : i32
  return %c0 : i32
}

// CHECK: int0=123, int1=456
