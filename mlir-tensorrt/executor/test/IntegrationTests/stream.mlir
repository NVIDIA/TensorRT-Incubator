// RUN: executor-opt %s -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-lua \
// RUN:   | executor-runner -input-type=lua -features=core,cuda | FileCheck %s

executor.func private @__cuda_stream_create() -> (!executor.ptr<host>)
executor.func private @__cuda_stream_sync(!executor.ptr<host>) -> ()

func.func @main() -> i32 {
  %c0 = executor.constant 0 : i32
  executor.print "start"()
  %0 = executor.call @__cuda_stream_create() : () -> (!executor.ptr<host>)
  executor.call @__cuda_stream_sync(%0) : (!executor.ptr<host>) -> ()
  executor.print "synchronized"()
  return %c0 : i32
}

//      CHECK: start
// CHECK-NEXT: synchronized
