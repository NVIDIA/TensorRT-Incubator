// REQUIRES: host-has-at-least-1-gpus
// RUN: mlir-tensorrt-opt %s -convert-cuda-to-executor -executor-lowering-pipeline \
// RUN:   | mlir-tensorrt-translate -mlir-to-lua \
// RUN:   | mlir-tensorrt-runner -input-type=lua | FileCheck %s

func.func @main() -> i32 {
  %c0 = executor.constant 0 : i32
  executor.print "start"()
  %0 = cuda.stream.create : !cuda.stream
  cuda.stream.sync %0 : !cuda.stream
  executor.print "synchronized"()
  return %c0 : i32
}

//      CHECK: start
// CHECK-NEXT: synchronized
