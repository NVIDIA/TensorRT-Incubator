// RUN: mlir-tensorrt-opt %s -executor-lowering-pipeline \
// RUN:   | mlir-tensorrt-translate -mlir-to-lua \
// RUN:   | not mlir-tensorrt-runner -input-type=lua 2>&1 | FileCheck %s

func.func @main() -> i32 {
  %c0 = executor.constant 0 : i32
  %true = executor.constant true
  %false = executor.constant false
  executor.assert %true, "true assertion message"
  executor.assert %false, "false assertion message"
  return %c0 : i32
}

// CHECK: error: InternalError: {{.*}}: false assertion message
// CHECK: stack traceback:
