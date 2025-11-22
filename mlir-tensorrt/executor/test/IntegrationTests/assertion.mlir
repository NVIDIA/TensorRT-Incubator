// RUN: executor-opt %s --executor-generate-abi-wrappers -executor-lowering-pipeline \
// RUN:   | executor-translate -mlir-to-runtime-executable \
// RUN:   | not executor-runner -input-type=rtexe --features=core 2>&1 | FileCheck %s

func.func @main() -> i32 {
  %c0 = executor.constant 0 : i32
  %true = executor.constant true
  %false = executor.constant false
  executor.assert %true, "true assertion message"
  executor.assert %false, "false assertion message"
  return %c0 : i32
}

// CHECK: false assertion message
// CHECK: stack traceback:
