// RUN: not executor-runner %s -input-type=rtexe -modules=core || FileCheck %s

// CHECK: error: failed to load executable from buffer: InvalidArgument: failed to verify that the provided buffer contains a valid MLIR-TRT Executable
