// REQUIRES: host-has-at-least-1-gpus
// RUN: not executor-runner %s -input-type=rtexe || FileCheck %s

// CHECK: error: failed to load executable from buffer: InvalidArgument: failed to verify that the provided buffer contains a valid MLIR-TRT Executable
