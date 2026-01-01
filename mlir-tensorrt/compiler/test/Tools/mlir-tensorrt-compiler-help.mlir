// RUN: mlir-tensorrt-compiler --help | FileCheck %s

// CHECK: General options:
// CHECK: --mlir-print-{{.*}}

// CHECK: MLIR-TensorRT Backend (KernelGen) Options:
// CHECK: MLIR-TensorRT Backend (TensorRT) Options:
// CHECK: MLIR-TensorRT Bufferization Options:
// CHECK: MLIR-TensorRT Optimization Options:
// CHECK: MLIR-to-TensorRT Translation Options:
