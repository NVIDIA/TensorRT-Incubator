// RUN: mlir-tensorrt-opt --help | FileCheck %s

// CHECK: -stablehlo-clustering-pipeline
// CHECK-NOT:     --debug
// CHECK:         --entrypoint=<string>
// CHECK-SAME: entrypoint function name
// CHECK:         --executor-index-bitwidth=<long>
// CHECK-SAME: executor index bitwidth
// CHECK:         --executor-use-packed-memref-cconv
// CHECK-SAME: whether to use packed or unpacked memref calling convention
