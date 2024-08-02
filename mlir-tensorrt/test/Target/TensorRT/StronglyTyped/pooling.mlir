// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../pooling.mlir | FileCheck %p/../pooling.mlir
