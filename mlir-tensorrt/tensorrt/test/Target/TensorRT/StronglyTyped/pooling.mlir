// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../pooling.mlir | FileCheck %p/../pooling.mlir
