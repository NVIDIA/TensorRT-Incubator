// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../shuffle.mlir | FileCheck %p/../shuffle.mlir
