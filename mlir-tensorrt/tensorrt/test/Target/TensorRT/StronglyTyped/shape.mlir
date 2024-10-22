// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../shape.mlir | FileCheck %p/../shape.mlir
