// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../shape.mlir | FileCheck %p/../shape.mlir
