// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../shuffle.mlir | FileCheck %p/../shuffle.mlir
