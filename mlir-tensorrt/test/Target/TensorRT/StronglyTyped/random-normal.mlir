// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../random-normal.mlir | FileCheck %p/../random-normal.mlir
