// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN: %p/../matrix-multiply.mlir | FileCheck %p/../matrix-multiply.mlir
