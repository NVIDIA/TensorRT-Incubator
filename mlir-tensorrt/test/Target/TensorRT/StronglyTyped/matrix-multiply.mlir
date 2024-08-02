// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN: %p/../matrix-multiply.mlir | FileCheck %p/../matrix-multiply.mlir
