// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN: %p/../resize.mlir | FileCheck %p/../resize.mlir
