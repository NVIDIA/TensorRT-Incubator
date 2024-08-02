// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../normalize.mlir | FileCheck %p/../normalize.mlir
