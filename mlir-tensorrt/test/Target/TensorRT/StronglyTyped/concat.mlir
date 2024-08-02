// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../concat.mlir | FileCheck %p/../concat.mlir
