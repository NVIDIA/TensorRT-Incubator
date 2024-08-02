// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../scatter.mlir | FileCheck %p/../scatter.mlir
