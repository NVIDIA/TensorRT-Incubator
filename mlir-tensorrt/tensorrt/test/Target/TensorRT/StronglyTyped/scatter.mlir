// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../scatter.mlir | FileCheck %p/../scatter.mlir
