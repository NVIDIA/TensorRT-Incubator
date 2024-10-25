// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN: %p/../resize.mlir | FileCheck %p/../resize.mlir
