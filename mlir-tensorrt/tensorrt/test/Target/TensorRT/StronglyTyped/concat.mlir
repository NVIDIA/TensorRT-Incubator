// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../concat.mlir | FileCheck %p/../concat.mlir
