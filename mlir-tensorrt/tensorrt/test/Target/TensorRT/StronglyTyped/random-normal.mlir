// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../random-normal.mlir | FileCheck %p/../random-normal.mlir
