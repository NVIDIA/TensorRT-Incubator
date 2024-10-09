// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../normalize.mlir | FileCheck %p/../normalize.mlir
