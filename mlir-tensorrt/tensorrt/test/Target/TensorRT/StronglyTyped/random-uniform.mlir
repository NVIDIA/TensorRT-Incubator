// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../random-uniform.mlir | FileCheck %p/../random-uniform.mlir
