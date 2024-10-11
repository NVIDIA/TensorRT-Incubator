// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../einsum.mlir | FileCheck %p/../einsum.mlir
