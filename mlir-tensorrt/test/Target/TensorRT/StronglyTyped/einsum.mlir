// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../einsum.mlir | FileCheck %p/../einsum.mlir
