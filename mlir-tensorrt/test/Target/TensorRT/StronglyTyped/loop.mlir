// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../loop.mlir | FileCheck %p/../loop.mlir
