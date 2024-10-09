// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../loop.mlir | FileCheck %p/../loop.mlir
