// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN: %p/../activation.mlir | FileCheck %p/../activation.mlir
