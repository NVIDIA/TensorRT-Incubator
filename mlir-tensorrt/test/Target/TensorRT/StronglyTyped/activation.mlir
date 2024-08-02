// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN: %p/../activation.mlir | FileCheck %p/../activation.mlir
