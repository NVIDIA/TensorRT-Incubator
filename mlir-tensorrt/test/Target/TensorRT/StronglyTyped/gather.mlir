// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../gather.mlir | FileCheck %p/../gather.mlir
