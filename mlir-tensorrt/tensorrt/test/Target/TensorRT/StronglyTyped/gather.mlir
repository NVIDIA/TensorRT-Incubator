// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../gather.mlir | FileCheck %p/../gather.mlir
