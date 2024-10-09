// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN: %p/../translate-to-tensorrt.mlir | FileCheck %p/../translate-to-tensorrt.mlir
