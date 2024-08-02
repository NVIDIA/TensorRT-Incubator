// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN: %p/../translate-to-tensorrt.mlir | FileCheck %p/../translate-to-tensorrt.mlir
