// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN: %p/../quantize.mlir | FileCheck %p/../quantize.mlir
