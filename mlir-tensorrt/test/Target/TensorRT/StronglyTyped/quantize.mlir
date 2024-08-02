// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN: %p/../quantize.mlir | FileCheck %p/../quantize.mlir
