// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN: %p/../topk.mlir | FileCheck %p/../topk.mlir
