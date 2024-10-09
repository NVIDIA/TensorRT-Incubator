// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN: %p/../topk.mlir | FileCheck %p/../topk.mlir
