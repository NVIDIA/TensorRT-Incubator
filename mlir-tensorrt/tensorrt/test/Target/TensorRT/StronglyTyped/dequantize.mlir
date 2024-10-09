// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../dequantize.mlir | FileCheck %p/../dequantize.mlir
