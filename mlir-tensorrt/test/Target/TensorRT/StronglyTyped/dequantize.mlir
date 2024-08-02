// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../dequantize.mlir | FileCheck %p/../dequantize.mlir
