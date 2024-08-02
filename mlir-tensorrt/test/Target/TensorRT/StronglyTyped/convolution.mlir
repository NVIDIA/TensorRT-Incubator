// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../convolution.mlir | FileCheck %p/../convolution.mlir
