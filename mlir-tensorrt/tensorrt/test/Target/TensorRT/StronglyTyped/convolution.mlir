// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../convolution.mlir | FileCheck %p/../convolution.mlir
