// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN: %p/../linspace.mlir | FileCheck %p/../linspace.mlir
