// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN: %p/../linspace.mlir | FileCheck %p/../linspace.mlir
