// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../identity.mlir | FileCheck %p/../identity.mlir
