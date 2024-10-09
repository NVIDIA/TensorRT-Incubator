// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../identity.mlir | FileCheck %p/../identity.mlir
