// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline \
// RUN:  %p/../identical-output.mlir | FileCheck %p/../identical-output.mlir
