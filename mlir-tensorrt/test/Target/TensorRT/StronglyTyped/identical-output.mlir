// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline \
// RUN:  %p/../identical-output.mlir | FileCheck %p/../identical-output.mlir
