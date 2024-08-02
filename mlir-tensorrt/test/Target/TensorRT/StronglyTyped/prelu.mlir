// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline %p/../prelu.mlir | FileCheck %p/../prelu.mlir
