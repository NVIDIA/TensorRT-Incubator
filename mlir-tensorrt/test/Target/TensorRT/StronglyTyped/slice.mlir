// RUN: %pick-one-gpu mlir-tensorrt-opt %flags -pass-pipeline="builtin.module(translate-tensorrt-to-engine{})" \
// RUN: %p/../slice.mlir | FileCheck %p/../slice.mlir
