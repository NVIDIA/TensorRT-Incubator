// RUN: %pick-one-gpu tensorrt-opt %flags -pass-pipeline="builtin.module(translate-tensorrt-to-engine{})" \
// RUN: %p/../slice.mlir | FileCheck %p/../slice.mlir
