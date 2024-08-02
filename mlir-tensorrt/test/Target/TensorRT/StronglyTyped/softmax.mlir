// RUN: %pick-one-gpu mlir-tensorrt-opt %flags -pass-pipeline="builtin.module(tensorrt-expand-ops,translate-tensorrt-to-engine)" \
// RUN: %p/../softmax.mlir
