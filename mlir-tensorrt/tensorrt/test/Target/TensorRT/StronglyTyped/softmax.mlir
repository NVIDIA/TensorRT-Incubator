// RUN: %pick-one-gpu tensorrt-opt %flags -pass-pipeline="builtin.module(tensorrt-expand-ops,translate-tensorrt-to-engine)" \
// RUN: %p/../softmax.mlir
