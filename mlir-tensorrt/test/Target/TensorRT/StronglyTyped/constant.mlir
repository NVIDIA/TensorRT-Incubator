// RUN: %pick-one-gpu mlir-tensorrt-opt %flags -pass-pipeline="builtin.module(func.func(tensorrt-legalize-int8),translate-tensorrt-to-engine)" \
// RUN:  %p/../constant.mlir | FileCheck %p/../constant.mlir
