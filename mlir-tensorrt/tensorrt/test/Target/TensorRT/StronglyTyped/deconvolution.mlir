// RUN: %pick-one-gpu tensorrt-opt %flags %pipeline  %p/../deconvolution.mlir | FileCheck %p/../deconvolution.mlir
