// RUN: %pick-one-gpu mlir-tensorrt-opt %flags %pipeline  %p/../deconvolution.mlir | FileCheck %p/../deconvolution.mlir
