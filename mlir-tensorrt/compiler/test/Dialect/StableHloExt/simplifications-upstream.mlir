// RUN: mlir-tensorrt-opt %stablehlo_src_dir/stablehlo/tests/transforms/stablehlo_aggressive_simplification.mlir -split-input-file \
// RUN:   --stablehlo-ext-simplifications="fold-op-element-limit=100"  -allow-unregistered-dialect | \
// RUN: FileCheck %stablehlo_src_dir/stablehlo/tests/transforms/stablehlo_aggressive_simplification.mlir
