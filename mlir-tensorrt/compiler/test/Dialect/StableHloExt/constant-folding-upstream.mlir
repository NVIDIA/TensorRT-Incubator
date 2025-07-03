// DEFINE: %{test_src} = %stablehlo_src_dir/stablehlo/tests/transforms/stablehlo_aggressive_folder.mlir
// RUN: mlir-tensorrt-opt %{test_src} -split-input-file -stablehlo-ext-constant-folding | FileCheck %{test_src}
