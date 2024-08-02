// Run the pass on upstream tests.
// RUN: mlir-tensorrt-opt -split-input-file -func-ext-duplicate-function-elimination %mlir_src_dir/test/Dialect/Func/duplicate-function-elimination.mlir | \
// RUN:  FileCheck %mlir_src_dir/test/Dialect/Func/duplicate-function-elimination.mlir
