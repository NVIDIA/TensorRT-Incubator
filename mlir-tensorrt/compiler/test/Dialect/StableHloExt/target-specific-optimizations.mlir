// RUN: mlir-tensorrt-opt %s --stablehlo-ext-target-specific-optimizations | FileCheck %s --check-prefix=DEFAULT
// RUN: mlir-tensorrt-opt %s --stablehlo-ext-target-specific-optimizations="disable-patterns=dot-general" | FileCheck %s --check-prefix=DISABLE-DG

// Optimizations for this pass are tested in other pass tests (e.g. 'canonicalize-dot-general.mlir').
// This test is just for the options of the pass for disabling/enabling optimizations.

// DEFAULT-LABEL: func.func @flatten_dot_general
// DISABLE-DG-LABEL: func.func @flatten_dot_general
func.func @flatten_dot_general(%arg0: tensor<1x1x32x32x256xf32>,
                               %arg1: tensor<1x8x256xf32>) -> tensor<1x1x32x32x8xf32> {
  // DEFAULT: stablehlo.reshape
  // DEFAULT: stablehlo.dot_general
  // DEFAULT-SAME: batching_dims = [0] x [0]
  // DEFAULT-SAME: contracting_dims = [2] x [2]

  // DISABLE-DG: stablehlo.dot_general
  // DISABLE-DG-SAME: batching_dims = [0] x [0]
  // DISABLE-DG-SAME: contracting_dims = [4] x [2]
  %0 = stablehlo.dot_general %arg0, %arg1,
    batching_dims = [0] x [0],
    contracting_dims = [4] x [2],
    precision = [DEFAULT, DEFAULT]
      : (tensor<1x1x32x32x256xf32>, tensor<1x8x256xf32>) -> tensor<1x1x32x32x8xf32>
  return %0 : tensor<1x1x32x32x8xf32>
}

