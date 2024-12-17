// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-ext-constant-folding --verify-diagnostics

func.func @simplify_reshape_broadcastindim_reshape(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x8x256xf16> {
  %0 = stablehlo.reshape %arg0 : (tensor<1x1x1x256xf16>) -> tensor<1x1x1x1x1x1x1x1x256xf16>
  // expected-error @below {{broadcast_dimensions size (10) does not match operand rank (9)}}
  %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] : (tensor<1x1x1x1x1x1x1x1x256xf16>) -> tensor<1x1x1x1x1x1x8x1x1x256xf16>
  %2 = stablehlo.reshape %1 : (tensor<1x1x1x1x1x1x8x1x1x256xf16>) -> tensor<1x1x8x256xf16>
  return %2 : tensor<1x1x8x256xf16>
}
