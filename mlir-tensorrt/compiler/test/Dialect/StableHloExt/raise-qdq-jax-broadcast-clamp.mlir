// RUN: mlir-tensorrt-opt %s -split-input-file -stablehlo-raise-qdq | FileCheck %s

// Test for JAX-generated FP8 quantization pattern where clamp min/max values
// are explicitly broadcasted using broadcast_in_dim. This is the pattern JAX
// generates when using AQT (Accurate Quantization Training) with FP8.
//
// JAX generates broadcast_in_dim for clamp min/max because StableHLO clamp
// requires min/operand/max to have the same type when all three are tensors.
// Unlike the implicit broadcast case (tensor<f32>, tensor<NxMxf32>, tensor<f32>),
// JAX explicitly broadcasts scalars to match the operand shape.

// -----

func.func @jax_fp8_quantize_with_broadcast_clamp(%arg0: tensor<1x968x1x8x256xbf16>) -> tensor<1x968x1x8x256xf8E4M3FN> {
  // Convert bf16 to f32 for quantization
  %0 = stablehlo.convert %arg0 : (tensor<1x968x1x8x256xbf16>) -> tensor<1x968x1x8x256xf32>

  // Scale constant
  %cst_scale = stablehlo.constant dense<7.429000e-03> : tensor<f32>
  %1 = stablehlo.broadcast_in_dim %cst_scale, dims = [] : (tensor<f32>) -> tensor<1x968x1x8x256xf32>
  %2 = stablehlo.divide %0, %1 : tensor<1x968x1x8x256xf32>

  // Round to nearest even
  %3 = stablehlo.round_nearest_even %2 : tensor<1x968x1x8x256xf32>

  // Clamp min/max with explicit broadcast (JAX pattern)
  %cst_min = stablehlo.constant dense<-4.480000e+02> : tensor<f32>
  %cst_max = stablehlo.constant dense<4.480000e+02> : tensor<f32>
  %4 = stablehlo.broadcast_in_dim %cst_min, dims = [] : (tensor<f32>) -> tensor<1x968x1x8x256xf32>
  %5 = stablehlo.broadcast_in_dim %cst_max, dims = [] : (tensor<f32>) -> tensor<1x968x1x8x256xf32>
  %6 = stablehlo.clamp %4, %3, %5 : tensor<1x968x1x8x256xf32>

  // Convert to FP8
  %7 = stablehlo.convert %6 : (tensor<1x968x1x8x256xf32>) -> tensor<1x968x1x8x256xf8E4M3FN>
  return %7 : tensor<1x968x1x8x256xf8E4M3FN>
}

// CHECK-LABEL: func.func @jax_fp8_quantize_with_broadcast_clamp
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x968x1x8x256xbf16>)
// CHECK: %[[CONVERT:.+]] = stablehlo.convert %[[ARG0]]
// CHECK: %[[COMPOSITE:.+]] = stablehlo.composite "tensorrt.pt_q" %[[CONVERT]]
// CHECK-SAME: {composite_attributes = {axis = -1 : i32, scale = dense<7.429000e-03> : tensor<f32>}, decomposition = @pt_q}
// CHECK: return %[[COMPOSITE]] : tensor<1x968x1x8x256xf8E4M3FN>

// CHECK-LABEL: func.func private @pt_q
// CHECK-SAME: attributes {plan.decomposition}

// -----

// Simpler test case with smaller shapes
func.func @jax_fp8_quantize_simple(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf8E4M3FN> {
  %cst_scale = stablehlo.constant dense<0.5> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %cst_scale, dims = [] : (tensor<f32>) -> tensor<2x3x4xf32>
  %1 = stablehlo.divide %arg0, %0 : tensor<2x3x4xf32>
  %2 = stablehlo.round_nearest_even %1 : tensor<2x3x4xf32>

  // Explicit broadcast for clamp min/max (JAX pattern)
  %cst_min = stablehlo.constant dense<-448.0> : tensor<f32>
  %cst_max = stablehlo.constant dense<448.0> : tensor<f32>
  %3 = stablehlo.broadcast_in_dim %cst_min, dims = [] : (tensor<f32>) -> tensor<2x3x4xf32>
  %4 = stablehlo.broadcast_in_dim %cst_max, dims = [] : (tensor<f32>) -> tensor<2x3x4xf32>
  %5 = stablehlo.clamp %3, %2, %4 : tensor<2x3x4xf32>

  %6 = stablehlo.convert %5 : (tensor<2x3x4xf32>) -> tensor<2x3x4xf8E4M3FN>
  return %6 : tensor<2x3x4xf8E4M3FN>
}

// CHECK-LABEL: func.func @jax_fp8_quantize_simple
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x3x4xf32>)
// CHECK: %[[COMPOSITE:.+]] = stablehlo.composite "tensorrt.pt_q" %[[ARG0]]
// CHECK-SAME: {composite_attributes = {axis = -1 : i32, scale = dense<5.000000e-01> : tensor<f32>}, decomposition = @pt_q}
// CHECK: return %[[COMPOSITE]] : tensor<2x3x4xf8E4M3FN>

// CHECK-LABEL: func.func private @pt_q
// CHECK-SAME: attributes {plan.decomposition}
