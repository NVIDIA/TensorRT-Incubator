// RUN: mlir-tensorrt-opt  %s --convert-stablehlo-to-tensorrt -split-input-file| FileCheck %s

func.func @test_raise_to_softmax(%arg0: tensor<16x20x80x40xf32>, %arg1: tensor<16x20x80x40xf32>) -> (tensor<16x80x20x20xf32>) {
  %0 = stablehlo.constant dense<0.111> : tensor<16x80x20x20xf32>
  %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %2 = stablehlo.reduce(%0 init: %1) across dimensions = [3] : (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %12 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
    stablehlo.return %12 : tensor<f32>
  }
  %3 = "stablehlo.broadcast_in_dim" (%2)
      {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<16x80x20xf32>) -> tensor<16x80x20x1xf32>
  %4 = "stablehlo.broadcast_in_dim" (%3)
     {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (tensor<16x80x20x1xf32>) -> tensor<16x80x20x20xf32>
  %5 = stablehlo.subtract %0, %4 : tensor<16x80x20x20xf32>
  %6 = stablehlo.exponential %5 : tensor<16x80x20x20xf32>
  %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %8 = stablehlo.reduce(%6 init: %7) across dimensions = [3] : (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %12 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %12 : tensor<f32>
  }
  %9 = "stablehlo.broadcast_in_dim" (%8)
    {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<16x80x20xf32>) -> tensor<16x80x20x1xf32>
  %10 = "stablehlo.broadcast_in_dim" (%9)
   {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (tensor<16x80x20x1xf32>) -> tensor<16x80x20x20xf32>
  %11 = stablehlo.divide %6, %10 : tensor<16x80x20x20xf32>
  return %11 : tensor<16x80x20x20xf32>
}
// CHECK-LABEL: @test_raise_to_softmax
//       CHECK: tensorrt.softmax
//  CHECK-SAME: axis = 3


// -----
func.func  @test_neg_incorrectBroadcast() -> (tensor<16x80x20x20xf32>) {
  %6 = stablehlo.constant dense<3.0> : tensor<1x1x1x1xf32>
  %7 = stablehlo.constant dense<4.0> : tensor<1x1x1x1xf32>
  %10 = "stablehlo.broadcast_in_dim" (%6) {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (tensor<1x1x1x1xf32>) -> tensor<16x80x20x20xf32>
  %11 = "stablehlo.broadcast_in_dim" (%7) {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (tensor<1x1x1x1xf32>) -> tensor<16x80x20x20xf32>
  %12 = stablehlo.divide %10, %11 : tensor<16x80x20x20xf32>
  return %12 : tensor<16x80x20x20xf32>
}
// CHECK-LABEL:  @test_neg_incorrectBroadcast
//   CHECK: tensorrt.broadcast
//   CHECK-NOT: tensorrt.softmax

// -----
func.func  @test_incorrect_softmax(%arg0: tensor<16x20x80x40xf32>, %arg1: tensor<16x20x80x40xf32>) -> (tensor<16x80x20x20xf32>) {
  %0 = stablehlo.constant dense<0.111> : tensor<16x80x20x20xf32>
  %1 = stablehlo.constant dense<0xFF811000> : tensor<f32>
  %2 = stablehlo.reduce(%0 init: %1) across dimensions = [3] : (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %12 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %12 : tensor<f32>
  }
  %3 = "stablehlo.broadcast_in_dim" (%2)
      {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<16x80x20xf32>) -> tensor<16x80x20x1xf32>
  %4 = "stablehlo.broadcast_in_dim" (%3)
     {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (tensor<16x80x20x1xf32>) -> tensor<16x80x20x20xf32>
  %5 = stablehlo.subtract %0, %4 : tensor<16x80x20x20xf32>
  %6 = stablehlo.exponential %5 : tensor<16x80x20x20xf32>
  %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %8 = stablehlo.reduce(%6 init: %7) across dimensions = [3] : (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %12 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %12 : tensor<f32>
  }
  %9 = "stablehlo.broadcast_in_dim" (%8)
    {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<16x80x20xf32>) -> tensor<16x80x20x1xf32>
  %10 = "stablehlo.broadcast_in_dim" (%9)
   {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (tensor<16x80x20x1xf32>) -> tensor<16x80x20x20xf32>
  %11 = stablehlo.divide %6, %10 : tensor<16x80x20x20xf32>
  return %11 : tensor<16x80x20x20xf32>
}
// CHECK-LABEL: @test_incorrect_softmax
//      CHECK: tensorrt.element_wise
//      CHECK: tensorrt.element_wise
//      CHECK-SAME: kDIV
//      CHECK-NOT:  tensorrt.softmax

// -----
// JAX numerically-safe softmax: has an extra maximum(-inf, reduce_max_result) clamp
// with broadcast_in_dim for -inf and broadcast_in_dim for rank expansion
func.func @test_jax_safe_softmax(%arg0: tensor<1x16x256x256xf32>) -> (tensor<1x16x256x256xf32>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x256x256xf32>, tensor<f32>) -> tensor<1x16x256xf32>
  %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x16x256xf32>
  %2 = stablehlo.maximum %1, %0 : tensor<1x16x256xf32>
  %3 = "stablehlo.broadcast_in_dim"(%2)
      {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x16x256xf32>) -> tensor<1x16x256x1xf32>
  %4 = "stablehlo.broadcast_in_dim"(%3)
      {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (tensor<1x16x256x1xf32>) -> tensor<1x16x256x256xf32>
  %5 = stablehlo.subtract %arg0, %4 : tensor<1x16x256x256xf32>
  %6 = stablehlo.exponential %5 : tensor<1x16x256x256xf32>
  %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x16x256x256xf32>, tensor<f32>) -> tensor<1x16x256xf32>
  %8 = "stablehlo.broadcast_in_dim"(%7)
      {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x16x256xf32>) -> tensor<1x16x256x1xf32>
  %9 = "stablehlo.broadcast_in_dim"(%8)
      {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (tensor<1x16x256x1xf32>) -> tensor<1x16x256x256xf32>
  %10 = stablehlo.divide %6, %9 : tensor<1x16x256x256xf32>
  return %10 : tensor<1x16x256x256xf32>
}
// CHECK-LABEL: @test_jax_safe_softmax
//       CHECK: tensorrt.softmax
//  CHECK-SAME: axis = 3

// -----
// JAX canonicalized softmax: maximum with constant -inf tensor (not broadcast),
// and reshape (not broadcast_in_dim) for rank expansion by one.
// This is the actual pattern produced by JAX after MLIR canonicalization.
func.func @test_jax_canonicalized_softmax(%arg0: tensor<1x16x256x256xf32>) -> (tensor<1x16x256x256xf32>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<1x16x256xf32>
  %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x256x256xf32>, tensor<f32>) -> tensor<1x16x256xf32>
  %1 = stablehlo.maximum %0, %cst_1 : tensor<1x16x256xf32>
  %2 = stablehlo.reshape %1 : (tensor<1x16x256xf32>) -> tensor<1x16x256x1xf32>
  %3 = "stablehlo.broadcast_in_dim"(%2)
      {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (tensor<1x16x256x1xf32>) -> tensor<1x16x256x256xf32>
  %4 = stablehlo.subtract %arg0, %3 : tensor<1x16x256x256xf32>
  %5 = stablehlo.exponential %4 : tensor<1x16x256x256xf32>
  %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x16x256x256xf32>, tensor<f32>) -> tensor<1x16x256xf32>
  %7 = stablehlo.reshape %6 : (tensor<1x16x256xf32>) -> tensor<1x16x256x1xf32>
  %8 = "stablehlo.broadcast_in_dim"(%7)
      {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (tensor<1x16x256x1xf32>) -> tensor<1x16x256x256xf32>
  %9 = stablehlo.divide %5, %8 : tensor<1x16x256x256xf32>
  return %9 : tensor<1x16x256x256xf32>
}
// CHECK-LABEL: @test_jax_canonicalized_softmax
//       CHECK: tensorrt.softmax
//  CHECK-SAME: axis = 3