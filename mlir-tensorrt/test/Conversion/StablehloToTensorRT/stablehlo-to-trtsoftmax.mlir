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