// RUN: mlir-tensorrt-opt %s -test-mtrt-stablehlo-matchers -split-input-file | FileCheck %s

func.func public @test_raise_to_softmax(%arg0: tensor<16x20x80x40xf32>, %arg1: tensor<16x20x80x40xf32>) -> (tensor<16x80x20x20xf32>) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<16x20x80x40xf32>, tensor<16x20x80x40xf32>) -> tensor<16x80x20x20xf32>
  %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %2 = stablehlo.reduce(%0 init: %1) across dimensions = [3] : (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %12 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
    stablehlo.return %12 : tensor<f32>
  }
  %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x80x20xf32>) -> tensor<16x80x20x1xf32>
  %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2, 3] : (tensor<16x80x20x1xf32>) -> tensor<16x80x20x20xf32>
  %5 = stablehlo.subtract %0, %4 : tensor<16x80x20x20xf32>
  %6 = stablehlo.exponential %5 : tensor<16x80x20x20xf32>
  %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %8 = stablehlo.reduce(%6 init: %7) across dimensions = [3] : (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %12 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %12 : tensor<f32>
  }
  %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<16x80x20xf32>) -> tensor<16x80x20x1xf32>
  %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2, 3] : (tensor<16x80x20x1xf32>) -> tensor<16x80x20x20xf32>
  %11 = stablehlo.divide %6, %10 : tensor<16x80x20x20xf32>
  return %11 : tensor<16x80x20x20xf32>
}

// CHECK-LABEL: @test_raise_to_softmax
// CHECK: stablehlo.divide
// CHECK-SAME: __matched__softmax__

// -----
func.func public @test_neg_incorrectReduceMaxDim(%arg0: tensor<16x20x80x40xf32>, %arg1: tensor<16x20x80x40xf32>) -> (tensor<16x80x20x20xf32>) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<16x20x80x40xf32>, tensor<16x20x80x40xf32>) -> tensor<16x80x20x20xf32>
  %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %2 = stablehlo.reduce(%0 init: %1) across dimensions = [2] : (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %12 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
    stablehlo.return %12 : tensor<f32>
  }
  %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x80x20xf32>) -> tensor<16x80x20x1xf32>
  %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2, 3] : (tensor<16x80x20x1xf32>) -> tensor<16x80x20x20xf32>
  %5 = stablehlo.subtract %0, %4 : tensor<16x80x20x20xf32>
  %6 = stablehlo.exponential %5 : tensor<16x80x20x20xf32>
  %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %8 = stablehlo.reduce(%6 init: %7) across dimensions = [3] : (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %12 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %12 : tensor<f32>
  }
  %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<16x80x20xf32>) -> tensor<16x80x20x1xf32>
  %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2, 3] : (tensor<16x80x20x1xf32>) -> tensor<16x80x20x20xf32>
  %11 = stablehlo.divide %6, %10 : tensor<16x80x20x20xf32>
  return %11 : tensor<16x80x20x20xf32>
}

// CHECK-LABEL: @test_neg_incorrectReduceMaxDim
// CHECK: stablehlo.divide
// CHECK-SAME: __not__softmax__


// -----
func.func public @test_neg_incorrectBroadcast(%arg0: tensor<16x20x80x40xf32>, %arg1: tensor<16x20x80x40xf32>) -> (tensor<16x80x20x20xf32>) {

  %6 = stablehlo.constant dense<0.000001e+00> : tensor<1x1x1x1xf32>
  %7 = stablehlo.constant dense<0.000001e+00> : tensor<1x1x1x1xf32>
  %10 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<16x80x20x20xf32>
  %11 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<16x80x20x20xf32>
  %12 = stablehlo.divide %10, %11 : tensor<16x80x20x20xf32>
  return %12 : tensor<16x80x20x20xf32>
}

// CHECK-LABEL: @test_neg_incorrectBroadcast
// CHECK: stablehlo.divide
// CHECK-SAME: __not__softmax__


// -----
func.func public @test_neg_incorrectReduceMaxBodyOp(%arg0: tensor<16x20x80x40xf32>, %arg1: tensor<16x20x80x40xf32>) -> (tensor<16x80x20x20xf32>) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<16x20x80x40xf32>, tensor<16x20x80x40xf32>) -> tensor<16x80x20x20xf32>
  %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %2 = stablehlo.reduce(%0 init: %1) across dimensions = [3] : (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %12 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %12 : tensor<f32>
  }
  %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x80x20xf32>) -> tensor<16x80x20x1xf32>
  %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2, 3] : (tensor<16x80x20x1xf32>) -> tensor<16x80x20x20xf32>
  %5 = stablehlo.subtract %0, %4 : tensor<16x80x20x20xf32>
  %6 = stablehlo.exponential %5 : tensor<16x80x20x20xf32>
  %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %8 = stablehlo.reduce(%6 init: %7) across dimensions = [3] : (tensor<16x80x20x20xf32>, tensor<f32>) -> tensor<16x80x20xf32>
   reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
    %12 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %12 : tensor<f32>
  }
  %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<16x80x20xf32>) -> tensor<16x80x20x1xf32>
  %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2, 3] : (tensor<16x80x20x1xf32>) -> tensor<16x80x20x20xf32>
  %11 = stablehlo.divide %6, %10 : tensor<16x80x20x20xf32>
  return %11 : tensor<16x80x20x20xf32>
}

// CHECK-LABEL: @test_neg_incorrectReduceMaxBodyOp
// CHECK: stablehlo.divide
// CHECK-SAME: __not__softmax__
