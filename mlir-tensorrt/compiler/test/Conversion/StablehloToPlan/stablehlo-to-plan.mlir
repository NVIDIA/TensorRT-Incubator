// RUN: mlir-tensorrt-opt %s -convert-stablehlo-to-plan -split-input-file | FileCheck %s

func.func @test_optimization_barrier(%arg0: tensor<1x1xf32>, %arg1: tensor<i8>) -> (tensor<1x1xf32>, tensor<i8>) {
  %0, %1 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<1x1xf32>, tensor<i8>
  return %0, %1 : tensor<1x1xf32>, tensor<i8>
}

// CHECK-LABEL: func.func @test_optimization_barrier(
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x1xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<i8>
// CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = plan.optimization_barrier %[[ARG0]], %[[ARG1]]
// CHECK-NEXT: return %[[OPT_BARRIER]]#0, %[[OPT_BARRIER]]#1 : tensor<1x1xf32>, tensor<i8>

// -----

func.func @test_optimization_barrier_with_tokens(%arg0: tensor<1x1xf32>, %arg1: !stablehlo.token) -> (tensor<1x1xf32>, !stablehlo.token) {
  %0, %1 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<1x1xf32>, !stablehlo.token
  return %0, %1 : tensor<1x1xf32>, !stablehlo.token
}

// CHECK-LABEL: func.func @test_optimization_barrier_with_tokens(
//   CHECK-NOT: plan.optimization_barrier

// -----

func.func @donation_attr(%arg0: tensor<2x2xf32> {tf.aliasing_output = 0 : i32}, %arg1 : tensor<2x2xf32>) -> tensor<2x2xf32>{
  %r = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// CHECK-LABEL: @donation_attr
//  CHECK-SAME: {plan.aliasing_output = 0 : i32}
//   CHECK-NOT: {tf.aliasing_output = 0 : i32}

// -----

func.func @no_donation_attr(%arg0: tensor<2x2xf32>, %arg1 : tensor<2x2xf32>) -> tensor<2x2xf32>{
  %r = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// CHECK-LABEL: @no_donation_attr
//   CHECK-NOT: {plan.aliasing_output = 0 : i32}
//   CHECK-NOT: {tf.aliasing_output = 0 : i32}
