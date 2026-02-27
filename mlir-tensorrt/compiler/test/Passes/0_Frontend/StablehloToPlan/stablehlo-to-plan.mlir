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

// -----

func.func @test_shape_assertion(%arg0: tensor<?x?xf32>) {
  %c = stablehlo.constant dense<1> : tensor<i32>
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
  %1 = stablehlo.compare GE, %0, %c, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%1, %0) {api_version = 2 : i32, error_message = "Expected value >= 1 for dimension variable 'K'.", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
  return
}

// CHECK-LABEL: func.func @test_shape_assertion(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<?x?xf32>
//       CHECK:   %[[CMP:.*]] = stablehlo.compare GE
//       CHECK:   %[[COND:.*]] = tensor.extract %[[CMP]][] : tensor<i1>
//       CHECK:   cf.assert %[[COND]], "Expected value >= 1 for dimension variable 'K'."
//       CHECK:   return

// -----

func.func @nvtx_push_pop_conversion(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
  %pre_1 = stablehlo.add %arg0, %arg0: tensor<2x4xf32>
  %pre_2 = stablehlo.add %arg1, %arg1: tensor<2x4xf32>
  %0, %1, %2 = "stablehlo.custom_call"(%pre_1, %pre_2)
  <{api_version = 2 : i32, backend_config = "",
  call_target_name = "nvtx.push", called_computations = [],
  has_side_effect = true, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>],
  result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>]}>
  {color = -65281 : i32, name = "softmax_subgraph"} : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<i64>)
  %3 = stablehlo.add %0, %1: tensor<2x4xf32>
  %4 = "stablehlo.custom_call"(%3, %2)
  <{api_version = 2 : i32, backend_config = "",
  call_target_name = "nvtx.pop", called_computations = [],
  has_side_effect = true, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<> : tensor<0xindex>],
  result_layouts = [dense<[1, 0]> : tensor<2xindex>]}>
  {color = -65281 : i32, name = "softmax_subgraph"} : (tensor<2x4xf32>, tensor<i64>) -> (tensor<2x4xf32>)
  return %4 : tensor<2x4xf32>
}

// CHECK-LABEL: @nvtx_push_pop_conversion
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2x4xf32>, %[[arg1:.+]]: tensor<2x4xf32>) -> tensor<2x4xf32>
//       CHECK:   %[[pre1:.+]] = stablehlo.add %[[arg0]], %[[arg0]] : tensor<2x4xf32>
//       CHECK:   %[[pre2:.+]] = stablehlo.add %[[arg1]], %[[arg1]] : tensor<2x4xf32>
//       CHECK:   %[[results:.+]]:2, %[[range_id:.+]] = plan.nvtx_push {color = -65281 : i32, name = "softmax_subgraph"} %[[pre1]], %[[pre2]] : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>, i64)
//       CHECK:   %[[wrapped:.+]] = tensor.from_elements %[[range_id]] : tensor<i64>
//       CHECK:   %[[add:.+]] = stablehlo.add %[[results]]#0, %[[results]]#1 : tensor<2x4xf32>
//       CHECK:   %[[extracted:.+]] = tensor.extract %[[wrapped]][] : tensor<i64>
//       CHECK:   %[[pop:.+]] = plan.nvtx_pop %[[add]], %[[extracted]] : (tensor<2x4xf32>, i64) -> tensor<2x4xf32>
//       CHECK:   return %[[pop]] : tensor<2x4xf32>

