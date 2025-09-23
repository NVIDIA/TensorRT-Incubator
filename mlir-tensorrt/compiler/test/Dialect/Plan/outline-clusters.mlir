// RUN: mlir-tensorrt-opt %s -split-input-file -plan-outline-clusters | FileCheck %s

// -----

#profile0 = #plan.bounds<shape, [1, 10], [40, 10]>

func.func @test_outline_closed_alloc_group(%arg0: tensor<?x10xf32> {plan.shape_bounds=#profile0}) -> tensor<?x10xf32> {
  %0 = plan.inline_closed_alloc_group
        target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
        inputs(%arg0 : tensor<?x10xf32>)
        in_attrs [#profile0] -> tensor<?x10xf32> {
  ^bb0(%in0: tensor<?x10xf32>):
    %1 = stablehlo.exponential %in0 : tensor<?x10xf32>
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %1, %c0 : tensor<?x10xf32>
    %2 = plan.with_shape %1(%dim, %c10) : (tensor<?x10xf32>, index, index) -> tensor<?x10xf32>
    yield %2 : tensor<?x10xf32>
  }
  return %0 : tensor<?x10xf32>
}

// CHECK-LABEL: func.func @test_outline_closed_alloc_group
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x10xf32> {plan.shape_bounds = #plan.bounds<shape, [1, 10], [40, 10]>}) -> tensor<?x10xf32>
//  CHECK-NEXT:   %[[v0:.+]] = tensorrt.call_alloc @trt_engines::@tensorrt_cluster(%[[arg0]] : tensor<?x10xf32>) -> tensor<?x10xf32>
//  CHECK-NEXT:   return %[[v0]] : tensor<?x10xf32>
//       CHECK: tensorrt.module @trt_engines
// CHECK-LABEL: func.func @tensorrt_cluster
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x10xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1, 10], opt = [20, 10], max = [40, 10]>}) -> tensor<?x10xf32>
//   CHECK-DAG:       %[[v0:.+]] = stablehlo.exponential %[[arg0]]
//   CHECK-DAG:       %[[c10:.+]] = arith.constant 10 :
//   CHECK-DAG:       %[[c0:.+]] = arith.constant 0 :
//   CHECK-DAG:       %[[dim:.+]] = tensor.dim %[[v0]], %[[c0]] :
//   CHECK-DAG:       %[[v1:.+]] = plan.with_shape %[[v0]](%[[dim]], %[[c10]]) :
//   CHECK-DAG:       return %[[v1]] : tensor<?x10xf32>
