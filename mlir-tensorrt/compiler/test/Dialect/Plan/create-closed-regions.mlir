// RUN: mlir-tensorrt-opt %s -plan-create-closed-regions -split-input-file | FileCheck %s
// RUN: mlir-tensorrt-opt %s -plan-create-closed-regions=test-pre-walk-order=true -split-input-file | FileCheck %s
// RUN: mlir-tensorrt-opt %s -plan-create-closed-regions=prefer-alloc-calling-convention=true -split-input-file | FileCheck %s --check-prefix=CHECK-ALLOC

func.func @test_simple_static(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) -> tensor<10xf32> {
    %1 = stablehlo.add %arg0, %arg1 : tensor<10xf32>
    yield %1 : tensor<10xf32>
  }
  return %0 : tensor<10xf32>
}

//       CHECK: #[[$nobounds:.+]] = #plan.bounds<none>
//       CHECK: #[[$bounds1:.+]] = #plan.bounds<shape, [10], [10]>
// CHECK-LABEL: @test_simple_static
//  CHECK-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>) -> tensor<10xf32>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<10xf32>
//       CHECK:     %[[v1:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[arg1]] : tensor<10xf32>, tensor<10xf32>)
//       CHECK:      outs(%[[v0]] : tensor<10xf32>)
//       CHECK:      in_attrs [#[[$nobounds]], #[[$nobounds]]]
//       CHECK:      res_attrs [#[[$bounds1]]] -> tensor<10xf32> {
//       CHECK:     ^bb0(%[[in:.+]]: tensor<10xf32>, %[[in_0:.+]]: tensor<10xf32>, %[[out:.+]]: tensor<10xf32>):
//       CHECK:     return %[[v1]] : tensor<10xf32>

//       CHECK-ALLOC: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-ALLOC-LABEL: @test_simple_static
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<10xf32>, %[[arg1:.+]]: tensor<10xf32>) -> tensor<10xf32>
//       CHECK-ALLOC:     %[[v1:.+]] = plan.alloc_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK-ALLOC:      inputs(%[[arg0]], %[[arg1]] : tensor<10xf32>, tensor<10xf32>)
//       CHECK-ALLOC:      in_attrs [#[[$nobounds]], #[[$nobounds]]]
//       CHECK-ALLOC:      -> tensor<10xf32> {
//       CHECK-ALLOC:     ^bb0(%[[in:.+]]: tensor<10xf32>, %[[in_0:.+]]: tensor<10xf32>):
//       CHECK-ALLOC:     return %[[v1]] : tensor<10xf32>

// -----

#profile0 = #plan.bounds<shape, [1, 10], [40, 10]>

func.func @test_simple_shape_bound(%arg0: tensor<?x10xf32> {plan.shape_bounds=#profile0}) -> tensor<?x10xf32> {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x10xf32>
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?x10xf32> {
    %1 = stablehlo.exponential %arg0 : tensor<?x10xf32>
    %2 = with_shape %1(%dim, %c10) : (tensor<?x10xf32>, index, index) -> tensor<?x10xf32>
    yield %2 : tensor<?x10xf32>
  }
  return %0 : tensor<?x10xf32>
}

//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
//   CHECK-DAG: #[[$arg0bounds:.+]] = #plan.bounds<shape, [1, 10], [40, 10]>
//   CHECK-DAG: #[[$map:.+]] = affine_map<()[s0] -> (s0 * 10)>
// CHECK-LABEL: @test_simple_shape_bound
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x10xf32> {{.*}}) -> tensor<?x10xf32> {
//       CHECK:     %[[c10:.+]] = arith.constant 10 : index
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x10xf32>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<400xf32>
//       CHECK:     %[[v1:.+]] = affine.apply #[[$map]]()[%[[dim]]]
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v0]][0] [%[[v1]]] [1] : tensor<400xf32> to tensor<?xf32>
//       CHECK:     %[[c10_0:.+]] = arith.constant 10 : index
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[dim]], %[[c10_0]] : tensor<2xindex>
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[extracted_slice]](%[[from_elements]]) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x10xf32>
//       CHECK:     %[[v2:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[c10]], %[[dim]] : tensor<?x10xf32>, index, index)
//       CHECK:      outs(%[[reshape]] : tensor<?x10xf32>)
//       CHECK:      in_attrs [#[[$arg0bounds]], #[[$nobounds]], #[[$nobounds]]]
//       CHECK:      res_attrs [#[[$arg0bounds]]] -> tensor<?x10xf32> {
//       CHECK:     ^bb0(%[[in:.+]]: tensor<?x10xf32>, %[[in_1:.+]]: index, %[[in_2:.+]]: index, %[[out:.+]]: tensor<?x10xf32>):
//       CHECK:       %[[v3:.+]] = stablehlo.exponential %[[in]] : tensor<?x10xf32>
//       CHECK:       %[[v4:.+]] = with_shape %[[v3]](%[[in_2]], %[[in_1]]) :
//       CHECK:       yield %[[v4]] : tensor<?x10xf32>
//       CHECK:     return %[[v2]] : tensor<?x10xf32>

//   CHECK-ALLOC-DAG: #[[$arg0bounds:.+]] = #plan.bounds<shape, [1, 10], [40, 10]>
//   CHECK-ALLOC-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-ALLOC-LABEL: @test_simple_shape_bound
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<?x10xf32> {{.*}}) -> tensor<?x10xf32> {
//       CHECK-ALLOC:     %[[c10:.+]] = arith.constant 10 : index
//       CHECK-ALLOC:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-ALLOC:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x10xf32>
//       CHECK-ALLOC:     %[[v2:.+]] = plan.alloc_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK-ALLOC:      inputs(%[[arg0]], %[[c10]], %[[dim]] : tensor<?x10xf32>, index, index)
//       CHECK-ALLOC:      in_attrs [#[[$arg0bounds]], #[[$nobounds]], #[[$nobounds]]]
//       CHECK-ALLOC:      -> tensor<?x10xf32> {
//       CHECK-ALLOC:     ^bb0(%[[in:.+]]: tensor<?x10xf32>, %[[in_1:.+]]: index, %[[in_2:.+]]: index):
//       CHECK-ALLOC:       %[[v3:.+]] = stablehlo.exponential %[[in]] : tensor<?x10xf32>
//       CHECK-ALLOC:       %[[v4:.+]] = with_shape %[[v3]](%[[in_2]], %[[in_1]]) :
//       CHECK-ALLOC:       yield %[[v4]] : tensor<?x10xf32>
//       CHECK-ALLOC:     return %[[v2]] : tensor<?x10xf32>

// -----

#profile0 = #plan.bounds<shape, [1], [40]>
#profile1 = #plan.bounds<value, dense<[1, 1]> : tensor<2xi32>, dense<[40, 40]> : tensor<2xi32>>

func.func @test_dynamic_reshape(%arg0: tensor<?xf32> {plan.shape_bounds = #profile0},
                                %arg1: tensor<2xi32> {plan.value_bounds = #profile1}) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %extracted = tensor.extract %arg1[%c0] : tensor<2xi32>
  %0 = arith.index_cast %extracted : i32 to index
  %extracted_0 = tensor.extract %arg1[%c1] : tensor<2xi32>
  %1 = arith.index_cast %extracted_0 : i32 to index
  %2 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?x?xf32> {
    %3 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    %4 = with_shape %3(%0, %1) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    yield %4 : tensor<?x?xf32>
  }
  return %2 : tensor<?x?xf32>
}

//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<shape, [1], [40]>
//   CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<value, dense<1> : tensor<2xi32>, dense<40> : tensor<2xi32>>
//   CHECK-DAG: #[[$map:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//   CHECK-DAG: #[[$bounds2:.+]] = #plan.bounds<shape, [1, 1], [40, 40]>
// CHECK-LABEL: @test_dynamic_reshape
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32> {{.*}}, %[[arg1:.+]]: tensor<2xi32> {{.*}}) -> tensor<?x?xf32> {
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg1]][%[[c0]]] : tensor<2xi32>
//       CHECK:     %[[v0:.+]] = arith.index_cast %[[extracted]] : i32 to index
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[arg1]][%[[c1]]] : tensor<2xi32>
//       CHECK:     %[[v1:.+]] = arith.index_cast %[[extracted_0]] : i32 to index
//       CHECK:     %[[v2:.+]] = tensor.empty() : tensor<1600xf32>
//       CHECK:     %[[v3:.+]] = affine.apply #[[$map]]()[%[[v0]], %[[v1]]]
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v2]][0] [%[[v3]]] [1] : tensor<1600xf32> to tensor<?xf32>
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v0]], %[[v1]] : tensor<2xindex>
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[extracted_slice]](%[[from_elements]]) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
//       CHECK:     %[[v4:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[arg1]], %[[v0]], %[[v1]] : tensor<?xf32>, tensor<2xi32>, index, index)
//       CHECK:      outs(%[[reshape]] : tensor<?x?xf32>)
//  CHECK-NEXT:      in_attrs [
//  CHECK-SAME:          #[[$bounds0]],
//  CHECK-SAME:          #[[$bounds1]],
//  CHECK-SAME:          #[[$nobounds]], #[[$nobounds]]]
//  CHECK-NEXT:      res_attrs [#[[$bounds2]]] -> tensor<?x?xf32>
//       CHECK:     ^bb0(%[[in:.+]]: tensor<?xf32>, %[[in_1:.+]]: tensor<2xi32>, %[[in_2:.+]]: index, %[[in_3:.+]]: index, %[[out:.+]]: tensor<?x?xf32>):
//       CHECK:       %[[v5:.+]] = stablehlo.dynamic_reshape %[[in]], %[[in_1]]
//       CHECK:       %[[v6:.+]] = with_shape %[[v5]](%[[in_2]], %[[in_3]])
//       CHECK:       yield %[[v6]] : tensor<?x?xf32>
//       CHECK:     return %[[v4]] : tensor<?x?xf32>


//   CHECK-ALLOC-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
//   CHECK-ALLOC-DAG: #[[$bounds0:.+]] = #plan.bounds<shape, [1], [40]>
//   CHECK-ALLOC-DAG: #[[$bounds1:.+]] = #plan.bounds<value, dense<1> : tensor<2xi32>, dense<40> : tensor<2xi32>>
// CHECK-ALLOC-LABEL: @test_dynamic_reshape
//  CHECK-ALLOC-SAME: (%[[arg0:.+]]: tensor<?xf32> {{.*}}, %[[arg1:.+]]: tensor<2xi32> {{.*}}) -> tensor<?x?xf32> {
//       CHECK-ALLOC:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK-ALLOC:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-ALLOC:     %[[extracted:.+]] = tensor.extract %[[arg1]][%[[c0]]] : tensor<2xi32>
//       CHECK-ALLOC:     %[[v0:.+]] = arith.index_cast %[[extracted]] : i32 to index
//       CHECK-ALLOC:     %[[extracted_0:.+]] = tensor.extract %[[arg1]][%[[c1]]] : tensor<2xi32>
//       CHECK-ALLOC:     %[[v1:.+]] = arith.index_cast %[[extracted_0]] : i32 to index
//       CHECK-ALLOC:     %[[v4:.+]] = plan.alloc_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK-ALLOC:      inputs(%[[arg0]], %[[arg1]], %[[v0]], %[[v1]] : tensor<?xf32>, tensor<2xi32>, index, index)
//  CHECK-ALLOC-NEXT:      in_attrs [
//  CHECK-ALLOC-SAME:          #[[$bounds0]],
//  CHECK-ALLOC-SAME:          #[[$bounds1]],
//  CHECK-ALLOC-SAME:          #[[$nobounds]], #[[$nobounds]]]
//  CHECK-ALLOC-NEXT:     -> tensor<?x?xf32>
//       CHECK-ALLOC:     ^bb0(%[[in:.+]]: tensor<?xf32>, %[[in_1:.+]]: tensor<2xi32>, %[[in_2:.+]]: index, %[[in_3:.+]]: index):
//       CHECK-ALLOC:       %[[v5:.+]] = stablehlo.dynamic_reshape %[[in]], %[[in_1]]
//       CHECK-ALLOC:       %[[v6:.+]] = with_shape %[[v5]](%[[in_2]], %[[in_3]])
//       CHECK-ALLOC:       yield %[[v6]] : tensor<?x?xf32>
//       CHECK-ALLOC:     return %[[v4]] : tensor<?x?xf32>

// -----

#profile0 = #plan.bounds<shape, [1, 1], [40, 40]>
#profile1 = #plan.bounds<shape, [1, 1], [60, 100]>

func.func @test_get_dim_size_max(%arg0: tensor<?x?xf32> {plan.shape_bounds=#profile0}, %arg1: tensor<?x?xf32> {plan.shape_bounds=#profile1}) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1xf32>
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %1 = arith.maxsi %dim, %dim_0 : index
  %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %dim_2 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %2 = arith.maxsi %dim_1, %dim_2 : index
  %3 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?x?xf32> {
    %4 = "stablehlo.get_dimension_size"(%arg0) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
    %6 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = "stablehlo.concatenate"(%5, %7) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
    %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
    %11 = "stablehlo.get_dimension_size"(%arg1) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = "stablehlo.concatenate"(%10, %12) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %14 = stablehlo.maximum %8, %13 : tensor<2xi32>
    %15 = "stablehlo.dynamic_broadcast_in_dim"(%0, %14) {broadcast_dimensions = array<i64: 0, 1>} : (tensor<1x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    %16 = with_shape %15(%1, %2) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    yield %16 : tensor<?x?xf32>
  }
  return %3 : tensor<?x?xf32>
}

//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<shape, [1, 1], [40, 40]>
//   CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<shape, [1, 1], [60, 100]>
//   CHECK-DAG: #[[$map:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK-LABEL: @test_get_dim_size_max
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32> {{.+}}, %[[arg1:.+]]: tensor<?x?xf32> {{.+}})
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<1x1xf32>
//       CHECK:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x?xf32>
//       CHECK:     %[[dim_0:.+]] = tensor.dim %[[arg1]], %[[c0]] : tensor<?x?xf32>
//       CHECK:     %[[v1:.+]] = arith.maxsi %[[dim]], %[[dim_0]] : index
//       CHECK:     %[[dim_1:.+]] = tensor.dim %[[arg0]], %[[c1]] : tensor<?x?xf32>
//       CHECK:     %[[dim_2:.+]] = tensor.dim %[[arg1]], %[[c1]] : tensor<?x?xf32>
//       CHECK:     %[[v2:.+]] = arith.maxsi %[[dim_1]], %[[dim_2]] : index
//       CHECK:     %[[v3:.+]] = tensor.empty() : tensor<6000xf32>
//       CHECK:     %[[v4:.+]] = affine.apply #[[$map]]()[%[[v1]], %[[v2]]]
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v3]][0] [%[[v4]]] [1] : tensor<6000xf32> to tensor<?xf32>
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v1]], %[[v2]] : tensor<2xindex>
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[extracted_slice]](%[[from_elements]]) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
//       CHECK:     %[[v5:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[arg1]], %[[v1]], %[[v2]] : tensor<?x?xf32>, tensor<?x?xf32>, index, index)
//       CHECK:      outs(%[[reshape]] : tensor<?x?xf32>)
//       CHECK:      in_attrs [
//  CHECK-SAME: #[[$bounds0]],
//  CHECK-SAME: #[[$bounds1]],
//  CHECK-SAME: #[[$nobounds]], #[[$nobounds]]]
//       CHECK:      res_attrs [#[[$bounds1]]] -> tensor<?x?xf32>
//       CHECK:     ^bb0(%[[in:.+]]: tensor<?x?xf32>, %[[in_3:.+]]: tensor<?x?xf32>, %[[in_4:.+]]: index, %[[in_5:.+]]: index, %[[out:.+]]: tensor<?x?xf32>):
//       CHECK:       %[[v18:.+]] = stablehlo.dynamic_broadcast_in_dim
//       CHECK:       %[[v19:.+]] = with_shape %[[v18]](%[[in_4]], %[[in_5]]) :
//       CHECK:       yield %[[v19]] : tensor<?x?xf32>
//       CHECK:     return %[[v5]] : tensor<?x?xf32>

// CHECK-ALLOC-LABEL: @test_get_dim_size_max

// -----

#profile0 = #plan.bounds<shape, [1], [100]>
#profile1 = #plan.bounds<value, dense<[1]> : tensor<1xindex>, dense<[100]> : tensor<1xindex>>
#profile2 = #plan.bounds<value, dense<[1]> : tensor<1xindex>, dense<[2]> : tensor<1xindex>>

func.func @real_dynamic_slice(%arg0: tensor<?xf32> {plan.shape_bounds = #profile0},
                              %arg1: tensor<1xindex> {plan.value_bounds = #profile1},
                              %arg2: tensor<1xindex> {plan.value_bounds = #profile1},
                              %arg3: tensor<1xindex> {plan.value_bounds = #profile2}) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %extracted = tensor.extract %arg1[%c0] : tensor<1xindex>
  %extracted_0 = tensor.extract %arg2[%c0] : tensor<1xindex>
  %extracted_1 = tensor.extract %arg3[%c0] : tensor<1xindex>
  %0 = arith.subi %extracted_0, %extracted : index
  %1 = arith.addi %extracted_1, %0 : index
  %2 = arith.subi %1, %c1 : index
  %3 = arith.divsi %2, %extracted_1 : index
  %4 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?xf32> {
    %5 = stablehlo.real_dynamic_slice %arg0, %arg1, %arg2, %arg3 : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
    %6 = with_shape %5(%3) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %6 : tensor<?xf32>
  }
  return %4 : tensor<?xf32>
}

//   CHECK-DAG:  #[[$bounds0:.+]] = #plan.bounds<shape, [1], [100]>
//   CHECK-DAG:  #[[$bounds1:.+]] = #plan.bounds<value, dense<1> : tensor<1xindex>, dense<100> : tensor<1xindex>>
//   CHECK-DAG:  #[[$bounds2:.+]] = #plan.bounds<value, dense<1> : tensor<1xindex>, dense<2> : tensor<1xindex>>
//   CHECK-DAG:  #[[$bounds3:.+]] = #plan.bounds<shape, [0], [100]>
//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-LABEL: @real_dynamic_slice
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32> {{.+}}, %[[arg1:.+]]: tensor<1xindex> {{.+}}, %[[arg2:.+]]: tensor<1xindex> {{.+}}, %[[arg3:.+]]: tensor<1xindex> {{.+}}) -> tensor<?xf32> {
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg1]][%[[c0]]] : tensor<1xindex>
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[arg2]][%[[c0]]] : tensor<1xindex>
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[arg3]][%[[c0]]] : tensor<1xindex>
//       CHECK:     %[[v0:.+]] = arith.subi %[[extracted_0]], %[[extracted]] : index
//       CHECK:     %[[v1:.+]] = arith.addi %[[extracted_1]], %[[v0]] : index
//       CHECK:     %[[v2:.+]] = arith.subi %[[v1]], %[[c1]] : index
//       CHECK:     %[[v3:.+]] = arith.divsi %[[v2]], %[[extracted_1]] : index
//       CHECK:     %[[v4:.+]] = tensor.empty() : tensor<100xf32>
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v4]][0] [%[[v3]]] [1] : tensor<100xf32> to tensor<?xf32>
//       CHECK:     %[[v5:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]], %[[v3]] : tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>, index)
//       CHECK:      outs(%[[extracted_slice]] : tensor<?xf32>)
//       CHECK:      in_attrs [
//  CHECK-SAME:           #[[$bounds0]],
//  CHECK-SAME:           #[[$bounds1]],
//  CHECK-SAME:           #[[$bounds1]],
//  CHECK-SAME:           #[[$bounds2]],
//  CHECK-SAME:           #[[$nobounds]]]
//  CHECK-NEXT:      res_attrs [#[[$bounds3]]] -> tensor<?xf32> {
//       CHECK:     ^bb0(%[[in:.+]]: tensor<?xf32>, %[[in_2:.+]]: tensor<1xindex>, %[[in_3:.+]]: tensor<1xindex>, %[[in_4:.+]]: tensor<1xindex>, %[[in_5:.+]]: index, %[[out:.+]]: tensor<?xf32>):
//       CHECK:       %[[v6:.+]] = stablehlo.real_dynamic_slice %[[in]], %[[in_2]], %[[in_3]], %[[in_4]]
//       CHECK:       %[[v7:.+]] = with_shape %[[v6]](%[[in_5]]) :
//       CHECK:       yield %[[v7]] : tensor<?xf32>
//       CHECK:     return %[[v5]] : tensor<?xf32>

// -----

#profile0 = #plan.bounds<shape, [1, 128, 128], [4, 512, 512]>
#profile1 = #plan.bounds<shape, [1, 128, 128], [4, 512, 512]>

func.func @dot_general_c12(%arg0: tensor<?x?x?xf32> {plan.shape_bounds = #profile0},
                           %arg1: tensor<?x?x?xf32> {plan.shape_bounds = #profile1})
                          -> tensor<?x?x?xf32> {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %dim_0 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %dim_1 = tensor.dim %arg1, %c2 : tensor<?x?x?xf32>
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?x?x?xf32> {
    %1 = "stablehlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %2 = with_shape %1(%dim, %dim_0, %dim_1) : (tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    yield %2 : tensor<?x?x?xf32>
  }
  return %0 : tensor<?x?x?xf32>
}

//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<shape, [1, 128, 128], [4, 512, 512]>
//       CHECK: #[[$nobounds:.+]] = #plan.bounds<none>
//       CHECK: #[[$map:.+]] = affine_map<()[s0, s1, s2] -> ((s0 * s1) * s2)>
// CHECK-LABEL: @dot_general_c12
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?xf32> {{.*}}, %[[arg1:.+]]: tensor<?x?x?xf32> {{.*}}) -> tensor<?x?x?xf32> {
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x?x?xf32>
//       CHECK:     %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c2]] : tensor<?x?x?xf32>
//       CHECK:     %[[dim_1:.+]] = tensor.dim %[[arg1]], %[[c2]] : tensor<?x?x?xf32>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<1048576xf32>
//       CHECK:     %[[v1:.+]] = affine.apply #[[$map]]()[%[[dim]], %[[dim_0]], %[[dim_1]]]
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v0]][0] [%[[v1]]] [1] : tensor<1048576xf32> to tensor<?xf32>
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[dim]], %[[dim_0]], %[[dim_1]] : tensor<3xindex>
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[extracted_slice]](%[[from_elements]]) : (tensor<?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
//       CHECK:     %[[v2:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[arg1]], %[[dim]], %[[dim_0]], %[[dim_1]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>, index, index, index)
//       CHECK:      outs(%[[reshape]] : tensor<?x?x?xf32>)
//       CHECK:      in_attrs [#[[$bounds0]], #[[$bounds0]], #[[$nobounds]], #[[$nobounds]], #[[$nobounds]]]
//       CHECK:      res_attrs [#[[$bounds0]]] -> tensor<?x?x?xf32> {
//       CHECK:     ^bb0(%[[in:.+]]: tensor<?x?x?xf32>, %[[in_2:.+]]: tensor<?x?x?xf32>, %[[in_3:.+]]: index, %[[in_4:.+]]: index, %[[in_5:.+]]: index, %[[out]]: tensor<?x?x?xf32>):
//       CHECK:       %[[v3:.+]] = stablehlo.dot_general %[[in]], %[[in_2]]
//       CHECK:       %[[v4:.+]] = with_shape %[[v3]](%[[in_3]], %[[in_4]], %[[in_5]]) :
//       CHECK:       yield %[[v4]] : tensor<?x?x?xf32>
//       CHECK:     return %[[v2]] : tensor<?x?x?xf32>

// -----

#profile0 = #plan.bounds<shape, [1], [100]>
#profile1 = #plan.bounds<value, dense<1> : tensor<1xindex>, dense<2> : tensor<1xindex>>

func.func @dynamic_pad(%arg0: tensor<?xf32> {plan.shape_bounds = #profile0},
                       %arg1: tensor<f32>,
                       %arg2: tensor<1xindex> {plan.value_bounds = #profile1},
                       %arg3: tensor<1xindex> {plan.value_bounds = #profile1},
                       %arg4: tensor<1xindex> {plan.value_bounds = #profile1}) -> tensor<?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %extracted = tensor.extract %arg2[%c0] : tensor<1xindex>
  %extracted_0 = tensor.extract %arg3[%c0] : tensor<1xindex>
  %extracted_1 = tensor.extract %arg4[%c0] : tensor<1xindex>
  %1 = arith.cmpi slt, %dim, %c1 : index
  %2 = arith.subi %dim, %c1 : index
  %3 = arith.select %1, %c0, %2 : index
  %4 = arith.muli %extracted_1, %3 : index
  %5 = arith.addi %4, %dim : index
  %6 = arith.addi %5, %extracted : index
  %7 = arith.addi %6, %extracted_0 : index
  %8 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?xf32> {
    %0 = stablehlo.dynamic_pad %arg0, %arg1, %arg2, %arg3, %arg4 : (tensor<?xf32>, tensor<f32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
    %9 = with_shape %0(%7) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %9 : tensor<?xf32>
  }
  return %8 : tensor<?xf32>
}

//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<value, dense<1> : tensor<1xindex>, dense<2> : tensor<1xindex>>
//   CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<shape, [3], [302]>
//   CHECK-DAG: #[[$bounds2:.+]] = #plan.bounds<shape, [1], [100]>
// CHECK-LABEL: @dynamic_pad
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32> {{.*}}, %[[arg1:.+]]: tensor<f32>, %[[arg2:.+]]: tensor<1xindex> {{.*}}, %[[arg3:.+]]: tensor<1xindex> {{.*}}, %[[arg4:.+]]: tensor<1xindex> {{.*}}) -> tensor<?xf32> {
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?xf32>
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg2]][%[[c0]]] : tensor<1xindex>
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[arg3]][%[[c0]]] : tensor<1xindex>
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[arg4]][%[[c0]]] : tensor<1xindex>
//       CHECK:     %[[v0:.+]] = arith.cmpi slt, %[[dim]], %[[c1]] : index
//       CHECK:     %[[v1:.+]] = arith.subi %[[dim]], %[[c1]] : index
//       CHECK:     %[[v2:.+]] = arith.select %[[v0]], %[[c0]], %[[v1]] : index
//       CHECK:     %[[v3:.+]] = arith.muli %[[extracted_1]], %[[v2]] : index
//       CHECK:     %[[v4:.+]] = arith.addi %[[v3]], %[[dim]] : index
//       CHECK:     %[[v5:.+]] = arith.addi %[[v4]], %[[extracted]] : index
//       CHECK:     %[[v6:.+]] = arith.addi %[[v5]], %[[extracted_0]] : index
//       CHECK:     %[[v7:.+]] = tensor.empty() : tensor<302xf32>
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v7]][0] [%[[v6]]] [1] : tensor<302xf32> to tensor<?xf32>
//       CHECK:     %[[v8:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]], %[[arg4]], %[[v6]] : tensor<?xf32>, tensor<f32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>, index)
//       CHECK:      outs(%[[extracted_slice]] : tensor<?xf32>)
//       CHECK:      in_attrs [
//  CHECK-SAME:        #[[$bounds2]],
//  CHECK-SAME:        #[[$nobounds]],
//  CHECK-SAME:        #[[$bounds0]],
//  CHECK-SAME:        #[[$bounds0]],
//  CHECK-SAME:        #[[$bounds0]],
//  CHECK-SAME:        #[[$nobounds]]]
//       CHECK:      res_attrs [#[[$bounds1]]] -> tensor<?xf32> {
//       CHECK:     ^bb0(%[[in:.+]]: tensor<?xf32>, %[[in_2:.+]]: tensor<f32>, %[[in_3:.+]]: tensor<1xindex>, %[[in_4:.+]]: tensor<1xindex>, %[[in_5:.+]]: tensor<1xindex>, %[[in_6:.+]]: index, %[[out]]: tensor<?xf32>):
//       CHECK:       %[[v9:.+]] = stablehlo.dynamic_pad %[[in]], %[[in_2]], %[[in_3]], %[[in_4]], %[[in_5]]
//       CHECK:       %[[v10:.+]] = with_shape %[[v9]](%[[in_6]]) :
//       CHECK:       yield %[[v10]] : tensor<?xf32>
//       CHECK:     return %[[v8]] : tensor<?xf32>

// -----

#profile0 = #plan.bounds<shape, [1], [6]>

func.func @broadcast(%arg0: tensor<?xi32> {plan.shape_bounds = #profile0}) -> tensor<1x2x?xi32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xi32>
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<1x2x?xi32> {
    %1 = "stablehlo.broadcast"(%arg0) {broadcast_sizes = array<i64: 1, 2>} : (tensor<?xi32>) -> tensor<1x2x?xi32>
    %2 = with_shape %1(%c1, %c2, %dim) : (tensor<1x2x?xi32>, index, index, index) -> tensor<1x2x?xi32>
    yield %2 : tensor<1x2x?xi32>
  }
  return %0 : tensor<1x2x?xi32>
}

// -----

#profile0 = #plan.bounds<shape, [1, 2, 3, 4], [4, 8, 12, 16]>

func.func @transpose(%arg0: tensor<?x?x?x?xi32> {plan.shape_bounds = #profile0}) -> tensor<?x?x?x?xi32> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<?x?x?x?xi32>
  %dim_0 = tensor.dim %arg0, %c0 : tensor<?x?x?x?xi32>
  %dim_1 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xi32>
  %dim_2 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xi32>
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?x?x?x?xi32> {
    %1 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 2>} : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
    %2 = with_shape %1(%dim, %dim_0, %dim_1, %dim_2) : (tensor<?x?x?x?xi32>, index, index, index, index) -> tensor<?x?x?x?xi32>
    yield %2 : tensor<?x?x?x?xi32>
  }
  return %0 : tensor<?x?x?x?xi32>
}

//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<shape, [1, 2, 3, 4], [4, 8, 12, 16]>
//   CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<shape, [2, 1, 4, 3], [8, 4, 16, 12]>
//   CHECK-DAG: #[[$map:.+]] = affine_map<()[s0, s1, s2, s3] -> (((s0 * s1) * s2) * s3)>
// CHECK-LABEL: @transpose
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?x?xi32> {{.*}}) -> tensor<?x?x?x?xi32> {
//       CHECK:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[c3:.+]] = arith.constant 3 : index
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c1]] : tensor<?x?x?x?xi32>
//       CHECK:     %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x?x?x?xi32>
//       CHECK:     %[[dim_1:.+]] = tensor.dim %[[arg0]], %[[c3]] : tensor<?x?x?x?xi32>
//       CHECK:     %[[dim_2:.+]] = tensor.dim %[[arg0]], %[[c2]] : tensor<?x?x?x?xi32>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<6144xi32>
//       CHECK:     %[[v1:.+]] = affine.apply #[[$map]]()[%[[dim]], %[[dim_0]], %[[dim_1]], %[[dim_2]]]
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v0]][0] [%[[v1]]] [1] : tensor<6144xi32> to tensor<?xi32>
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[dim]], %[[dim_0]], %[[dim_1]], %[[dim_2]] : tensor<4xindex>
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[extracted_slice]](%[[from_elements]]) : (tensor<?xi32>, tensor<4xindex>) -> tensor<?x?x?x?xi32>
//       CHECK:     %[[v2:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[dim]], %[[dim_0]], %[[dim_1]], %[[dim_2]] : tensor<?x?x?x?xi32>, index, index, index, index)
//       CHECK:      outs(%[[reshape]] : tensor<?x?x?x?xi32>)
//       CHECK:      in_attrs [#[[$bounds0]], #[[$nobounds]], #[[$nobounds]], #[[$nobounds]], #[[$nobounds]]]
//       CHECK:      res_attrs [#[[$bounds1]]]
//       CHECK:     ^bb0(%[[in:.+]]: tensor<?x?x?x?xi32>, %[[in_3:.+]]: index, %[[in_4:.+]]: index, %[[in_5:.+]]: index, %[[in_6:.+]]: index, %[[out]]: tensor<?x?x?x?xi32>):
//       CHECK:       %[[v3:.+]] = stablehlo.transpose %[[in]]
//       CHECK:       %[[v4:.+]] = with_shape %[[v3]](%[[in_3]], %[[in_4]], %[[in_5]], %[[in_6]]) :
//       CHECK:       yield %[[v4]] : tensor<?x?x?x?xi32>
//       CHECK:     return %[[v2]] : tensor<?x?x?x?xi32>

// -----

#profile0 = #plan.bounds<value, dense<1> : tensor<1xindex>, dense<6> : tensor<1xindex>>

func.func @dynamic_iota(%arg0: tensor<1xindex> {plan.value_bounds = #profile0}) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %extracted = tensor.extract %arg0[%c0] : tensor<1xindex>
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?xf32> {
    %1 = "stablehlo.dynamic_iota"(%arg0) {iota_dimension = 0 : i64} : (tensor<1xindex>) -> tensor<?xf32>
    %2 = with_shape %1(%extracted) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %2 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<shape, [1], [6]>
//   CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<value, dense<1> : tensor<1xindex>, dense<6> : tensor<1xindex>>

// CHECK-LABEL: @dynamic_iota
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xindex> {{.*}}) -> tensor<?xf32> {
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<1xindex>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<6xf32>
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v0]][0] [%[[extracted]]] [1] : tensor<6xf32> to tensor<?xf32>
//       CHECK:     %[[v1:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[extracted]] : tensor<1xindex>, index)
//       CHECK:      outs(%[[extracted_slice]] : tensor<?xf32>)
//       CHECK:      in_attrs [
//  CHECK-SAME:        #[[$bounds1]],
//  CHECK-SAME:        #[[$nobounds]]]
//       CHECK:      res_attrs [#[[$bounds0]]] -> tensor<?xf32> {
//       CHECK:     ^bb0(%[[in:.+]]: tensor<1xindex>, %[[in_0:.+]]: index, %[[out]]: tensor<?xf32>):
//       CHECK:       %[[v2:.+]] = stablehlo.dynamic_iota %[[in]]
//       CHECK:       %[[v3:.+]] = with_shape %[[v2]](%[[in_0]]) :
//       CHECK:       yield %[[v3]] : tensor<?xf32>
//       CHECK:     return %[[v1]] : tensor<?xf32>

// -----

#profile0 = #plan.bounds<value, dense<1> : tensor<i64>, dense<6> : tensor<i64>>
#profile1 = #plan.bounds<shape, [1, 4], [6, 4]>
#profile2 = #plan.bounds<shape, [2, 1, 4], [2, 6, 4]>

func.func @add_dynamic(%arg0: tensor<i64> {plan.value_bounds = #profile0},
                       %arg1: tensor<?x4xf32> {plan.shape_bounds = #profile1},
                       %arg2: tensor<2x?x4xf32> {plan.shape_bounds = #profile2}) -> tensor<2x?x4xf32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = stablehlo.constant dense<2> : tensor<1xi32>
  %1 = stablehlo.constant dense<4> : tensor<1xi32>
  %2 = stablehlo.constant dense<1> : tensor<1xi32>
  %3 = plan.cluster target(#plan.host_backend<benefit = 1>) -> tensor<i32> {
    %6 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %v = tensor.extract %arg0[] : tensor<i64>
    %v_i32 = arith.trunci %v : i64 to i32
    %7 = plan.with_values %6(%v_i32) : tensor<i32>
    yield %7 : tensor<i32>
  }
  %extracted = tensor.extract %arg0[] : tensor<i64>
  %4 = arith.index_cast %extracted : i64 to index
  %5 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<2x?x4xf32> {
    %6 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
    %7 = "stablehlo.concatenate"(%2, %6, %1) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %8 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %7) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<1x?x4xf32>
    %9 = with_shape %8(%c1, %4, %c4) : (tensor<1x?x4xf32>, index, index, index) -> tensor<1x?x4xf32>
    %10 = "stablehlo.concatenate"(%0, %6, %1) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %11 = "stablehlo.dynamic_broadcast_in_dim"(%9, %10) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<1x?x4xf32>, tensor<3xi32>) -> tensor<2x?x4xf32>
    %12 = with_shape %11(%c2, %4, %c4) : (tensor<2x?x4xf32>, index, index, index) -> tensor<2x?x4xf32>
    %13 = stablehlo.add %12, %arg2 : tensor<2x?x4xf32>
    %14 = with_shape %13(%c2, %4, %c4) : (tensor<2x?x4xf32>, index, index, index) -> tensor<2x?x4xf32>
    yield %14 : tensor<2x?x4xf32>
  }
  return %5 : tensor<2x?x4xf32>
}

//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
//   CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<value, dense<1> : tensor<i32>, dense<6> : tensor<i32>>
//   CHECK-DAG: #[[$bounds2:.+]] = #plan.bounds<shape, [1, 4], [6, 4]>
//   CHECK-DAG: #[[$bounds3:.+]] = #plan.bounds<shape, [2, 1, 4], [2, 6, 4]>
// CHECK-LABEL: @add_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i64> {{.*}}, %[[arg1:.+]]: tensor<?x4xf32> {{.*}}, %[[arg2:.+]]: tensor<2x?x4xf32> {{.*}}) -> tensor<2x?x4xf32> {
//       CHECK:   %[[c2:.+]] = arith.constant 2 : index
//       CHECK:   %[[c4:.+]] = arith.constant 4 : index
//       CHECK:   %[[c1:.+]] = arith.constant 1 : index
//       CHECK:   %[[v0:.+]] = stablehlo.constant dense<2> : tensor<1xi32>
//       CHECK:   %[[v1:.+]] = stablehlo.constant dense<4> : tensor<1xi32>
//       CHECK:   %[[v2:.+]] = stablehlo.constant dense<1> : tensor<1xi32>
//       CHECK:   %[[v3:.+]] = plan.alloc_cluster target(#plan.host_backend<benefit = 1>)
//       CHECK:      yield
//       CHECK:   %[[extracted:.+]] = tensor.extract %[[arg0]][] : tensor<i64>
//       CHECK:   %[[v4:.+]] = arith.index_cast %[[extracted]] : i64 to index
//       CHECK:   %[[v5:.+]] = tensor.empty() : tensor<48xf32>
//       CHECK:   %[[v6:.+]] = affine.apply #[[$map]]()[%[[v4]]]
//       CHECK:   %[[extracted_slice:.+]] = tensor.extract_slice %[[v5]][0] [%[[v6]]] [1] : tensor<48xf32> to tensor<?xf32>
//       CHECK:   %[[c2_0:.+]] = arith.constant 2 : index
//       CHECK:   %[[c4_1:.+]] = arith.constant 4 : index
//       CHECK:   %[[from_elements:.+]] = tensor.from_elements %[[c2_0]], %[[v4]], %[[c4_1]] : tensor<3xindex>
//       CHECK:   %[[reshape:.+]] = tensor.reshape %[[extracted_slice]](%[[from_elements]]) : (tensor<?xf32>, tensor<3xindex>) -> tensor<2x?x4xf32>
//       CHECK:   %[[v7:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//  CHECK-NEXT:    inputs(%[[arg1]], %[[arg2]], %[[c2]], %[[c4]], %[[c1]], %[[v3]], %[[v4]] :
//  CHECK-NEXT:    outs(%[[reshape]] : tensor<2x?x4xf32>)
//       CHECK:   return %[[v7]] : tensor<2x?x4xf32>


// -----

#profile0 = #plan.bounds<value, dense<1> : tensor<i64>, dense<6> : tensor<i64>>
#profile1 = #plan.bounds<shape, [1, 1, 5, 1, 7], [6, 6, 5, 6, 7]>

func.func @collapse_dynamic(%arg0: tensor<i64> {plan.value_bounds = #profile0},
                            %arg1: tensor<i64> {plan.value_bounds = #profile0},
                            %arg2: tensor<i64> {plan.value_bounds = #profile0},
                            %arg3: tensor<?x?x5x?x7xf32> {plan.shape_bounds = #profile1}) -> tensor<?x?x7xf32> {
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index
  %0 = stablehlo.constant dense<7> : tensor<1xi32>
  %1 = stablehlo.constant dense<5> : tensor<i64>
  %2 = plan.cluster target(#plan.host_backend<benefit = 1>) -> tensor<i32> {
    %10 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %v = tensor.extract %arg0[] : tensor<i64>
    %v_i32 = arith.trunci %v : i64 to i32
    %100 = plan.with_values %10(%v_i32) : tensor<i32>
    yield %100 : tensor<i32>
  }
  %3 = plan.cluster target(#plan.host_backend<benefit = 1>) -> tensor<i32> {
    %10 = stablehlo.multiply %arg1, %arg2 : tensor<i64>
    %11 = stablehlo.multiply %10, %1 : tensor<i64>
    %12 = stablehlo.convert %11 : (tensor<i64>) -> tensor<i32>
    %lhs = tensor.extract %arg1[] : tensor<i64>
    %rhs = tensor.extract %arg2[] : tensor<i64>
    %m = arith.muli %lhs, %rhs : i64
    %m_i64 = arith.trunci %m : i64 to i32
    %121 = plan.with_values %12(%m_i64) : tensor<i32>
    yield %121 : tensor<i32>
  }
  %extracted = tensor.extract %arg0[] : tensor<i64>
  %4 = arith.index_cast %extracted : i64 to index
  %extracted_0 = tensor.extract %arg1[] : tensor<i64>
  %extracted_1 = tensor.extract %arg2[] : tensor<i64>
  %5 = arith.index_cast %extracted_0 : i64 to index
  %6 = arith.index_cast %extracted_1 : i64 to index
  %7 = arith.muli %5, %6 : index
  %8 = arith.muli %7, %c5 : index
  %9 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?x?x7xf32> {
    %10 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %11 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
    %12 = "stablehlo.concatenate"(%10, %11, %0) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %13 = stablehlo.dynamic_reshape %arg3, %12 : (tensor<?x?x5x?x7xf32>, tensor<3xi32>) -> tensor<?x?x7xf32>
    %14 = with_shape %13(%4, %8, %c7) : (tensor<?x?x7xf32>, index, index, index) -> tensor<?x?x7xf32>
    yield %14 : tensor<?x?x7xf32>
  }
  return %9 : tensor<?x?x7xf32>
}

//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<value, dense<1> : tensor<i32>, dense<6> : tensor<i32>>
//   CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<shape, [1, 5, 7], [6, 180, 7]>
//   CHECK-DAG: #[[$bounds2:.+]] = #plan.bounds<value, dense<1> : tensor<i32>, dense<36> : tensor<i32>>
//   CHECK-DAG: #[[$bounds3:.+]] = #plan.bounds<shape, [1, 1, 5, 1, 7], [6, 6, 5, 6, 7]>
// CHECK-LABEL: @collapse_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i64> {{.+}}, %[[arg1:.+]]: tensor<i64> {{.*}}, %[[arg2:.+]]: tensor<i64> {{.*}}, %[[arg3:.+]]: tensor<?x?x5x?x7xf32> {{.*}}) -> tensor<?x?x7xf32> {
//   CHECK-DAG:     %[[c7:.+]] = arith.constant 7 : index
//   CHECK-DAG:     %[[c5:.+]] = arith.constant 5 : index
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<7> : tensor<1xi32>
//       CHECK:     %[[v1:.+]] = stablehlo.constant dense<5> : tensor<i64>
//       CHECK:     %[[v2:.+]] = plan.alloc_cluster target(#plan.host_backend<benefit = 1>)
//       CHECK:        yield
//       CHECK:     %[[v3:.+]] = plan.alloc_cluster target(#plan.host_backend<benefit = 1>)
//       CHECK:        yield
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][] : tensor<i64>
//       CHECK:     %[[v4:.+]] = arith.index_cast %[[extracted]] : i64 to index
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[arg1]][] : tensor<i64>
//       CHECK:     %[[extracted_1:.+]] = tensor.extract %[[arg2]][] : tensor<i64>
//       CHECK:     %[[v5:.+]] = arith.index_cast %[[extracted_0]] : i64 to index
//       CHECK:     %[[v6:.+]] = arith.index_cast %[[extracted_1]] : i64 to index
//       CHECK:     %[[v7:.+]] = arith.muli %[[v5]], %[[v6]] : index
//       CHECK:     %[[v8:.+]] = arith.muli %[[v7]], %[[c5]] : index
//       CHECK:     %[[v9:.+]] = tensor.empty() : tensor<7560xf32>
//       CHECK:     %[[v10:.+]] = affine.apply #[[$map]]()[%[[v4]], %[[v8]]]
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v9]][0] [%[[v10]]] [1] : tensor<7560xf32> to tensor<?xf32>
//       CHECK:     %[[c7_2:.+]] = arith.constant 7 : index
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[v4]], %[[v8]], %[[c7_2]] : tensor<3xindex>
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[extracted_slice]](%[[from_elements]]) : (tensor<?xf32>, tensor<3xindex>) -> tensor<?x?x7xf32>
//       CHECK:     %[[v11:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg3]], %[[c7]], %[[v2]], %[[v3]], %[[v4]], %[[v8]] :
//       CHECK:      outs(%[[reshape]] : tensor<?x?x7xf32>)
//       CHECK:      in_attrs [#[[$bounds3]],
// CHECK-SAME:         #[[$nobounds]],
// CHECK-SAME:         #[[$bounds0]], #[[$bounds2]], #[[$nobounds]], #[[$nobounds]]]
//       CHECK:      res_attrs [#[[$bounds1]]] -> tensor<?x?x7xf32> {
//       CHECK:     ^bb0(%[[in:.+]]: tensor<?x?x5x?x7xf32>, %[[in_4:.+]]: index, %[[in_5:.+]]: tensor<i32>, %[[in_6:.+]]: tensor<i32>, %[[in_7:.+]]: index, %[[in_8:.+]]: index, %[[out]]: tensor<?x?x7xf32>):
//       CHECK:       %[[v12:.+]] = stablehlo.constant dense<7> : tensor<1xi32>
//       CHECK:       %[[v13:.+]] = stablehlo.reshape %[[in_5]] : (tensor<i32>) -> tensor<1xi32>
//       CHECK:       %[[v14:.+]] = stablehlo.reshape %[[in_6]] : (tensor<i32>) -> tensor<1xi32>
//       CHECK:       %[[v15:.+]] = stablehlo.concatenate %[[v13]], %[[v14]], %[[v12]]
//       CHECK:       %[[v16:.+]] = stablehlo.dynamic_reshape %[[in]], %[[v15]]
//       CHECK:       %[[v17:.+]] = with_shape %[[v16]](%[[in_7]], %[[in_8]], %[[in_4]]) :
//       CHECK:       yield %[[v17]] : tensor<?x?x7xf32>
//       CHECK:     return %[[v11]] : tensor<?x?x7xf32>

// -----

#profile0 = #plan.bounds<shape, [1], [10]>
#profile1 = #plan.bounds<value, dense<2> : tensor<index>, dense<6> : tensor<index>>

func.func @test_separated(%arg0: tensor<?xf32> {plan.shape_bounds = #profile0},
                          %arg1: index {plan.value_bounds  = #profile1})
    -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?xf32> {
    %2 = stablehlo.exponential %arg0 : tensor<?xf32>
    %3 = with_shape {tag = "with_shape0"} %2(%dim) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %3 : tensor<?xf32>
  }
  %extracted_slice = tensor.extract_slice %0[0] [%arg1] [1] {tag = "extract_slice"} : tensor<?xf32> to tensor<?xf32>
  %extracted_slice2 = plan.with_shape %extracted_slice (%arg1) : (tensor<?xf32>, index) -> tensor<?xf32>
  %1 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?xf32> {
    %2 = stablehlo.exponential %extracted_slice : tensor<?xf32>
    %3 = with_shape {tag = "with_shape1"} %2(%arg1) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %3 : tensor<?xf32>
  }
  return %1 : tensor<?xf32>
}

//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<shape, [1], [10]>
//   CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<shape, [2], [6]>
//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-LABEL: @test_separated
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32> {{.*}}, %[[arg1:.+]]: index {{.*}}) -> tensor<?xf32> {
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?xf32>
//       CHECK:     %[[v0:.+]] = tensor.empty() : tensor<10xf32>
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v0]][0] [%[[dim]]] [1] : tensor<10xf32> to tensor<?xf32>
//       CHECK:     %[[v1:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[dim]] : tensor<?xf32>, index)
//       CHECK:      outs(%[[extracted_slice]] : tensor<?xf32>)
//       CHECK:      in_attrs [#[[$bounds0]], #[[$nobounds]]]
//       CHECK:      res_attrs [#[[$bounds0]]] -> tensor<?xf32> {
//       CHECK:     ^bb0(%[[in:.+]]: tensor<?xf32>, %[[in_2:.+]]: index, %[[out]]: tensor<?xf32>):
//       CHECK:       %[[v5:.+]] = stablehlo.exponential %[[in]] : tensor<?xf32>
//       CHECK:       %[[v6:.+]] = with_shape {tag = "with_shape0"} %[[v5]](%[[in_2]]) :
//       CHECK:       yield %[[v6]] : tensor<?xf32>
//       CHECK:     %[[extracted_slice_0:.+]] = tensor.extract_slice %[[v1]][0] [%[[arg1]]] [1] {tag = "extract_slice"} : tensor<?xf32> to tensor<?xf32>
//       CHECK:     %[[v2:.+]] = plan.with_shape %[[extracted_slice_0]](%[[arg1]]) :
//       CHECK:     %[[v3:.+]] = tensor.empty() : tensor<6xf32>
//       CHECK:     %[[extracted_slice_1:.+]] = tensor.extract_slice %[[v3]][0] [%[[arg1]]] [1] : tensor<6xf32> to tensor<?xf32>
//       CHECK:     %[[v4:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg1]], %[[extracted_slice_0]] : index, tensor<?xf32>)
//       CHECK:      outs(%[[extracted_slice_1]] : tensor<?xf32>)
//       CHECK:      in_attrs [#[[$nobounds]], #[[$bounds1]]]
//       CHECK:      res_attrs [#[[$bounds1]]] -> tensor<?xf32> {
//       CHECK:     ^bb0(%[[in:.+]]: index, %[[in_2:.+]]: tensor<?xf32>, %[[out]]: tensor<?xf32>):
//       CHECK:       %[[v5:.+]] = stablehlo.exponential %[[in_2]] : tensor<?xf32>
//       CHECK:       %[[v6:.+]] = with_shape {tag = "with_shape1"} %[[v5]](%[[in]]) :
//       CHECK:       yield %[[v6]] : tensor<?xf32>
//       CHECK:     return %[[v4]] : tensor<?xf32>


// -----

// A convoluted example where `stablehlo.dynamic_broadcast_in_dim` was not canonicalized
// into its static version. Normally this would never occur, but we should still handle this
// situation gracefully.

#profile0 = #plan.bounds<shape, [1], [1]>

func.func @test_unneeded_dynamism(%arg0: tensor<?xf32> {plan.shape_bounds = #profile0}) -> tensor<?xf32> {
  %0 = stablehlo.constant dense<[1]> : tensor<1xi32>
  %c1 = arith.constant 1 : index
  %1 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) -> tensor<?xf32> {
    %1 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %0) {broadcast_dimensions = array<i64: 0>} : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
    %2 = with_shape %1 (%c1) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %2 : tensor<?xf32>
  }
  return %1 : tensor<?xf32>
}

//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<shape, [1], [1]>
//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-LABEL: @test_unneeded_dynamism
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32> {{.*}}) -> tensor<?xf32> {
//       CHECK:     %[[v0:.+]] = stablehlo.constant dense<1> : tensor<1xi32>
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[v1:.+]] = tensor.empty() : tensor<1xf32>
//       CHECK:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v1]][0] [1] [1] : tensor<1xf32> to tensor<1xf32>
//       CHECK:     %[[c1_0:.+]] = arith.constant 1 : index
//       CHECK:     %[[from_elements:.+]] = tensor.from_elements %[[c1_0]] : tensor<1xindex>
//       CHECK:     %[[reshape:.+]] = tensor.reshape %[[extracted_slice]](%[[from_elements]]) : (tensor<1xf32>, tensor<1xindex>) -> tensor<?xf32>
//       CHECK:     %[[v2:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[c1]] : tensor<?xf32>, index)
//       CHECK:      outs(%[[reshape]] : tensor<?xf32>)
//       CHECK:      in_attrs [#[[$bounds0]], #[[$nobounds]]]
//       CHECK:      res_attrs [#[[$bounds0]]] -> tensor<?xf32> {
//       CHECK:     ^bb0(%[[in:.+]]: tensor<?xf32>, %[[in_1:.+]]: index, %[[out]]: tensor<?xf32>):
//       CHECK:       %[[v3:.+]] = stablehlo.constant dense<1> : tensor<1xi32>
//       CHECK:       %[[v4:.+]] = stablehlo.dynamic_broadcast_in_dim %[[in]], %[[v3]]
//       CHECK:       %[[v5:.+]] = with_shape %[[v4]](%[[in_1]]) :
//       CHECK:       yield %[[v5]] : tensor<?xf32>
//       CHECK:     return %[[v2]] : tensor<?xf32>

// -----

#profile0 = #plan.bounds<shape, [1], [10]>
#profile1 = #plan.bounds<shape, [2], [10]>

// Connected regions verifies that the bounds analysis result is correctly updated
// as we rewrite the IR.

func.func @test_connected_regions(%arg0: tensor<?xf32> {plan.shape_bounds = #profile0},
                                  %arg1: tensor<?xf32> {plan.shape_bounds = #profile1})
    -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?xf32> {
    %2 = stablehlo.exponential %arg0 : tensor<?xf32>
    %3 = with_shape %2(%dim) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %3 : tensor<?xf32>
  }
  %dim1 = tensor.dim %arg1, %c0 : tensor<?xf32>
  %1 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?xf32> {
    %2 = stablehlo.add %0, %arg1 : tensor<?xf32>
    %3 = with_shape %2 (%dim1) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %3 : tensor<?xf32>
  }
  return %1 : tensor<?xf32>
}

//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<shape, [1], [10]>
//   CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<shape, [2], [10]>
//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>

// CHECK-LABEL: func.func @test_connected_regions
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32> {{.*}}, %[[arg1:.+]]: tensor
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?xf32>
//   CHECK-DAG:     %[[v0:.+]] = tensor.empty() : tensor<10xf32>
//   CHECK-DAG:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v0]][0] [%[[dim]]] [1] : tensor<10xf32> to tensor<?xf32>
//       CHECK:     %[[v1:.+]] = plan.dps_cluster
//  CHECK-NEXT:      inputs(%[[arg0]], %[[dim]] :
//  CHECK-NEXT:      outs(%[[extracted_slice]] :
//  CHECK-NEXT:      in_attrs [#[[$bounds0]], #[[$nobounds]]]
//  CHECK-NEXT:      res_attrs [#[[$bounds0]]] -> tensor<?xf32> {
//  CHECK-NEXT:     ^bb0(%[[in:.+]]: tensor<?xf32>, %[[in_2:.+]]: index, %[[out:.+]]: tensor<?xf32>):
//  CHECK-NEXT:       %[[v4:.+]] = stablehlo.exponential %[[in]] : tensor<?xf32>
//  CHECK-NEXT:       %[[v5:.+]] = with_shape %[[v4]](%[[in_2]]) :
//  CHECK-NEXT:       yield %[[v5]] : tensor<?xf32>
//   CHECK-DAG:     %[[dim_0:.+]] = tensor.dim %[[arg1]], %[[c0]] : tensor<?xf32>
//   CHECK-DAG:     %[[v2:.+]] = tensor.empty() : tensor<10xf32>
//   CHECK-DAG:     %[[extracted_slice_1:.+]] = tensor.extract_slice %[[v2]][0] [%[[dim_0]]] [1] : tensor<10xf32> to tensor<?xf32>
//       CHECK:     %[[v3:.+]] = plan.dps_cluster
//  CHECK-NEXT:      inputs(%[[arg1]], %[[v1]], %[[dim_0]] :
//  CHECK-NEXT:      outs(%[[extracted_slice_1]] : tensor<?xf32>)
//  CHECK-NEXT:      in_attrs [#[[$bounds1]], #[[$bounds0]], #[[$nobounds]]]
//  CHECK-NEXT:      res_attrs [#[[$bounds1]]] -> tensor<?xf32> {
//  CHECK-NEXT:     ^bb0(%[[in:.+]]: tensor<?xf32>, %[[in_2:.+]]: tensor<?xf32>, %[[in_3:.+]]: index, %[[out:.+]]: tensor<?xf32>):
//  CHECK-NEXT:       %[[v4:.+]] = stablehlo.add %[[in_2]], %[[in]] : tensor<?xf32>
//  CHECK-NEXT:       %[[v5:.+]] = with_shape %[[v4]](%[[in_3]]) :
//  CHECK-NEXT:       yield %[[v5]] : tensor<?xf32>
//       CHECK:     return %[[v3]] : tensor<?xf32>

// -----

// Connected regions verifies that the bounds analysis result is correctly updated
// as we rewrite the IR.

#profile0 = #plan.bounds<value, dense<0> : tensor<i32>, dense<123> : tensor<i32>>


func.func @test_connected_regions_host_values(
                  %arg0: tensor<128xf32>,
                  %arg1: tensor<4xf32>,
                  %arg2: tensor<i32> {tensorrt.host_tensor, plan.value_bounds = #profile0})
                  -> tensor<128xf32> {
  %0:2 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<128xf32>, tensor<i32> {
    %1 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2 : (tensor<128xf32>, tensor<4xf32>, tensor<i32>) -> tensor<128xf32>
    yield %1, %arg2 : tensor<128xf32>, tensor<i32>
  }
  %1 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<128xf32> {
    %2 = stablehlo.dynamic_update_slice %arg0, %arg1, %0#1 : (tensor<128xf32>, tensor<4xf32>, tensor<i32>) -> tensor<128xf32>
    yield %2 : tensor<128xf32>
  }
  return %1 : tensor<128xf32>
}

//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<shape, [128], [128]>
//   CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<value, dense<0> : tensor<i32>, dense<123> : tensor<i32>>
//   CHECK-DAG: #[[$bounds2:.+]] = #plan.bounds<shape, [], []>
//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-LABEL: @test_connected_regions_host_values
//       CHECK:     %[[v2:.+]]:2 = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//  CHECK-NEXT:        inputs({{.+}} : tensor<128xf32>, tensor<4xf32>, tensor<i32>)
//  CHECK-NEXT:        outs(%{{.+}} : tensor<128xf32>, tensor<i32>)
//  CHECK-NEXT:        in_attrs [#[[$nobounds]], #[[$nobounds]], #[[$bounds1]]]
//  CHECK-NEXT:        res_attrs [#[[$bounds0]], #[[$bounds2]]] -> tensor<128xf32>, tensor<i32> {
//       CHECK:     plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//  CHECK-NEXT:        inputs({{.*}}, %[[v2]]#1 : tensor<128xf32>, tensor<4xf32>, tensor<i32>)
//       CHECK:        outs(%{{.+}} : tensor<128xf32>)
//  CHECK-NEXT:        in_attrs [#[[$nobounds]], #[[$nobounds]], #[[$bounds1]]]
//  CHECK-NEXT:        res_attrs [#[[$bounds0]]] -> tensor<128xf32>

// -----


func.func @shape_calc(%arg0: tensor<?xf32> {plan.shape_bounds = #plan.bounds<shape,
[1], [4]>},
                      %arg1: tensor<2xi32> {plan.value_bounds = #plan.bounds<value, dense<[1, 1]> : tensor<2xi32>, dense<[2, 2]> : tensor<2xi32>>},
                      %arg2: tensor<2xi32> {plan.value_bounds = #plan.bounds<value, dense<[1, 1]> : tensor<2xi32>, dense<[2, 2]> : tensor<2xi32>>})
                      -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %extracted = tensor.extract %arg2[%c0] : tensor<2xi32>
  %extracted_0 = tensor.extract %arg2[%c1] : tensor<2xi32>
  %extracted_1 = tensor.extract %arg1[%c0] : tensor<2xi32>
  %extracted_2 = tensor.extract %arg1[%c1] : tensor<2xi32>
  %0 = arith.addi %extracted_1, %extracted : i32
  %1 = arith.addi %extracted_2, %extracted_0 : i32
  %2 = arith.muli %0, %0 : i32
  %3 = arith.muli %1, %1 : i32
  %4 = arith.index_cast %extracted_1 : i32 to index
  %5 = arith.index_cast %extracted : i32 to index
  %6 = arith.addi %4, %5 : index
  %7 = arith.muli %6, %6 : index
  %8 = arith.index_cast %extracted_2 : i32 to index
  %9 = arith.index_cast %extracted_0 : i32 to index
  %10 = arith.addi %8, %9 : index
  %11 = arith.muli %10, %10 : index
  %12 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) attributes {__cluster_target__ = #plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>} -> tensor<?x?xf32> {
    %13 = with_values %arg2(%extracted, %extracted_0) : tensor<2xi32>
    %14 = with_values %arg1(%extracted_1, %extracted_2) : tensor<2xi32>
    %15 = stablehlo.add %14, %13 : tensor<2xi32>
    %16 = with_values %15(%0, %1) : tensor<2xi32>
    %17 = stablehlo.multiply %16, %16 : tensor<2xi32>
    %18 = with_values %17(%2, %3) : tensor<2xi32>
    %19 = stablehlo.dynamic_reshape %arg0, %18 : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    %20 = with_shape %19(%7, %11) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    yield %20 : tensor<?x?xf32>
  }
  return %12 : tensor<?x?xf32>
}

//   CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<value, dense<1> : tensor<2xi32>, dense<2> : tensor<2xi32>>
//   CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<shape, [4, 4], [16, 16]>
//   CHECK-DAG: #[[$bounds2:.+]] = #plan.bounds<shape, [1], [4]>
//   CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
//   CHECK-DAG: #[[$map:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK-LABEL: func.func @shape_calc
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>{{.*}}, %[[arg1:.+]]: tensor<2xi32>{{.*}}, %[[arg2:.+]]: tensor<2xi32>{{.*}})
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg2]][%[[c0]]] : tensor<2xi32>
//   CHECK-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg2]][%[[c1]]] : tensor<2xi32>
//   CHECK-DAG:     %[[extracted_1:.+]] = tensor.extract %[[arg1]][%[[c0]]] : tensor<2xi32>
//   CHECK-DAG:     %[[extracted_2:.+]] = tensor.extract %[[arg1]][%[[c1]]] : tensor<2xi32>
//   CHECK-DAG:     %[[v0:.+]] = arith.addi %[[extracted_1]], %[[extracted]] : i32
//   CHECK-DAG:     %[[v1:.+]] = arith.addi %[[extracted_2]], %[[extracted_0]] : i32
//   CHECK-DAG:     %[[v2:.+]] = arith.muli %[[v0]], %[[v0]] : i32
//   CHECK-DAG:     %[[v3:.+]] = arith.muli %[[v1]], %[[v1]] : i32
//   CHECK-DAG:     %[[v4:.+]] = arith.index_cast %[[extracted_1]] : i32 to index
//   CHECK-DAG:     %[[v5:.+]] = arith.index_cast %[[extracted]] : i32 to index
//   CHECK-DAG:     %[[v6:.+]] = arith.addi %[[v4]], %[[v5]] : index
//   CHECK-DAG:     %[[v7:.+]] = arith.muli %[[v6]], %[[v6]] : index
//   CHECK-DAG:     %[[v8:.+]] = arith.index_cast %[[extracted_2]] : i32 to index
//   CHECK-DAG:     %[[v9:.+]] = arith.index_cast %[[extracted_0]] : i32 to index
//   CHECK-DAG:     %[[v10:.+]] = arith.addi %[[v8]], %[[v9]] : index
//   CHECK-DAG:     %[[v11:.+]] = arith.muli %[[v10]], %[[v10]] : index
//   CHECK-DAG:     %[[v12:.+]] = tensor.empty() : tensor<256xf32>
//   CHECK-DAG:     %[[v13:.+]] = affine.apply #[[$map]]()[%[[v7]], %[[v11]]]
//   CHECK-DAG:     %[[extracted_slice:.+]] = tensor.extract_slice %[[v12]][0] [%[[v13]]] [1] : tensor<256xf32> to tensor<?xf32>
//   CHECK-DAG:     %[[from_elements:.+]] = tensor.from_elements %[[v7]], %[[v11]] : tensor<2xindex>
//   CHECK-DAG:     %[[reshape:.+]] = tensor.reshape %[[extracted_slice]](%[[from_elements]]) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
//       CHECK:     %[[v14:.+]] = plan.dps_cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
//       CHECK:      inputs(%[[arg0]], %[[arg1]], %[[arg2]], %[[extracted]], %[[extracted_0]], %[[extracted_1]], %[[extracted_2]], %[[v0]], %[[v1]], %[[v2]], %[[v3]], %[[v7]], %[[v11]] :
//       CHECK:      outs(%[[reshape]] : tensor<?x?xf32>)
//       CHECK:      in_attrs [#[[$bounds2]], #[[$bounds0]], #[[$bounds0]],
//  CHECK-SAME:                #[[$nobounds]], #[[$nobounds]], #[[$nobounds]], #[[$nobounds]], #[[$nobounds]],
//  CHECK-SAME:                #[[$nobounds]], #[[$nobounds]], #[[$nobounds]], #[[$nobounds]], #[[$nobounds]]]
//       CHECK:      res_attrs [#[[$bounds1]]] -> tensor<?x?xf32> {
//       CHECK:     return %[[v14]] : tensor<?x?xf32>

// -----

func.func @float_tensor_host_access(
                      %arg0: tensor<2xf32>)
                      -> (tensor<2xf32>, f32) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %extracted = tensor.extract %arg0[%c0] : tensor<2xf32>
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
        -> tensor<2xf32> {
    %1 = stablehlo.add %arg0, %arg0 : tensor<2xf32>
    yield %1 : tensor<2xf32>
  }
  return %0, %extracted : tensor<2xf32>, f32
}

// CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<shape, [2], [2]>
// CHECK-LABEL: func.func @float_tensor_host_access
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xf32>) -> (tensor<2xf32>, f32) {
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<2xf32>
//   CHECK-DAG:     %[[v0:.+]] = tensor.empty() : tensor<2xf32>
//       CHECK:     %[[v1:.+]] = plan.dps_cluster
//  CHECK-NEXT:      inputs(%[[arg0]] : tensor<2xf32>)
//  CHECK-NEXT:      outs(%[[v0]] : tensor<2xf32>)
//  CHECK-NEXT:      in_attrs [#[[$nobounds]]]
//       CHECK:      res_attrs [#[[$bounds0]]]

// -----

// CHECK-DAG: #[[$bounds0:.+]] = #plan.bounds<value, dense<-2147483648> : tensor<i32>, dense<2147483647> : tensor<i32>>
// CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-DAG: #[[$bounds1:.+]] = #plan.bounds<shape, [10], [10]>

// CHECK-LABEL: @scf_while_unused_result
func.func @scf_while_unused_result(%arg0: tensor<i32>) -> (tensor<i32>) {
  %c20_i32 = arith.constant 20 : i32
  %c1_i32 = stablehlo.constant dense<1> : tensor<i32>
  %c2 = stablehlo.constant dense<0.0> : tensor<10xf32>
  %c0 = arith.constant 0 : index
  // CHECK: scf.while
  %1:2 = scf.while (%arg1 = %arg0, %arg2 = %c2)
      : (tensor<i32>, tensor<10xf32>) -> (tensor<i32>, tensor<10xf32>) {
    %extracted_0 = tensor.extract %arg1[] : tensor<i32>
    %cond = arith.cmpi eq, %extracted_0, %c20_i32 : i32
    scf.condition(%cond) %arg1, %arg2 : tensor<i32>, tensor<10xf32>
  } do {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<10xf32>):
    // CHECK: plan.dps_cluster
    // CHECK: in_attrs [#[[$bounds0]], #[[$nobounds]]]
    // CHECK: res_attrs
    %3, %4 = plan.cluster
        target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>)
        -> tensor<i32>, tensor<10xf32> {
      %1 = stablehlo.add %arg1, %c1_i32 : tensor<i32>
      plan.yield %1, %arg2  : tensor<i32>, tensor<10xf32>
    }
    scf.yield %3, %4 : tensor<i32>, tensor<10xf32>
  }
  return %1#0 : tensor<i32>
}
