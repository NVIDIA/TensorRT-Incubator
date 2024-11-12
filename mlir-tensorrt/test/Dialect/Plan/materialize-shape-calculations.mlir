// RUN: mlir-tensorrt-opt %s -split-input-file -plan-materialize-shape-calculations -plan-create-shape-funcs | FileCheck %s --check-prefix=SHAPE
// RUN: mlir-tensorrt-opt %s -split-input-file -plan-materialize-shape-calculations | FileCheck %s

func.func @test_simple(%arg0: tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = stablehlo.exponential %arg0 : tensor<?x10xf32>
  return %0 : tensor<?x10xf32>
}

// CHECK-LABEL: @test_simple
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x10xf32>)
//   CHECK-DAG:     %[[c10:.+]] = arith.constant 10 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x10xf32>
//   CHECK-DAG:     %[[arg0_:.+]] = plan.with_shape %[[arg0]]
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.exponential %[[arg0_]] : tensor<?x10xf32>
//  CHECK-NEXT:     %[[v1:.+]] = plan.with_shape %[[v0]](%[[dim]], %[[c10]]) :
//  CHECK-NEXT:     return %[[v1]] :

// SHAPE-LABEL: @shape_test_simple_result_0
//  SHAPE-SAME: (%[[arg0:.+]]: index {plan.shape_func_arg = {argument = 0 : index, dimension = 0 : index}})
//  SHAPE-NEXT:     %[[c10:.+]] = arith.constant 10 : index
//  SHAPE-NEXT:     return %[[arg0]], %[[c10]] : index, index
// SHAPE-LABEL: @test_simple_get_shapes
//  SHAPE-SAME: (%[[arg0:.+]]: tensor<2xindex, #plan.memory_space<host>> {tensorrt.host_tensor})
//  SHAPE-NEXT:     %[[c0:.+]] = arith.constant 0 : index
//  SHAPE-NEXT:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<2xindex, #plan.memory_space<host>>
//  SHAPE-NEXT:     %[[v0:.+]]:2 = call @shape_test_simple_result_0(%[[extracted]]) : (index) -> (index, index)
//  SHAPE-NEXT:     %[[from_elements:.+]] = tensor.from_elements %[[v0]]#0, %[[v0]]#1 : tensor<2xindex, #plan.memory_space<host>>
//  SHAPE-NEXT:     return %[[from_elements]]


// -----

func.func @test_dynamic_reshape(%arg0: tensor<4xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<4xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @test_dynamic_reshape
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4xf32>, %[[arg1:.+]]: tensor<2xi32>)
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg1]][%[[c0]]] : tensor<2xi32>
//   CHECK-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg1]][%[[c1]]] : tensor<2xi32>
//   CHECK-DAG:     %[[v0:.+]] = plan.with_values %[[arg1]](%[[extracted]], %[[extracted_0]]) : tensor<2xi32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.dynamic_reshape %[[arg0]], %[[v0]] :
//   CHECK-DAG:     %[[v2:.+]] = plan.with_shape %[[v1]](%[[extracted]], %[[extracted_0]])
//   CHECK-DAG:     return %[[v2]] : tensor<?x?xf32>

// SHAPE-LABEL: @shape_test_dynamic_reshape_result_0
//  SHAPE-SAME: (%[[arg0:.+]]: i32 {plan.shape_func_arg = {argument = 1 : index, indices = array<i64: 0>}},
//  SHAPE-SAME:  %[[arg1:.+]]: i32 {plan.shape_func_arg = {argument = 1 : index, indices = array<i64: 1>}})
//       SHAPE:     return %[[arg0]], %[[arg1]] : i32, i32
// SHAPE-LABEL: @test_dynamic_reshape_get_shapes
//  SHAPE-SAME: (%[[arg0:.+]]: tensor<1xindex, #plan.memory_space<host>> {tensorrt.host_tensor}, %[[arg1:.+]]: tensor<2xi32> {tensorrt.host_tensor}) -> (tensor<2xindex, #plan.memory_space<host>> {tensorrt.host_tensor})
//       SHAPE:     %[[c0:.+]] = arith.constant 0 : index
//       SHAPE:     %[[extracted:.+]] = tensor.extract %[[arg1]][%[[c0]]] : tensor<2xi32>
//       SHAPE:     %[[c1:.+]] = arith.constant 1 : index
//       SHAPE:     %[[extracted_0:.+]] = tensor.extract %[[arg1]][%[[c1]]] : tensor<2xi32>
//       SHAPE:     %[[v0:.+]]:2 = call @shape_test_dynamic_reshape_result_0(%[[extracted]], %[[extracted_0]])
//       SHAPE:     %[[v1:.+]] = arith.index_cast %[[v0]]#0 : i32 to index
//       SHAPE:     %[[v2:.+]] = arith.index_cast %[[v0]]#1 : i32 to index
//       SHAPE:     %[[from_elements:.+]] = tensor.from_elements %[[v1]], %[[v2]] : tensor<2xindex, #plan.memory_space<host>>
//       SHAPE:     return %[[from_elements]]

// -----

func.func @test_get_dim_size_max(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<1x1xf32>

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

  %shape = stablehlo.maximum %8, %13 : tensor<2xi32>

  %result = stablehlo.dynamic_broadcast_in_dim %0, %shape, dims = [0, 1]
    : (tensor<1x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>

  return %result : tensor<?x?xf32>
}

// CHECK-LABEL: @test_get_dim_size_max
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<?x?xf32>)
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<1x1xf32>
//   CHECK-DAG:     %[[v12:.+]] = stablehlo.dynamic_broadcast_in_dim %[[v0]], %{{.+}}
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x?xf32>
//   CHECK-DAG:     %[[dim_i32:.+]] = arith.index_cast %[[dim]] : index to i32
//   CHECK-DAG:     %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c1]] : tensor<?x?xf32>
//   CHECK-DAG:     %[[dim_0_i32:.+]] = arith.index_cast %[[dim_0]] : index to i32

//   CHECK-DAG:     %[[dim_1:.+]] = tensor.dim %[[arg1]], %[[c0]] : tensor<?x?xf32>
//   CHECK-DAG:     %[[dim_1_i32:.+]] = arith.index_cast %[[dim_1]] : index to i32
//   CHECK-DAG:     %[[dim_2:.+]] = tensor.dim %[[arg1]], %[[c1]] : tensor<?x?xf32>
//   CHECK-DAG:     %[[dim_2_i32:.+]] = arith.index_cast %[[dim_2]] : index to i32

//   CHECK-DAG:     %[[v13:.+]] = arith.maxsi %[[dim_i32]], %[[dim_1_i32]] : i32
//   CHECK-DAG:     %[[v14:.+]] = arith.maxsi %[[dim_0_i32]], %[[dim_2_i32]] : i32
//   CHECK-DAG:     %[[v15:.+]] = plan.with_shape %[[v12]](%[[v13]], %[[v14]]) :
//   CHECK-DAG:     return %[[v15]] : tensor<?x?xf32>

// SHAPE-LABEL: func.func private @shape_test_get_dim_size_max_result_0
//  SHAPE-SAME: (%[[arg0:.+]]: index {plan.shape_func_arg = {argument = 0 : index, dimension = 0 : index}}, %[[arg1:.+]]: index {plan.shape_func_arg = {argument = 0 : index, dimension = 1 : index}}, %[[arg2:.+]]: index {plan.shape_func_arg = {argument = 1 : index, dimension = 0 : index}}, %[[arg3:.+]]: index {plan.shape_func_arg = {argument = 1 : index, dimension = 1 : index}}) -> (i32, i32) attributes {plan.shapes_func_marker = "plan.shapes_func_marker"} {
//   SHAPE-DAG:     %[[v0:.+]] = arith.index_cast %[[arg0]] : index to i32
//   SHAPE-DAG:     %[[v1:.+]] = arith.index_cast %[[arg1]] : index to i32
//   SHAPE-DAG:     %[[v2:.+]] = arith.index_cast %[[arg2]] : index to i32
//   SHAPE-DAG:     %[[v3:.+]] = arith.index_cast %[[arg3]] : index to i32
//   SHAPE-DAG:     %[[v4:.+]] = arith.maxsi %[[v0]], %[[v2]] : i32
//   SHAPE-DAG:     %[[v5:.+]] = arith.maxsi %[[v1]], %[[v3]] : i32
//       SHAPE:     return %[[v4]], %[[v5]] : i32, i32

// SHAPE-LABEL: func.func @test_get_dim_size_max_get_shapes
//  SHAPE-SAME: (%[[arg0:.+]]: tensor<2xindex, #plan.memory_space<host>> {tensorrt.host_tensor}, %[[arg1:.+]]: tensor<2xindex, #plan.memory_space<host>> {tensorrt.host_tensor}) -> (tensor<2xindex, #plan.memory_space<host>> {tensorrt.host_tensor})
//   SHAPE-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   SHAPE-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<2xindex, #plan.memory_space<host>>
//   SHAPE-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   SHAPE-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c1]]] : tensor<2xindex, #plan.memory_space<host>>
//   SHAPE-DAG:     %[[c0_1:.+]] = arith.constant 0 : index
//   SHAPE-DAG:     %[[extracted_2:.+]] = tensor.extract %[[arg1]][%[[c0_1]]] : tensor<2xindex, #plan.memory_space<host>>
//   SHAPE-DAG:     %[[c1_3:.+]] = arith.constant 1 : index
//   SHAPE-DAG:     %[[extracted_4:.+]] = tensor.extract %[[arg1]][%[[c1_3]]] : tensor<2xindex, #plan.memory_space<host>>
//   SHAPE-DAG:     %[[v0]]:2 = call @shape_test_get_dim_size_max_result_0(%[[extracted]], %[[extracted_0]], %[[extracted_2]], %[[extracted_4]]) : (index, index, index, index) -> (i32, i32)
//   SHAPE-DAG:     %[[v1:.+]] = arith.index_cast %[[v0]]#0 : i32 to index
//   SHAPE-DAG:     %[[v2:.+]] = arith.index_cast %[[v0]]#1 : i32 to index
//   SHAPE-DAG:     %[[from_elements:.+]] = tensor.from_elements %[[v1]], %[[v2]] : tensor<2xindex, #plan.memory_space<host>>
//   SHAPE-DAG:     return %[[from_elements]] : tensor<2xindex, #plan.memory_space<host>>

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %result = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %result: tensor<?x?x?xf32>
}

// CHECK-LABEL: @dot_general
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?xf32>, %[[arg1:.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x?x?xf32>
//   CHECK-DAG:     %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c2]] : tensor<?x?x?xf32>
//   CHECK-DAG:     %[[dim_1:.+]] = tensor.dim %[[arg1]], %[[c2]] : tensor<?x?x?xf32>
//       CHECK:     %[[v0:.+]] = stablehlo.dot_general
//       CHECK:     %[[v1:.+]] = plan.with_shape %[[v0]](%[[dim]], %[[dim_0]], %[[dim_1]]) :
//       CHECK:     return %[[v1]] : tensor<?x?x?xf32>

// -----

func.func @dynamic_pad(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>, %arg4: tensor<1xindex>) -> tensor<?xf32> {
  %result = "stablehlo.dynamic_pad"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<?xf32>, tensor<f32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  return %result: tensor<?xf32>
}

// CHECK-LABEL: @dynamic_pad
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<f32>, %[[arg2:.+]]: tensor<1xindex>, %[[arg3:.+]]: tensor<1xindex>, %[[arg4:.+]]: tensor<1xindex>) -> tensor<?xf32> {
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[arg0_:.+]] = plan.with_shape %[[arg0]]
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg2]][%[[c0]]] : tensor<1xindex>
//   CHECK-DAG:     %[[v0:.+]] = plan.with_values %[[arg2]](%[[extracted]]) : tensor<1xindex>
//   CHECK-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg3]][%[[c0]]] : tensor<1xindex>
//   CHECK-DAG:     %[[v1:.+]] = plan.with_values %[[arg3]](%[[extracted_0]]) : tensor<1xindex>
//   CHECK-DAG:     %[[extracted_1:.+]] = tensor.extract %[[arg4]][%[[c0]]] : tensor<1xindex>
//   CHECK-DAG:     %[[v2:.+]] = plan.with_values %[[arg4]](%[[extracted_1]]) : tensor<1xindex>
//   CHECK-DAG:     %[[v3:.+]] = stablehlo.dynamic_pad %[[arg0_]], %[[arg1]], %[[v0]], %[[v1]], %[[v2]] : (tensor<?xf32>, tensor<f32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?xf32>
//   CHECK-DAG:     %[[v4:.+]] = arith.cmpi slt, %[[dim]], %[[c1]] : index
//   CHECK-DAG:     %[[v5:.+]] = arith.subi %[[dim]], %[[c1]] : index
//   CHECK-DAG:     %[[v6:.+]] = arith.select %[[v4]], %[[c0]], %[[v5]] : index
//   CHECK-DAG:     %[[v7:.+]] = arith.muli %[[extracted_1]], %[[v6]] : index
//   CHECK-DAG:     %[[v8:.+]] = arith.addi %[[v7]], %[[dim]] : index
//   CHECK-DAG:     %[[v9:.+]] = arith.addi %[[v8]], %[[extracted]] : index
//   CHECK-DAG:     %[[v10:.+]] = arith.addi %[[v9]], %[[extracted_0]] : index
//   CHECK-DAG:     %[[v11:.+]] = plan.with_shape %[[v3]](%[[v10]]) :
//   CHECK-DAG:     return %[[v11]] : tensor<?xf32>

// SHAPE-LABEL: @shape_dynamic_pad_result_0
//  SHAPE-SAME: (%[[arg0:.+]]: index {plan.shape_func_arg = {argument = 0 : index, dimension = 0 : index}},
//  SHAPE-SAME:  %[[arg1:.+]]: index {plan.shape_func_arg = {argument = 4 : index, indices = array<i64: 0>}},
//  SHAPE-SAME:  %[[arg2:.+]]: index {plan.shape_func_arg = {argument = 2 : index, indices = array<i64: 0>}},
//  SHAPE-SAME:  %[[arg3:.+]]: index {plan.shape_func_arg = {argument = 3 : index, indices = array<i64: 0>}})
//   SHAPE-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   SHAPE-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   SHAPE-DAG:     %[[v0:.+]] = arith.cmpi slt, %[[arg0]], %[[c1]] : index
//   SHAPE-DAG:     %[[v1:.+]] = arith.subi %[[arg0]], %[[c1]] : index
//   SHAPE-DAG:     %[[v2:.+]] = arith.select %[[v0]], %[[c0]], %[[v1]] : index
//   SHAPE-DAG:     %[[v3:.+]] = arith.muli %[[arg1]], %[[v2]] : index
//   SHAPE-DAG:     %[[v4:.+]] = arith.addi %[[v3]], %[[arg0]] : index
//   SHAPE-DAG:     %[[v5:.+]] = arith.addi %[[v4]], %[[arg2]] : index
//   SHAPE-DAG:     %[[v6:.+]] = arith.addi %[[v5]], %[[arg3]] : index
//   SHAPE-DAG:     return %[[v6]]
// SHAPE-LABEL: @dynamic_pad_get_shapes
//  SHAPE-SAME: (%[[arg0:.+]]: tensor<1xindex, #plan.memory_space<host>> {tensorrt.host_tensor}, %[[arg1:.+]]: tensor<1xindex, #plan.memory_space<host>> {tensorrt.host_tensor},
//  SHAPE-SAME:  %[[arg2:.+]]: tensor<1xindex> {tensorrt.host_tensor}, %[[arg3:.+]]: tensor<1xindex> {tensorrt.host_tensor}, %[[arg4:.+]]: tensor<1xindex> {tensorrt.host_tensor})
//       SHAPE:     %[[c0:.+]] = arith.constant 0 : index
//       SHAPE:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<1xindex, #plan.memory_space<host>>
//       SHAPE:     %[[c0_0:.+]] = arith.constant 0 : index
//       SHAPE:     %[[extracted_1:.+]] = tensor.extract %[[arg4]][%[[c0_0]]] : tensor<1xindex>
//       SHAPE:     %[[c0_2:.+]] = arith.constant 0 : index
//       SHAPE:     %[[extracted_3:.+]] = tensor.extract %[[arg2]][%[[c0_2]]] : tensor<1xindex>
//       SHAPE:     %[[c0_4:.+]] = arith.constant 0 : index
//       SHAPE:     %[[extracted_5:.+]] = tensor.extract %[[arg3]][%[[c0_4]]] : tensor<1xindex>
//       SHAPE:     %[[v0:.+]] = call @shape_dynamic_pad_result_0(%[[extracted]], %[[extracted_1]], %[[extracted_3]], %[[extracted_5]])
//       SHAPE:     %[[from_elements:.+]] = tensor.from_elements %[[v0]] : tensor<1xindex, #plan.memory_space<host>>
//       SHAPE:     return %[[from_elements]] : tensor<1xindex, #plan.memory_space<host>>

// -----

func.func @broadcast(%arg0: tensor<?xi32>) -> tensor<1x2x?xi32> {
  %result = "stablehlo.broadcast"(%arg0) {broadcast_sizes = array<i64: 1, 2>} : (tensor<?xi32>) -> tensor<1x2x?xi32>
  return %result: tensor<1x2x?xi32>
}

// CHECK-LABEL: @broadcast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xi32>)
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?xi32>
//       CHECK:     %[[v0:.+]] = stablehlo.broadcast
//       CHECK:     %[[v1:.+]] = plan.with_shape %[[v0]](%[[c1]], %[[c2]], %[[dim]]) :
//       CHECK:     return %[[v1]] :

// -----


func.func @transpose(%arg0: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %result = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 2>} : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %result: tensor<?x?x?x?xi32>
}

// CHECK-LABEL: @transpose
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c1]] : tensor<?x?x?x?xi32>
//   CHECK-DAG:     %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x?x?x?xi32>
//   CHECK-DAG:     %[[dim_1:.+]] = tensor.dim %[[arg0]], %[[c3]] : tensor<?x?x?x?xi32>
//   CHECK-DAG:     %[[dim_2:.+]] = tensor.dim %[[arg0]], %[[c2]] : tensor<?x?x?x?xi32>
//   CHECK-DAG:     %[[arg0_:.+]] = plan.with_shape %[[arg0]]
//       CHECK:     %[[v0:.+]] = stablehlo.transpose %[[arg0_]], dims = [1, 0, 3, 2]
//       CHECK:     %[[v1:.+]] = plan.with_shape %[[v0]](%[[dim]], %[[dim_0]], %[[dim_1]], %[[dim_2]])
//       CHECK:     return %[[v1]] :

// -----

func.func @dynamic_iota(%arg0: tensor<1xindex>) -> tensor<?xf32> {
  %result = "stablehlo.dynamic_iota"(%arg0) {
    iota_dimension = 0 : i64
  } : (tensor<1xindex>) -> tensor<?xf32>
  func.return %result: tensor<?xf32>
}

// CHECK-LABEL: @dynamic_iota
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xindex>)
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]]
//   CHECK-DAG:     %[[with_val:.+]] = plan.with_values %[[arg0]](%[[extracted]])
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.dynamic_iota %[[with_val]], dim = 0 :
//   CHECK-DAG:     %[[v1:.+]] = plan.with_shape %[[v0]](%[[extracted]])
//   CHECK-DAG:     return %[[v1]] : tensor<?xf32>

// -----

func.func @add_dynamic(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>, %arg2: tensor<2x?x4xf32>) -> tensor<2x?x4xf32> {
  %0 = stablehlo.constant dense<1> : tensor<1xi32>
  %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
  %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
  %3 = stablehlo.constant dense<4> : tensor<1xi32>
  %4 = stablehlo.concatenate %0, %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %5 = stablehlo.dynamic_broadcast_in_dim %arg1, %4, dims = [1, 2] : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<1x?x4xf32>
  %6 = stablehlo.constant dense<2> : tensor<1xi32>
  %7 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
  %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
  %9 = stablehlo.constant dense<4> : tensor<1xi32>
  %10 = stablehlo.concatenate %6, %8, %9, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %11 = stablehlo.dynamic_broadcast_in_dim %5, %10, dims = [0, 1, 2] : (tensor<1x?x4xf32>, tensor<3xi32>) -> tensor<2x?x4xf32>
  %12 = stablehlo.add %11, %arg2 : tensor<2x?x4xf32>
  return %12 : tensor<2x?x4xf32>
}

// CHECK-LABEL: @add_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i64>, %[[arg1:.+]]: tensor<?x4xf32>, %[[arg2:.+]]: tensor<2x?x4xf32>)
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//       CHECK:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][] : tensor<i64>
//       CHECK:     %[[v2:.+]] = arith.trunci %[[extracted_0]] : i64 to i32
//       CHECK:     %[[v15:.+]] = stablehlo.add
//       CHECK:     %[[v16:.+]] = plan.with_shape %[[v15]](%[[c2]], %[[v2]], %[[c4]]) :
//       CHECK:     return %[[v16]] : tensor<2x?x4xf32>

// -----

// Similar to the above `add_dynamic`, except that the shape
// is derived rather than passed as an argument.

func.func @add_dynamic_derive_shape(
            %arg0: tensor<?xf32>,
            %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = stablehlo.constant dense<1> : tensor<1xi32>
  %1 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
  %3 = stablehlo.compare  EQ, %2, %0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  %4 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
  %6 = stablehlo.select %3, %5, %2 : tensor<1xi1>, tensor<1xi32>
  %7 = stablehlo.dynamic_broadcast_in_dim %arg0, %6, dims = [0] : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
  %8 = stablehlo.dynamic_broadcast_in_dim %arg1, %6, dims = [0] : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
  %9 = stablehlo.add %7, %8 : tensor<?xf32>
  return %9 : tensor<?xf32>
}

// CHECK-LABEL: func.func @add_dynamic_derive_shape
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<?xf32>)
//   CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?xf32>
//   CHECK-DAG:     %[[arg0_:.+]] = plan.with_shape %[[arg0]]
//   CHECK-DAG:     %[[dim_0:.+]] = tensor.dim %[[arg1]], %[[c0]] : tensor<?xf32>
//   CHECK-DAG:     %[[arg1_:.+]] = plan.with_shape %[[arg1]]
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.constant dense<1> : tensor<1xi32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.get_dimension_size %[[arg0_]], dim = 0 : (tensor<?xf32>) -> tensor<i32>
//   CHECK-DAG:     %[[v2:.+]] = arith.index_cast %[[dim]] : index to i32
//   CHECK-DAG:     %[[v3:.+]] = plan.with_values %[[v1]](%[[v2]]) : tensor<i32>
//   CHECK-DAG:     %[[v4:.+]] = stablehlo.reshape %[[v3]] : (tensor<i32>) -> tensor<1xi32>
//   CHECK-DAG:     %[[v5:.+]] = plan.with_values %[[v4]](%[[v2]]) : tensor<1xi32>
//   CHECK-DAG:     %[[v6:.+]] = stablehlo.compare  EQ, %[[v5]], %[[v0]] : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
//   CHECK-DAG:     %[[v7:.+]] = arith.cmpi eq, %[[v2]], %[[c1_i32]] : i32
//   CHECK-DAG:     %[[v8:.+]] = plan.with_values %[[v6]](%[[v7]]) : tensor<1xi1>
//   CHECK-DAG:     %[[v9:.+]] = stablehlo.get_dimension_size %[[arg1_]], dim = 0 : (tensor<?xf32>) -> tensor<i32>
//   CHECK-DAG:     %[[v10:.+]] = arith.index_cast %[[dim_0]] : index to i32
//   CHECK-DAG:     %[[v11:.+]] = plan.with_values %[[v9]](%[[v10]]) : tensor<i32>
//   CHECK-DAG:     %[[v12:.+]] = stablehlo.reshape %[[v11]] : (tensor<i32>) -> tensor<1xi32>
//   CHECK-DAG:     %[[v13:.+]] = plan.with_values %[[v12]](%[[v10]]) : tensor<1xi32>
//   CHECK-DAG:     %[[v14:.+]] = stablehlo.select %[[v8]], %[[v13]], %[[v5]] : tensor<1xi1>, tensor<1xi32>
//   CHECK-DAG:     %[[v15:.+]] = arith.select %[[v7]], %[[v10]], %[[v2]] : i32
//   CHECK-DAG:     %[[v16:.+]] = plan.with_values %[[v14]](%[[v15]]) : tensor<1xi32>
//   CHECK-DAG:     %[[v17:.+]] = stablehlo.dynamic_broadcast_in_dim %[[arg0_]], %[[v16]], dims = [0] : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
//   CHECK-DAG:     %[[v19:.+]] = plan.with_shape %[[v17]](%[[v15]]) :
//   CHECK-DAG:     %[[v20:.+]] = stablehlo.dynamic_broadcast_in_dim %[[arg1_]], %[[v16]], dims = [0] : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
//   CHECK-DAG:     %[[v21:.+]] = plan.with_shape %[[v20]](%[[v15]]) :
//   CHECK-DAG:     %[[v22:.+]] = stablehlo.add %[[v19]], %[[v21]] : tensor<?xf32>
//   CHECK-DAG:     %[[v23:.+]] = plan.with_shape %[[v22]](%[[v15]]) :
//   CHECK-DAG:     return %[[v23]] : tensor<?xf32>

// SHAPE-LABEL: func.func private @shape_add_dynamic_derive_shape_result_0
//  SHAPE-SAME: (%[[arg0:.+]]: index {plan.shape_func_arg = {argument = 0 : index, dimension = 0 : index}},
//  SHAPE-SAME:  %[[arg1:.+]]: index {plan.shape_func_arg = {argument = 1 : index, dimension = 0 : index}})
//   SHAPE-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   SHAPE-DAG:     %[[v0:.+]] = arith.index_cast %[[arg0]] : index to i32
//   SHAPE-DAG:     %[[v1:.+]] = arith.cmpi eq, %[[v0]], %[[c1_i32]] : i32
//   SHAPE-DAG:     %[[v2:.+]] = arith.index_cast %[[arg1]] : index to i32
//   SHAPE-DAG:     %[[v3:.+]] = arith.select %[[v1]], %[[v2]], %[[v0]] : i32
//   SHAPE-DAG:     return %[[v3]] : i32

// SHAPE-LABEL: func.func @add_dynamic_derive_shape_get_shapes
//  SHAPE-SAME: (%[[arg0:.+]]: tensor<1xindex, #plan.memory_space<host>> {tensorrt.host_tensor}, %[[arg1:.+]]: tensor<1xindex, #plan.memory_space<host>> {tensorrt.host_tensor}) -> (tensor<1xindex, #plan.memory_space<host>> {tensorrt.host_tensor})
//       SHAPE:     %[[c0:.+]] = arith.constant 0 : index
//       SHAPE:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<1xindex, #plan.memory_space<host>>
//       SHAPE:     %[[c0_0:.+]] = arith.constant 0 : index
//       SHAPE:     %[[extracted_1:.+]] = tensor.extract %[[arg1]][%[[c0_0]]] : tensor<1xindex, #plan.memory_space<host>>
//       SHAPE:     %[[v0:.+]] = call @shape_add_dynamic_derive_shape_result_0(%[[extracted]], %[[extracted_1]]) : (index, index) -> i32
//       SHAPE:     %[[v1]] = arith.index_cast %[[v0]] : i32 to index
//       SHAPE:     %[[from_elements:.+]] = tensor.from_elements %[[v1]] : tensor<1xindex, #plan.memory_space<host>>
//       SHAPE:     return %[[from_elements]] : tensor<1xindex, #plan.memory_space<host>>
//       SHAPE:   }

// -----

func.func @real_dynamic_slice(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>) -> tensor<?xf32> {
  %result = "stablehlo.real_dynamic_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  func.return %result: tensor<?xf32>
}

// CHECK-LABEL: @real_dynamic_slice
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<1xindex>, %[[arg2:.+]]: tensor<1xindex>, %[[arg3:.+]]: tensor<1xindex>)
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg1]][%[[c0]]] : tensor<1xindex>
//   CHECK-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg2]][%[[c0]]] : tensor<1xindex>
//   CHECK-DAG:     %[[extracted_1:.+]] = tensor.extract %[[arg3]][%[[c0]]] : tensor<1xindex>
//   CHECK-DAG:     %[[arg0_:.+]] = plan.with_shape %[[arg0]]
//   CHECK-DAG:     %[[offt:.+]] = plan.with_values %[[arg1]](%[[extracted]]) : tensor<1xindex>
//   CHECK-DAG:     %[[limit:.+]] = plan.with_values %[[arg2]](%[[extracted_0]]) : tensor<1xindex>
//   CHECK-DAG:     %[[stride:.+]] = plan.with_values %[[arg3]](%[[extracted_1]]) : tensor<1xindex>
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.real_dynamic_slice %[[arg0_]], %[[offt]], %[[limit]], %[[stride]]
// This is "floordiv(end-start, step)":
//   CHECK-DAG:     %[[v1:.+]] = arith.subi %[[extracted_0]], %[[extracted]] : index
//   CHECK-DAG:     %[[v2:.+]] = arith.addi %[[extracted_1]], %[[v1]] : index
//   CHECK-DAG:     %[[v3:.+]] = arith.subi %[[v2]], %[[c1]] : index
//   CHECK-DAG:     %[[v4:.+]] = arith.divsi %[[v3]], %[[extracted_1]] : index
//   CHECK-DAG:     %[[v5:.+]] = plan.with_shape %[[v0]](%[[v4]]) :
//   CHECK-DAG:     return %[[v5]]

// -----

func.func @dynamic_reshape_collapse(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<?x?x5x?x7xf32>) -> tensor<?x?x7xf32> {
  %0 = stablehlo.multiply %arg1, %arg2 : tensor<i64>
  %1 = stablehlo.constant dense<5> : tensor<i64>
  %2 = stablehlo.multiply %0, %1 : tensor<i64>
  %3 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
  %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
  %5 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
  %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
  %7 = stablehlo.constant dense<7> : tensor<1xi32>
  %8 = stablehlo.concatenate %4, %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  %9 = stablehlo.dynamic_reshape %arg3, %8 : (tensor<?x?x5x?x7xf32>, tensor<3xi32>) -> tensor<?x?x7xf32>
  return %9 : tensor<?x?x7xf32>
}

// CHECK-LABEL: @dynamic_reshape_collapse
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i64>, %[[arg1:.+]]: tensor<i64>, %[[arg2:.+]]: tensor<i64>, %[[arg3:.+]]: tensor<?x?x5x?x7xf32>)
//   CHECK-DAG:     %[[c7:.+]] = arith.constant 7 : index
//   CHECK-DAG:     %[[c5:.+]] = arith.constant 5 : i64
//   CHECK-DAG:     %[[v9:.+]] = stablehlo.dynamic_reshape
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][] : tensor<i64>
//   CHECK-DAG:     %[[extracted_1:.+]] = tensor.extract %[[arg2]][] : tensor<i64>
//   CHECK-DAG:     %[[extracted_2:.+]] = tensor.extract %[[arg1]][] : tensor<i64>
//   CHECK-DAG:     %[[v13:.+]] = arith.muli %[[extracted_2]], %[[extracted_1]] : i64
//   CHECK-DAG:     %[[v14:.+]] = arith.muli %[[v13]], %[[c5]] : i64
//   CHECK-DAG:     %[[v14_i32:.+]] = arith.trunci %[[v14]] : i64 to i32
//   CHECK-DAG:     %[[extracted_i32:.+]] = arith.trunci %[[extracted]] : i64 to i32
//   CHECK-DAG:     %[[v15:.+]] = plan.with_shape %[[v9]](%[[extracted_i32]], %[[v14_i32]], %[[c7]]) :
//   CHECK-DAG:     return %[[v15]] : tensor<?x?x7xf32>

// -----

func.func @stablehlo_roll_axis_example(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>, %arg2: tensor<i64>) -> tensor<?x4xf32> {
  %0 = stablehlo.constant dense<1> : tensor<1xi32>
  %1 = stablehlo.constant dense<4> : tensor<1xi32>
  %2 = stablehlo.constant dense<false> : tensor<i1>
  %3 = stablehlo.constant dense<4> : tensor<i32>
  %4 = stablehlo.constant dense<0> : tensor<i32>
  %5 = stablehlo.constant dense<2> : tensor<i64>
  %6 = stablehlo.constant dense<1> : tensor<i32>
  %7 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i64>) -> tensor<1xi64>
  %8 = stablehlo.slice %7 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
  %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
  %10 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
  %11 = stablehlo.convert %9 : (tensor<i64>) -> tensor<i32>
  %12 = stablehlo.maximum %10, %6 : tensor<i32>
  %13 = stablehlo.compare  EQ, %12, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %14 = stablehlo.select %13, %6, %12 : tensor<i1>, tensor<i32>
  %15 = stablehlo.remainder %11, %14 : tensor<i32>
  %16 = stablehlo.compare  NE, %15, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %17 = stablehlo.compare  LT, %15, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %18 = stablehlo.compare  LT, %14, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %19 = stablehlo.compare  NE, %17, %18,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %20 = stablehlo.and %19, %16 : tensor<i1>
  %21 = stablehlo.add %15, %14 : tensor<i32>
  %22 = stablehlo.select %20, %21, %15 : tensor<i1>, tensor<i32>
  %23 = stablehlo.concatenate %arg1, %arg1, dim = 0 : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %24 = stablehlo.subtract %10, %22 : tensor<i32>
  %25 = stablehlo.multiply %arg0, %5 : tensor<i64>
  %26 = stablehlo.convert %25 : (tensor<i64>) -> tensor<i32>
  %27 = stablehlo.compare  LT, %24, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %28 = stablehlo.add %24, %26 : tensor<i32>
  %29 = stablehlo.select %27, %28, %24 : tensor<i1>, tensor<i32>
  %30 = stablehlo.add %4, %3 : tensor<i32>
  %31 = stablehlo.select %2, %30, %4 : tensor<i1>, tensor<i32>
  %32 = stablehlo.convert %29 : tensor<i32>
  %33 = stablehlo.reshape %32 : (tensor<i32>) -> tensor<1xi32>
  %34 = stablehlo.convert %31 : tensor<i32>
  %35 = stablehlo.reshape %34 : (tensor<i32>) -> tensor<1xi32>
  %36 = stablehlo.concatenate %33, %35, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %37 = stablehlo.convert %29 : tensor<i32>
  %38 = stablehlo.reshape %37 : (tensor<i32>) -> tensor<1xi32>
  %39 = stablehlo.convert %31 : tensor<i32>
  %40 = stablehlo.reshape %39 : (tensor<i32>) -> tensor<1xi32>
  %41 = stablehlo.concatenate %38, %40, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %42 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
  %43 = stablehlo.reshape %42 : (tensor<i32>) -> tensor<1xi32>
  %44 = stablehlo.concatenate %43, %1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %45 = stablehlo.add %41, %44 : tensor<2xi32>
  %46 = stablehlo.concatenate %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %47 = stablehlo.real_dynamic_slice %23, %36, %45, %46 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
  return %47 : tensor<?x4xf32>
}

// CHECK-LABEL: @stablehlo_roll_axis_example
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i64>, %[[arg1:.+]]: tensor<?x4xf32>, %[[arg2:.+]]: tensor<i64>) -> tensor<?x4xf32> {
//   CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][] : tensor<i64>
//   CHECK-DAG:     %[[v9:.+]] = arith.trunci %[[extracted]] : i64 to i32
//   CHECK-DAG:     %[[arg1_:.+]] = plan.with_shape %[[arg1]]
//   CHECK-DAG:     %[[v55:.+]] = stablehlo.concatenate %[[arg1_]], %[[arg1_]], dim = 0 :
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg1]], %[[c0]] : tensor<?x4xf32>
//   CHECK-DAG:     %[[v56:.+]] = arith.addi %[[dim]], %[[dim]] : index
//   CHECK-DAG:     %[[v57:.+]] = plan.with_shape %[[v55]](%[[v56]], %[[c4]]) :
//   CHECK-DAG:     %[[v99:.+]] = stablehlo.real_dynamic_slice %[[v57]],
//   CHECK-DAG:     %[[v100:.+]] = plan.with_shape %[[v99]](%[[v9]], %[[c4]]) :
//   CHECK-DAG:     return %[[v100]] : tensor<?x4xf32>

// -----

func.func @real_dynamic_slice_add_negative(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = stablehlo.constant dense<-1> : tensor<i64>
  %1 = stablehlo.add %0, %arg0 : tensor<i64>
  %2 = stablehlo.constant dense<0> : tensor<1xi32>
  %3 = stablehlo.constant dense<0> : tensor<1xi32>
  %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %5 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
  %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
  %7 = stablehlo.constant dense<4> : tensor<1xi32>
  %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %9 = stablehlo.constant dense<1> : tensor<1xi32>
  %10 = stablehlo.constant dense<1> : tensor<1xi32>
  %11 = stablehlo.concatenate %9, %10, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %12 = stablehlo.real_dynamic_slice %arg1, %4, %8, %11 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
  return %12 : tensor<?x4xf32>
}

// CHECK-LABEL: func.func @real_dynamic_slice_add_negative
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i64>, %[[arg1:.+]]: tensor<?x4xf32>) -> tensor<?x4xf32> {
//   CHECK-DAG:     %[[cst:.+]] = arith.constant dense<1> : tensor<2xi32>
//   CHECK-DAG:     %[[cst_0:.+]] = arith.constant dense<0> : tensor<2xi32>
//   CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//   CHECK-DAG:     %[[c4_i32:.+]] = arith.constant 4 : i32
//   CHECK-DAG:     %[[cn1_i64:.+]] = arith.constant -1 : i64
//   CHECK-DAG:     %[[c_0:.+]] = stablehlo.constant dense<4> : tensor<1xi32>
//   CHECK-DAG:     %[[c_2:.+]] = stablehlo.constant dense<-1> : tensor<i64>
//   CHECK-DAG:     %[[arg1_:.+]] = plan.with_shape %[[arg1]]
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][] : tensor<i64>
//   CHECK-DAG:     %[[v0:.+]] = plan.with_values %[[arg0]](%[[extracted]]) : tensor<i64>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.add %[[c_2]], %[[v0]] : tensor<i64>
//   CHECK-DAG:     %[[v2:.+]] = arith.addi %[[extracted]], %[[cn1_i64]] : i64
//   CHECK-DAG:     %[[v3:.+]] = plan.with_values %[[v1]](%[[v2]]) : tensor<i64>
//   CHECK-DAG:     %[[v6:.+]] = stablehlo.convert %[[v3]] : (tensor<i64>) -> tensor<i32>
//   CHECK-DAG:     %[[v7:.+]] = arith.trunci %[[v2]] : i64 to i32
//   CHECK-DAG:     %[[v8:.+]] = plan.with_values %[[v6]](%[[v7]]) : tensor<i32>
//   CHECK-DAG:     %[[v9:.+]] = stablehlo.reshape %[[v8]] : (tensor<i32>) -> tensor<1xi32>
//   CHECK-DAG:     %[[v10:.+]] = plan.with_values %[[v9]](%[[v7]]) : tensor<1xi32>
//   CHECK-DAG:     %[[v11:.+]] = stablehlo.concatenate %[[v10]], %[[c_0]], dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v12:.+]] = plan.with_values %[[v11]](%[[v7]], %[[c4_i32]]) : tensor<2xi32>
//   CHECK-DAG:     %[[v15:.+]] = stablehlo.real_dynamic_slice %[[arg1_]], %[[cst_0]], %[[v12]], %[[cst]] : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
//   CHECK-DAG:     %[[v16:.+]] = plan.with_shape %[[v15]](%[[v7]], %[[c4]]) : (tensor<?x4xf32>, i32, index) -> tensor<?x4xf32>
//   CHECK-DAG:     return %[[v16]] : tensor<?x4xf32>

// -----

func.func @real_dynamic_slice_stride_2(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = stablehlo.constant dense<0> : tensor<1xi32>
  %1 = stablehlo.constant dense<0> : tensor<1xi32>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %3 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
  %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
  %5 = stablehlo.constant dense<4> : tensor<1xi32>
  %6 = stablehlo.concatenate %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %7 = stablehlo.constant dense<2> : tensor<1xi32>
  %8 = stablehlo.constant dense<1> : tensor<1xi32>
  %9 = stablehlo.concatenate %7, %8, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %10 = stablehlo.real_dynamic_slice %arg1, %2, %6, %9 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
  return %10 : tensor<?x4xf32>
}

// CHECK-LABEL: func.func @real_dynamic_slice_stride_2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<i64>, %[[arg1:.+]]: tensor<?x4xf32>) -> tensor<?x4xf32> {
//   CHECK-DAG:     %[[cst:.+]] = arith.constant dense<[2, 1]> : tensor<2xi32>
//   CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//   CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c2_i32:.+]] = arith.constant 2 : i32
//   CHECK-DAG:     %[[cst_0:.+]] = arith.constant dense<0> : tensor<2xi32>
//   CHECK-DAG:     %[[c4_i32:.+]] = arith.constant 4 : i32
//   CHECK-DAG:     %[[c_1:.+]] = stablehlo.constant dense<4> : tensor<1xi32>
//   CHECK-DAG:     %[[arg1_:.+]] = plan.with_shape %[[arg1]]
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][] : tensor<i64>
//   CHECK-DAG:     %[[v0:.+]] = plan.with_values %[[arg0]](%[[extracted]]) : tensor<i64>
//   CHECK-DAG:     %[[v3:.+]] = stablehlo.convert %[[v0]] : (tensor<i64>) -> tensor<i32>
//   CHECK-DAG:     %[[v4:.+]] = arith.trunci %[[extracted]] : i64 to i32
//   CHECK-DAG:     %[[v5:.+]] = plan.with_values %[[v3]](%[[v4]]) : tensor<i32>
//   CHECK-DAG:     %[[v6:.+]] = stablehlo.reshape %[[v5]] : (tensor<i32>) -> tensor<1xi32>
//   CHECK-DAG:     %[[v7:.+]] = plan.with_values %[[v6]](%[[v4]]) : tensor<1xi32>
//   CHECK-DAG:     %[[v8:.+]] = stablehlo.concatenate %[[v7]], %[[c_1]], dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
//   CHECK-DAG:     %[[v9:.+]] = plan.with_values %[[v8]](%[[v4]], %[[c4_i32]]) : tensor<2xi32>
//   CHECK-DAG:     %[[v12:.+]] = stablehlo.real_dynamic_slice %[[arg1_]], %[[cst_0]], %[[v9]], %[[cst]] : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
//   CHECK-DAG:     %[[v13:.+]] = arith.addi %[[v4]], %[[c1_i32]] : i32
//   CHECK-DAG:     %[[v14:.+]] = arith.divsi %[[v13]], %[[c2_i32]] : i32
//   CHECK-DAG:     %[[v15:.+]] = plan.with_shape %[[v12]](%[[v14]], %[[c4]]) : (tensor<?x4xf32>, i32, index) -> tensor<?x4xf32>
//   CHECK-DAG:     return %[[v15]] : tensor<?x4xf32>

// -----

func.func @gather(%operand : tensor<3x4x2xi32>, %start_indices : tensor<?x3x2xi64>) -> tensor<?x3x2x2xi32> {
  %result = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
      slice_sizes = array<i64: 1, 2, 2>,
      indices_are_sorted = false
  } : (tensor<3x4x2xi32>, tensor<?x3x2xi64>) -> tensor<?x3x2x2xi32>
  return %result : tensor<?x3x2x2xi32>
}

// CHECK-LABEL: @gather
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x4x2xi32>, %[[arg1:.+]]: tensor<?x3x2xi64>)
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg1]], %[[c0]] : tensor<?x3x2xi64>
//   CHECK-DAG:     %[[v0:.+]] = "stablehlo.gather"
//   CHECK-DAG:     %[[v1:.+]] = plan.with_shape %[[v0]](%[[dim]], %[[c3]], %[[c2]], %[[c2]])
//       CHECK:     return %[[v1]]

// -----

func.func @reverse(%a : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "stablehlo.reverse"(%a) {
    dimensions = array<i64: 1, 3>
  } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @reverse
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
//   CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x?x?x?xf32>
//   CHECK-DAG:     %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c1]] : tensor<?x?x?x?xf32>
//   CHECK-DAG:     %[[dim_1:.+]] = tensor.dim %[[arg0]], %[[c2]] : tensor<?x?x?x?xf32>
//   CHECK-DAG:     %[[dim_2:.+]] = tensor.dim %[[arg0]], %[[c3]] : tensor<?x?x?x?xf32>
//   CHECK-DAG:     %[[arg0_:.+]] = plan.with_shape %[[arg0]]
//       CHECK:     %[[v0:.+]] = stablehlo.reverse %[[arg0_]]
//       CHECK:     %[[v1:.+]] = plan.with_shape %[[v0]](%[[dim]], %[[dim_0]], %[[dim_1]], %[[dim_2]])
//       CHECK:     return %[[v1]] : tensor<?x?x?x?xf32>

// -----

func.func @dot_general2(%arg0: tensor<?x?x?xf32>,
                  %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @dot_general2
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?xf32>, %[[arg1:.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c1]] : tensor<?x?x?xf32>
//   CHECK-DAG:     %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x?x?xf32>
//   CHECK-DAG:     %[[arg0_:.+]] = plan.with_shape %[[arg0]]
//   CHECK-DAG:     %[[dim_1:.+]] = tensor.dim %[[arg1]], %[[c0]] : tensor<?x?x?xf32>
//   CHECK-DAG:     %[[arg1_:.+]] = plan.with_shape %[[arg1]]
//       CHECK:     %[[v0:.+]] = stablehlo.dot_general
//       CHECK:     %[[v1:.+]] = plan.with_shape %[[v0]](%[[dim]], %[[dim_0]], %[[dim_1]]) :
//       CHECK:     return %[[v1]] : tensor<?x?x?xf32>

// -----

#profile0 = #tensorrt.shape_profile<min=[1], opt=[2], max=[4]>
#profile1 = #tensorrt.shape_profile<min=[2], opt=[4], max=[6]>

func.func @test_loop_concat(
                            %arg0: tensor<1xf32>,
                            %arg1: tensor<1xi32> {tensorrt.value_bounds = #profile0},
                            %arg2: tensor<?xf32> {tensorrt.shape_profile  = #profile1},
                            %arg3: tensor<1024xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %lb = tensor.dim %arg2, %c0 : tensor<?xf32>
  %ub = tensor.extract %arg1[%c0] : tensor<1xi32>
  %ub_index = arith.index_cast %ub : i32 to index
  %0 = scf.for %i = %lb to %ub_index step %c1 iter_args(%iter_arg = %arg2) -> tensor<?xf32> {
    %accum = "stablehlo.concatenate"(%iter_arg, %arg0) {dimension = 0} : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
    scf.yield %accum : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @test_loop_concat
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xf32>, %[[arg1:[a-zA-Z0-9]+]]: tensor<1xi32>{{.*}}, %[[arg2:[a-zA-Z0-9]+]]: tensor<?xf32>{{.*}}, %[[arg3:[a-zA-Z0-9]+]]: tensor<1024xf32>)
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[dim:.+]] = tensor.dim %[[arg2]], %[[c0]] : tensor<?xf32>
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg1]][%[[c0]]] : tensor<1xi32>
//       CHECK:     %[[v0:.+]] = arith.index_cast %[[extracted]] : i32 to index
//       CHECK:     %[[v1:.+]] = scf.for {{.+}} iter_args(%[[arg5:.+]] =
//       CHECK:       %[[v3:.+]] = stablehlo.concatenate %[[arg5]], %[[arg0]]
//       CHECK:       %[[dim_1:.+]] = tensor.dim %[[arg5]], %[[c0]] : tensor<?xf32>
//       CHECK:       %[[v4:.+]] = arith.addi %[[dim_1]], %[[c1]] : index
//       CHECK:       %[[v5:.+]] = plan.with_shape %[[v3]](%[[v4]]) :
//       CHECK:       scf.yield %[[v5]] : tensor<?xf32>
//       CHECK:     %[[dim_0:.+]] = tensor.dim %[[v1]], %[[c0]] : tensor<?xf32>
//       CHECK:     %[[v2:.+]] = plan.with_shape %[[v1]](%[[dim_0]]) :
//       CHECK:     return %[[v2]] : tensor<?xf32>


// Currently, we can't generate end-to-end shape function when loops like the above
// are involved.
// SHAPE-LABEL: @test_loop_concat
// SHAPE-NOT: @test_loop_concat_get_shapes

// -----

#profile0 = #tensorrt.shape_profile<min=[1], opt=[2], max=[4]>

func.func @bufferization_aloc_tensor(%arg0: tensor<1xindex>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.extract %arg0[%c0] : tensor<1xindex>
  %0 = bufferization.alloc_tensor(%dim) : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func.func @bufferization_aloc_tensor
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xindex>) -> tensor<?xf32>
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<1xindex>
//       CHECK:     %[[v0:.+]] = bufferization.alloc_tensor(%[[extracted]]) : tensor<?xf32>
//       CHECK:     %[[v1:.+]] = plan.with_shape %[[v0]](%[[extracted]]) :
//       CHECK:     return %[[v1]] : tensor<?xf32>

// SHAPE-LABEL: @bufferization_aloc_tensor
// SHAPE-LABEL: @shape_bufferization_aloc_tensor_result_0
//  SHAPE-SAME: (%[[arg0:.+]]: index {
//  SHAPE-NEXT:     return %[[arg0]] :

// SHAPE-LABEL: @bufferization_aloc_tensor_get_shapes
//  SHAPE-SAME: (%[[arg0:.+]]: tensor<1xindex> {tensorrt.host_tensor}) -> (tensor<1xindex, #plan.memory_space<host>> {tensorrt.host_tensor})
//       SHAPE:     %[[c0:.+]] = arith.constant 0 : index
//       SHAPE:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<1xindex>
//       SHAPE:     %[[v0:.+]] = call @shape_bufferization_aloc_tensor_result_0(%[[extracted]]) :
//       SHAPE:     %[[from_elements:.+]] = tensor.from_elements %[[v0]] :
//       SHAPE:     return %[[from_elements]] : tensor<1xindex, #plan.memory_space<host>>

// -----

#profile0 = #tensorrt.shape_profile<min=[1], opt=[2], max=[4]>

func.func @bufferization_materialize_in_destination(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = bufferization.alloc_tensor(%dim) : tensor<?xf32>
  %1 = bufferization.materialize_in_destination %arg0 in %0 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @bufferization_materialize_in_destination
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>) -> tensor<?xf32> {
//   CHECK-DAG:     %[[arg0_:.+]] = plan.with_shape %[[arg0]]
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?xf32>
//   CHECK-DAG:     %[[v0:.+]] = bufferization.alloc_tensor(%[[dim]]) : tensor<?xf32>
//   CHECK-DAG:     %[[v1:.+]] = plan.with_shape %[[v0]](%[[dim]]) :
//   CHECK-DAG:     %[[v2:.+]] = bufferization.materialize_in_destination %[[arg0_]] in %[[v1]] :
//   CHECK-DAG:     %[[v3:.+]] = plan.with_shape %[[v2]](%[[dim]]) :
//       CHECK:     return %[[v3]] : tensor<?xf32>


// -----

func.func @zero_slice_slice(%arg4: tensor<1xi32>,
                %arg6: tensor<1xi32> {tensorrt.value_bounds = #tensorrt.shape_profile<min = [1], opt = [1], max = [1]>},
                %arg7: tensor<1xi32> {tensorrt.value_bounds = #tensorrt.shape_profile<min = [1], opt = [1], max = [1]>},
                %arg8: tensor<1xi32> {tensorrt.value_bounds = #tensorrt.shape_profile<min = [1], opt = [1], max = [1]>},
                %arg9: tensor<1xi32> {tensorrt.shape_profile = #tensorrt.shape_profile<min = [1], opt = [1], max = [1]>}) -> tensor<?xi32> {
  %1 = stablehlo.real_dynamic_slice %arg9, %arg6, %arg7, %arg8 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2 = stablehlo.concatenate %arg4, %1, dim = 0 : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
  return %2 : tensor<?xi32>
}

// CHECK-LABEL: func.func @zero_slice_slice
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>, %[[arg1:.+]]: tensor<1xi32> {plan.value_bounds = #plan.bounds<value, dense<1> : tensor<1xi32>, dense<1> : tensor<1xi32>>}, %[[arg2:.+]]: tensor<1xi32> {plan.value_bounds = #plan.bounds<value, dense<1> : tensor<1xi32>, dense<1> : tensor<1xi32>>}, %[[arg3:.+]]: tensor<1xi32> {plan.value_bounds = #plan.bounds<value, dense<1> : tensor<1xi32>, dense<1> : tensor<1xi32>>}, %[[arg4:.+]]: tensor<1xi32> {plan.shape_profile = #plan.bounds<shape, [1], [1]>})
//   CHECK-DAG:     %[[cst:.+]] = arith.constant dense<1> : tensor<1xi32>
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg3]][%[[c0]]] : tensor<1xi32>
//   CHECK-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg2]][%[[c0]]] : tensor<1xi32>
//   CHECK-DAG:     %[[extracted_1:.+]] = tensor.extract %[[arg1]][%[[c0]]] : tensor<1xi32>
//   CHECK-DAG:     %[[v3:.+]] = stablehlo.real_dynamic_slice %[[arg4]], %[[cst]], %[[cst]], %[[cst]] : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
//   CHECK-DAG:     %[[v4:.+]] = arith.subi %[[extracted_0]], %[[extracted_1]] : i32
//   CHECK-DAG:     %[[v5:.+]] = arith.addi %[[extracted]], %[[v4]] : i32
//   CHECK-DAG:     %[[v6:.+]] = arith.subi %[[v5]], %[[c1_i32]] : i32
//   CHECK-DAG:     %[[v7:.+]] = arith.divsi %[[v6]], %[[extracted]] : i32
//   CHECK-DAG:     %[[v8:.+]] = arith.index_cast %[[v7]] : i32 to index
//   CHECK-DAG:     %[[v9:.+]] = plan.with_shape %[[v3]](%[[v7]]) : (tensor<?xi32>, i32) -> tensor<?xi32>
//   CHECK-DAG:     %[[v10:.+]] = stablehlo.concatenate %[[arg0]], %[[v9]], dim = 0 : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
//   CHECK-DAG:     %[[v11:.+]] = arith.addi %[[v8]], %[[c1]] : index
//   CHECK-DAG:     %[[v12:.+]] = plan.with_shape %[[v10]](%[[v11]]) : (tensor<?xi32>, index) -> tensor<?xi32>
//   CHECK-DAG:     return %[[v12]] : tensor<?xi32>


// -----

#profile0 = #tensorrt.shape_profile<min=[1], opt=[2], max=[4]>
#profile1 = #tensorrt.shape_profile<min=[1, 1], opt=[2, 2], max=[2, 2]>

func.func @shape_calc(%arg0: tensor<?xf32> {tensorrt.shape_profile = #profile0},
                      %arg1: tensor<2xi32> {tensorrt.value_bounds = #profile1},
                      %arg2: tensor<2xi32> {tensorrt.value_bounds = #profile1}) -> tensor<?x?xf32> {
  %0 = stablehlo.add %arg1, %arg2 : tensor<2xi32>
  %1 = stablehlo.multiply %0, %0 : tensor<2xi32>
  %2 = stablehlo.dynamic_reshape %arg0, %1 : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @shape_calc
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32> {{.+}}, %[[arg1:.+]]: tensor<2xi32> {{.+}}, %[[arg2:.+]]: tensor<2xi32> {{.+}})
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[arg0_:.+]] = plan.with_shape %[[arg0]]
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg2]][%[[c0]]] : tensor<2xi32>
//   CHECK-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg2]][%[[c1]]] : tensor<2xi32>
//   CHECK-DAG:     %[[v0:.+]] = plan.with_values %[[arg2]](%[[extracted]], %[[extracted_0]]) : tensor<2xi32>
//   CHECK-DAG:     %[[extracted_1:.+]] = tensor.extract %[[arg1]][%[[c0]]] : tensor<2xi32>
//   CHECK-DAG:     %[[extracted_2:.+]] = tensor.extract %[[arg1]][%[[c1]]] : tensor<2xi32>
//   CHECK-DAG:     %[[v1:.+]] = plan.with_values %[[arg1]](%[[extracted_1]], %[[extracted_2]]) : tensor<2xi32>
//   CHECK-DAG:     %[[v2:.+]] = stablehlo.add %[[v1]], %[[v0]] : tensor<2xi32>
//   CHECK-DAG:     %[[v3:.+]] = arith.addi %[[extracted_1]], %[[extracted]] : i32
//   CHECK-DAG:     %[[v4:.+]] = arith.addi %[[extracted_2]], %[[extracted_0]] : i32
//   CHECK-DAG:     %[[v5:.+]] = plan.with_values %[[v2]](%[[v3]], %[[v4]]) : tensor<2xi32>
//   CHECK-DAG:     %[[v6:.+]] = stablehlo.multiply %[[v5]], %[[v5]] : tensor<2xi32>
//   CHECK-DAG:     %[[v7:.+]] = arith.muli %[[v3]], %[[v3]] : i32
//   CHECK-DAG:     %[[v8:.+]] = arith.muli %[[v4]], %[[v4]] : i32
//   CHECK-DAG:     %[[v9:.+]] = plan.with_values %[[v6]](%[[v7]], %[[v8]]) : tensor<2xi32>
//   CHECK-DAG:     %[[v10:.+]] = stablehlo.dynamic_reshape %[[arg0_]], %[[v9]] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
//   CHECK-DAG:     %[[v11:.+]] = plan.with_shape %[[v10]](%[[v7]], %[[v8]]) : (tensor<?x?xf32>, i32, i32) -> tensor<?x?xf32>
//   CHECK-DAG:     return %[[v11]] :

// -----

func.func @slice_with_repetetive_max(%arg0: tensor<2xi32>, %arg1: tensor<1xf32>) -> tensor<?xf32> {
  %0 = stablehlo.slice %arg0 [0:1:1] : (tensor<2xi32>) -> tensor<1xi32>
  %1 = stablehlo.slice %arg0 [1:2:1] : (tensor<2xi32>) -> tensor<1xi32>
  %2 = stablehlo.maximum %0, %1 : tensor<1xi32>
  %3 = stablehlo.maximum %2, %1 : tensor<1xi32>
  %4 = stablehlo.dynamic_broadcast_in_dim %arg1, %3, dims=[0] : (tensor<1xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %4 : tensor<?xf32>
}

// CHECK-LABEL: func.func @slice_with_repetetive_max
//  CHECK-SAME: (%[[arg0:.+]]: tensor<2xi32>, %[[arg1:.+]]: tensor<1xf32>) -> tensor<?xf32> {
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<2xi32>
//   CHECK-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c1]]] : tensor<2xi32>
//   CHECK-DAG:     %[[v0:.+]] = plan.with_values %[[arg0]](%[[extracted]], %[[extracted_0]]) : tensor<2xi32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.slice %[[v0]] [0:1] : (tensor<2xi32>) -> tensor<1xi32>
//   CHECK-DAG:     %[[v2:.+]] = plan.with_values %[[v1]](%[[extracted]]) : tensor<1xi32>
//   CHECK-DAG:     %[[v3:.+]] = stablehlo.slice %[[v0]] [1:2] : (tensor<2xi32>) -> tensor<1xi32>
//   CHECK-DAG:     %[[v4:.+]] = plan.with_values %[[v3]](%[[extracted_0]]) : tensor<1xi32>
//   CHECK-DAG:     %[[v5:.+]] = stablehlo.maximum %[[v2]], %[[v4]] : tensor<1xi32>
//   CHECK-DAG:     %[[v6:.+]] = arith.maxsi %[[extracted]], %[[extracted_0]] : i32
//   CHECK-DAG:     %[[v7:.+]] = plan.with_values %[[v5]](%[[v6]]) : tensor<1xi32>
//   CHECK-DAG:     %[[v8:.+]] = stablehlo.maximum %[[v7]], %[[v4]] : tensor<1xi32>
//   CHECK-DAG:     %[[v9:.+]] = plan.with_values %[[v8]](%[[v6]]) : tensor<1xi32>
//   CHECK-DAG:     %[[v10:.+]] = stablehlo.dynamic_broadcast_in_dim %[[arg1]], %[[v9]], dims = [0] : (tensor<1xf32>, tensor<1xi32>) -> tensor<?xf32>
//   CHECK-DAG:     %[[v11:.+]] = plan.with_shape %[[v10]](%[[v6]]) : (tensor<?xf32>, i32) -> tensor<?xf32>
//   CHECK-DAG:     return %[[v11]] : tensor<?xf32>

// SHAPE-LABEL: func.func private @shape_slice_with_repetetive_max_result_0
//  SHAPE-SAME: (%[[arg0:.+]]: i32 {plan.shape_func_arg = {argument = 0 : index, indices = array<i64: 0>}}, %[[arg1:.+]]: i32 {plan.shape_func_arg = {argument = 0 : index, indices = array<i64: 1>}})
//   SHAPE-DAG:     %[[v0:.+]] = arith.maxsi %[[arg0]], %[[arg1]] : i32
//   SHAPE-DAG:     return %[[v0]] : i32
// SHAPE-LABEL: func.func @slice_with_repetetive_max_get_shapes
//  SHAPE-SAME: (%[[arg0:.+]]: tensor<2xi32> {tensorrt.host_tensor}, %[[arg1:.+]]: tensor<1xindex, #plan.memory_space<host>> {tensorrt.host_tensor}) -> (tensor<1xindex, #plan.memory_space<host>> {tensorrt.host_tensor})
//   SHAPE-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   SHAPE-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<2xi32>
//   SHAPE-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   SHAPE-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c1]]] : tensor<2xi32>
//   SHAPE-DAG:     %[[v0:.+]] = call @shape_slice_with_repetetive_max_result_0(%[[extracted]], %[[extracted_0]]) : (i32, i32) -> i32
//   SHAPE-DAG:     %[[v1:.+]] = arith.index_cast %[[v0]] : i32 to index
//   SHAPE-DAG:     %[[from_elements:.+]] = tensor.from_elements %[[v1]] : tensor<1xindex, #plan.memory_space<host>>
//   SHAPE-DAG:     return %[[from_elements]] : tensor<1xindex, #plan.memory_space<host>>

// -----

func.func @extract_from_slice_strided(%arg0: tensor<6x7x8xi32>, %arg1: tensor<1xf32>) -> tensor<?x?x?x?x?x?xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = stablehlo.slice %arg0 [1:6:2, 2:7:3, 3:4:1] : (tensor<6x7x8xi32>) -> tensor<3x2x1xi32>
  %2 = stablehlo.reshape %0 : (tensor<3x2x1xi32>) -> tensor<6xi32>
  %3 = stablehlo.dynamic_broadcast_in_dim %arg1, %2, dims=[0] : (tensor<1xf32>, tensor<6xi32>) -> tensor<?x?x?x?x?x?xf32>
  return %3 : tensor<?x?x?x?x?x?xf32>
}

// CHECK-LABEL: func.func @extract_from_slice_strided
//  CHECK-SAME: (%[[arg0:.+]]: tensor<6x7x8xi32>, %[[arg1:.+]]: tensor<1xf32>)
//   CHECK-DAG:     %[[c5:.+]] = arith.constant 5 : index
//   CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.slice %[[arg0]] [1:6:2, 2:7:3, 3:4] : (tensor<6x7x8xi32>) -> tensor<3x2x1xi32>
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c1]], %[[c2]], %[[c3]]] : tensor<6x7x8xi32>
//   CHECK-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c1]], %[[c5]], %[[c3]]] : tensor<6x7x8xi32>
//   CHECK-DAG:     %[[extracted_1:.+]] = tensor.extract %[[arg0]][%[[c3]], %[[c2]], %[[c3]]] : tensor<6x7x8xi32>
//   CHECK-DAG:     %[[extracted_2:.+]] = tensor.extract %[[arg0]][%[[c3]], %[[c5]], %[[c3]]] : tensor<6x7x8xi32>
//   CHECK-DAG:     %[[extracted_3:.+]] = tensor.extract %[[arg0]][%[[c5]], %[[c2]], %[[c3]]] : tensor<6x7x8xi32>
//   CHECK-DAG:     %[[extracted_4:.+]] = tensor.extract %[[arg0]][%[[c5]], %[[c5]], %[[c3]]] : tensor<6x7x8xi32>
//   CHECK-DAG:     %[[v4:.+]] = stablehlo.dynamic_broadcast_in_dim
//   CHECK-DAG:     %[[v5:.+]] = plan.with_shape %[[v4]](%[[extracted]], %[[extracted_0]], %[[extracted_1]], %[[extracted_2]], %[[extracted_3]], %[[extracted_4]]) :
//   CHECK-DAG:     return %[[v5]] :

// -----

// Checks that we can propgate through `tensor.cast`, which may be leftover from input preprocessing pipeline.

func.func @simplify_extract_from_cast(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>) -> tensor<?xi32> {
  %c_0 = stablehlo.constant dense<0> : tensor<1xi32>
  %c_2 = stablehlo.constant dense<0> : tensor<1xi32>
  %3 = stablehlo.minimum %arg0, %arg1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  %cast = tensor.cast %3 : tensor<?xi32> to tensor<1xi32>
  %4 = stablehlo.real_dynamic_slice %arg2, %c_0, %cast, %c_2 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %4 : tensor<?xi32>
}

// CHECK-LABEL: func.func @simplify_extract_from_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi32>, %[[arg1:.+]]: tensor<1xi32>, %[[arg2:.+]]: tensor<1xi32>) -> tensor<?xi32> {
//   CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c:.+]] = stablehlo.constant dense<0> : tensor<1xi32>
//   CHECK-DAG:     %[[v0:.+]] = stablehlo.minimum %[[arg0]], %[[arg1]] : (tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
//   CHECK-DAG:     %[[v1:.+]] = plan.with_shape %[[v0]](%[[c1]]) : (tensor<?xi32>, index) -> tensor<?xi32>
//   CHECK-DAG:     %[[cast:.+]] = tensor.cast %[[v1]] : tensor<?xi32> to tensor<1xi32>
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[v1]][%[[c0]]] : tensor<?xi32>
//   CHECK-DAG:     %[[v2:.+]] = plan.with_values %[[cast]](%[[extracted]]) : tensor<1xi32>
//   CHECK-DAG:     %[[v3:.+]] = stablehlo.real_dynamic_slice %[[arg2]], %[[c]], %[[v2]], %[[c]] :
//   CHECK-DAG:     %[[v4:.+]] = arith.subi %[[extracted]], %[[c1_i32]] : i32
//   CHECK-DAG:     %[[v5:.+]] = arith.divsi %[[v4]], %[[c0_i32]] : i32
//   CHECK-DAG:     %[[v6:.+]] = plan.with_shape %[[v3]](%[[v5]]) : (tensor<?xi32>, i32) -> tensor<?xi32>
//   CHECK-DAG:     return %[[v6]] : tensor<?xi32>

// SHAPE-LABEL: func.func private @shape_simplify_extract_from_cast_result_0
//  SHAPE-SAME: (%[[arg0:.+]]: i32) -> i32
//   SHAPE-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
//   SHAPE-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
//   SHAPE-DAG:     %[[v0:.+]] = arith.subi %[[arg0]], %[[c1_i32]] : i32
//   SHAPE-DAG:     %[[v1:.+]] = arith.divsi %[[v0]], %[[c0_i32]] : i32
//   SHAPE-DAG:     return %[[v1]] : i32

// -----

func.func @direct_return_arg(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  return %arg0: tensor<?xf32>
}

// CHECK-LABEL: @direct_return_arg
//  CHECK-SAME: %[[arg0:.+]]: tensor<?xf32>
//   CHECK-DAG: %[[v0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[dim:.+]] = tensor.dim %[[arg0]], %[[v0]]
//   CHECK-DAG: %[[arg0_:.+]] = plan.with_shape %[[arg0]](%[[dim]])
//   CHECK-DAG: return %[[arg0_]]

// -----

func.func @conv_input_dynamnic(
        %arg0: tensor<?x?x?x?xf32>,
        %arg1: tensor<256x256x1x1xf32>) -> (tensor<?x?x?x?xf32>) {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<?x?x?x?xf32>, tensor<256x256x1x1xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @conv_input_dynamnic
// CHECK-SAME: (%[[arg0:.+]]: tensor<?x?x?x?xf32>, %[[arg1:.+:]] tensor<256x256x1x1xf32>)
//  CHECK-DAG: %[[c256:.+]] = arith.constant 256 : index
//  CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
//  CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
//  CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
//  CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
//  CHECK-DAG: %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x?x?x?xf32>
//  CHECK-DAG: %[[dim_0:.+]] = tensor.dim %[[arg0]], %[[c1]] : tensor<?x?x?x?xf32>
//  CHECK-DAG: %[[dim_1:.+]] = tensor.dim %[[arg0]], %[[c2]] : tensor<?x?x?x?xf32>
//  CHECK-DAG: %[[dim_2:.+]] = tensor.dim %[[arg0]], %[[c3]] : tensor<?x?x?x?xf32>
//  CHECK-DAG: %[[v0:.+]] = plan.with_shape %[[arg0]](%[[dim]], %[[dim_0]], %[[dim_1]], %[[dim_2]]) :
//  CHECK-DAG: %[[v1:.+]] = stablehlo.convolution
//  CHECK-DAG: %[[v2:.+]] = arith.maxsi %[[dim_1]], %[[c0]] : index
//  CHECK-DAG: %[[v3:.+]] = arith.maxsi %[[dim_2]], %[[c0]] : index
//  CHECK-DAG: %[[v4:.+]] = plan.with_shape %[[v1]](%[[dim]], %[[c256]], %[[v2]], %[[v3]]) :
//  CHECK-DAG: return %[[v4]]

// -----

#profile = #tensorrt.shape_profile<min=[128, 128], opt=[256, 128], max=[512, 128]>

func.func @refine_based_on_profile(%arg0: tensor<?x?xi32> {tensorrt.shape_profile = #profile})
     -> tensor<?x?xi32> {
  %result = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<?x?xi32>) -> tensor<?x?xi32>
  return %result: tensor<?x?xi32>
}

// CHECK-LABEL: func.func @refine_based_on_profile
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xi32>
//   CHECK-DAG:     %[[c128:.+]] = arith.constant 128 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?x?xi32>
//   CHECK-DAG:     %[[v0:.+]] = plan.with_shape %[[arg0]](%[[dim]], %[[c128]]) : (tensor<?x?xi32>, index, index) -> tensor<?x?xi32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.transpose %[[v0]], dims = [1, 0] :
//   CHECK-DAG:     %[[v2:.+]] = plan.with_shape %[[v1]](%[[c128]], %[[dim]]) : (tensor<?x?xi32>, index, index) -> tensor<?x?xi32>
//   CHECK-DAG:     return %[[v2]] : tensor<?x?xi32>
