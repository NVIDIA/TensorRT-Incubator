// RUN: mlir-tensorrt-opt %s -split-input-file -plan-materialize-shape-calculations -plan-create-shape-funcs=abi-version=0 | FileCheck %s --check-prefix=SHAPE
// RUN: mlir-tensorrt-opt %s -split-input-file -plan-materialize-shape-calculations | FileCheck %s


func.func @composite(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = stablehlo.composite "foo.bar" %arg0, %arg1 {
    decomposition = @dynamic_reshape
  } : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

func.func private @dynamic_reshape(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @composite
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<2xi32>)
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?xf32>
//   CHECK-DAG:     %[[v0:.+]] = plan.with_shape %[[arg0]](%[[dim]]) :
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.composite "foo.bar" %[[v0]], %[[arg1]]
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg1]][%[[c0]]] : tensor<2xi32>
//   CHECK-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg1]][%[[c1]]] : tensor<2xi32>
//   CHECK-DAG:     %[[v2:.+]] = plan.with_shape %[[v1]](%[[extracted]], %[[extracted_0]]) : (tensor<?x?xf32>, i32, i32) -> tensor<?x?xf32>
//   CHECK-DAG:     return %[[v2]] : tensor<?x?xf32>

// -----

func.func @stablehlo_composite_static_shapes(%arg0: tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8> {
  %cst = stablehlo.constant dense_resource<__elided__> : tensor<1x3x200x200xf32>
  %0 = stablehlo.add %arg0, %cst : (tensor<?x3x?x?xf32>, tensor<1x3x200x200xf32>) -> tensor<1x3x200x200xf32>
  %1 = stablehlo.composite "tensorrt.pt_q" %0 {composite_attributes = {axis = -1 : i32, scale = dense<8.000000e-01> : tensor<f32>}, decomposition = @pt_q} : (tensor<1x3x200x200xf32>) -> tensor<?x3x?x?xi8>
  return %1 : tensor<?x3x?x?xi8>
}
func.func private @pt_q(%arg0: tensor<1x3x200x200xf32>) -> tensor<?x3x?x?xi8> {
  %0 = stablehlo.convert %arg0 : (tensor<1x3x200x200xf32>) -> tensor<?x3x?x?xi8>
  return %0 : tensor<?x3x?x?xi8>
}

// CHECK-LABEL: stablehlo_composite_static_shapes
//  CHECK-SAME: %[[arg0:.+]]: tensor<?x3x?x?xf32>
//   CHECK-DAG: %[[v0:.+]] = arith.constant 200 : index
//   CHECK-DAG: %[[v1:.+]] = arith.constant 3 : index
//   CHECK-DAG: %[[v2:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[v3:.+]] = stablehlo.constant
//   CHECK-DAG: %[[arg0_:.+]] = plan.with_shape %[[arg0]]
//   CHECK-DAG: %[[v4:.+]] = stablehlo.add %[[arg0_]], %[[v3]]
//   CHECK-DAG: %[[v5:.+]] = stablehlo.composite "tensorrt.pt_q" %[[v4]]
//   CHECK-DAG: %[[v6:.+]] = plan.with_shape %[[v5]](%[[v2]], %[[v1]], %[[v0]], %[[v0]])
//  CHECK-NEXT: return %[[v6]]

// -----

func.func @stablehlo_composite_dynamic_shapes(%arg0: tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8> {
    %0 = stablehlo.composite "tensorrt.pt_q" %arg0 {composite_attributes = {axis = -1 : i32, scale = dense<8.000000e-01> : tensor<f32>}, decomposition = @pt_q} : (tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8>
    return %0 : tensor<?x3x?x?xi8>
}
func.func private @pt_q(%arg0: tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8> attributes {plan.decomposition} {
  %0 = stablehlo.convert %arg0 : (tensor<?x3x?x?xf32>) -> tensor<?x3x?x?xi8>
  return %0 : tensor<?x3x?x?xi8>
}

// CHECK-LABEL: stablehlo_composite_dynamic_shapes
//  CHECK-SAME: %[[arg0:.+]]: tensor<?x3x?x?xf32>
//   CHECK-DAG: %[[v0:.+]] = arith.constant 3 : index
//   CHECK-DAG: %[[v1:.+]] = arith.constant 2 : index
//   CHECK-DAG: %[[v2:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[arg0_:.+]] = plan.with_shape %[[arg0]]
//   CHECK-DAG: %[[v3:.+]] = stablehlo.composite "tensorrt.pt_q" %[[arg0_]]
//   CHECK-DAG: %[[v4:.+]] = tensor.dim %[[arg0]], %[[v2]]
//   CHECK-DAG: %[[v5:.+]] = tensor.dim %[[arg0]], %[[v1]]
//   CHECK-DAG: %[[v6:.+]] = tensor.dim %[[arg0]], %[[v0]]
//   CHECK-DAG: %[[v7:.+]] = plan.with_shape %[[v3]](%[[v4]], %[[v0]], %[[v5]], %[[v6]])
//  CHECK-NEXT: return %[[v7]]

// SHAPE-LABEL: func.func @stablehlo_composite_dynamic_shapes
// SHAPE-LABEL: func.func private @shape_stablehlo_composite_dynamic_shapes_result_0
//  SHAPE-SAME: (%[[arg0:.+]]: index {plan.shape_func_arg = {argument = 0 : index, dimension = 0 : index}},
//  SHAPE-SAME:  %[[arg1:.+]]: index {plan.shape_func_arg = {argument = 0 : index, dimension = 2 : index}},
//  SHAPE-SAME:  %[[arg2:.+]]: index {plan.shape_func_arg = {argument = 0 : index, dimension = 3 : index}})
//  SHAPE-SAME:  -> (index, index, index, index) attributes {plan.shapes_func_marker} {
//       SHAPE:     %[[c3:.+]] = arith.constant 3 : index
//       SHAPE:     return %[[arg0]], %[[c3]], %[[arg1]], %[[arg2]] : index, index, index, index
//       SHAPE:   }
// SHAPE-LABEL: func.func @stablehlo_composite_dynamic_shapes_get_shapes
//  SHAPE-SAME: (%[[arg0:.+]]: tensor<4xindex, #plan.memory_space<host>>)
//   SHAPE-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   SHAPE-DAG:     %[[extracted:.+]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<4xindex, #plan.memory_space<host>>
//   SHAPE-DAG:     %[[c2:.+]] = arith.constant 2 : index
//   SHAPE-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg0]][%[[c2]]] : tensor<4xindex, #plan.memory_space<host>>
//   SHAPE-DAG:     %[[c3:.+]] = arith.constant 3 : index
//   SHAPE-DAG:     %[[extracted_1:.+]] = tensor.extract %[[arg0]][%[[c3]]] : tensor<4xindex, #plan.memory_space<host>>
//   SHAPE-DAG:     %[[v0:.+]]:4 = call @shape_stablehlo_composite_dynamic_shapes_result_0(%[[extracted]], %[[extracted_0]], %[[extracted_1]]) :
//   SHAPE-DAG:     %[[from_elements:.+]] = tensor.from_elements %[[v0]]#0, %[[v0]]#1, %[[v0]]#2, %[[v0]]#3 :
//   SHAPE-DAG:     return %[[from_elements]] : tensor<4xindex, #plan.memory_space<host>>

// -----

// Deomonstrates that extra re-materialization of intermediate results is required
// when the result cannot be determined from inputs. This can/will be eliminated if
// the composite is replaced with the decomposition and CSE is performed.
func.func @composite_data_dependent(%arg0: tensor<?xf32>, %arg1: tensor<2x2xi32>) -> tensor<?x?xf32> {
  %0 = stablehlo.composite "foo.bar" %arg0, %arg1, %arg1 {
    decomposition = @data_dependent
  } : (tensor<?xf32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

func.func private @data_dependent(%arg0: tensor<?xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<2x2xi32>) -> tensor<?x?xf32> {
  %1 = stablehlo.dot %arg1, %arg2 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  %2 = stablehlo.slice %1[0:1, 0:2] : (tensor<2x2xi32>) -> tensor<1x2xi32>
  %3 = stablehlo.reshape %2 : (tensor<1x2xi32>) -> tensor<2xi32>
  %0 = stablehlo.dynamic_reshape %arg0, %3 : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @composite_data_dependent
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?xf32>, %[[arg1:.+]]: tensor<2x2xi32>) -> tensor<?x?xf32> {
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]] : tensor<?xf32>
//   CHECK-DAG:     %[[v0:.+]] = plan.with_shape %[[arg0]](%[[dim]]) : (tensor<?xf32>, index) -> tensor<?xf32>
//   CHECK-DAG:     %[[v1:.+]] = stablehlo.composite "foo.bar" %[[v0]], %[[arg1]], %[[arg1]]
//   CHECK-DAG:     %[[extracted:.+]] = tensor.extract %[[arg1]][%[[c0]], %[[c0]]] : tensor<2x2xi32>
//   CHECK-DAG:     %[[extracted_0:.+]] = tensor.extract %[[arg1]][%[[c0]], %[[c1]]] : tensor<2x2xi32>
//   CHECK-DAG:     %[[extracted_1:.+]] = tensor.extract %[[arg1]][%[[c1]], %[[c0]]] : tensor<2x2xi32>
//   CHECK-DAG:     %[[extracted_2:.+]] = tensor.extract %[[arg1]][%[[c1]], %[[c1]]] : tensor<2x2xi32>
//   CHECK-DAG:     %[[v2:.+]] = plan.with_values %[[arg1]](%[[extracted]], %[[extracted_0]], %[[extracted_1]], %[[extracted_2]]) : tensor<2x2xi32>
//   CHECK-DAG:     %[[v3:.+]] = stablehlo.dot %[[v2]], %[[v2]] : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
//   CHECK-DAG:     %[[extracted_3:.+]] = tensor.extract %[[v3]][%[[c0]], %[[c0]]] : tensor<2x2xi32>
//   CHECK-DAG:     %[[extracted_4:.+]] = tensor.extract %[[v3]][%[[c0]], %[[c1]]] : tensor<2x2xi32>
//   CHECK-DAG:     %[[v4:.+]] = plan.with_shape %[[v1]](%[[extracted_3]], %[[extracted_4]]) : (tensor<?x?xf32>, i32, i32) -> tensor<?x?xf32>
//   CHECK-DAG:     return %[[v4]] : tensor<?x?xf32>
