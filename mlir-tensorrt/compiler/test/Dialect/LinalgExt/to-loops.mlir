// RUN: mlir-tensorrt-opt %s -convert-to-loops -split-input-file -cse -canonicalize | FileCheck %s

func.func @linalg_generic_to_loops(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1, 0)>,
      affine_map<(d0, d1) -> (d1, d0)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg2 : tensor<4x4xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %0 = arith.addf %arg3, %arg4 : f32
    %1 = arith.addf %0, %arg5 : f32
    linalg.yield %1 : f32
  } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @linalg_generic_to_loops
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x4xf32>, %[[arg1:.+]]: tensor<4x4xf32>, %[[arg2:.+]]: tensor<4x4xf32>)
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0
//   CHECK-DAG:     %[[c4:.+]] = arith.constant 4
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1
//       CHECK:     %[[v0:.+]] = scf.for %[[arg3:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[arg4:.+]] = %[[arg2]])
//       CHECK:       %[[v1:.+]] = scf.for %[[arg5:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[arg6:.+]] = %[[arg4]])
//   CHECK-DAG:         %[[extracted:.+]] = tensor.extract %[[arg0]][%[[arg3]], %[[arg5]]]
//   CHECK-DAG:         %[[extracted_0:.+]] = tensor.extract %[[arg1]][%[[arg5]], %[[c0]]]
//   CHECK-DAG:         %[[extracted_1:.+]] = tensor.extract %[[arg6]][%[[arg5]], %[[arg3]]]
//   CHECK-DAG:         %[[v2:.+]] = arith.addf %[[extracted]], %[[extracted_0]]
//   CHECK-DAG:         %[[v3:.+]] = arith.addf %[[v2]], %[[extracted_1]]
//   CHECK-DAG:         %[[inserted:.+]] = tensor.insert %[[v3]] into %[[arg6]][%[[arg5]], %[[arg3]]]
//       CHECK:         scf.yield %[[inserted]]
//       CHECK:       scf.yield %[[v1]]
//       CHECK:     return %[[v0]]

// -----

func.func @contraction(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>)
    -> tensor<4xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1, d0)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg2 : tensor<4xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %0 = arith.addf %arg3, %arg4 : f32
    %1 = arith.addf %0, %arg5 : f32
    linalg.yield %1 : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @c
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x4xf32>, %[[arg1:.+]]: tensor<4x4xf32>, %[[arg2:.+]]: tensor<4xf32>)
//       CHECK:     %[[c0:.+]] = arith.constant 0
//       CHECK:     %[[c4:.+]] = arith.constant 4
//       CHECK:     %[[c1:.+]] = arith.constant 1
//       CHECK:     %[[v0:.+]] = scf.for %[[arg3:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[arg4:.+]] = %[[arg2]])
//       CHECK:       %[[v1:.+]] = scf.for %[[arg5:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[arg6:.+]] = %[[arg4]])
//       CHECK:         %[[extracted:.+]] = tensor.extract %[[arg0]][%[[arg3]], %[[arg5]]]
//       CHECK:         %[[extracted_0:.+]] = tensor.extract %[[arg1]][%[[arg5]], %[[arg3]]]
//       CHECK:         %[[extracted_1:.+]] = tensor.extract %[[arg6]][%[[arg3]]]
//       CHECK:         %[[v2:.+]] = arith.addf %[[extracted]], %[[extracted_0]]
//       CHECK:         %[[v3:.+]] = arith.addf %[[v2]], %[[extracted_1]]
//       CHECK:         %[[inserted:.+]] = tensor.insert %[[v3]] into %[[arg6]][%[[arg3]]]
//       CHECK:         scf.yield %[[inserted]]
//       CHECK:       scf.yield %[[v1]]
//       CHECK:     return %[[v0]]

// -----

func.func @linalg_index_op(%arg0: tensor<4x4xindex>) -> tensor<4x4xindex> {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %3 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> ()>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%c1 : index) outs(%arg0 : tensor<4x4xindex>) {
    ^bb0(%arg1: index, %arg2: index):
      %l1 = linalg.index 0 : index
      %l2 = linalg.index 1 : index
      %0 = arith.muli %l1, %c4 : index
      %1 = arith.addi %l2, %arg1 : index
      %2 = arith.addi %0, %1 : index
      linalg.yield %2 : index
  } -> tensor<4x4xindex>
  return %3 : tensor<4x4xindex>
}

// CHECK-LABEL: func.func @linalg_index_op
//  CHECK-SAME: (%[[arg0:.+]]: tensor<4x4xindex>)
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[v0:.+]] = scf.for %[[arg1:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[arg2:.+]] = %[[arg0]])
//       CHECK:       %[[v1:.+]] = scf.for %[[arg3:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[arg4:.+]] = %[[arg2]])
//   CHECK-DAG:         %[[v2:.+]] = arith.muli %[[arg1]], %[[c4]]
//   CHECK-DAG:         %[[v3:.+]] = arith.addi %[[arg3]], %[[c1]]
//   CHECK-DAG:         %[[v4:.+]] = arith.addi %[[v2]], %[[v3]]
//   CHECK-DAG:         %[[inserted:.+]] = tensor.insert %[[v4]] into %[[arg4]][%[[arg1]], %[[arg3]]] : tensor<4x4xindex>
//   CHECK-DAG:         scf.yield %[[inserted]] : tensor<4x4xindex>
//       CHECK:       scf.yield %[[v1]]
//       CHECK:     return %[[v0]]

// -----

func.func @linalg_map(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg.map {arith.addf} ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg2: tensor<4x4xf32>)
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @linalg_map
// CHECK-COUNT-2: scf.for
// CHECK-COUNT-2: tensor.extract
//         CHECK: arith.addf
//         CHECK: tensor.insert
// CHECK-COUNT-2: scf.yield

// -----

// CHECK-LABEL: func.func @regression_dynamic_fill_and_reduce
// CHECK-SAME: (%[[arg0:.+]]: tensor<?x?xf32>, %[[arg1:.+]]: tensor<?xf32>)
func.func @regression_dynamic_fill_and_reduce(
              %arg0: tensor<?x?xf32>,
              %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[fill:.+]] = arith.constant 0.0{{.*}} : f32
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[dim:.+]] = tensor.dim %[[arg0]], %[[c0]]
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  // CHECK-DAG: %[[empty:.+]] = tensor.empty(%[[dim]])
  %0 = tensor.empty(%dim) : tensor<?xf32>
  // CHECK: %[[v1:.+]] = scf.for %[[iter:.+]] = %[[c0]] to %[[dim]] step %[[c1]] iter_args(%[[carry:.+]] = %[[empty]])
  // CHECK:   %[[inserted:.+]] = tensor.insert %[[fill]] into %[[carry]][%[[iter]]]
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
  outs(%0 : tensor<?xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  } -> tensor<?xf32>
  // CHECK: %[[dim_1:.+]] = tensor.dim %[[arg0]], %[[c1]]
  // CHECK: %[[outer:.+]] = scf.for {{.*}} to %[[dim]] step %[[c1]] iter_args(%[[carry0:.+]] = %[[v1]])
  // CHECK:  %[[inner:.+]] = scf.for {{.*}} to %[[dim_1]] step %[[c1]] iter_args(%[[carry1:.+]] = %[[carry0]])
  // CHECK:  scf.yield %{{.*}}
  // CHECK: scf.yield %[[inner]]
  // CHECK: return %[[outer]]
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map
    <(d0, d1) -> (d1)>,
    affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%1 : tensor<?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<?xf32>
  return %2 : tensor<?xf32>
}
