// RUN: kernel-opt %s --transform-interpreter -split-input-file  | FileCheck %s

func.func @scatter1(%arg0: tensor<1xi64>, %arg1: tensor<128x1x1215xf32> , %arg2: tensor<128x1xf32> ) -> tensor<128x1x1215xf32> {
  %1 = kernel.scatter updates(%arg2  : tensor<128x1xf32>) into(%arg1  : tensor<128x1x1215xf32>) at(%arg0 : tensor<1xi64>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %2 = arith.minimumf %arg3, %arg4 : f32
    kernel.yield %2 : f32
  } {
    update_window_dims = array<i64: 0, 1>,
    inserted_window_dims = array<i64: 2>,
    scatter_dims_to_operand_dims = array<i64: 2>,
    unique_indices,
    index_vector_dim = 0
  } : tensor<128x1x1215xf32>
  return %1 : tensor<128x1x1215xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %1 = transform.kernel.lower_to_loops %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
// CHECK-LABEL: func.func @scatter1
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi64>, %[[arg1:.+]]: tensor<128x1x1215xf32>, %[[arg2:.+]]: tensor<128x1xf32>)
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c128:.+]] = arith.constant 128 : index
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[v0:.+]] = scf.for %[[arg3:.+]] = %[[c0]] to %[[c128]] step %[[c1]] iter_args(%[[arg4:.+]] = %[[arg1]]) -> (tensor<128x1x1215xf32>) {
//       CHECK:       %[[c0_0:.+]] = arith.constant 0 : index
//       CHECK:       %[[c1_1:.+]] = arith.constant 1 : index
//       CHECK:       %[[c1_2:.+]] = arith.constant 1 : index
//       CHECK:       %[[v1:.+]] = scf.for %[[arg5:.+]] = %[[c0_0]] to %[[c1_1]] step %[[c1_2]] iter_args(%[[arg6:.+]] = %[[arg4]]) -> (tensor<128x1x1215xf32>)
//   CHECK-DAG:         %[[extracted:.+]] = tensor.extract %[[arg2]][%[[arg3]], %[[arg5]]]
//   CHECK-DAG:         %[[c0_3:.+]] = arith.constant 0 : index
//   CHECK-DAG:         %[[extracted_4:.+]] = tensor.extract %[[arg0]][%[[c0_3]]]
//   CHECK-DAG:         %[[v2:.+]] = arith.index_cast %[[extracted_4]] : i64 to index
//   CHECK-DAG:         %[[v3:.+]] = affine.apply #[[$map]](%[[c0_3]], %[[c0_3]], %[[arg3]])
//   CHECK-DAG:         %[[v4:.+]] = affine.apply #[[$map]](%[[c0_3]], %[[c0_3]], %[[arg5]])
//   CHECK-DAG:         %[[v5:.+]] = affine.apply #[[$map]](%[[v2]], %[[c0_3]], %[[c0_3]])
//   CHECK-DAG:         %[[true:.+]] = arith.constant true
//   CHECK-DAG:         %[[c0_5:.+]] = arith.constant 0 : index
//   CHECK-DAG:         %[[c1215:.+]] = arith.constant 1215 : index
//   CHECK-DAG:         %[[v6:.+]] = arith.cmpi slt, %[[v5]], %[[c1215]] : index
//   CHECK-DAG:         %[[v7:.+]] = arith.cmpi sge, %[[v5]], %[[c0_5]] : index
//   CHECK-DAG:         %[[v8:.+]] = arith.andi %[[v6]], %[[v7]] : i1
//   CHECK-DAG:         %[[v9:.+]] = arith.andi %[[v8]], %[[true]] : i1
//   CHECK-DAG:         %[[v10:.+]] = scf.if %[[v9]]
//       CHECK:           %[[extracted_6:.+]] = tensor.extract %[[arg6]][%[[v3]], %[[v4]], %[[v5]]]
//       CHECK:           %[[v11:.+]] = arith.minimumf %[[extracted_6]], %[[extracted]] : f32
//       CHECK:           %[[inserted:.+]] = tensor.insert %[[v11]] into %[[arg6]][%[[v3]], %[[v4]], %[[v5]]]
//       CHECK:           scf.yield %[[inserted]] : tensor<128x1x1215xf32>
//       CHECK:         else
//       CHECK:           scf.yield %[[arg6]] : tensor<128x1x1215xf32>
//       CHECK:         scf.yield %[[v10]] : tensor<128x1x1215xf32>
//       CHECK:       scf.yield %[[v1]]
//       CHECK:     return %[[v0]]


// -----

func.func @scatter_update_scalar(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                            %arg2: tensor<1xi32>) -> tensor<3xi32> {
  %0 = kernel.scatter updates(%arg2 : tensor<1xi32>) into (%arg0 : tensor<3xi32>) at (%arg1 : tensor<1x1xi32>) {
  ^bb0(%arg3: i32, %arg4: i32):
    kernel.yield %arg4 : i32
  } {
      update_window_dims = array<i64>,
      inserted_window_dims = array<i64: 0>,
      scatter_dims_to_operand_dims = array<i64: 0>,
      index_vector_dim = 1
  } : tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %1 = transform.kernel.lower_to_loops %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
// CHECK-LABEL: func.func @scatter_update_scalar
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3xi32>, %[[arg1:.+]]: tensor<1x1xi32>, %[[arg2:.+]]: tensor<1xi32>)
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[c1_0:.+]] = arith.constant 1 : index
//       CHECK:     %[[v0:.+]] = scf.for %[[arg3:.+]] = %[[c0]] to %[[c1]] step %[[c1_0]] iter_args(%[[arg4:.+]] = %[[arg0]])
//   CHECK-DAG:       %[[extracted:.+]] = tensor.extract %[[arg2]][%[[arg3]]]
//   CHECK-DAG:       %[[c0_1:.+]] = arith.constant 0 : index
//   CHECK-DAG:       %[[extracted_2:.+]] = tensor.extract %[[arg1]][%[[arg3]], %[[c0_1]]]
//   CHECK-DAG:       %[[v1:.+]] = arith.index_cast %[[extracted_2]] : i32 to index
//   CHECK-DAG:       %[[v2:.+]] = affine.apply #[[$map]](%[[v1]], %[[c0_1]], %[[c0_1]])
//   CHECK-DAG:       %[[true:.+]] = arith.constant true
//   CHECK-DAG:       %[[c0_3:.+]] = arith.constant 0 : index
//   CHECK-DAG:       %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:       %[[v3:.+]] = arith.cmpi slt, %[[v2]], %[[c3]] : index
//   CHECK-DAG:       %[[v4:.+]] = arith.cmpi sge, %[[v2]], %[[c0_3]] : index
//   CHECK-DAG:       %[[v5:.+]] = arith.andi %[[v3]], %[[v4]] : i1
//   CHECK-DAG:       %[[v6:.+]] = arith.andi %[[v5]], %[[true]] : i1
//   CHECK-DAG:       %[[v7:.+]] = scf.if %[[v6]]
//       CHECK:         %[[extracted_4:.+]] = tensor.extract %[[arg4]][%[[v2]]] : tensor<3xi32>
//       CHECK:         %[[inserted:.+]] = tensor.insert %[[extracted]] into %[[arg4]][%[[v2]]] : tensor<3xi32>
//       CHECK:         scf.yield %[[inserted]] : tensor<3xi32>
//       CHECK:       else
//       CHECK:         scf.yield %[[arg4]] : tensor<3xi32>
//       CHECK:       scf.yield %[[v7]]
//       CHECK:     return %[[v0]]

// -----

func.func @test_scatter_no_batch(%input0: tensor<200x100x300xf32>,
                                 %input1: tensor<200x100x300xf32>,
                                 %indices: tensor<10x2xi64>,
                                 %updates0: tensor<10x300xf32>,
                                 %updates1: tensor<10x300xf32>) -> (tensor<200x100x300xf32>, tensor<200x100x300xf32>) {
  %0:2 = kernel.scatter updates(%updates0, %updates1 : tensor<10x300xf32>, tensor<10x300xf32>)
    into (%input0, %input1 : tensor<200x100x300xf32>, tensor<200x100x300xf32>)
    at (%indices : tensor<10x2xi64>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
    %2 = arith.addf %arg3, %arg4 : f32
    %3 = arith.addf %arg5, %arg6 : f32
    kernel.yield %2, %3 : f32, f32
  } {
    update_window_dims = array<i64: 1>,
    inserted_window_dims = array<i64: 0, 1>,
    scatter_dims_to_operand_dims = array<i64: 0, 1>,
    index_vector_dim = 1,
    unique_indices
  } : tensor<200x100x300xf32>, tensor<200x100x300xf32>
  return %0#0, %0#1 : tensor<200x100x300xf32>, tensor<200x100x300xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %1 = transform.kernel.lower_to_loops %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
// CHECK-LABEL: func.func @test_scatter_no_batch
//  CHECK-SAME: (%[[arg0:.+]]: tensor<200x100x300xf32>, %[[arg1:.+]]: tensor<200x100x300xf32>, %[[arg2:.+]]: tensor<10x2xi64>, %[[arg3:.+]]: tensor<10x300xf32>, %[[arg4:.+]]: tensor<10x300xf32>) -> (tensor<200x100x300xf32>, tensor<200x100x300xf32>) {
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c10:.+]] = arith.constant 10 : index
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[v0]]:2 = scf.for %[[arg5:.+]] = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[arg6:.+]] = %[[arg0:.+]], %[[arg7:.+]] = %[[arg1]]) -> (tensor<200x100x300xf32>, tensor<200x100x300xf32>) {
//       CHECK:       %[[c0_0:.+]] = arith.constant 0 : index
//       CHECK:       %[[c300:.+]] = arith.constant 300 : index
//       CHECK:       %[[c1_1:.+]] = arith.constant 1 : index
//       CHECK:       %[[v1]]:2 = scf.for %[[arg8:.+]] = %[[c0_0]] to %[[c300]] step %[[c1_1]] iter_args(%[[arg9:.+]] = %[[arg6:.+]], %[[arg10:.+]] = %[[arg7]]) -> (tensor<200x100x300xf32>, tensor<200x100x300xf32>) {
//   CHECK-DAG:         %[[extracted:.+]] = tensor.extract %[[arg3]][%[[arg5]], %[[arg8]]]
//   CHECK-DAG:         %[[extracted_2:.+]] = tensor.extract %[[arg4]][%[[arg5]], %[[arg8]]]
//   CHECK-DAG:         %[[c0_3:.+]] = arith.constant 0 : index
//   CHECK-DAG:         %[[extracted_4:.+]] = tensor.extract %[[arg2]][%[[arg5]], %[[c0_3]]]
//   CHECK-DAG:         %[[v2:.+]] = arith.index_cast %[[extracted_4]] : i64 to index
//   CHECK-DAG:         %[[c1_5:.+]] = arith.constant 1 : index
//   CHECK-DAG:         %[[extracted_6:.+]] = tensor.extract %[[arg2]][%[[arg5]], %[[c1_5]]]
//   CHECK-DAG:         %[[v3:.+]] = arith.index_cast %[[extracted_6]] : i64 to index
//   CHECK-DAG:         %[[v4:.+]] = affine.apply #[[$map]](%[[v2]], %[[c0_3]], %[[c0_3]])
//   CHECK-DAG:         %[[v5:.+]] = affine.apply #[[$map]](%[[v3]], %[[c0_3]], %[[c0_3]])
//   CHECK-DAG:         %[[v6:.+]] = affine.apply #[[$map]](%[[c0_3]], %[[c0_3]], %[[arg8]])
//   CHECK-DAG:         %[[true:.+]] = arith.constant true
//   CHECK-DAG:         %[[c0_7:.+]] = arith.constant 0 : index
//   CHECK-DAG:         %[[c200:.+]] = arith.constant 200 : index
//   CHECK-DAG:         %[[v7:.+]] = arith.cmpi slt, %[[v4]], %[[c200]] : index
//   CHECK-DAG:         %[[v8:.+]] = arith.cmpi sge, %[[v4]], %[[c0_7]] : index
//   CHECK-DAG:         %[[v9:.+]] = arith.andi %[[v7]], %[[v8]] : i1
//   CHECK-DAG:         %[[v10:.+]] = arith.andi %[[v9]], %[[true]] : i1
//   CHECK-DAG:         %[[c100:.+]] = arith.constant 100 : index
//   CHECK-DAG:         %[[v11:.+]] = arith.cmpi slt, %[[v5]], %[[c100]] : index
//   CHECK-DAG:         %[[v12:.+]] = arith.cmpi sge, %[[v5]], %[[c0_7]] : index
//   CHECK-DAG:         %[[v13:.+]] = arith.andi %[[v11]], %[[v12]] : i1
//   CHECK-DAG:         %[[v14:.+]] = arith.andi %[[v13]], %[[v10]] : i1
//   CHECK-DAG:         %[[v15:.+]]:2 = scf.if %[[v14]]
//   CHECK-DAG:           %[[extracted_8:.+]] = tensor.extract %[[arg9]][%[[v4]], %[[v5]], %[[v6]]]
//   CHECK-DAG:           %[[extracted_9:.+]] = tensor.extract %[[arg10]][%[[v4]], %[[v5]], %[[v6]]] :
//   CHECK-DAG:           %[[v16:.+]] = arith.addf %[[extracted_8]], %[[extracted]] : f32
//   CHECK-DAG:           %[[v17:.+]] = arith.addf %[[extracted_9]], %[[extracted_2]] : f32
//   CHECK-DAG:           %[[inserted:.+]] = tensor.insert %[[v16]] into %[[arg9]][%[[v4]], %[[v5]], %[[v6]]]
//   CHECK-DAG:           %[[inserted_10:.+]] = tensor.insert %[[v17]] into %[[arg10]][%[[v4]], %[[v5]], %[[v6]]]
//       CHECK:           scf.yield %[[inserted]], %[[inserted_10]]
//       CHECK:          else
//       CHECK:           scf.yield %[[arg9]], %[[arg10]] : tensor<200x100x300xf32>, tensor<200x100x300xf32>
//       CHECK:         scf.yield %[[v15]]#0, %[[v15]]#1
//       CHECK:       scf.yield %[[v1]]#0, %[[v1]]#1
//       CHECK:     return %[[v0]]#0, %[[v0]]#1


// -----

func.func @test_scatter_batch(%input_tensor: tensor<15x200x100x300xf32>,
                              %scatter_indices: tensor<15x10x2xi32>,
                              %updates: tensor<15x10x300xf32>) -> tensor<15x200x100x300xf32> {
  %0 = kernel.scatter updates(%updates : tensor<15x10x300xf32>) into (%input_tensor : tensor<15x200x100x300xf32>) at (%scatter_indices : tensor<15x10x2xi32>) {
    ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 2>,
      inserted_window_dims = array<i64: 1, 2>,
      input_batching_dims = array<i64: 0>,
      scatter_indices_batching_dims = array<i64: 0>,
      scatter_dims_to_operand_dims = array<i64: 1, 2>,
      index_vector_dim = 2,
      indices_are_sorted,
      unique_indices
  } : tensor<15x200x100x300xf32>
  func.return %0 : tensor<15x200x100x300xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %1 = transform.kernel.lower_to_loops %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
// CHECK-LABEL: func.func @test_scatter_batch
//  CHECK-SAME: (%[[arg0:.+]]: tensor<15x200x100x300xf32>, %[[arg1:.+]]: tensor<15x10x2xi32>, %[[arg2:.+]]: tensor<15x10x300xf32>)
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c15:.+]] = arith.constant 15 : index
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[v0:.+]] = scf.for %[[arg3:.+]] = %[[c0]] to %[[c15]] step %[[c1]] iter_args(%[[arg4:.+]] = %[[arg0]]) -> (tensor<15x200x100x300xf32>) {
//       CHECK:       %[[c0_0:.+]] = arith.constant 0 : index
//       CHECK:       %[[c10:.+]] = arith.constant 10 : index
//       CHECK:       %[[c1_1:.+]] = arith.constant 1 : index
//       CHECK:       %[[v1:.+]] = scf.for %[[arg5:.+]] = %[[c0_0]] to %[[c10]] step %[[c1_1]] iter_args(%[[arg6:.+]] = %[[arg4]]) -> (tensor<15x200x100x300xf32>) {
//       CHECK:         %[[c0_2:.+]] = arith.constant 0 : index
//       CHECK:         %[[c300:.+]] = arith.constant 300 : index
//       CHECK:         %[[c1_3:.+]] = arith.constant 1 : index
//       CHECK:         %[[v2:.+]] = scf.for %[[arg7:.+]] = %[[c0_2]] to %[[c300]] step %[[c1_3]] iter_args(%[[arg8:.+]] = %[[arg6]]) -> (tensor<15x200x100x300xf32>) {
//   CHECK-DAG:           %[[extracted:.+]] = tensor.extract %[[arg2]][%[[arg3]], %[[arg5]], %[[arg7]]]
//   CHECK-DAG:           %[[c0_4:.+]] = arith.constant 0 : index
//   CHECK-DAG:           %[[extracted_5:.+]] = tensor.extract %[[arg1]][%[[arg3]], %[[arg5]], %[[c0_4]]]
//   CHECK-DAG:           %[[v3:.+]] = arith.index_cast %[[extracted_5]] : i32 to index
//   CHECK-DAG:           %[[c1_6:.+]] = arith.constant 1 : index
//   CHECK-DAG:           %[[extracted_7:.+]] = tensor.extract %[[arg1]][%[[arg3]], %[[arg5]], %[[c1_6]]]
//   CHECK-DAG:           %[[v4:.+]] = arith.index_cast %[[extracted_7]] : i32 to index
//   CHECK-DAG:           %[[v5:.+]] = affine.apply #[[$map]](%[[c0_4]], %[[arg3]], %[[c0_4]])
//   CHECK-DAG:           %[[v6:.+]] = affine.apply #[[$map]](%[[v3]], %[[c0_4]], %[[c0_4]])
//   CHECK-DAG:           %[[v7:.+]] = affine.apply #[[$map]](%[[v4]], %[[c0_4]], %[[c0_4]])
//   CHECK-DAG:           %[[v8:.+]] = affine.apply #[[$map]](%[[c0_4]], %[[c0_4]], %[[arg7]])
//   CHECK-DAG:           %[[true:.+]] = arith.constant true
//   CHECK-DAG:           %[[c0_8:.+]] = arith.constant 0 : index
//   CHECK-DAG:           %[[c200:.+]] = arith.constant 200 : index
//   CHECK-DAG:           %[[v9:.+]] = arith.cmpi slt, %[[v6]], %[[c200]] : index
//   CHECK-DAG:           %[[v10:.+]] = arith.cmpi sge, %[[v6]], %[[c0_8]] : index
//   CHECK-DAG:           %[[v11:.+]] = arith.andi %[[v9]], %[[v10]] : i1
//   CHECK-DAG:           %[[v12:.+]] = arith.andi %[[v11]], %[[true]] : i1
//   CHECK-DAG:           %[[c100:.+]] = arith.constant 100 : index
//   CHECK-DAG:           %[[v13:.+]] = arith.cmpi slt, %[[v7]], %[[c100]] : index
//   CHECK-DAG:           %[[v14:.+]] = arith.cmpi sge, %[[v7]], %[[c0_8]] : index
//   CHECK-DAG:           %[[v15:.+]] = arith.andi %[[v13]], %[[v14]] : i1
//   CHECK-DAG:           %[[v16:.+]] = arith.andi %[[v15]], %[[v12]] : i1
//   CHECK-DAG:           %[[v17:.+]] = scf.if %[[v16]] -> (tensor<15x200x100x300xf32>) {
//       CHECK:             %[[extracted_9:.+]] = tensor.extract %[[arg8]][%[[v5]], %[[v6]], %[[v7]], %[[v8]]]
//       CHECK:             %[[v18:.+]] = arith.addf %[[extracted_9]], %[[extracted]] : f32
//       CHECK:             %[[inserted:.+]] = tensor.insert %[[v18]] into %[[arg8]][%[[v5]], %[[v6]], %[[v7]], %[[v8]]]
//       CHECK:             scf.yield %[[inserted]] : tensor<15x200x100x300xf32>
//       CHECK:            else
//       CHECK:             scf.yield %[[arg8]] : tensor<15x200x100x300xf32>
//       CHECK:         scf.yield %[[v2]]
//       CHECK:       scf.yield %[[v1]]
//       CHECK:     return %[[v0]]

// -----

func.func @scatter_with_batching_no_index_vector_dim(%arg0: tensor<3x2x4x9xi32>, %arg1: tensor<4x3x5xi32>, %arg2: tensor<4x3x5x8xi32>) -> tensor<3x2x4x9xi32> {
  %0 = kernel.scatter updates(%arg2 : tensor<4x3x5x8xi32>) into (%arg0 : tensor<3x2x4x9xi32>) at (%arg1 : tensor<4x3x5xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      kernel.yield %arg4 : i32
  } {
      update_window_dims = array<i64: 3>,
      inserted_window_dims = array<i64: 1>,
      input_batching_dims = array<i64: 0, 2>,
      scatter_indices_batching_dims = array<i64: 1, 0>,
      scatter_dims_to_operand_dims = array<i64: 1>,
      index_vector_dim = 3,
      unique_indices
  } : tensor<3x2x4x9xi32>
  func.return %0 : tensor<3x2x4x9xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %1 = transform.kernel.lower_to_loops %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
// CHECK-LABEL: func.func @scatter_with_batching_no_index_vector_dim
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x2x4x9xi32>, %[[arg1:.+]]: tensor<4x3x5xi32>, %[[arg2:.+]]: tensor<4x3x5x8xi32>)
//   CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[v0:.+]] = scf.for %[[arg3:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[arg4:.+]] = %[[arg0]]) -> (tensor<3x2x4x9xi32>) {
//   CHECK-DAG:       %[[c0_0:.+]] = arith.constant 0 : index
//   CHECK-DAG:       %[[c3:.+]] = arith.constant 3 : index
//   CHECK-DAG:       %[[c1_1:.+]] = arith.constant 1 : index
//   CHECK-DAG:       %[[v1:.+]] = scf.for %[[arg5:.+]] = %[[c0_0]] to %[[c3]] step %[[c1_1]] iter_args(%[[arg6:.+]] = %[[arg4]]) -> (tensor<3x2x4x9xi32>) {
//   CHECK-DAG:         %[[c0_2:.+]] = arith.constant 0 : index
//   CHECK-DAG:         %[[c5:.+]] = arith.constant 5 : index
//   CHECK-DAG:         %[[c1_3:.+]] = arith.constant 1 : index
//   CHECK-DAG:         %[[v2:.+]] = scf.for %[[arg7:.+]] = %[[c0_2]] to %[[c5]] step %[[c1_3]] iter_args(%[[arg8:.+]] = %[[arg6]]) -> (tensor<3x2x4x9xi32>) {
//   CHECK-DAG:           %[[c0_4:.+]] = arith.constant 0 : index
//   CHECK-DAG:           %[[c8:.+]] = arith.constant 8 : index
//   CHECK-DAG:           %[[c1_5:.+]] = arith.constant 1 : index
//   CHECK-DAG:           %[[v3:.+]] = scf.for %[[arg9:.+]] = %[[c0_4]] to %[[c8]] step %[[c1_5]] iter_args(%[[arg10:.+]] = %[[arg8]]) -> (tensor<3x2x4x9xi32>) {
//   CHECK-DAG:             %[[extracted:.+]] = tensor.extract %[[arg2]][%[[arg3]], %[[arg5]], %[[arg7]], %[[arg9]]]
//   CHECK-DAG:             %[[c0_6:.+]] = arith.constant 0 : index
//   CHECK-DAG:             %[[extracted_7:.+]] = tensor.extract %[[arg1]][%[[arg3]], %[[arg5]], %[[arg7]]]
//   CHECK-DAG:             %[[v4:.+]] = arith.index_cast %[[extracted_7]] : i32 to index
//   CHECK-DAG:             %[[v5:.+]] = affine.apply #[[$map]](%[[c0_6]], %[[arg5]], %[[c0_6]])
//   CHECK-DAG:             %[[v6:.+]] = affine.apply #[[$map]](%[[v4]], %[[c0_6]], %[[c0_6]])
//   CHECK-DAG:             %[[v7:.+]] = affine.apply #[[$map]](%[[c0_6]], %[[arg3]], %[[c0_6]])
//   CHECK-DAG:             %[[v8:.+]] = affine.apply #[[$map]](%[[c0_6]], %[[c0_6]], %[[arg9]])
//   CHECK-DAG:             %[[true:.+]] = arith.constant true
//   CHECK-DAG:             %[[c0_8:.+]] = arith.constant 0 : index
//   CHECK-DAG:             %[[c2:.+]] = arith.constant 2 : index
//   CHECK-DAG:             %[[v9:.+]] = arith.cmpi slt, %[[v6]], %[[c2]] : index
//   CHECK-DAG:             %[[v10:.+]] = arith.cmpi sge, %[[v6]], %[[c0_8]] : index
//   CHECK-DAG:             %[[v11:.+]] = arith.andi %[[v9]], %[[v10]] : i1
//   CHECK-DAG:             %[[v12:.+]] = arith.andi %[[v11]], %[[true]] : i1
//   CHECK-DAG:             %[[v13:.+]] = scf.if %[[v12]] -> (tensor<3x2x4x9xi32>) {
//       CHECK:               %[[extracted_9:.+]] = tensor.extract %[[arg10]][%[[v5]], %[[v6]], %[[v7]], %[[v8]]] : tensor<3x2x4x9xi32>
//       CHECK:               %[[inserted:.+]] = tensor.insert %[[extracted]] into %[[arg10]][%[[v5]], %[[v6]], %[[v7]], %[[v8]]] : tensor<3x2x4x9xi32>
//       CHECK:               scf.yield %[[inserted]] : tensor<3x2x4x9xi32>
//       CHECK:             } else {
//       CHECK:               scf.yield %[[arg10]]

//   CHECK-DAG:           scf.yield %[[v3]]
//   CHECK-DAG:         scf.yield %[[v2]]
//   CHECK-DAG:       scf.yield %[[v1]]
//   CHECK-DAG:     return %[[v0]]

// -----

func.func @scatter_batching_dim_dynamic_scatter_indices(%arg0: tensor<?x2x4x7x9xi32>, %arg1: tensor<4x?x5x2xi32>, %arg2: tensor<4x?x5x8xi32>) -> tensor<?x2x4x7x9xi32> {
  %0 = kernel.scatter updates(%arg2 : tensor<4x?x5x8xi32>) into (%arg0 : tensor<?x2x4x7x9xi32>) at (%arg1 : tensor<4x?x5x2xi32>) {
  ^bb0(%arg3: i32, %arg4: i32):
    kernel.yield %arg4 : i32
  }  {
    update_window_dims = array<i64: 3>,
    inserted_window_dims = array<i64: 1, 3>,
    input_batching_dims = array<i64: 0, 2>,
    scatter_indices_batching_dims = array<i64: 1, 0>,
    scatter_dims_to_operand_dims = array<i64: 1, 3>,
    index_vector_dim = 3
  } : tensor<?x2x4x7x9xi32>
  func.return %0 : tensor<?x2x4x7x9xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %1 = transform.kernel.lower_to_loops %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
// CHECK-LABEL: func.func @scatter_batching_dim_dynamic_scatter_indices
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x2x4x7x9xi32>, %[[arg1:.+]]: tensor<4x?x5x2xi32>, %[[arg2:.+]]: tensor<4x?x5x8xi32>)
//       CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK-DAG:     %[[dim:.+]] = tensor.dim %[[arg2]], %[[c1]]
//       CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-DAG:     %[[c4:.+]] = arith.constant 4 : index
//       CHECK-DAG:     %[[c1_0:.+]] = arith.constant 1 : index
//       CHECK-DAG:     %[[v0:.+]] = scf.for %[[arg3:.+]] = %[[c0]] to %[[c4]] step %[[c1_0]] iter_args(%[[arg4:.+]] = %[[arg0]]) -> (tensor<?x2x4x7x9xi32>) {
//       CHECK-DAG:       %[[c0_1:.+]] = arith.constant 0 : index
//       CHECK-DAG:       %[[c1_2:.+]] = arith.constant 1 : index
//       CHECK-DAG:       %[[v1:.+]] = scf.for %[[arg5:.+]] = %[[c0_1]] to %[[dim]] step %[[c1_2]] iter_args(%[[arg6:.+]] = %[[arg4]]) -> (tensor<?x2x4x7x9xi32>) {
//       CHECK-DAG:         %[[c0_3:.+]] = arith.constant 0 : index
//       CHECK-DAG:         %[[c5:.+]] = arith.constant 5 : index
//       CHECK-DAG:         %[[c1_4:.+]] = arith.constant 1 : index
//       CHECK-DAG:         %[[v2:.+]] = scf.for %[[arg7:.+]] = %[[c0_3]] to %[[c5]] step %[[c1_4]] iter_args(%[[arg8:.+]] = %[[arg6]]) -> (tensor<?x2x4x7x9xi32>) {
//       CHECK-DAG:           %[[c0_5:.+]] = arith.constant 0 : index
//       CHECK-DAG:           %[[c8:.+]] = arith.constant 8 : index
//       CHECK-DAG:           %[[c1_6:.+]] = arith.constant 1 : index
//       CHECK-DAG:           %[[v3:.+]] = scf.for %[[arg9:.+]] = %[[c0_5]] to %[[c8]] step %[[c1_6]] iter_args(%[[arg10:.+]] = %[[arg8]]) -> (tensor<?x2x4x7x9xi32>) {
//       CHECK-DAG:             %[[extracted:.+]] = tensor.extract %[[arg2]][%[[arg3]], %[[arg5]], %[[arg7]], %[[arg9]]]
//       CHECK-DAG:             %[[c0_7:.+]] = arith.constant 0 : index
//       CHECK-DAG:             %[[extracted_8:.+]] = tensor.extract %[[arg1]][%[[arg3]], %[[arg5]], %[[arg7]], %[[c0_7]]]
//       CHECK-DAG:             %[[v4:.+]] = arith.index_cast %[[extracted_8]] : i32 to index
//       CHECK-DAG:             %[[c1_9:.+]] = arith.constant 1 : index
//       CHECK-DAG:             %[[extracted_10:.+]] = tensor.extract %[[arg1]][%[[arg3]], %[[arg5]], %[[arg7]], %[[c1_9]]]
//       CHECK-DAG:             %[[v5:.+]] = arith.index_cast %[[extracted_10]] : i32 to index
//       CHECK-DAG:             %[[v6:.+]] = affine.apply #[[$map]](%[[c0_7]], %[[arg5]], %[[c0_7]])
//       CHECK-DAG:             %[[v7:.+]] = affine.apply #[[$map]](%[[v4]], %[[c0_7]], %[[c0_7]])
//       CHECK-DAG:             %[[v8:.+]] = affine.apply #[[$map]](%[[c0_7]], %[[arg3]], %[[c0_7]])
//       CHECK-DAG:             %[[v9:.+]] = affine.apply #[[$map]](%[[v5]], %[[c0_7]], %[[c0_7]])
//       CHECK-DAG:             %[[v10:.+]] = affine.apply #[[$map]](%[[c0_7]], %[[c0_7]], %[[arg9]])
//       CHECK-DAG:             %[[true:.+]] = arith.constant true
//       CHECK-DAG:             %[[c0_11:.+]] = arith.constant 0 : index
//       CHECK-DAG:             %[[c0_12:.+]] = arith.constant 0 : index
//       CHECK-DAG:             %[[dim_13:.+]] = tensor.dim %[[arg10]], %[[c0_12]]
//       CHECK-DAG:             %[[v11:.+]] = arith.cmpi slt, %[[v6]], %[[dim_13]] : index
//       CHECK-DAG:             %[[v12:.+]] = arith.cmpi sge, %[[v6]], %[[c0_11]] : index
//       CHECK-DAG:             %[[v13:.+]] = arith.andi %[[v11]], %[[v12]] : i1
//       CHECK-DAG:             %[[v14:.+]] = arith.andi %[[v13]], %[[true]] : i1
//       CHECK-DAG:             %[[c2:.+]] = arith.constant 2 : index
//       CHECK-DAG:             %[[v15:.+]] = arith.cmpi slt, %[[v7]], %[[c2]] : index
//       CHECK-DAG:             %[[v16:.+]] = arith.cmpi sge, %[[v7]], %[[c0_11]] : index
//       CHECK-DAG:             %[[v17:.+]] = arith.andi %[[v15]], %[[v16]] : i1
//       CHECK-DAG:             %[[v18:.+]] = arith.andi %[[v17]], %[[v14]] : i1
//       CHECK-DAG:             %[[c7:.+]] = arith.constant 7 : index
//       CHECK-DAG:             %[[v19:.+]] = arith.cmpi slt, %[[v9]], %[[c7]] : index
//       CHECK-DAG:             %[[v20:.+]] = arith.cmpi sge, %[[v9]], %[[c0_11]] : index
//       CHECK-DAG:             %[[v21:.+]] = arith.andi %[[v19]], %[[v20]] : i1
//       CHECK-DAG:             %[[v22:.+]] = arith.andi %[[v21]], %[[v18]] : i1
//       CHECK-DAG:             %[[v23:.+]] = scf.if %[[v22]] -> (tensor<?x2x4x7x9xi32>) {
//           CHECK:               %[[extracted_14:.+]] = tensor.extract %[[arg10]][%[[v6]], %[[v7]], %[[v8]], %[[v9]], %[[v10]]] : tensor<?x2x4x7x9xi32>
//           CHECK:               %[[inserted:.+]] = tensor.insert %[[extracted]] into %[[arg10]][%[[v6]], %[[v7]], %[[v8]], %[[v9]], %[[v10]]] : tensor<?x2x4x7x9xi32>
//           CHECK:               scf.yield %[[inserted]] : tensor<?x2x4x7x9xi32>
//           CHECK:             } else {
//           CHECK:               scf.yield %[[arg10]] : tensor<?x2x4x7x9xi32>
//           CHECK:             }
//           CHECK:             scf.yield %[[v23]] : tensor<?x2x4x7x9xi32>
//       CHECK-DAG:           scf.yield %[[v3]]
//       CHECK-DAG:         scf.yield %[[v2]]
//       CHECK-DAG:       scf.yield %[[v1]]
//       CHECK-DAG:     return %[[v0]]

// -----


func.func @overlapping_windows_check(%arg0: tensor<3x4x2xi64>, %arg1: tensor<2x3x2xi64>, %arg2: tensor<2x3x2x2xi64>) -> tensor<3x4x2xi64> {
  %0 = kernel.scatter updates(%arg2 : tensor<2x3x2x2xi64>) into (%arg0 : tensor<3x4x2xi64>) at (%arg1 : tensor<2x3x2xi64>) {
    ^bb0(%arg3: i64, %arg4: i64):
      %0 = arith.addi %arg3, %arg4 : i64
      kernel.yield %0 : i64
  } {
      update_window_dims = array<i64: 2, 3>,
      inserted_window_dims = array<i64: 0>,
      scatter_dims_to_operand_dims = array<i64: 1, 0>,
      index_vector_dim = 2
  } : tensor<3x4x2xi64>
  return %0 : tensor<3x4x2xi64>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %1 = transform.kernel.lower_to_loops %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

//       CHECK: #[[$map:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
// CHECK-LABEL: func.func @overlapping_windows_check
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x4x2xi64>, %[[arg1:.+]]: tensor<2x3x2xi64>, %[[arg2:.+]]: tensor<2x3x2x2xi64>)
//       CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK-DAG:     %[[v0:.+]] = scf.for %[[arg3:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[arg4:.+]] = %[[arg0]]) -> (tensor<3x4x2xi64>) {
//       CHECK-DAG:       %[[c0_0:.+]] = arith.constant 0 : index
//       CHECK-DAG:       %[[c3:.+]] = arith.constant 3 : index
//       CHECK-DAG:       %[[c1_1:.+]] = arith.constant 1 : index
//       CHECK-DAG:       %[[v1:.+]] = scf.for %[[arg5:.+]] = %[[c0_0]] to %[[c3]] step %[[c1_1]] iter_args(%[[arg6:.+]] = %[[arg4]]) -> (tensor<3x4x2xi64>) {
//       CHECK-DAG:         %[[c0_2:.+]] = arith.constant 0 : index
//       CHECK-DAG:         %[[c2_3:.+]] = arith.constant 2 : index
//       CHECK-DAG:         %[[c1_4:.+]] = arith.constant 1 : index
//       CHECK-DAG:         %[[v2:.+]] = scf.for %[[arg7:.+]] = %[[c0_2]] to %[[c2_3]] step %[[c1_4]] iter_args(%[[arg8:.+]] = %[[arg6]]) -> (tensor<3x4x2xi64>) {
//       CHECK-DAG:           %[[c0_5:.+]] = arith.constant 0 : index
//       CHECK-DAG:           %[[c2_6:.+]] = arith.constant 2 : index
//       CHECK-DAG:           %[[c1_7:.+]] = arith.constant 1 : index
//       CHECK-DAG:           %[[v3:.+]] = scf.for %[[arg9:.+]] = %[[c0_5]] to %[[c2_6]] step %[[c1_7]] iter_args(%[[arg10:.+]] = %[[arg8]]) -> (tensor<3x4x2xi64>) {
//       CHECK-DAG:             %[[extracted:.+]] = tensor.extract %[[arg2]][%[[arg3]], %[[arg5]], %[[arg7]], %[[arg9]]]
//       CHECK-DAG:             %[[c0_8:.+]] = arith.constant 0 : index
//       CHECK-DAG:             %[[extracted_9:.+]] = tensor.extract %[[arg1]][%[[arg3]], %[[arg5]], %[[c0_8]]]
//       CHECK-DAG:             %[[v4:.+]] = arith.index_cast %[[extracted_9]] : i64 to index
//       CHECK-DAG:             %[[c1_10:.+]] = arith.constant 1 : index
//       CHECK-DAG:             %[[extracted_11:.+]] = tensor.extract %[[arg1]][%[[arg3]], %[[arg5]], %[[c1_10]]]
//       CHECK-DAG:             %[[v5:.+]] = arith.index_cast %[[extracted_11]] : i64 to index
//       CHECK-DAG:             %[[v6:.+]] = affine.apply #[[$map]](%[[v5]], %[[c0_8]], %[[c0_8]])
//       CHECK-DAG:             %[[v7:.+]] = affine.apply #[[$map]](%[[v4]], %[[c0_8]], %[[arg7]])
//       CHECK-DAG:             %[[v8:.+]] = affine.apply #[[$map]](%[[c0_8]], %[[c0_8]], %[[arg9]])
//       CHECK-DAG:             %[[true:.+]] = arith.constant true
//       CHECK-DAG:             %[[c0_12:.+]] = arith.constant 0 : index
//       CHECK-DAG:             %[[c3_13:.+]] = arith.constant 3 : index
//       CHECK-DAG:             %[[v9:.+]] = arith.cmpi slt, %[[v6]], %[[c3_13]] : index
//       CHECK-DAG:             %[[v10:.+]] = arith.cmpi sge, %[[v6]], %[[c0_12]] : index
//       CHECK-DAG:             %[[v11:.+]] = arith.andi %[[v9]], %[[v10]] : i1
//       CHECK-DAG:             %[[v12:.+]] = arith.andi %[[v11]], %[[true]] : i1
//       CHECK-DAG:             %[[c4:.+]] = arith.constant 4 : index
//       CHECK-DAG:             %[[v13:.+]] = arith.cmpi slt, %[[v7]], %[[c4]] : index
//       CHECK-DAG:             %[[v14:.+]] = arith.cmpi sge, %[[v7]], %[[c0_12]] : index
//       CHECK-DAG:             %[[v15:.+]] = arith.andi %[[v13]], %[[v14]] : i1
//       CHECK-DAG:             %[[v16:.+]] = arith.andi %[[v15]], %[[v12]] : i1
//       CHECK-DAG:             %[[v17:.+]] = scf.if %[[v16]] -> (tensor<3x4x2xi64>) {
//           CHECK:               %[[extracted_14:.+]] = tensor.extract %[[arg10]][%[[v6]], %[[v7]], %[[v8]]]
//           CHECK:               %[[v18:.+]] = arith.addi %[[extracted_14]], %[[extracted]] : i64
//           CHECK:               %[[inserted:.+]] = tensor.insert %[[v18]] into %[[arg10]][%[[v6]], %[[v7]], %[[v8]]]
//           CHECK:               scf.yield %[[inserted]] : tensor<3x4x2xi64>
//           CHECK:             } else {
//       CHECK-DAG:               scf.yield %[[arg10]]
//       CHECK-DAG:           scf.yield %[[v3]]
//       CHECK-DAG:         scf.yield %[[v2]]
//       CHECK-DAG:       scf.yield %[[v1]]
//       CHECK-DAG:     return %[[v0]]
