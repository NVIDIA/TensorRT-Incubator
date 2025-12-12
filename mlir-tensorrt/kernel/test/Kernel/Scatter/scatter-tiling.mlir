// RUN: kernel-opt %s --transform-interpreter -verify-diagnostics -split-input-file | FileCheck %s


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

// CHECK-LABEL: func.func @scatter1
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xi64>, %[[arg1:.+]]: tensor<128x1x1215xf32>, %[[arg2:.+]]: tensor<128x1xf32>)
//       CHECK:     %[[v1:.+]] = scf.forall (%[[arg3:.+]]) in (128) shared_outs(%[[arg4:.+]] = %[[arg1]]) -> (tensor<128x1x1215xf32>) {
//   CHECK-DAG:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg2]][%[[arg3]], 0] [1, 1] [1, 1] : tensor<128x1xf32> to tensor<1x1xf32>
//   CHECK-DAG:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg0]][0] [1] [1] : tensor<1xi64> to tensor<1xi64>
//   CHECK-DAG:       %[[extracted_slice_1:.+]] = tensor.extract_slice %[[arg4]][%[[arg3]], 0, 0] [1, 1, 1215] [1, 1, 1] : tensor<128x1x1215xf32> to tensor<1x1x1215xf32>
//       CHECK:       %[[v1:.+]] = kernel.scatter updates(%[[extracted_slice]] : tensor<1x1xf32>) into(%[[extracted_slice_1]] : tensor<1x1x1215xf32>) at(%[[extracted_slice_0]] : tensor<1xi64>)
//   CHECK-DAG:         tensor.parallel_insert_slice %[[v1]] into %[[arg4]][%[[arg3]], 0, 0] [1, 1, 1215] [1, 1, 1] : tensor<1x1x1215xf32> into tensor<128x1x1215xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0   num_threads [] tile_sizes [1, 0] :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

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

// CHECK-LABEL: func.func @scatter_update_scalar
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3xi32>, %[[arg1:.+]]: tensor<1x1xi32>, %[[arg2:.+]]: tensor<1xi32>)
//       CHECK:     %[[v1:.+]] = scf.forall (%[[arg3:.+]]) in (1) shared_outs(%[[arg4:.+]] = %[[arg0]]) -> (tensor<3xi32>) {
//   CHECK-DAG:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg2]][%[[arg3]]] [1] [1] : tensor<1xi32> to tensor<1xi32>
//   CHECK-DAG:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg1]][%[[arg3]], 0] [1, 1] [1, 1] : tensor<1x1xi32> to tensor<1x1xi32>
//   CHECK-DAG:       %[[extracted_slice_1:.+]] = tensor.extract_slice %[[arg4]][0] [3] [1] : tensor<3xi32> to tensor<3xi32>
//       CHECK:       %[[v1:.+]] = kernel.scatter updates(%[[extracted_slice]] : tensor<1xi32>) into(%[[extracted_slice_1]] : tensor<3xi32>) at(%[[extracted_slice_0]] : tensor<1x1xi32>)
//       CHECK:         tensor.parallel_insert_slice %[[v1]] into %[[arg4]][0] [3] [1] : tensor<3xi32> into tensor<3xi32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0   num_threads [] tile_sizes [1]
      :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @test_scatter_no_batch(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi64>, %arg2: tensor<10x300xf32>) -> (tensor<200x100x300xf32>, tensor<200x100x300xf32>) {
  %0:2 = kernel.scatter updates(%arg2, %arg2 : tensor<10x300xf32>, tensor<10x300xf32>) into (%arg0, %arg0 : tensor<200x100x300xf32>, tensor<200x100x300xf32>) at (%arg1 : tensor<10x2xi64>) {
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

//       CHECK: #[[$map:.+]] = affine_map<(d0) -> (d0 * 150)>
// CHECK-LABEL: func.func @test_scatter_no_batch
//  CHECK-SAME: (%[[arg0:.+]]: tensor<200x100x300xf32>, %[[arg1:.+]]: tensor<10x2xi64>, %[[arg2:.+]]: tensor<10x300xf32>)
//       CHECK:     %[[v2:.+]]:2 = scf.forall (%[[arg3:.+]]) in (2) shared_outs(%[[arg4:.+]] = %{{.+}}, %[[arg5:.+]] = %{{.+}})
//   CHECK-DAG:       %[[v3:.+]] = affine.apply #[[$map]](%[[arg3]])
//   CHECK-DAG:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg2]][0, %[[v3]]] [10, 150] [1, 1] : tensor<10x300xf32> to tensor<10x150xf32>
//   CHECK-DAG:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg2]][0, %[[v3]]] [10, 150] [1, 1] : tensor<10x300xf32> to tensor<10x150xf32>
//   CHECK-DAG:       %[[extracted_slice_1:.+]] = tensor.extract_slice %[[arg1]][0, 0] [10, 2] [1, 1] : tensor<10x2xi64> to tensor<10x2xi64>
//   CHECK-DAG:       %[[extracted_slice_2:.+]] = tensor.extract_slice %[[arg4]][0, 0, %[[v3]]] [200, 100, 150] [1, 1, 1] : tensor<200x100x300xf32> to tensor<200x100x150xf32>
//   CHECK-DAG:       %[[extracted_slice_3:.+]] = tensor.extract_slice %[[arg5]][0, 0, %[[v3]]] [200, 100, 150] [1, 1, 1] : tensor<200x100x300xf32> to tensor<200x100x150xf32>
//       CHECK:       %[[v4:.+]]:2 = kernel.scatter updates(%[[extracted_slice]], %[[extracted_slice_0]] : tensor<10x150xf32>, tensor<10x150xf32>) into(%[[extracted_slice_2]], %[[extracted_slice_3]] : tensor<200x100x150xf32>, tensor<200x100x150xf32>) at(%[[extracted_slice_1]] : tensor<10x2xi64>)
//       CHECK:         tensor.parallel_insert_slice %[[v4]]#0 into %[[arg4]][0, 0, %[[v3]]] [200, 100, 150] [1, 1, 1] : tensor<200x100x150xf32> into tensor<200x100x300xf32>
//       CHECK:         tensor.parallel_insert_slice %[[v4]]#1 into %[[arg5]][0, 0, %[[v3]]] [200, 100, 150] [1, 1, 1] : tensor<200x100x150xf32> into tensor<200x100x300xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0   num_threads [0, 2] tile_sizes []
      :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

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

//  CHECK-DAG: #[[$map:.+]] = affine_map<(d0) -> (d0 * 5)>
//  CHECK-DAG: #[[$map1:.+]] = affine_map<(d0) -> (d0 * 30)>
// CHECK-LABEL: func.func @test_scatter_batch
//  CHECK-SAME: (%[[arg0:.+]]: tensor<15x200x100x300xf32>, %[[arg1:.+]]: tensor<15x10x2xi32>, %[[arg2:.+]]: tensor<15x10x300xf32>)
//       CHECK:     %[[v1:.+]] = scf.forall (%[[arg3:.+]], %[[arg4:.+]], %[[arg5:.+]]) in (3, 10, 10) shared_outs(%[[arg6:.+]] = %[[arg0]])
//   CHECK-DAG:       %[[v2:.+]] = affine.apply #[[$map]](%[[arg3]])
//   CHECK-DAG:       %[[v3:.+]] = affine.apply #[[$map1]](%[[arg5]])
//   CHECK-DAG:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg2]][%[[v2]], %[[arg4]], %[[v3]]] [5, 1, 30] [1, 1, 1] : tensor<15x10x300xf32> to tensor<5x1x30xf32>
//   CHECK-DAG:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg1]][%[[v2]], %[[arg4]], 0] [5, 1, 2] [1, 1, 1] : tensor<15x10x2xi32> to tensor<5x1x2xi32>
//   CHECK-DAG:       %[[extracted_slice_1:.+]] = tensor.extract_slice %[[arg6]][%[[v2]], 0, 0, %[[v3]]] [5, 200, 100, 30] [1, 1, 1, 1] : tensor<15x200x100x300xf32> to tensor<5x200x100x30xf32>
//       CHECK:       %[[v4:.+]] = kernel.scatter updates(%[[extracted_slice]] : tensor<5x1x30xf32>) into(%[[extracted_slice_1]] : tensor<5x200x100x30xf32>) at(%[[extracted_slice_0]] : tensor<5x1x2xi32>)
//       CHECK:         tensor.parallel_insert_slice %[[v4]] into %[[arg6]][%[[v2]], 0, 0, %[[v3]]] [5, 200, 100, 30] [1, 1, 1, 1] : tensor<5x200x100x30xf32> into tensor<15x200x100x300xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0   num_threads [3, 10, 10] tile_sizes []
      :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

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
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0 num_threads [2, 3, 2, 2] tile_sizes []
      :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

//   CHECK-DAG: #[[$map:.+]] = affine_map<(d0) -> (d0 * 2)>
//   CHECK-DAG: #[[$map1:.+]] = affine_map<(d0) -> (d0 * 3)>
//   CHECK-DAG: #[[$map2:.+]] = affine_map<(d0) -> (d0 * -3 + 5)>
//   CHECK-DAG: #[[$map3:.+]] = affine_map<(d0) -> (d0 * -3 + 5, 3)>
//   CHECK-DAG: #[[$map4:.+]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-LABEL: func.func @scatter_with_batching_no_index_vector_dim
//  CHECK-SAME: (%[[arg0:.+]]: tensor<3x2x4x9xi32>, %[[arg1:.+]]: tensor<4x3x5xi32>, %[[arg2:.+]]: tensor<4x3x5x8xi32>)
//       CHECK:     %[[v0:.+]] = scf.forall (%[[arg3:.+]], %[[arg4:.+]], %[[arg5:.+]], %[[arg6:.+]]) in (2, 3, 2, 2) shared_outs(%[[arg7:.+]] = %[[arg0]]) -> (tensor<3x2x4x9xi32>) {
//       CHECK:       %[[v1:.+]] = affine.apply #[[$map]](%[[arg3]])
//       CHECK:       %[[v2:.+]] = affine.apply #[[$map1]](%[[arg5]])
//       CHECK:       %[[v3:.+]] = affine.apply #[[$map2]](%[[arg5]])
//       CHECK:       %[[v4:.+]] = affine.min #[[$map3]](%[[arg5]])
//       CHECK:       %[[v5:.+]] = affine.apply #[[$map4]](%[[arg6]])
//       CHECK:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg2]][%[[v1]], %[[arg4]], %[[v2]], %[[v5]]] [2, 1, %[[v4]], 4] [1, 1, 1, 1] : tensor<4x3x5x8xi32> to tensor<2x1x?x4xi32>
//       CHECK:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg1]][%[[v1]], %[[arg4]], %[[v2]]] [2, 1, %[[v4]]] [1, 1, 1] : tensor<4x3x5xi32> to tensor<2x1x?xi32>
//       CHECK:       %[[extracted_slice_1:.+]] = tensor.extract_slice %[[arg7]][%[[arg4]], 0, %[[v1]], %[[v5]]] [1, 2, 2, 4] [1, 1, 1, 1] : tensor<3x2x4x9xi32> to tensor<1x2x2x4xi32>
//       CHECK:       %[[v6:.+]] = kernel.scatter updates(%[[extracted_slice]] : tensor<2x1x?x4xi32>) into(%[[extracted_slice_1]] : tensor<1x2x2x4xi32>) at(%[[extracted_slice_0]] : tensor<2x1x?xi32>)
//       CHECK:       scf.forall.in_parallel
//       CHECK:         tensor.parallel_insert_slice %[[v6]] into %[[arg7]][%[[arg4]], 0, %[[v1]], %[[v5]]] [1, 2, 2, 4] [1, 1, 1, 1] :
//       CHECK:     return %[[v0]] : tensor<3x2x4x9xi32>

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

//       CHECK: #[[$map:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: func.func @scatter_batching_dim_dynamic_scatter_indices
//  CHECK-SAME: (%[[arg0:.+]]: tensor<?x2x4x7x9xi32>, %[[arg1:.+]]: tensor<4x?x5x2xi32>, %[[arg2:.+]]: tensor<4x?x5x8xi32>)
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[dim:.+]] = tensor.dim %[[arg2]], %[[c1]] : tensor<4x?x5x8xi32>
//       CHECK:     %[[v0:.+]] = scf.forall (%[[arg3]], %[[arg4]]) in (2, %[[dim]]) shared_outs(%[[arg5]] = %[[arg0]])
//       CHECK:       %[[v1:.+]] = affine.apply #[[$map]](%[[arg3]])
//       CHECK:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg2]][%[[v1]], %[[arg4]], 0, 0] [2, 1, 5, 8] [1, 1, 1, 1] : tensor<4x?x5x8xi32> to tensor<2x1x5x8xi32>
//       CHECK:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[arg1]][%[[v1]], %[[arg4]], 0, 0] [2, 1, 5, 2] [1, 1, 1, 1] : tensor<4x?x5x2xi32> to tensor<2x1x5x2xi32>
//       CHECK:       %[[extracted_slice_1:.+]] = tensor.extract_slice %[[arg5]][%[[arg4]], 0, %[[v1]], 0, 0] [1, 2, 2, 7, 8] [1, 1, 1, 1, 1] : tensor<?x2x4x7x9xi32> to tensor<1x2x2x7x8xi32>
//       CHECK:       %[[v2:.+]] = kernel.scatter updates(%[[extracted_slice]] : tensor<2x1x5x8xi32>) into(%[[extracted_slice_1]] : tensor<1x2x2x7x8xi32>) at(%[[extracted_slice_0]] : tensor<2x1x5x2xi32>)
//       CHECK:       scf.forall.in_parallel
//       CHECK:         tensor.parallel_insert_slice %[[v2]] into %[[arg5]][%[[arg4]], 0, %[[v1]], 0, 0] [1, 2, 2, 7, 8] [1, 1, 1, 1, 1] : tensor<1x2x2x7x8xi32> into tensor<?x2x4x7x9xi32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0 num_threads [] tile_sizes [2, 1, 0, 0]
      :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----


func.func @overlapping_windows_check(%arg0: tensor<3x4x2xi64>, %arg1: tensor<2x3x2xi64>, %arg2: tensor<2x3x2x2xi64>) -> tensor<3x4x2xi64> {
  // expected-warning @below {{tiling is not thread safe at axis #0}}
  // expected-warning @below {{tiling is not thread safe at axis #1}}
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
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0 num_threads [] tile_sizes [1, 1, 0, 0]
      :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// -----

func.func @scatter_scalar_constant_update_add(%arg0: tensor<1xf16>, %arg1: tensor<6xf16>) -> tensor<6xf16> {
  %cst = arith.constant dense<5> : tensor<1xi32>
  %0 = kernel.scatter updates(%arg0 : tensor<1xf16>) into(%arg1 : tensor<6xf16>) at(%cst : tensor<1xi32>) {
  ^bb0(%arg2: f16, %arg3: f16):
    %1 = arith.addf %arg2, %arg3 : f16
    kernel.yield %1 : f16
  } {
    index_vector_dim = 0 : i64,
    indices_are_sorted,
    scatter_dims_to_operand_dims = array<i64: 0>,
    unique_indices,
    update_window_dims = array<i64: 0>
  } : tensor<6xf16>
  return %0 : tensor<6xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.read_only}) {
    %0 = transform.structured.match ops {["kernel.scatter"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %0 num_threads [1]
      :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @scatter_scalar_constant_update_add
//  CHECK-SAME: (%[[arg0:.+]]: tensor<1xf16>, %[[arg1:.+]]: tensor<6xf16>)
//       CHECK:     %[[cst:.+]] = arith.constant dense<5> : tensor<1xi32>
//       CHECK:     scf.forall (%[[arg2]]) in (1) shared_outs(%[[arg3]] = %[[arg1]]) -> (tensor<6xf16>)
//   CHECK-DAG:       %[[extracted_slice:.+]] = tensor.extract_slice %[[arg0]][%[[arg2]]] [1] [1] : tensor<1xf16> to tensor<1xf16>
//   CHECK-DAG:       %[[extracted_slice_0:.+]] = tensor.extract_slice %[[cst]][0] [1] [1] : tensor<1xi32> to tensor<1xi32>
//   CHECK-DAG:       %[[extracted_slice_1:.+]] = tensor.extract_slice %[[arg3]][0] [6] [1] : tensor<6xf16> to tensor<6xf16>
//   CHECK-DAG:       kernel.scatter updates(%[[extracted_slice]] : tensor<1xf16>) into(%[[extracted_slice_1]] : tensor<6xf16>) at(%[[extracted_slice_0]] : tensor<1xi32>)
