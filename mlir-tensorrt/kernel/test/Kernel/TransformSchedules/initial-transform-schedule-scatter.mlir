// RUN: kernel-opt %s -split-input-file -kernel-initial-transform-schedule="generator-benefit={scatter:100} \
// RUN:  device-compute-capability=80 device-max-smem-per-block=48 device-max-registers-per-block=65536" \
// RUN: -verify-diagnostics | FileCheck %s

// RUN: kernel-opt -split-input-file %s -verify-diagnostics -kernel-linalg-codegen-pipeline="generator-benefit={scatter:100} \
// RUN:  device-compute-capability=80 device-max-smem-per-block=48 device-max-registers-per-block=65536" \
// RUN: | FileCheck %s --check-prefix=E2E

func.func @scatter1(%arg0: tensor<1xi64>, %arg1: tensor<128x1x1215xf32>, %arg2: tensor<128x1xf32>) -> tensor<128x1x1215xf32> {
  %1 = kernel.scatter
    updates(%arg2  : tensor<128x1xf32>)
    into(%arg1  : tensor<128x1x1215xf32>)
    at(%arg0 : tensor<1xi64>) {
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

// CHECK-LABEL: @scatter1
// CHECK: kernel.scatter
// CHECK: kernel.parameters = #kernel.scatter_parameters<gpu_target = #nvvm.target<chip = "sm_80">,
// CHECK-SAME: cta_workload_shape = [32, 1],
// CHECK-SAME: cta_blocking_shape = [32, 1],
// CHECK-SAME: thread_tile_shape = [1, 1],
// CHECK-SAME: grid_shape = [4, 1],
// CHECK-SAME: cta_shape = [32, 1]>


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

// CHECK-LABEL: @scatter_update_scalar
// CHECK: kernel.scatter
// CHECK: kernel.parameters = #kernel.scatter_parameters<gpu_target = #nvvm.target<chip = "sm_80">,
// CHECK-SAME: cta_workload_shape = [1],
// CHECK-SAME: cta_blocking_shape = [1],
// CHECK-SAME: thread_tile_shape = [1],
// CHECK-SAME: grid_shape = [1],
// CHECK-SAME: cta_shape = [1]>

// E2E-LABEL: @scatter_update_scalar
//       E2E: gpu.module.kernels.ptx_data

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

// CHECK-LABEL: @test_scatter_no_batch
// CHECK: kernel.scatter
// CHECK: kernel.parameters = #kernel.scatter_parameters<gpu_target = #nvvm.target<chip = "sm_80">,
// CHECK-SAME: cta_workload_shape = [10, 75],
// CHECK-SAME: cta_blocking_shape = [10, 75],
// CHECK-SAME: thread_tile_shape = [1, 1],
// CHECK-SAME: grid_shape = [1, 4],
// CHECK-SAME: cta_shape = [10, 75]>

// E2E-LABEL: @test_scatter_no_batch
//       E2E: gpu.module.kernels.ptx_data

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

// CHECK-LABEL: @test_scatter_batch
// CHECK: kernel.scatter
// CHECK: kernel.parameters = #kernel.scatter_parameters<gpu_target = #nvvm.target<chip = "sm_80">,
// CHECK-SAME: cta_workload_shape = [15, 10, 30],
// CHECK-SAME: cta_blocking_shape = [15, 10, 30],
// CHECK-SAME: thread_tile_shape = [1, 2, 3],
// CHECK-SAME: grid_shape = [1, 1, 10],
// CHECK-SAME: cta_shape = [15, 5, 10]>

// E2E-LABEL: @test_scatter_batch
//       E2E: gpu.module.kernels.ptx_data

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

// CHECK-LABEL: @scatter_with_batching_no_index_vector_dim
// CHECK: kernel.scatter
// CHECK: kernel.parameters = #kernel.scatter_parameters<gpu_target = #nvvm.target<chip = "sm_80">,
// CHECK-SAME: cta_workload_shape = [4, 3, 5, 8],
// CHECK-SAME: cta_blocking_shape = [4, 3, 5, 8],
// CHECK-SAME: thread_tile_shape = [1, 1, 1, 2],
// CHECK-SAME: grid_shape = [1, 1, 1, 1],
// CHECK-SAME: cta_shape = [4, 3, 5, 4]>

// E2E-LABEL: @scatter_with_batching_no_index_vector_dim
//       E2E: gpu.module.kernels.ptx_data

// -----


func.func @scatter_batching_dim_dynamic_scatter_indices(%arg0: tensor<?x2x4x7x9xi32>, %arg1: tensor<4x?x5x2xi32>, %arg2: tensor<4x?x5x8xi32>) -> tensor<?x2x4x7x9xi32> {
  // expected-error @below {{'kernel.scatter' op failed to determine schedule parameters}}
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


// CHECK-LABEL: @overlapping_windows_check
// CHECK: kernel.scatter
// CHECK: kernel.parameters = #kernel.scatter_parameters<gpu_target = #nvvm.target<chip = "sm_80">,
// CHECK-SAME: cta_workload_shape = [2, 3, 2, 2],
// CHECK-SAME: cta_blocking_shape = [2, 3, 2, 2],
// CHECK-SAME: thread_tile_shape = [2, 3, 2, 2],
// CHECK-SAME: grid_shape = [1, 1, 1, 1],
// CHECK-SAME: cta_shape = [1, 1, 1, 1]>

// E2E-LABEL: @overlapping_windows_check
//       E2E: gpu.module.kernels.ptx_data

// -----

func.func @single_update_non_unique(
    %arg0: tensor<1x1xi32>,
    %arg1: tensor<1x2x8x3xi32>,
    %arg2: tensor<10x8x3xi32>)
    -> tensor<10x8x3xi32> {
  %0 = kernel.scatter updates(%arg1 : tensor<1x2x8x3xi32>) into(%arg2 : tensor<10x8x3xi32>) at(%arg0 : tensor<1x1xi32>) {
  ^bb0(%arg3: i32, %arg4: i32):
    kernel.yield %arg4 : i32
  } {index_vector_dim = 1 : i64,
     scatter_dims_to_operand_dims = array<i64: 0>,
     update_window_dims = array<i64: 1, 2, 3>
  } : tensor<10x8x3xi32>
  return %0 : tensor<10x8x3xi32>
}

// CHECK-LABEL: @single_update_non_unique
// CHECK: kernel.scatter
// CHECK: kernel.parameters = #kernel.scatter_parameters<gpu_target = #nvvm.target<chip = "sm_80">,
// CHECK-SAME: cta_workload_shape = [1, 2, 8, 3],
// CHECK-SAME: cta_blocking_shape = [1, 2, 8, 3],
// CHECK-SAME: thread_tile_shape = [1, 1, 1, 1],
// CHECK-SAME: grid_shape = [1, 1, 1, 1],
// CHECK-SAME: cta_shape = [1, 2, 8, 3]>

// E2E-LABEL: @single_update_non_unique
//       E2E: gpu.module.kernels.ptx_data

// -----

func.func @scatter_scalar_update(
    %arg0: tensor<1xi32>,
    %arg1: tensor<complex<f32>>,
    %arg2: tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>> attributes {cluster.codegen} {
  %0 = kernel.scatter updates(%arg1 : tensor<complex<f32>>) into(%arg2 : tensor<3xcomplex<f32>>) at(%arg0 : tensor<1xi32>) {
  ^bb0(%arg3: complex<f32>, %arg4: complex<f32>):
    kernel.yield %arg4 : complex<f32>
  } {
    index_vector_dim = 0 : i64,
    indices_are_sorted,
    inserted_window_dims = array<i64: 0>,
    scatter_dims_to_operand_dims = array<i64: 0>,
    unique_indices,
    update_window_dims = array<i64>} : tensor<3xcomplex<f32>>
  return %0 : tensor<3xcomplex<f32>>
}

// CHECK-LABEL: @scatter_scalar_update
// CHECK: kernel.scatter
// CHECK: kernel.parameters = #kernel.scatter_parameters<gpu_target = #nvvm.target<chip = "sm_80">,
// CHECK-SAME: cta_workload_shape = [],
// CHECK-SAME: cta_blocking_shape = [],
// CHECK-SAME: thread_tile_shape = [],
// CHECK-SAME: grid_shape = [],
// CHECK-SAME: cta_shape = []>

// E2E-LABEL: @scatter_scalar_update
//       E2E: gpu.module.kernels.ptx_data
