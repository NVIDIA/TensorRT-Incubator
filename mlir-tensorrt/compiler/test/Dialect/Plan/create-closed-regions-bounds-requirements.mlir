// RUN: mlir-tensorrt-opt %s -plan-create-closed-regions -split-input-file | FileCheck %s

// Test that HostBackend succeeds even when bounds can't be derived for
// dynamically-shaped inputs. The HostBackend uses the default implementation
// of requiresInputBoundsForDynamicShapes which returns false.

// CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-LABEL: @host_backend_no_bounds_required
// CHECK: plan.alloc_cluster target(#plan.host_backend<benefit = 1>)
// CHECK: in_attrs [#[[$nobounds]], #[[$nobounds]]]
func.func @host_backend_no_bounds_required(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = plan.cluster target(#plan.host_backend<benefit = 1>) -> tensor<?xf32> {
    %1 = stablehlo.exponential %arg0 : tensor<?xf32>
    %2 = plan.with_shape %1(%dim) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %2 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

// -----

// Test with multiple dynamic inputs without bounds - HostBackend should still succeed

// CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-LABEL: @host_backend_multiple_dynamic_inputs_no_bounds
// CHECK: plan.alloc_cluster target(#plan.host_backend<benefit = 1>)
// CHECK: in_attrs [#[[$nobounds]], #[[$nobounds]], #[[$nobounds]]]
func.func @host_backend_multiple_dynamic_inputs_no_bounds(
    %arg0: tensor<?xf32>,
    %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = plan.cluster target(#plan.host_backend<benefit = 1>) -> tensor<?xf32> {
    %1 = stablehlo.add %arg0, %arg1 : tensor<?xf32>
    %2 = plan.with_shape %1(%dim) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %2 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

// -----

// Test that TensorRTBackend succeeds when bounds ARE provided via plan.shape_bounds

#profile0 = #plan.bounds<shape, [1], [100]>

// CHECK-DAG: #[[$bounds:.+]] = #plan.bounds<shape, [1], [100]>
// CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-LABEL: @tensorrt_backend_with_bounds
// CHECK: plan.dps_cluster target(#plan.tensorrt_backend
// CHECK: in_attrs [#[[$bounds]], #[[$nobounds]]]
func.func @tensorrt_backend_with_bounds(%arg0: tensor<?xf32> {plan.shape_bounds = #profile0}) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) -> tensor<?xf32> {
    %1 = stablehlo.exponential %arg0 : tensor<?xf32>
    %2 = plan.with_shape %1(%dim) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %2 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

// -----

// Test that KernelBackend (which uses default implementation returning false)
// succeeds without bounds

// CHECK-DAG: #[[$nobounds:.+]] = #plan.bounds<none>
// CHECK-LABEL: @kernel_backend_no_bounds_required
// CHECK: plan.alloc_cluster target(#plan.kernel_backend<benefit = 1>)
// CHECK: in_attrs [#[[$nobounds]], #[[$nobounds]]]
func.func @kernel_backend_no_bounds_required(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = plan.cluster target(#plan.kernel_backend<benefit = 1>) -> tensor<?xf32> {
    %1 = stablehlo.exponential %arg0 : tensor<?xf32>
    %2 = plan.with_shape %1(%dim) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %2 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}
