// RUN: mlir-tensorrt-opt %s -plan-create-closed-regions -split-input-file -verify-diagnostics

// Test that TensorRTBackend fails when bounds can't be derived for
// dynamically-shaped inputs without plan.shape_bounds attribute.
// TensorRTBackend::requiresInputBoundsForDynamicShapes returns true.

func.func @tensorrt_backend_requires_bounds(
    // expected-error @below {{failed to derive upper bound for}}
    %arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  // expected-error @below {{failed to compute input attribute}}
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) -> tensor<?xf32> {
    %1 = stablehlo.exponential %arg0 : tensor<?xf32>
    %2 = plan.with_shape %1(%dim) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %2 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

// -----

// Test that TensorRTBackend fails for multiple dynamic inputs without bounds

func.func @tensorrt_backend_multiple_dynamic_requires_bounds(
    // expected-error @below {{failed to derive upper bound for}}
    %arg0: tensor<?xf32>,
    %arg1: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  // expected-error @below {{failed to compute input attribute}}
  %0 = plan.cluster target(#plan.tensorrt_backend<disallow_shape_tensor_calculations = false, benefit = 1>) -> tensor<?xf32> {
    %1 = stablehlo.exponential %arg0 : tensor<?xf32>
    %2 = plan.with_shape %1(%dim) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %2 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}
