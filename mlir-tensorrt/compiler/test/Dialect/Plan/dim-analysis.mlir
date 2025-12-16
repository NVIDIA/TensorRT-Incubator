// RUN: mlir-tensorrt-opt %s -split-input-file -test-dim-analysis 2>&1 | FileCheck %s

// Test basic symbol mapping through function call
func.func private @callee(%arg0: tensor<i32> {jax.global_constant = "n"},
                          %arg1: tensor<i32> {jax.global_constant = "K_obj"}) -> tensor<i32> {
  return %arg0 : tensor<i32>
}

func.func @test_call_symbol_mapping(%arg0: tensor<?xf32>) -> tensor<i32> {
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  %c1 = stablehlo.constant dense<10> : tensor<i32>
  %result = call @callee(%0, %c1) {test_tag = "call_op"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}

// CHECK: === Analysis for entrypoint: test_call_symbol_mapping ===
// CHECK: Dimension Equivalence Classes:
// CHECK-DAG: Class [{{.*}}Symbol(n){{.*}}]
// CHECK-DAG: Class [{{.*}}Symbol(K_obj){{.*}}]
// CHECK-NOT: Class [{{.*}}]
// CHECK: Symbol mappings:
// CHECK:   Function: {{callee$}}
// CHECK:     Arg 0 is symbol: n
// CHECK:     Arg 1 is symbol: K_obj

// -----

// Test get_dimension_size linking tensor dims to SSA values
func.func private @callee2(%arg0: tensor<i32> {jax.global_constant = "n"}) -> tensor<i32> {
  return %arg0 : tensor<i32>
}

func.func @test_get_dimension_size(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<i32> {
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 {test_tag = "dim_size_0"} : (tensor<?xf32>) -> tensor<i32>
  %1 = stablehlo.get_dimension_size %arg1, dim = 1 {test_tag = "dim_size_1"} : (tensor<?x?xf32>) -> tensor<i32>
  %result = call @callee2(%0) : (tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}

// CHECK: === Analysis for entrypoint: test_get_dimension_size ===
// CHECK: Dimension Equivalence Classes:
// CHECK: Class [{{.*}}]
// CHECK: Operation: dim_size_0
// CHECK:   Result symbol: n
// CHECK: Operation: dim_size_1
// CHECK:   Result symbol: <unknown>
// CHECK: Symbol mappings:
// CHECK:   Function: callee2
// CHECK:     Arg 0 is symbol: n

// -----

// Test shape assertion equality constraints
func.func @test_shape_assertion(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<i32> {
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 {test_tag = "dim_arg0_0"} : (tensor<?xf32>) -> tensor<i32>
  %1 = stablehlo.get_dimension_size %arg1, dim = 1 {test_tag = "dim_arg1_1"} : (tensor<?x?xf32>) -> tensor<i32>
  %2 = stablehlo.compare EQ, %0, %1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%2, %0, %1) {api_version = 2 : i32, error_message = "dims must be equal", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> ()
  return %0 : tensor<i32>
}

// CHECK: === Analysis for entrypoint: test_shape_assertion ===
// CHECK: Dimension Equivalence Classes:
// The two dimension sizes should be in the same equivalence class
// CHECK-NEXT: Class [{{.*}}Dim({{.*}}, 0){{.*}}Dim({{.*}}, 1){{.*}}]
//  CHECK-NOT: Class [{{.*}}]
// CHECK: Symbol mappings:

// -----

// Test transitive equality through symbols
func.func private @callee3(%arg0: tensor<i32> {jax.global_constant = "n"},
                           %arg1: tensor<i32> {jax.global_constant = "n"}) -> tensor<i32> {
  return %arg0 : tensor<i32>
}

func.func @test_transitive_equality(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<i32> {
  // Both dimension sizes are passed as "n", so they should be equal
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 {test_tag = "dim_0"} : (tensor<?xf32>) -> tensor<i32>
  %1 = stablehlo.get_dimension_size %arg1, dim = 1 {test_tag = "dim_1"} : (tensor<?x?xf32>) -> tensor<i32>
  %result = call @callee3(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}

// CHECK: === Analysis for entrypoint: test_transitive_equality ===
// CHECK: Dimension Equivalence Classes:
// Both dims should be in the same class with Symbol(n)
// CHECK-NEXT:  Class [{{.*}}Symbol(n){{.*}}]
// CHECK-NOT: Class [{{.*}}]
// CHECK: Operation: dim_0
// CHECK:   Result symbol: n
// CHECK: Operation: dim_1
// CHECK:   Result symbol: n

// -----

// Test multiple symbols in a complex function
func.func private @wrapped_main(%arg0: tensor<i32> {jax.global_constant = "K_obj"},
                                %arg1: tensor<i32> {jax.global_constant = "n"},
                                %arg2: tensor<i32> {jax.global_constant = "num_eqs"},
                                %arg3: tensor<?xf32>,
                                %arg4: tensor<?x?xf32>,
                                %arg5: tensor<?x?xf32>) -> tensor<?xf32> {
  return %arg3 : tensor<?xf32>
}

func.func @test_multiple_symbols(%arg0: tensor<?xf32>,
                                  %arg1: tensor<?x?xf32>,
                                  %arg2: tensor<?x?xf32>) -> tensor<?xf32> {
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 {test_tag = "n_dim"} : (tensor<?xf32>) -> tensor<i32>
  %1 = stablehlo.get_dimension_size %arg1, dim = 0 {test_tag = "K_obj_dim"} : (tensor<?x?xf32>) -> tensor<i32>
  %2 = stablehlo.get_dimension_size %arg1, dim = 1 {test_tag = "n_dim2"} : (tensor<?x?xf32>) -> tensor<i32>
  %3 = stablehlo.get_dimension_size %arg2, dim = 0 {test_tag = "num_eqs_dim"} : (tensor<?x?xf32>) -> tensor<i32>

  // Shape assertion: arg1.shape[1] == arg0.shape[0] (both are 'n')
  %cmp = stablehlo.compare EQ, %2, %0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  stablehlo.custom_call @shape_assertion(%cmp, %2, %0) {api_version = 2 : i32, error_message = "n must match", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> ()

  %result = call @wrapped_main(%1, %0, %3, %arg0, %arg1, %arg2) {test_tag = "main_call"} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// CHECK: === Analysis for entrypoint: test_multiple_symbols ===
// CHECK: Dimension Equivalence Classes:
// n_dim and n_dim2 should be in the same class with Symbol(n)
// CHECK-DAG: Class [{{.*}}Symbol(n){{.*}}]
// CHECK-DAG: Class [{{.*}}Symbol(K_obj){{.*}}]
// CHECK-DAG: Class [{{.*}}Symbol(num_eqs){{.*}}]
// CHECK: Operation: n_dim
// CHECK:   Result symbol: n
// CHECK: Operation: K_obj_dim
// CHECK:   Result symbol: K_obj
// CHECK: Operation: n_dim2
// CHECK:   Result symbol: n
// CHECK: Operation: num_eqs_dim
// CHECK:   Result symbol: num_eqs
// CHECK: Symbol mappings:
// CHECK:   Function: wrapped_main
// CHECK:     Arg 0 is symbol: K_obj
// CHECK:     Arg 1 is symbol: n
// CHECK:     Arg 2 is symbol: num_eqs

// -----

// Test plan.with_shape, arith.index_cast, and tensor.from_elements linking
func.func private @callee_with_shape(%arg0: tensor<i32> {jax.global_constant = "n"},
                                     %arg1: tensor<i32> {jax.global_constant = "m"}) -> tensor<i32> {
  return %arg0 : tensor<i32>
}

func.func @test_with_shape(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim {test_tag = "with_shape_d0"}  %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim {test_tag = "with_shape_d1"} %arg0, %c1 : tensor<?x?xf32>

  // plan.with_shape ties the tensor to its dimension values
  %shaped = plan.with_shape %arg0(%d0, %d1) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>

  // Convert to i32 for the call - arith.index_cast propagates equality
  %d0_i32 = arith.index_cast %d0 : index to i32
  %d1_i32 = arith.index_cast %d1 : index to i32
  // tensor.from_elements propagates equality for single-element tensors
  %d0_tensor = tensor.from_elements %d0_i32 : tensor<i32>
  %d1_tensor = tensor.from_elements %d1_i32 : tensor<i32>

  // Call with dimension values - should link d0 to "n" and d1 to "m"
  %result = call @callee_with_shape(%d0_tensor, %d1_tensor) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  return %shaped : tensor<?x?xf32>
}

// CHECK: Dimension Equivalence Classes:
// Should have exactly 2 classes: one for dim 0 (n) and one for dim 1 (m)
// All values flow through: tensor.dim -> plan.with_shape -> arith.index_cast -> tensor.from_elements -> func.call
// CHECK-DAG: Class [{{.*}}Dim({{.*}}, 0){{.*}}Symbol(n){{.*}}]
// CHECK-DAG: Class [{{.*}}Dim({{.*}}, 1){{.*}}Symbol(m){{.*}}]
// CHECK-NOT: Class [{{.*}}]
// CHECK: Operation: with_shape_d0
// CHECK: Operation: with_shape_d1

// -----

// Test stablehlo.reshape propagating equality for single-element tensors
func.func private @callee_reshape(%arg0: tensor<i32> {jax.global_constant = "dim_sym"}) -> tensor<i32> {
  return %arg0 : tensor<i32>
}

func.func @test_reshape(%arg0: tensor<?xf32>) -> tensor<i32> {
  // Get dimension as tensor<i32>
  %dim = stablehlo.get_dimension_size %arg0, dim = 0 {test_tag = "reshape_dim"} : (tensor<?xf32>) -> tensor<i32>

  // Reshape to tensor<1xi32> and back - should preserve equality
  %reshaped_1 = stablehlo.reshape %dim : (tensor<i32>) -> tensor<1xi32>
  %reshaped_back = stablehlo.reshape %reshaped_1 : (tensor<1xi32>) -> tensor<i32>

  // Call with the reshaped value - should link to "dim_sym"
  %result = call @callee_reshape(%reshaped_back) : (tensor<i32>) -> tensor<i32>

  return %result : tensor<i32>
}

// CHECK: Dimension Equivalence Classes:
// The reshape chain should preserve equality
// CHECK: Class [{{.*}}Symbol(dim_sym){{.*}}]
// CHECK-NOT: Class [{{.*}}]
// CHECK: Operation: reshape_dim
// CHECK:   Result symbol: dim_sym

// -----

// Negative test: assertions nested in scf.if are NOT processed
// This is intentional - we only analyze top-level operations to avoid
// constraints from conditional branches that may not always execute.
func.func @test_nested_assertion_ignored(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %cond: i1) -> tensor<i32> {
  %d0 = stablehlo.get_dimension_size %arg0, dim = 0 {test_tag = "nested_d0"} : (tensor<?xf32>) -> tensor<i32>
  %d1 = stablehlo.get_dimension_size %arg1, dim = 0 {test_tag = "nested_d1"} : (tensor<?xf32>) -> tensor<i32>

  // This assertion is inside scf.if, so it should be IGNORED
  scf.if %cond {
    %cmp = stablehlo.compare EQ, %d0, %d1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%cmp, %d0, %d1) {api_version = 2 : i32, error_message = "nested assertion", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> ()
  }

  return %d0 : tensor<i32>
}

// CHECK: === Analysis for entrypoint: test_nested_assertion_ignored ===
// CHECK: Dimension Equivalence Classes:
// d0 and d1 should be in SEPARATE classes since the assertion is nested
// CHECK: Class [{{.*}}Dim({{.*}}, 0){{.*}}]
// CHECK: Class [{{.*}}Dim({{.*}}, 0){{.*}}]
// CHECK-NOT: Class [{{.*}}]
// CHECK: Operation: nested_d0
// The dimensions should NOT have symbols since they're not linked
// CHECK:   Result symbol: <unknown>
// CHECK: Operation: nested_d1
// CHECK:   Result symbol: <unknown>
