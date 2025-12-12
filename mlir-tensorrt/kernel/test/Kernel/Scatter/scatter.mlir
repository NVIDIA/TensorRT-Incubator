// RUN: kernel-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// CHECK: func @scatter
!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

// CHECK: func @scatter_with_batching_dims
!input_type = tensor<5x200x100x300xf32>
!updates_type = tensor<5x10x300xf32>
!indices_type = tensor<5x10x2xi32>
func.func @scatter_with_batching_dims(%input_tensor: tensor<5x200x100x300xf32>,
    %scatter_indices: tensor<5x10x2xi32>, %updates: tensor<5x10x300xf32>) ->
      tensor<5x200x100x300xf32> {
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
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
  } : tensor<5x200x100x300xf32>
  func.return %0 : tensor<5x200x100x300xf32>
}

// -----

// CHECK: func @valid_scatter_dimensions_with_dynamic_index_vector_dim
!input_type = tensor<?x?x?xf32>
!updates_type = tensor<?x?xf32>
!indices_type = tensor<10x?xi32>
func.func @valid_scatter_dimensions_with_dynamic_index_vector_dim(
    %input_tensor: tensor<?x?x?xf32>, %scatter_indices: tensor<10x?xi32>,
    %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1, 2>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_with_promotable_types(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf64> {
  // expected-error @below {{'kernel.scatter' op expected type of operand #2 ('tensor<200x100x300xf32>') to match type of corresponding result ('tensor<200x100x300xf64>')}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f64, %rhs: f64):
    %add = arith.addf %lhs, %rhs : f64
    kernel.yield %add : f64
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf64>
  func.return %0 : tensor<200x100x300xf64>
}

// -----

!update_type = tensor<1xi32>
!input_type = tensor<3xi32>
!indices_type = tensor<1x1xi32>

func.func @scatter_c1(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                            %arg2: tensor<1xi32>) -> tensor<3xi32> {  
  // expected-error @below {{Not all inputs have compatible shapes.}}
  %0, %1 = kernel.scatter updates(%arg2, %arg2 : !update_type, !update_type) 
     into (%arg0, %arg2 : !input_type, !update_type) at (%arg1 : !indices_type) {
  ^bb0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32):
    kernel.yield %arg3, %arg5 : i32, i32
  } {    
      update_window_dims = array<i64>,
      inserted_window_dims = array<i64: 0>,
      scatter_dims_to_operand_dims = array<i64: 0>,
      index_vector_dim = 1    
  } : tensor<3xi32>, tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<?x?xf32>
!indices_type = tensor<?x?xi32>
func.func @scatter_c2(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<?x?xi32>, %updates: tensor<?x?xf32>) -> !input_type {
  // expected-error @+1 {{Expects rank-of operand to match size-of('update_window_dims') + size-of('inserted_window_dims') + size-of('input_batching_dims') i.e. 4 but got 3.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1, 2>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : !input_type
  func.return %0 : !input_type
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<?x?x?xf32>
!indices_type = tensor<?x?x?xi32>
func.func @scatter_c2(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) -> !input_type {
  // expected-error @+1 {{Expects rank-of operand to match size-of('update_window_dims') + size-of('inserted_window_dims') + size-of('input_batching_dims') i.e. 4 but got 3.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 2>,
      inserted_window_dims = array<i64: 1, 2>,
      input_batching_dims = array<i64: 0>,
      scatter_indices_batching_dims = array<i64: 0>,
      scatter_dims_to_operand_dims = array<i64: 1>,
      index_vector_dim = 2,
    indices_are_sorted,
    unique_indices 
  } : !input_type
  return %0 : !input_type
}

// -----

!input_type = tensor<3xi32>
!updates_type = tensor<1xi32>
!indices_type = tensor<1x1xi32>

func.func @scatter_c3(%input_tensor: tensor<3xi32>, %scatter_indices: tensor<1x1xi32>,
                            %updates: tensor<1xi32>) -> tensor<3xi32> {
  // expected-error @+1 {{Not all updates have compatible shapes.}}
  %0, %1 = kernel.scatter updates(%input_tensor, %updates : !input_type, !updates_type )
    into (%input_tensor, %input_tensor : !input_type, !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32):
    kernel.yield %arg3, %arg5 : i32, i32
  } {
    update_window_dims = array<i64>,
    inserted_window_dims = array<i64: 0>,
    scatter_dims_to_operand_dims = array<i64: 0>,
    index_vector_dim = 1    
  } : !input_type, !input_type
  func.return %0 : !input_type
}

// -----


!input_type = tensor<?x?x?xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c4(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{expects updates tensor must be of rank 3 ( == rank-of('scatter_indices') - 1 + size-of('update_window_dims'), where 'scatter_indices' is expanded by a trailing 1 dimension if 'index_vector_dim' == rank-of('scatter_indices')), but got 2.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1, 0>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c4(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{expects updates tensor must be of rank 3 ( == rank-of('scatter_indices') - 1 + size-of('update_window_dims'), where 'scatter_indices' is expanded by a trailing 1 dimension if 'index_vector_dim' == rank-of('scatter_indices')), but got 2.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 2,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x301xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c4(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x301xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{expects bounds of the window dimensions of updates to not exceed the bounds of the corresponding dimensions of operand. For dimension 1, updates bound is 301, operand bound is 300.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<11x2xi32>
func.func @scatter_c4(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<11x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{expects bounds of the scatter dimensions of updates to be same as the bounds of the corresponding dimensions of scatter indices. For scatter dimension 0, updates bound is 10 , scatter_indices bound is 11.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}


// -----

!input_type = tensor<5x200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<5x10x2xi32>
func.func @scatter_c4(%input_tensor: tensor<5x200x100x300xf32>,
    %scatter_indices: tensor<5x10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<5x200x100x300xf32> {
  // expected-error @+1 {{expects updates tensor must be of rank 3 ( == rank-of('scatter_indices') - 1 + size-of('update_window_dims'), where 'scatter_indices' is expanded by a trailing 1 dimension if 'index_vector_dim' == rank-of('scatter_indices')), but got 2.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 1, 2>,
      input_batching_dims = array<i64: 0>,
      scatter_indices_batching_dims = array<i64: 0>,
      scatter_dims_to_operand_dims = array<i64: 1, 2>,
      index_vector_dim = 2,
    indices_are_sorted,
    unique_indices 
  } : tensor<5x200x100x300xf32>
  func.return %0 : tensor<5x200x100x300xf32>
}


// -----

// Note: this test is different from 'stablehlo.scatter' because
// we don't allow type promotion of the input.

!input_type = tensor<200x100x300xi32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c6_c23_c24(%input_tensor: tensor<200x100x300xi32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {  
  
  // expected-error @below {{'kernel.scatter' op expected type of operand #2 ('tensor<200x100x300xi32>') to match type of corresponding result ('tensor<200x100x300xf32>')}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs :  f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<512x1x6400x6400xf32>
!updates_type = tensor<512x1x6400x6400xf32>
!indices_type = tensor<1xi32>

func.func @scatter_c7() ->  tensor<512x1x6400x6400xf32> {
  %base = arith.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = arith.constant dense<0> : tensor<1xi32>
  %update = arith.constant dense<1.000000e+00> : tensor<512x1x6400x6400xf32>
  
  // expected-error @below {{Expects update_window_dims to be sorted; got: [0, 1, 3, 2].}}
  %scatter = kernel.scatter updates(%update : !updates_type) into (%base : !input_type) at (%index : !indices_type) {
  ^bb0(%arg5: f32, %arg6: f32):
    kernel.yield %arg6 : f32
  } {
    indices_are_sorted,
    update_window_dims = array<i64: 0, 1, 3, 2>,
    scatter_dims_to_operand_dims = array<i64: 3>,
    index_vector_dim = 0,
    unique_indices } :
    tensor<512x1x6400x6400xf32>
  func.return %scatter : tensor<512x1x6400x6400xf32>
}


// -----


!input_type = tensor<512x1x6400x6400xf32>
!updates_type = tensor<512x1x6400x6400xf32>
!indices_type = tensor<1xi32>
func.func @scatter_c7() ->  tensor<512x1x6400x6400xf32> {
  %base = arith.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = arith.constant dense<0> : tensor<1xi32>
  %update = arith.constant dense<1.000000e+00> : tensor<512x1x6400x6400xf32>
  // expected-error @+1 {{Expects update_window_dims to be sorted; got: [0, 1, 3, 2].}}
  %scatter = kernel.scatter updates(%update : !updates_type) into (%base : !input_type) 
    at (%index : !indices_type) {
  ^bb0(%arg5: f32, %arg6: f32):
    kernel.yield %arg6 : f32
  } {
    indices_are_sorted,
      update_window_dims = array<i64: 0, 1, 3, 2>,
      scatter_dims_to_operand_dims = array<i64: 3>,
      index_vector_dim = 0,
      unique_indices } :
    tensor<512x1x6400x6400xf32>
  func.return %scatter : tensor<512x1x6400x6400xf32>
}

// -----

!input_type = tensor<512x1x6400x6400xf32>
!updates_type = tensor<512x1x6400x6400xf32>
!indices_type = tensor<1xi32>

func.func @scatter_c7() ->  tensor<512x1x6400x6400xf32> {
  %base = arith.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = arith.constant dense<0> : tensor<1xi32>
  %update = arith.constant dense<1.000000e+00> : tensor<512x1x6400x6400xf32>
  // expected-error @+1 {{Expects update_window_dims to not repeat; got: [0, 1, 2, 2].}}
  %scatter = kernel.scatter updates(%update : !updates_type) into (%base : !input_type) 
    at (%index : !indices_type) {
  ^bb0(%arg5: f32, %arg6: f32):
    kernel.yield %arg6 : f32
  } {
    indices_are_sorted,
      update_window_dims = array<i64: 0, 1, 2, 2>,
      scatter_dims_to_operand_dims = array<i64: 3>,
      index_vector_dim = 0,
      unique_indices } :
    tensor<512x1x6400x6400xf32>
  func.return %scatter : tensor<512x1x6400x6400xf32>
}

// -----

!input_type = tensor<512x1x6400x6400xf32>
!updates_type = tensor<512x1x6400x6400xf32>
!indices_type = tensor<1xi32>

func.func @scatter_c8() ->  tensor<512x1x6400x6400xf32> {
  %base = arith.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = arith.constant dense<0> : tensor<1xi32>
  %update = arith.constant dense<1.000000e+00> : tensor<512x1x6400x6400xf32>
  // expected-error @+1 {{Expects each element of update_window_dims to be in range [0, rank-of('updates')) i.e. [0, 4). got: -1.}}
  %scatter = kernel.scatter updates(%update : !updates_type) into (%base : !input_type) 
    at (%index : !indices_type) {
  ^bb0(%arg5: f32, %arg6: f32):
    kernel.yield %arg6 : f32
  } {
      indices_are_sorted,
      update_window_dims = array<i64: -1, 0, 1, 2>,
      scatter_dims_to_operand_dims = array<i64: 3>,
      index_vector_dim = 0,
      unique_indices
    } :
    tensor<512x1x6400x6400xf32>
  func.return %scatter : tensor<512x1x6400x6400xf32>
}

// -----

!input_type = tensor<512x1x6400x6400xf32>
!updates_type = tensor<512x1x6400x6400xf32>
!indices_type = tensor<1xi32>

func.func @scatter_c8() ->  tensor<512x1x6400x6400xf32> {
  %base = arith.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = arith.constant dense<0> : tensor<1xi32>
  %update = arith.constant dense<1.000000e+00> : tensor<512x1x6400x6400xf32>
  // expected-error @+1 {{Expects each element of update_window_dims to be in range [0, rank-of('updates')) i.e. [0, 4). got: 4.}}
  %scatter = kernel.scatter updates(%update : !updates_type) into (%base : !input_type) 
    at (%index : !indices_type) {
  ^bb0(%arg5: f32, %arg6: f32):
    kernel.yield %arg6 : f32
  } {
    indices_are_sorted,
      update_window_dims = array<i64: 0, 1, 2, 4>,
      scatter_dims_to_operand_dims = array<i64: 3>,
      index_vector_dim = 0,
      unique_indices } :
    tensor<512x1x6400x6400xf32>
  func.return %scatter : tensor<512x1x6400x6400xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>

func.func @scatter_c9(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{has duplicated dimension from inserted_window_dims and input_batching_dims: 1}}
  %0 = kernel.scatter 
      updates(%updates : !updates_type) 
      into (%input_tensor : !input_type) 
      at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 1, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c9(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
// expected-error @+1 {{has duplicated dimension from inserted_window_dims and input_batching_dims: 0}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0>,
      input_batching_dims = array<i64: 0>,
      scatter_indices_batching_dims = array<i64: 0>,
      scatter_dims_to_operand_dims = array<i64: 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<?x?x?xf32>
!updates_type = tensor<?x?xf32>
!indices_type = tensor<?x?xi32>
func.func @scatter_c10(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?xi32>, %updates: tensor<?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects inserted_window_dims to be sorted; got: [1, 0].}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 1, 0>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}


// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c11(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Expects each element of inserted_window_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: -1.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: -1, 3>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c11(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Expects each element of inserted_window_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: 3.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 3>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<?x?x?xf32>
!updates_type = tensor<?x?xf32>
!indices_type = tensor<?x?xi32>
func.func @scatter_c12(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?xi32>, %updates: tensor<?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects input_batching_dims to be sorted; got: [1, 0].}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      input_batching_dims = array<i64: 1, 0>,
      scatter_indices_batching_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 2>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}




// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c13(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Expects each element of input_batching_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: -1.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 1>,
      input_batching_dims = array<i64: -1>,
      scatter_indices_batching_dims = array<i64: 0>,
      scatter_dims_to_operand_dims = array<i64: 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c13(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Expects each element of input_batching_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: 3.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 1>,
      input_batching_dims = array<i64: 3>,
      scatter_indices_batching_dims = array<i64: 0>,
      scatter_dims_to_operand_dims = array<i64: 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<?x?x?xf32>
!updates_type = tensor<?x?x?xf32>
!indices_type = tensor<?x?x?xi32>
func.func @scatter_c14(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects scatter_indices_batching_dims to not repeat; got: [1, 0, 1].}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 2>,
      input_batching_dims = array<i64: 0, 1>,
      scatter_indices_batching_dims = array<i64: 1, 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 2>,
      index_vector_dim = 2,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

!input_type = tensor<?x?x?xf32>
!updates_type = tensor<?x?x?xf32>
!indices_type = tensor<?x?x?xi32>
func.func @scatter_c15(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects each element of scatter_indices_batching_dims to be in range [0, rank-of('scatter_indices')) i.e. [0, 3). got: -1.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 2>,
      input_batching_dims = array<i64: 0, 1>,
      scatter_indices_batching_dims = array<i64: 1, -1>,
      scatter_dims_to_operand_dims = array<i64: 2>,
      index_vector_dim = 2,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

!input_type = tensor<?x?x?xf32>
!updates_type = tensor<?x?x?xf32>
!indices_type = tensor<?x?x?xi32>
func.func @scatter_c15(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects each element of scatter_indices_batching_dims to be in range [0, rank-of('scatter_indices')) i.e. [0, 3). got: 3.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 2>,
      input_batching_dims = array<i64: 0, 1>,
      scatter_indices_batching_dims = array<i64: 1, 3>,
      scatter_dims_to_operand_dims = array<i64: 2>,
      index_vector_dim = 2,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

!input_type = tensor<?x?x?xf32>
!updates_type = tensor<?x?x?xf32>
!indices_type = tensor<?x?x?xi32>
func.func @scatter_c16(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{expects scatter_indices_batching_dims not to include index_vector_dim 2.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 2>,
      input_batching_dims = array<i64: 0, 1>,
      scatter_indices_batching_dims = array<i64: 1, 2>,
      scatter_dims_to_operand_dims = array<i64: 2>,
      index_vector_dim = 2,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x5x300xf32>
!indices_type = tensor<10x5x2xi32>
func.func @scatter_c17(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x5x2xi32>, %updates: tensor<10x5x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{input_batching_dims and scatter_indices_batching_dims should have the same size.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 2>,
      inserted_window_dims = array<i64: 1>,
      input_batching_dims = array<i64: 0>,
      scatter_indices_batching_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 1>,
      index_vector_dim = 2,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<5x100x300xf32>
!updates_type = tensor<10x5x300xf32>
!indices_type = tensor<10x5x2xi32>
func.func @scatter_c18(%input_tensor: tensor<5x100x300xf32>,
    %scatter_indices: tensor<10x5x2xi32>, %updates: tensor<10x5x300xf32>) ->
      tensor<5x100x300xf32> {
  // expected-error @+1 {{input_batching_dims[1] and scatter_indices_batching_dims[1] must have compatible sizes, but got 100 and 10.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 2>,
      input_batching_dims = array<i64: 0, 1>,
      scatter_indices_batching_dims = array<i64: 1, 0>,
      scatter_dims_to_operand_dims = array<i64: 2>,
      index_vector_dim = 2,
    indices_are_sorted,
    unique_indices 
  } : tensor<5x100x300xf32>
  func.return %0 : tensor<5x100x300xf32>
}

// -----

!input_type = tensor<?x?x?xf32>
!updates_type = tensor<?x?xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c19(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Scatter op has 3 elements in scatter_dims_to_operand_dims and the bound of dimension index_vector_dim=1 of scatter_indices is 2. These two numbers must be equal.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1, 2>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

!input_type = tensor<?x?x?xf32>
!updates_type = tensor<?x?x?xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c19(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Scatter op has 3 elements in scatter_dims_to_operand_dims and the bound of dimension index_vector_dim=2 of scatter_indices is 1. These two numbers must be equal.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1, 2>,
      index_vector_dim = 2,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c20(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{has duplicated dimension from scatter_dims_to_operand_dims and input_batching_dims: 0}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 0>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<200x10x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c20(%input_tensor: tensor<200x10x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x10x300xf32> {
  // expected-error @+1 {{has duplicated dimension from scatter_dims_to_operand_dims and input_batching_dims: 1}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0>,
      input_batching_dims = array<i64: 1>,
      scatter_indices_batching_dims = array<i64: 0>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x10x300xf32>
  func.return %0 : tensor<200x10x300xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<?x?x?xf32>
!indices_type = tensor<?x?x?xi32>
func.func @scatter_c21(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<?x?x?xi32>, %updates: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects each element of scatter_dims_to_operand_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: -1.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: -1, 0>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<?x?xf32>
!indices_type = tensor<?x?xi32>
func.func @scatter_c21(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<?x?xi32>, %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects each element of scatter_dims_to_operand_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: 3.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 3>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

!input_type = tensor<?x?x?xf32>
!updates_type = tensor<?x?x?xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c22(%input_tensor: tensor<?x?x?xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<?x?x?xf32>) ->
      tensor<?x?x?xf32> {
  // expected-error @+1 {{Expects index_vector_dim to be in range [0, rank-of('scatter_indices')] i.e. [0, 2]. got: 3.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 3,
    indices_are_sorted,
    unique_indices 
  } : tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c22(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Expects index_vector_dim to be in range [0, rank-of('scatter_indices')] i.e. [0, 2]. got: -1.}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = -1,    
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Reduction-region must take 2 parameters, but takes 1 parameter(s)}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32):
    %add = arith.addf %lhs, %lhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{The reduction-region expected to return some value(s)}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    "kernel.yield"() : () -> ()
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error @+1 {{Reduction-region here must produce 1 values, but produces 2 instead}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    kernel.yield %add, %add : f32, f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}


// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error@+1 {{The type of reduction-region's parameter at index 0 is different than the corresponding result type: 'f32' vs 'i32'}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs :  f32
    %cst = arith.constant -1 : i32
    kernel.yield %cst : i32    
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// -----

!input_type = tensor<200x100x300xf32>
!updates_type = tensor<10x300xf32>
!indices_type = tensor<10x2xi32>
func.func @scatter_c23(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  // expected-error@+1 {{The type of reduction-region's parameter at index 1 is different than the corresponding result type: 'i32' vs 'f32'}}
  %0 = kernel.scatter updates(%updates : !updates_type) into (%input_tensor : !input_type) at (%scatter_indices : !indices_type) {
  ^bb0(%lhs: f32, %rhs: i32):
    %add = arith.addf %lhs, %lhs : f32
    kernel.yield %add : f32
  } {
      update_window_dims = array<i64: 1>,
      inserted_window_dims = array<i64: 0, 1>,
      scatter_dims_to_operand_dims = array<i64: 0, 1>,
      index_vector_dim = 1,
    indices_are_sorted,
    unique_indices 
  } : tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

