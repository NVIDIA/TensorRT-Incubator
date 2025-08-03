// RUN: mlir-tensorrt-opt %s -split-input-file -verify-diagnostics


// expected-error @below {{max_shape size (2) must equal min_shape size (3)}}
#plan_shape_bounds = #plan.bounds<shape, [1, 2, 3], [4, 5]>

// -----

// expected-error @below {{min_shape must be pointwise less-than-or-equal-to max_shape, but min_shape[2] = 3 > max_shape[2] = 2}}
#plan_shape_bounds = #plan.bounds<shape, [1, 2, 3], [4, 5, 2]>

// -----

// expected-error @below {{invalid kind of attribute specified}}
#plan_value_bounds = #plan.bounds<value, [1, 2, 3], [4, 5]>

// -----


// expected-error @below {{min_values type ('tensor<3xi64>') and max_values type ('tensor<2xi64>') must be the same}}
#plan_value_bounds = #plan.bounds<value, dense<[1, 2, 3]> : tensor<3xi64>, dense<[4, 5]> : tensor<2xi64>>

// -----

// expected-error @below {{min_values must be pointwise less-than-or-equal-to max_values, but min_values[0] = 1 > max_values[0] = 0}}
#plan_value_bounds = #plan.bounds<value, dense<[1, 2, 3]> : tensor<3xi64>, dense<[0, 1, 2]> : tensor<3xi64>>

// -----

// expected-error @below {{min_values must be pointwise less-than-or-equal-to max_values, but min_values[1, 1] = 5 > max_values[1, 1] = 4}}
#plan_value_bounds = #plan.bounds<value, dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>, dense<[[7, 8, 9], [10, 4, 7]]> : tensor<2x3xi64>>

// -----

// expected-error @below {{min_values must be pointwise less-than-or-equal-to max_values, but min_values[1, 1] = 5.00 > max_values[1, 1] = 4.00}}
#plan_value_bounds = #plan.bounds<value, dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>,
                                         dense<[[7., 8., 9.], [10., 4., 7.]]> : tensor<2x3xf32>>

// -----

#bounds = #plan.bounds<value, dense<10> : tensor<1x10xi32>, dense<20> : tensor<1x10xi32>>
// expected-error @below {{'func.func' op arg #0 expected type of values bounds elements ('tensor<1x10xi32>') to be compatible with the type ('tensor<1x11xi32>')}}
func.func @value_bounds_shape_mismatch(%arg0: tensor<1x11xi32> {plan.value_bounds = #bounds}) {
  return
}


// -----

#bounds = #plan.bounds<shape, [10], [20]>

// expected-error @below {{'func.func' op arg #0 has type 'tensor<1x?xi32>', whose rank is not equal to the rank of the corresponding shape bounds #plan.bounds<shape, [10], [20]>}}
func.func @value_bounds_shape_mismatch(%arg0: tensor<1x?xi32> {plan.shape_profile = #bounds}) {
  return
}

// -----

#bounds = #plan.bounds<shape, [], []>
func.func @value_bounds_shape_0d_match(%arg0: tensor<i32> {plan.shape_profile = #bounds}) {
  return
}

// -----

#bounds = #plan.bounds<shape, [10], [20]>
// expected-error @below {{'func.func' op expected only value bounds or none bounds for scalar arg #0 of type 'i32', but got #plan.bounds<shape, [10], [20]>}}
func.func @value_bounds_shape_mismatch(%arg0: i32 {plan.shape_profile = #bounds}) {
  return
}

// -----

#bounds = #plan.bounds<value,  dense<10> : tensor<1xi32>, dense<20> : tensor<1xi32>>

// expected-error @below {{'func.func' op arg #0 expected type of values bounds elements ('tensor<1xi32>') to be compatible with the type ('tensor<i32>')}}
func.func @value_bounds_0rank_shape_mismatch(%arg0: tensor<i32> {plan.value_bounds = #bounds}) {
  return
}

// -----

#bounds = #plan.bounds<value, dense<10> : tensor<1x11xi32>, dense<20> : tensor<1x11xi32>>

// expected-error @below {{'func.func' op arg #0 expected element type of value bounds elements ('i32') to be compatible with the type ('tensor<1x11xi64>')}}
func.func @value_bounds_element_type_mismatch(%arg0: tensor<1x11xi64> {plan.value_bounds = #bounds}) {
  return
}

// -----

#bounds = #plan.bounds<value, dense<10> : tensor<1xi32>,dense<20> : tensor<1xi32>>

// expected-error @below {{'func.func' op arg #0 type expects rank-0 value bounds type, but got 'tensor<1xi32>'}}
func.func @value_bounds_scalar_shape_mismatch(%arg0: i32 {plan.value_bounds = #bounds}) {
  return
}

// -----

#bounds = #plan.bounds<value,  dense<10> : tensor<i32>, dense<20> : tensor<i32>>

func.func @value_bounds_scalar_shape_ok(%arg0: i32 {plan.value_bounds = #bounds}) {
  return
}


// -----

func.func @plan_inline_group_mismatched_result_types(%arg0: tensor<10xf32>, %arg1: index) {
  // expected-error @below {{'plan.inline_group' op expected types of yielded operands ('tensor<10xf32>', 'index') to equal types of results ('index', 'tensor<10xf32>')}}
  plan.inline_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>) -> index, tensor<10xf32> {
    yield %arg0, %arg1 : tensor<10xf32>, index
  }
  return
}

// -----

func.func @inline_closed_group_wrong_num_block_args(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'plan.inline_closed_group' op  region control flow edge from parent operands to Region #0: source has 3 operands, but target successor needs 4}}
  %2 = plan.inline_closed_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<?xf32>, index)
    outs(%arg2 : tensor<?xf32>)
    in_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>]
    res_attrs [#plan.bounds<shape, [10], [20]>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index, %out0: tensor<?xf32>, %out1: tensor<?xf32>):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}

// -----

func.func @inline_closed_alloc_group_wrong_num_block_args(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'plan.inline_closed_alloc_group' op  region control flow edge from parent operands to Region #0: source has 2 operands, but target successor needs 3}}
  %2 = plan.inline_closed_alloc_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<?xf32>, index)
    in_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index, %out0: tensor<?xf32>):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}

// -----

func.func @inline_closed_group_wrong_size_bounds_attrs(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'plan.inline_closed_group' op expected number of inputs (2) to equal the number of input_attrs BoundsAttrs (0)}}
  %2 = plan.inline_closed_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<?xf32>, index)
    outs(%arg2 : tensor<?xf32>)
    in_attrs []
    res_attrs [#plan.bounds<shape, [10], [20]>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index, %out0: tensor<?xf32>):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}

// -----

func.func @inline_closed_alloc_group_wrong_size_bounds_attrs(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'plan.inline_closed_alloc_group' op expected number of inputs (2) to equal the number of input_attrs BoundsAttrs (0)}}
  %2 = plan.inline_closed_alloc_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<?xf32>, index)
    in_attrs [] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}

// -----

func.func @inline_closed_group_wrong_scalar_bounds_type(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'plan.inline_closed_group' op expected only value bounds or none bounds for scalar inputs #0 of type 'index', but got #plan.bounds<shape, [10], [20]>}}
  %2 = plan.inline_closed_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg1, %arg0 : index, tensor<?xf32>)
    outs(%arg2 : tensor<?xf32>)
    in_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>]
    res_attrs [#plan.bounds<shape, [10], [20]>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index, %out0: tensor<?xf32>):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}

// -----

func.func @inline_closed_group_alloc_wrong_scalar_bounds_type(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'plan.inline_closed_alloc_group' op expected only value bounds or none bounds for scalar inputs #0 of type 'index', but got #plan.bounds<shape, [10], [20]>}}
  %2 = plan.inline_closed_alloc_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg1, %arg0 : index, tensor<?xf32>)
    in_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}

// -----

func.func @inline_closed_group_wrong_bounds_type(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'plan.inline_closed_group' op inputs #0 has type 'tensor<?xf32>', whose rank is not equal to the rank of the corresponding shape bounds #plan.bounds<shape, [10, 10], [20, 20]>}}
  %2 = plan.inline_closed_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<?xf32>, index)
    outs(%arg2 : tensor<?xf32>)
    in_attrs [#plan.bounds<shape, [10, 10], [20, 20]>, #plan.bounds<none>]
    res_attrs [#plan.bounds<shape, [10], [20]>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index, %out0: tensor<?xf32>):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}
// -----

func.func @inline_closed_alloc_group_wrong_bounds_type(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'plan.inline_closed_alloc_group' op inputs #0 has type 'tensor<?xf32>', whose rank is not equal to the rank of the corresponding shape bounds #plan.bounds<shape, [10, 10], [20, 20]>}}
  %2 = plan.inline_closed_alloc_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<?xf32>, index)
    in_attrs [#plan.bounds<shape, [10, 10], [20, 20]>, #plan.bounds<none>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}

// -----

func.func @inline_closed_group_wrong_bounds_type(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'plan.inline_closed_group' op inputs #0 has type 'tensor<?xf32>', but has a corresponding bounds attribute of 'value' kind, which is only allowed for staticly shaped operands}}
  %2 = plan.inline_closed_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<?xf32>, index)
    outs(%arg2 : tensor<?xf32>)
    in_attrs [#plan.bounds<value, dense<[10]> : tensor<1xi64>, dense<[10]> : tensor<1xi64>>,
              #plan.bounds<none>]
    res_attrs [#plan.bounds<shape, [10], [20]>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index, %out0: tensor<?xf32>):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}

// -----

func.func @inline_closed_alloc_group_wrong_bounds_type(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{'plan.inline_closed_alloc_group' op inputs #0 has type 'tensor<?xf32>', but has a corresponding bounds attribute of 'value' kind, which is only allowed for staticly shaped operands}}
  %2 = plan.inline_closed_alloc_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<?xf32>, index)
    in_attrs [#plan.bounds<value, dense<[10]> : tensor<1xi64>, dense<[10]> : tensor<1xi64>>,
              #plan.bounds<none>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: index):
    %2 = plan.with_shape %in0 (%in1) : (tensor<?xf32>, index) -> tensor<?xf32>
    %res = stablehlo.exponential %2 : tensor<?xf32>
    yield %res : tensor<?xf32>
  }
  return %2 : tensor<?xf32>
}

// -----

func.func @inline_closed_group_wrong_bounds_type(%arg0: tensor<1xf32>, %arg1: index, %arg2: tensor<1xf32>) -> tensor<1xf32> {
  // expected-error @below {{'plan.inline_closed_group' op inputs #0 expected element type of value bounds elements ('i64') to be compatible with the type ('tensor<1xf32>')}}
  %2 = plan.inline_closed_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<1xf32>, index)
    outs(%arg2 : tensor<1xf32>)
    in_attrs [#plan.bounds<value, dense<[10]> : tensor<1xi64>, dense<[10]> : tensor<1xi64>>,
              #plan.bounds<none>]
    res_attrs [#plan.bounds<shape, [10], [20]>] -> tensor<1xf32> {
  ^bb0(%in0: tensor<1xf32>, %in1: index, %out0: tensor<1xf32>):
    %2 = plan.with_shape %in0 (%in1) : (tensor<1xf32>, index) -> tensor<1xf32>
    %res = stablehlo.exponential %2 : tensor<1xf32>
    yield %res : tensor<1xf32>
  }
  return %2 : tensor<1xf32>
}


// -----

func.func @inline_closed_alloc_group_wrong_bounds_type(%arg0: tensor<1xf32>, %arg1: index, %arg2: tensor<1xf32>) -> tensor<1xf32> {
  // expected-error @below {{'plan.inline_closed_alloc_group' op inputs #0 expected element type of value bounds elements ('i64') to be compatible with the type ('tensor<1xf32>')}}
  %2 = plan.inline_closed_alloc_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<1xf32>, index)
    in_attrs [#plan.bounds<value, dense<[10]> : tensor<1xi64>, dense<[10]> : tensor<1xi64>>,
              #plan.bounds<none>] -> tensor<1xf32> {
  ^bb0(%in0: tensor<1xf32>, %in1: index):
    %2 = plan.with_shape %in0 (%in1) : (tensor<1xf32>, index) -> tensor<1xf32>
    %res = stablehlo.exponential %2 : tensor<1xf32>
    yield %res : tensor<1xf32>
  }
  return %2 : tensor<1xf32>
}

// -----

func.func @inline_closed_group_wrong_bounds_type(%arg0: tensor<1xf32>, %arg1: index, %arg2: tensor<1xf32>) -> tensor<1xf32> {
  // expected-error @below {{'plan.inline_closed_group' op inputs #0 expected type of values bounds elements ('tensor<2xf32>') to be compatible with the type ('tensor<1xf32>')}}
  %2 = plan.inline_closed_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<1xf32>, index)
    outs(%arg2 : tensor<1xf32>)
    in_attrs [#plan.bounds<value, dense<[10., 20.]> : tensor<2xf32>, dense<[20., 30.]> : tensor<2xf32>>,
              #plan.bounds<none>]
    res_attrs [#plan.bounds<shape, [10], [20]>] -> tensor<1xf32> {
  ^bb0(%in0: tensor<1xf32>, %in1: index, %out0: tensor<1xf32>):
    %2 = plan.with_shape %in0 (%in1) : (tensor<1xf32>, index) -> tensor<1xf32>
    %res = stablehlo.exponential %2 : tensor<1xf32>
    yield %res : tensor<1xf32>
  }
  return %2 : tensor<1xf32>
}

// -----

func.func @inline_closed_alloc_group_wrong_bounds_type(%arg0: tensor<1xf32>, %arg1: index, %arg2: tensor<1xf32>) -> tensor<1xf32> {
  // expected-error @below {{'plan.inline_closed_alloc_group' op inputs #0 expected type of values bounds elements ('tensor<2xf32>') to be compatible with the type ('tensor<1xf32>')}}
  %2 = plan.inline_closed_alloc_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0, %arg1 : tensor<1xf32>, index)
    in_attrs [#plan.bounds<value, dense<[10., 20.]> : tensor<2xf32>, dense<[20., 30.]> : tensor<2xf32>>,
              #plan.bounds<none>] -> tensor<1xf32> {
  ^bb0(%in0: tensor<1xf32>, %in1: index):
    %2 = plan.with_shape %in0 (%in1) : (tensor<1xf32>, index) -> tensor<1xf32>
    %res = stablehlo.exponential %2 : tensor<1xf32>
    yield %res : tensor<1xf32>
  }
  return %2 : tensor<1xf32>
}

// -----

func.func @inline_closed_alloc_group_missing_input_attr(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %1 = plan.inline_closed_alloc_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
  // expected-error @below {{'plan.inline_closed_alloc_group' expected 'in_attrs'}}
    inputs(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  ^bb0(%in0: tensor<1xf32>):
    yield %in0 : tensor<1xf32>
  }
  return %1 : tensor<1xf32>
}

// -----

func.func @inline_closed_alloc_group_unallowed_res_attr(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %1 = plan.inline_closed_alloc_group target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>)
    inputs(%arg0: tensor<?xf32>)
  // expected-error @below {{expected '->'}}
    in_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>]
    res_attrs [#plan.bounds<shape, [10], [20]>, #plan.bounds<none>] -> tensor<?xf32> {
  ^bb0(%in0: tensor<?xf32>):
    yield %in0 : tensor<?xf32>
  }
  return %1 : tensor<?xf32>
}

// -----

func.func @with_shape_invalid(%arg0: index, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @below {{'plan.with_shape' op expected number of shape dimension extent values (1) to equal the operand type and result type rank (2)}}
  %0 = plan.with_shape %arg1 (%arg0) : (tensor<?x?xf32>, index) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @with_shape_invalid(%arg0: index, %arg1: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %c1 = arith.constant 1 : index
  // expected-error @below {{'plan.with_shape' op dimension #1 is equal to 2, but the corresponding index value can be constant-folded to 1 : index}}
  %0 = plan.with_shape %arg1 (%arg0, %c1) : (tensor<?x2xf32>, index, index) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}
