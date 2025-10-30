// RUN: mlir-tensorrt-opt %s -split-input-file -verify-diagnostics

func.func @valid_bounds_attributes(%arg0: !executor.ptr<host> {
    executor.abi = #executor.arg<byval, memref<?xf32>>,
    plan.shape_bounds = #plan.bounds<shape, [10], [20]>
  },
  %arg1: !executor.ptr<host> {
    executor.abi = #executor.arg<byval, memref<i32>>,
    plan.value_bounds = #plan.bounds<value, dense<1> : tensor<i32>, dense<4> : tensor<i32>>
  },
  %arg2: !executor.ptr<host> {
    executor.abi = #executor.arg<byref, memref<?xf32>>,
    plan.shape_bounds = #plan.bounds<shape, [10], [20]>
  }) attributes {
    executor.func_abi = (memref<?xf32>, memref<i32>) -> (memref<?xf32>)
} {
  return
}

// -----

// expected-error @below {{'func.func' op arg #0 has type 'memref<10x?xf32>', whose rank is not equal to the rank of the corresponding shape bounds #plan.bounds<shape, [10], [20]>}}
func.func @invalid_shape_bound(%arg0: !executor.ptr<host> {
    executor.abi = #executor.arg<byval, memref<10x?xf32>>,
    plan.shape_bounds = #plan.bounds<shape, [10], [20]>
  },
  %arg2: !executor.ptr<host> {
    executor.abi = #executor.arg<byref, memref<?xf32>>,
    plan.shape_bounds = #plan.bounds<shape, [10], [20]>
  }) attributes {
    executor.func_abi = (memref<10x?xf32>) -> (memref<?xf32>)
} {
  return
}
