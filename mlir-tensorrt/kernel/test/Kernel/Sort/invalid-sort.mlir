// RUN: kernel-opt -verify-diagnostics -split-input-file %s

func.func @invalid_sort_no_inputs() {
  // expected-error @+1 {{expected one or two input operands}}
  kernel.sort () <block_threads = 128, items_per_thread = 4>
  return
}

// -----

func.func @too_many_args(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>)
  -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
  // expected-error @below {{'kernel.sort' op expected one or two input operands}}
  %0, %1, %2 = kernel.sort (%arg0, %arg1, %arg2) <block_threads = 128, items_per_thread = 4>
    : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
  return %0, %1, %2 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
}

// -----

func.func @invlid_result_type(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>)
   -> (tensor<10xf32>, tensor<10xf32>) {
  // expected-error @below {{'kernel.sort' op expected result types to match input types}}
  %0, %1 = "kernel.sort"(%arg0, %arg1) {
    block_threads = 128 : i64, items_per_thread = 4 : i64
  } : (tensor<?xf32>, tensor<?xf32>) -> (tensor<10xf32>, tensor<10xf32>)
  return %0, %1 : tensor<10xf32>, tensor<10xf32>
}
