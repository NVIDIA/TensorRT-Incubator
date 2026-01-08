// RUN: mlir-tensorrt-opt %s -split-input-file \
// RUN:  -plan-clustering -plan-create-closed-regions -plan-outline-clusters \
// RUN: | FileCheck %s

builtin.module attributes {
  plan.backends = [#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>]
} {

  func.func @dont_offload_slices_of_block_arguments(%arg0: tensor<4xf32>) -> (tensor<2xf32>) {
    %1 = stablehlo.slice %arg0[0:2] : (tensor<4xf32>) -> tensor<2xf32>
    return %1: tensor<2xf32>
  }
}

// CHECK-LABEL: func.func @dont_offload_slices_of_block_arguments
// CHECK-NOT: #plan.tensorrt_backend

// -----

builtin.module attributes {
  plan.backends = [#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>]
} {

  func.func @dont_offload_dynamic_slices_of_block_arguments(%arg0: tensor<16xf32>, %arg1: tensor<i32>) -> (tensor<4xf32>) {
    %0 = "stablehlo.dynamic_slice"(%arg0, %arg1) {
      slice_sizes = array<i64: 4>
    } : (tensor<16xf32>, tensor<i32>) -> tensor<4xf32>
    return %0: tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @dont_offload_dynamic_slices_of_block_arguments
// CHECK-NOT: #plan.tensorrt_backend

// -----

builtin.module attributes {
  plan.backends = [#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>]
} {

  func.func @dont_offload_dynamic_update_slices_of_block_arguments(%arg0: tensor<128xf32>, %arg1: tensor<4xf32>, %arg2: tensor<i32>) -> (tensor<128xf32>) {
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2 : (tensor<128xf32>, tensor<4xf32>, tensor<i32>) -> tensor<128xf32>
    return %0: tensor<128xf32>
  }
}

// CHECK-LABEL: func.func @dont_offload_dynamic_update_slices_of_block_arguments
// CHECK-NOT: #plan.tensorrt_backend

// -----

builtin.module attributes {
  plan.backends = [#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>]
} {

  func.func @dont_offload_real_dynamic_slices_of_block_arguments(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>) -> (tensor<?xf32>) {
    %0 = stablehlo.real_dynamic_slice %arg0, %arg1, %arg2, %arg3 : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
    return %0: tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @dont_offload_real_dynamic_slices_of_block_arguments
// CHECK-NOT: #plan.tensorrt_backend

// -----

// Test that operations yielded from loops are not offloaded to TensorRT.

builtin.module attributes {
  plan.backends = [#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>]
} {

  func.func @dont_offload_ops_yielded_from_loops(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>)
      -> (tensor<128xf32>, tensor<128xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %result:2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %arg0, %acc1 = %arg1)
      -> (tensor<128xf32>, tensor<128xf32>) { // FIXME: change to (tensor<128xf32>, tensor<16x8xf32>)
      %0 = stablehlo.add %acc, %arg1 : tensor<128xf32>
      %1 = stablehlo.reshape %acc1 : (tensor<128xf32>) -> tensor<16x8xf32>
      %2 = stablehlo.reshape %0 : (tensor<128xf32>) -> tensor<16x8xf32>
      %3 = stablehlo.add %1, %2 : tensor<16x8xf32>
      %4 = stablehlo.reshape %3 : (tensor<16x8xf32>) -> tensor<128xf32>
      scf.yield %0, %4 : tensor<128xf32>, tensor<128xf32>
    }
    return %result#0, %result#1 : tensor<128xf32>, tensor<128xf32>
  }
}

// CHECK-LABEL: func.func @dont_offload_ops_yielded_from_loops
// CHECK-NOT: #plan.tensorrt_backend
