// RUN: mlir-tensorrt-opt %s -split-input-file -test-bounds-analysis 2>&1 | FileCheck %s

func.func @test_simple_static(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %0 = plan.cluster target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>) attributes {tag = "inline_group"} -> tensor<10xf32> {
    %1 = stablehlo.add %arg0, %arg1 : tensor<10xf32>
    yield %1 : tensor<10xf32>
  }
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: func test_simple_static:
// CHECK-NEXT:  arg #0: <[10, 10]>
// CHECK-NEXT:  arg #1: <[10, 10]>
// CHECK-NEXT: test_tag: inline_group:
// CHECK-NEXT:  result #0: <[10, 10]>

// -----

#profile0 = #plan.bounds<value, dense<10> : tensor<index>, dense<20> : tensor<index>>

func.func @test_forward_backward(%arg0: tensor<?xf32>, %arg1: index {plan.value_bounds = #profile0}) -> tensor<?xf32> {
  %0 = plan.with_shape {tag = "with_shape0"} %arg0(%arg1) : (tensor<?xf32>, index) -> tensor<?xf32>
  %1 = stablehlo.exponential %0 : tensor<?xf32>
  %2 = plan.with_shape {tag = "with_shape1"} %1(%arg1) : (tensor<?xf32>, index) -> tensor<?xf32>
  return {tag = "return"} %2 : tensor<?xf32>
}

// CHECK-LABEL: func test_forward_backward:
// CHECK-NEXT:  arg #0: <[10, 20]>
// CHECK-NEXT:  arg #1: <<uninitialized>>
// CHECK-NEXT: test_tag: with_shape0:
// CHECK-NEXT:  operand #0: <[10, 20]>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: <[10, 20]>
// CHECK-NEXT: test_tag: with_shape1:
// CHECK-NEXT:  operand #0: <[10, 20]>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: <[10, 20]>
// CHECK-NEXT: test_tag: return:
// CHECK-NEXT:  operand #0: <[10, 20]>

// -----

#profile0 = #plan.bounds<shape, [1, 128, 128], [4, 512, 512]>
#profile1 = #plan.bounds<shape, [1, 128, 128], [4, 512, 512]>

func.func @dot_general_c12(%arg0: tensor<?x?x?xf32> {plan.shape_bounds = #profile0},
                           %arg1: tensor<?x?x?xf32> {plan.shape_bounds = #profile1})
                          -> tensor<?x?x?xf32> {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %dim_0 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %dim_1 = tensor.dim %arg1, %c2 : tensor<?x?x?xf32>
  %0 = plan.cluster target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>) attributes {tag = "inline_group"} -> tensor<?x?x?xf32> {
    %1 = "stablehlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %2 = with_shape {tag = "with_shape"} %1(%dim, %dim_0, %dim_1) : (tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    yield %2 : tensor<?x?x?xf32>
  }
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: func dot_general_c12:
// CHECK-NEXT:  arg #0: <[1, 4], [128, 512], [128, 512]>
// CHECK-NEXT:  arg #1: <[1, 4], [128, 512], [128, 512]>
// CHECK-NEXT: test_tag: inline_group:
// CHECK-NEXT:  result #0: <[1, 4], [128, 512], [128, 512]>
// CHECK-NEXT:   Region #0:
// CHECK-NEXT: test_tag: with_shape:
// CHECK-NEXT:  operand #0: <[1, 4], [128, 512], [128, 512]>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  operand #2: <<uninitialized>>
// CHECK-NEXT:  operand #3: <<uninitialized>>
// CHECK-NEXT:  result #0: <[1, 4], [128, 512], [128, 512]>



// -----

#profile0 = #plan.bounds<shape, [1], [1]>

func.func @test_unneeded_dynamism(%arg0: tensor<?xf32> {plan.shape_bounds = #profile0}) -> tensor<?xf32> {
  %0 = stablehlo.constant dense<[1]> : tensor<1xi32>
  %c1 = arith.constant 1 : index
  %1 = plan.cluster target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>) -> tensor<?xf32> {
    %1 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %0) {broadcast_dimensions = array<i64: 0>, tag = "broadcast"} : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
    %2 = with_shape {tag = "with_shape"} %1 (%c1) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %2 : tensor<?xf32>
  }
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func test_unneeded_dynamism:
// CHECK-NEXT:  arg #0: <[1, 1]>
// CHECK-NEXT: test_tag: broadcast:
// CHECK-NEXT:  operand #0: <[1, 1]>
// CHECK-NEXT:  operand #1: <[1, 1]>
// CHECK-NEXT:  result #0: <[1, 1]>
// CHECK-NEXT: test_tag: with_shape:
// CHECK-NEXT:  operand #0: <[1, 1]>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: <[1, 1]>

// -----

func.func @test_loop_concat(
    %arg0: tensor<1xf32>,
    %arg1: tensor<1xi32>
      {plan.value_bounds = #plan.bounds<value, dense<[1]> : tensor<1xi32>, dense<[4]> : tensor<1xi32>>},
    %arg2: tensor<?xf32>
      {plan.shape_bounds = #plan.bounds<shape, [2], [6]>})
    -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %extracted = tensor.extract %arg1[%c0] : tensor<1xi32>
  %0 = arith.index_cast %extracted : i32 to index
  %1 = scf.for %arg3 = %c0 to %0 step %c1 iter_args(%arg4 = %arg2) -> (tensor<?xf32>) {
    %dim = tensor.dim %arg4, %c0 : tensor<?xf32>
    %2 = arith.addi %dim, %c1 : index
    %4 = "stablehlo.concatenate"(%arg4, %arg0) {dimension = 0 : i64, tag = "concat"} : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
    %5 = plan.with_shape {tag = "with_shape"} %4(%2) : (tensor<?xf32>, index) -> tensor<?xf32>
    scf.yield %5 : tensor<?xf32>
  } {tag = "for"}
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func test_loop_concat:
// CHECK-NEXT:  arg #0: <[1, 1]>
// CHECK-NEXT:  arg #1: <[1, 1]>
// CHECK-NEXT:  arg #2: <[2, 6]>
// CHECK-NEXT: test_tag: for:
// CHECK-NEXT:  operand #0: <<uninitialized>>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  operand #2: <<uninitialized>>
// CHECK-NEXT:  operand #3: <[2, 6]>
// CHECK-NEXT:  result #0: <[1, 2147483647]>
// CHECK-NEXT:   Region #0:
// CHECK-NEXT:     arg #0: <<uninitialized>>
// CHECK-NEXT:     arg #1: <[1, 2147483647]>
// CHECK-NEXT: test_tag: concat:
// CHECK-NEXT:  operand #0: <[1, 2147483647]>
// CHECK-NEXT:  operand #1: <[1, 1]>
// CHECK-NEXT:  result #0: <[0, 2147483647]>
// CHECK-NEXT: test_tag: with_shape:
// CHECK-NEXT:  operand #0: <[0, 2147483647]>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: <[1, 2147483647]>

// -----

#profile0 = #plan.bounds<shape, [1], [10]>
#profile1 = #plan.bounds<value, dense<2> : tensor<index>, dense<6> : tensor<index>>

func.func @test_separated(%arg0: tensor<?xf32> {plan.shape_bounds = #profile0},
                          %arg1: index {plan.value_bounds  = #profile1})
    -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = plan.cluster target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>) -> tensor<?xf32> {
    %2 = stablehlo.exponential %arg0 : tensor<?xf32>
    %3 = with_shape {tag = "with_shape0"} %2(%dim) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %3 : tensor<?xf32>
  }
  %extracted_slice = tensor.extract_slice %0[0] [%arg1] [1] {tag = "extract_slice"} : tensor<?xf32> to tensor<?xf32>
  %extracted_slice2 = plan.with_shape %extracted_slice (%arg1) : (tensor<?xf32>, index) -> tensor<?xf32>
  %1 = plan.cluster target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>) -> tensor<?xf32> {
    %2 = stablehlo.exponential %extracted_slice : tensor<?xf32>
    %3 = with_shape {tag = "with_shape1"} %2(%arg1) : (tensor<?xf32>, index) -> tensor<?xf32>
    yield %3 : tensor<?xf32>
  }
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func test_separated:
// CHECK-NEXT:  arg #0: <[1, 10]>
// CHECK-NEXT:  arg #1: <<uninitialized>>
// CHECK-NEXT: test_tag: with_shape0:
// CHECK-NEXT:  operand #0: <[1, 10]>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: <[1, 10]>
// CHECK-NEXT: test_tag: extract_slice:
// CHECK-NEXT:  operand #0: <[1, 10]>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: <[2, 6]>
// CHECK-NEXT: test_tag: with_shape1:
// CHECK-NEXT:  operand #0: <[2, 6]>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: <[2, 6]>

// -----

#profile0 = #plan.bounds<shape, [1], [40]>
#profile1 = #plan.bounds<value, dense<[1, 1]> : tensor<2xi32>, dense<[40, 40]> : tensor<2xi32>>

func.func @test_reshape(%arg0: tensor<?xf32> {plan.shape_bounds = #profile0},
                        %arg1: tensor<2xi32> {plan.value_bounds = #profile1}) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %extracted = tensor.extract %arg1[%c0] : tensor<2xi32>
  %0 = arith.index_cast %extracted : i32 to index
  %extracted_0 = tensor.extract %arg1[%c1] : tensor<2xi32>
  %1 = arith.index_cast %extracted_0 : i32 to index
  %2 = plan.cluster target(#plan.tensorrt_backend<benefit=1, disallow_shape_tensor_calculations=false>) attributes {tag = "inline_group"} -> tensor<?x?xf32> {
    %3 = stablehlo.dynamic_reshape %arg0, %arg1 {tag = "dynamic_reshape"} : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    %4 = with_shape %3(%0, %1) : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    yield %4 : tensor<?x?xf32>
  }
  return %2 : tensor<?x?xf32>
}

// CHECK-LABEL: func test_reshape:
// CHECK-NEXT:  arg #0: <[1, 40]>
// CHECK-NEXT:  arg #1: <[2, 2]>
// CHECK-NEXT: test_tag: inline_group:
// CHECK-NEXT:  result #0: <[1, 40], [1, 40]>
// CHECK-NEXT:   Region #0:
// CHECK-NEXT: test_tag: dynamic_reshape:
// CHECK-NEXT:  operand #0: <[1, 40]>
// CHECK-NEXT:  operand #1: <[2, 2]>
// CHECK-NEXT:  result #0: <[1, 40], [1, 40]>
