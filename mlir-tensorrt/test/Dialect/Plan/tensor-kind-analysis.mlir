// RUN: mlir-tensorrt-opt -split-input-file -test-tensor-kind-analysis %s 2>&1 | FileCheck %s

func.func @test_reshape_partition(%arg0: tensor<128xf32>, %arg1: index, %arg2: index) -> (tensor<?x?xf32>, tensor<2xindex>) {
  %shape = tensor.from_elements %arg1, %arg2 {tag = "from_elements"} : tensor<2xindex>
  %0 = tensor.reshape %arg0(%shape) {tag = "reshape"} : (tensor<128xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = bufferization.alloc_tensor() copy(%shape) {
    memory_space = #plan.memory_space<device>,
    tag = "alloc_copy"
  } : tensor<2xindex>
  return %0, %1 : tensor<?x?xf32>, tensor<2xindex>
}

// CHECK-LABEL: func test_reshape_partition:
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT:  arg #1: <<uninitialized>>
// CHECK-NEXT:  arg #2: <<uninitialized>>
// CHECK-NEXT: test_tag: from_elements:
// CHECK-NEXT:  operand #0: <<uninitialized>>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: reshape:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: device
// CHECK-NEXT: test_tag: alloc_copy:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  result #0: device

// -----


func.func @test_inline_group(%arg0: tensor<1xi32>,
                             %arg1: tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32> {tensorrt.host_tensor}) {
  %1:2 = plan.inline_group target(#plan.tensorrt_cluster<benefit=1, disallow_shape_tensor_calculations=false>) attributes {tag = "inline_group"} -> tensor<1xi32>, tensor<1xf32> {
    yield %arg0, %arg1 : tensor<1xi32>, tensor<1xf32>
  }
  return %1#0, %1#1 : tensor<1xi32>, tensor<1xf32>
}


// CHECK-LABEL: func test_inline_group:
// CHECK:  arg #0: device
// CHECK:  arg #1: host
// CHECK: test_tag: inline_group:
// CHECK:  result #0: device
// CHECK:  result #1: host

// -----

func.func @test_inline_closed_group(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>, %arg2: index, %arg3: index) -> tensor<?x?xf32> {
  %empty = tensor.empty(%arg2, %arg3) : tensor<?x?xf32>
  %1 = plan.inline_closed_group  target(#plan.tensorrt_cluster<benefit=1,disallow_shape_tensor_calculations=false>)
    inputs( %arg0, %arg1, %arg2, %arg3 : tensor<?xf32>, tensor<2xi32>, index, index)
    outs(%empty : tensor<?x?xf32> )
    in_attrs [#plan.bounds<none>, #plan.bounds<none>, #plan.bounds<none>, #plan.bounds<none>]
    res_attrs [#plan.bounds<none>] attributes {
      tag = "inline_closed_group"
    } -> tensor<?x?xf32> {
  ^bb0(%in0: tensor<?xf32>, %in1: tensor<2xi32>, %in2: index, %in3: index, %out: tensor<?x?xf32>):
    %2 = stablehlo.dynamic_reshape %in0, %in1 : (tensor<?xf32>, tensor<2xi32>) -> (tensor<?x?xf32>)
    yield %2 : tensor<?x?xf32>
  }
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func test_inline_closed_group:
// CHECK:  arg #0: device
// CHECK:  arg #1: host
// CHECK:  arg #2: <<uninitialized>>
// CHECK:  arg #3: <<uninitialized>>
// CHECK: test_tag: inline_closed_group:
// CHECK:  operand #0: device
// CHECK:  operand #1: host
// CHECK:  operand #2: <<uninitialized>>
// CHECK:  operand #3: <<uninitialized>>
// CHECK:  operand #4: <<uninitialized>>
// CHECK:  result #0: device
// CHECK:   Region #0:
// CHECK:     arg #0: device
// CHECK:     arg #1: host
// CHECK:     arg #2: <<uninitialized>>
// CHECK:     arg #3: <<uninitialized>>
// CHECK:     arg #4: <<uninitialized>>

// -----

func.func @test_while(%arg0: tensor<i32>) -> (tensor<i32>) {
  %c20_i32 = arith.constant 20 : i32
  %c1_i32 = stablehlo.constant dense<1> : tensor<i32>
  %c0 = arith.constant 0 : index
  %1 = scf.while (%arg1 = %arg0) : (tensor<i32>) -> (tensor<i32>) {
    %extracted_0 = tensor.extract %arg1[] : tensor<i32>
    %cond = arith.cmpi eq, %extracted_0, %c20_i32 : i32
    scf.condition(%cond) %arg1 : tensor<i32>
  } do {
  ^bb0(%arg1: tensor<i32>):
    %1 = stablehlo.add %arg1, %c1_i32 {tag = "body_add"} : tensor<i32>
    scf.yield %1 : tensor<i32>
  } attributes {tag = "while"}
  return %1 : tensor<i32>
}

// CHECK-LABEL: func test_while:
//  CHECK-NEXT:  arg #0: both
//  CHECK-NEXT: test_tag: while:
//  CHECK-NEXT:  operand #0: both
//  CHECK-NEXT:  result #0: device
//  CHECK-NEXT:   Region #0:
//  CHECK-NEXT:     arg #0: both
//  CHECK-NEXT:   Region #1:
//  CHECK-NEXT:     arg #0: both
//  CHECK-NEXT: test_tag: body_add:
//  CHECK-NEXT:  operand #0: both
//  CHECK-NEXT:  operand #1: both
//  CHECK-NEXT:  result #0: both

// -----

func.func @device_extract(%arg0: tensor<128xi1>, %arg1: index) -> i1 {
  %1 = tensor.extract %arg0[%arg1] : tensor<128xi1>
  return %1 : i1
}

// CHECK-LABEL: func device_extract:
//       CHECK:  arg #0: both
//       CHECK:  arg #1: <<uninitialized>>

// -----

func.func @test_reshape(%arg0: tensor<128xf32>, %arg1: index, %arg2: index) -> tensor<?x?xf32> {
  %shape = tensor.from_elements %arg1, %arg2 {tag = "from_elements"} : tensor<2xindex>
  %0 = tensor.reshape %arg0(%shape) {tag = "reshape"} : (tensor<128xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func test_reshape:
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT:  arg #1: <<uninitialized>>
// CHECK-NEXT:  arg #2: <<uninitialized>>
// CHECK-NEXT: test_tag: from_elements:
// CHECK-NEXT:  operand #0: <<uninitialized>>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: reshape:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: device

// -----

func.func @test_reshape_both(%arg0: tensor<128xf32>, %arg1: index, %arg2: index) -> (tensor<?x?xf32>, tensor<2xindex>) {
  %shape = tensor.from_elements %arg1, %arg2 {tag = "from_elements"} : tensor<2xindex>
  %0 = tensor.reshape %arg0(%shape) {tag = "reshape"} : (tensor<128xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0, %shape : tensor<?x?xf32>, tensor<2xindex>
}

// CHECK-LABEL: func test_reshape_both:
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT:  arg #1: <<uninitialized>>
// CHECK-NEXT:  arg #2: <<uninitialized>>
// CHECK-NEXT: test_tag: from_elements:
// CHECK-NEXT:  operand #0: <<uninitialized>>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: both
// CHECK-NEXT: test_tag: reshape:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: both
// CHECK-NEXT:  result #0: device

// -----

func.func @test_constant(%arg0: tensor<128xf32>) -> tensor<?x?xf32> {
  %shape = arith.constant {tag = "constant"} dense<[16, 8]> : tensor<2xindex>
  %0 = tensor.reshape %arg0(%shape) {tag = "reshape"} : (tensor<128xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func test_constant:
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT: test_tag: constant:
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: reshape:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: device
// -----

func.func @test_loop_extract(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c128 step %c1 {
    %val = tensor.extract %arg0[%i] {tag = "extract"} : tensor<128xf32>
    executor.print "%f"(%val : f32)
  } {tag = "for"}
  return {tag = "return"} %arg0 : tensor<128xf32>
}

// CHECK-LABEL: func test_loop_extract:
// CHECK-NEXT:  arg #0: both
// CHECK-NEXT: test_tag: for:
// CHECK-NEXT:  operand #0: <<uninitialized>>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  operand #2: <<uninitialized>>
// CHECK-NEXT:   Region #0:
// CHECK-NEXT:     arg #0: <<uninitialized>>
// CHECK-NEXT: test_tag: extract:
// CHECK-NEXT:  operand #0: both
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: <<uninitialized>>
// CHECK-NEXT: test_tag: return:
// CHECK-NEXT:  operand #0: both

// -----

func.func @test_loop_extract_with_copy(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c1 {
    %1 = bufferization.alloc_tensor() copy(%arg0) {memory_space = #plan.memory_space<host_pinned>} : tensor<128xf32>
    %extracted = tensor.extract %1[%arg2] {tag = "extract"} : tensor<128xf32>
    executor.print "%f"(%extracted : f32)
  } {tag = "for"}
  %0 = bufferization.materialize_in_destination %arg0 in %arg1 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return {tag = "return"} %0 : tensor<128xf32>
}

// CHECK-LABEL: func test_loop_extract_with_copy:
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT:  arg #1: device
// CHECK-NEXT: test_tag: for:
// CHECK-NEXT:  operand #0: <<uninitialized>>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  operand #2: <<uninitialized>>
// CHECK-NEXT:   Region #0:
// CHECK-NEXT:     arg #0: <<uninitialized>>
// CHECK-NEXT: test_tag: extract:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  result #0: <<uninitialized>>
// CHECK-NEXT: test_tag: return:
// CHECK-NEXT:  operand #0: device
