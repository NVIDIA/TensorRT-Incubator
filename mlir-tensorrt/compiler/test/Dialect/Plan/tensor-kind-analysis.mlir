// RUN: mlir-tensorrt-opt -split-input-file -test-tensor-kind-analysis %s 2>&1 | FileCheck %s --check-prefixes=CHECK,BOTH
// RUN: mlir-tensorrt-opt -split-input-file -test-tensor-kind-analysis="interprocedural=true" %s 2>&1 | FileCheck %s --check-prefixes=BOTH,INTERP

func.func @test_reshape_partition(%arg0: tensor<128xf32>, %arg1: index, %arg2: index) -> (tensor<?x?xf32>, tensor<2xindex>) {
  %shape = tensor.from_elements %arg1, %arg2 {tag = "from_elements"} : tensor<2xindex>
  %0 = tensor.reshape %arg0(%shape) {tag = "reshape"} : (tensor<128xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = bufferization.alloc_tensor() copy(%shape) {
    memory_space = #plan.memory_space<device>,
    tag = "alloc_copy"
  } : tensor<2xindex>
  return %0, %1 : tensor<?x?xf32>, tensor<2xindex>
}

// BOTH-LABEL: func test_reshape_partition:
// BOTH-NEXT:  arg #0: device
// BOTH-NEXT:  arg #1: <<uninitialized>>
// BOTH-NEXT:  arg #2: <<uninitialized>>
// BOTH-NEXT: test_tag: from_elements:
// BOTH-NEXT:  operand #0: <<uninitialized>>
// BOTH-NEXT:  operand #1: <<uninitialized>>
// BOTH-NEXT:  result #0: host
// BOTH-NEXT: test_tag: reshape:
// BOTH-NEXT:  operand #0: device
// BOTH-NEXT:  operand #1: host
// BOTH-NEXT:  result #0: device
// BOTH-NEXT: test_tag: alloc_copy:
// BOTH-NEXT:  operand #0: host
// BOTH-NEXT:  result #0: device

// -----


func.func @test_inline_group(%arg0: tensor<1xi32>,
                             %arg1: tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32> {tensorrt.host_tensor}) {
  %1:2 = plan.inline_group target(#plan.tensorrt_cluster<benefit=1, disallow_shape_tensor_calculations=false>) attributes {tag = "inline_group"} -> tensor<1xi32>, tensor<1xf32> {
    yield %arg0, %arg1 : tensor<1xi32>, tensor<1xf32>
  }
  return %1#0, %1#1 : tensor<1xi32>, tensor<1xf32>
}


// BOTH-LABEL: func test_inline_group:
// BOTH:  arg #0: device
// BOTH:  arg #1: host
// BOTH: test_tag: inline_group:
// BOTH:  result #0: device
// BOTH:  result #1: host

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

// BOTH-LABEL: func test_inline_closed_group:
// BOTH:  arg #0: device
// BOTH:  arg #1: host
// BOTH:  arg #2: <<uninitialized>>
// BOTH:  arg #3: <<uninitialized>>
// BOTH: test_tag: inline_closed_group:
// BOTH:  operand #0: device
// BOTH:  operand #1: host
// BOTH:  operand #2: <<uninitialized>>
// BOTH:  operand #3: <<uninitialized>>
// BOTH:  operand #4: <<uninitialized>>
// BOTH:  result #0: device
// BOTH:   Region #0:
// BOTH:     arg #0: device
// BOTH:     arg #1: host
// BOTH:     arg #2: <<uninitialized>>
// BOTH:     arg #3: <<uninitialized>>
// BOTH:     arg #4: <<uninitialized>>

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

// BOTH-LABEL: func test_while:
//  BOTH-NEXT:  arg #0: both
//  BOTH-NEXT: test_tag: while:
//  BOTH-NEXT:  operand #0: both
//  BOTH-NEXT:  result #0: device
//  BOTH-NEXT:   Region #0:
//  BOTH-NEXT:     arg #0: both
//  BOTH-NEXT:   Region #1:
//  BOTH-NEXT:     arg #0: both
//  BOTH-NEXT: test_tag: body_add:
//  BOTH-NEXT:  operand #0: both
//  BOTH-NEXT:  operand #1: both
//  BOTH-NEXT:  result #0: both

// -----

func.func @device_extract(%arg0: tensor<128xi1>, %arg1: index) -> i1 {
  %1 = tensor.extract %arg0[%arg1] : tensor<128xi1>
  return %1 : i1
}

// BOTH-LABEL: func device_extract:
//       BOTH:  arg #0: both
//       BOTH:  arg #1: <<uninitialized>>

// -----

func.func @test_reshape(%arg0: tensor<128xf32>, %arg1: index, %arg2: index) -> tensor<?x?xf32> {
  %shape = tensor.from_elements %arg1, %arg2 {tag = "from_elements"} : tensor<2xindex>
  %0 = tensor.reshape %arg0(%shape) {tag = "reshape"} : (tensor<128xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// BOTH-LABEL: func test_reshape:
// BOTH-NEXT:  arg #0: device
// BOTH-NEXT:  arg #1: <<uninitialized>>
// BOTH-NEXT:  arg #2: <<uninitialized>>
// BOTH-NEXT: test_tag: from_elements:
// BOTH-NEXT:  operand #0: <<uninitialized>>
// BOTH-NEXT:  operand #1: <<uninitialized>>
// BOTH-NEXT:  result #0: host
// BOTH-NEXT: test_tag: reshape:
// BOTH-NEXT:  operand #0: device
// BOTH-NEXT:  operand #1: host
// BOTH-NEXT:  result #0: device

// -----

func.func @test_reshape_both(%arg0: tensor<128xf32>, %arg1: index, %arg2: index) -> (tensor<?x?xf32>, tensor<2xindex>) {
  %shape = tensor.from_elements %arg1, %arg2 {tag = "from_elements"} : tensor<2xindex>
  %0 = tensor.reshape %arg0(%shape) {tag = "reshape"} : (tensor<128xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0, %shape : tensor<?x?xf32>, tensor<2xindex>
}

// BOTH-LABEL: func test_reshape_both:
// BOTH-NEXT:  arg #0: device
// BOTH-NEXT:  arg #1: <<uninitialized>>
// BOTH-NEXT:  arg #2: <<uninitialized>>
// BOTH-NEXT: test_tag: from_elements:
// BOTH-NEXT:  operand #0: <<uninitialized>>
// BOTH-NEXT:  operand #1: <<uninitialized>>
// BOTH-NEXT:  result #0: both
// BOTH-NEXT: test_tag: reshape:
// BOTH-NEXT:  operand #0: device
// BOTH-NEXT:  operand #1: both
// BOTH-NEXT:  result #0: device

// -----

func.func @test_constant(%arg0: tensor<128xf32>) -> tensor<?x?xf32> {
  %shape = arith.constant {tag = "constant"} dense<[16, 8]> : tensor<2xindex>
  %0 = tensor.reshape %arg0(%shape) {tag = "reshape"} : (tensor<128xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// BOTH-LABEL: func test_constant:
// BOTH-NEXT:  arg #0: device
// BOTH-NEXT: test_tag: constant:
// BOTH-NEXT:  result #0: host
// BOTH-NEXT: test_tag: reshape:
// BOTH-NEXT:  operand #0: device
// BOTH-NEXT:  operand #1: host
// BOTH-NEXT:  result #0: device
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

// BOTH-LABEL: func test_loop_extract:
// BOTH-NEXT:  arg #0: both
// BOTH-NEXT: test_tag: for:
// BOTH-NEXT:  operand #0: <<uninitialized>>
// BOTH-NEXT:  operand #1: <<uninitialized>>
// BOTH-NEXT:  operand #2: <<uninitialized>>
// BOTH-NEXT:   Region #0:
// BOTH-NEXT:     arg #0: <<uninitialized>>
// BOTH-NEXT: test_tag: extract:
// BOTH-NEXT:  operand #0: both
// BOTH-NEXT:  operand #1: <<uninitialized>>
// BOTH-NEXT:  result #0: <<uninitialized>>
// BOTH-NEXT: test_tag: return:
// BOTH-NEXT:  operand #0: both

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

// BOTH-LABEL: func test_loop_extract_with_copy:
// BOTH-NEXT:  arg #0: device
// BOTH-NEXT:  arg #1: device
// BOTH-NEXT: test_tag: for:
// BOTH-NEXT:  operand #0: <<uninitialized>>
// BOTH-NEXT:  operand #1: <<uninitialized>>
// BOTH-NEXT:  operand #2: <<uninitialized>>
// BOTH-NEXT:   Region #0:
// BOTH-NEXT:     arg #0: <<uninitialized>>
// BOTH-NEXT: test_tag: extract:
// BOTH-NEXT:  operand #0: host
// BOTH-NEXT:  operand #1: <<uninitialized>>
// BOTH-NEXT:  result #0: <<uninitialized>>
// BOTH-NEXT: test_tag: return:
// BOTH-NEXT:  operand #0: device

// -----

func.func private @external(%arg0: tensor<128xf32>) -> tensor<128xf32>

func.func @test_ext_call(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = call @external(%arg0) {tag = "call"} : (tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// BOTH-LABEL: func external:
// BOTH-NEXT:  func test_ext_call:
// BOTH-NEXT:   arg #0: unknown
// BOTH-NEXT:  test_tag: call:
// BOTH-NEXT:   operand #0: unknown
// BOTH-NEXT:   result #0: device

// -----

func.func private @internal(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  return %arg0 : tensor<128xf32>
}

func.func @test_call(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = call @internal(%arg0) {tag = "call"} : (tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// INTERP-LABEL: func internal:
// INTERP-NEXT:   arg #0: device
// INTERP-NEXT:  func test_call:
// INTERP-NEXT:   arg #0: device
// INTERP-NEXT:  test_tag: call:
// INTERP-NEXT:   operand #0: device
// INTERP-NEXT:   result #0: device

// CHECK-LABEL: func internal:
// CHECK-NEXT:   arg #0: device
// CHECK-NEXT:  func test_call:
// CHECK-NEXT:   arg #0: unknown
// CHECK-NEXT:  test_tag: call:
// CHECK-NEXT:   operand #0: unknown
// CHECK-NEXT:   result #0: device
