// RUN: mlir-tensorrt-opt -split-input-file -test-tensor-kind-analysis %s 2>&1 >/dev/null | FileCheck %s

func.func @test_ewise(%arg0: tensor<1xi32>) -> tensor<i1> {
  %cst_i32 = stablehlo.constant dense<10> : tensor<1xi32>
  %0 = stablehlo.compare LT, %arg0, %cst_i32 {tag = "ewise"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  %1 = stablehlo.reshape  %0 {tag = "collapse_shape"} : (tensor<1xi1>) -> tensor<i1>
  return %1 : tensor<i1>
}

// CHECK-LABEL: func test_ewise
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT: test_tag: ewise:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT:  result #0: device
// CHECK-NEXT: test_tag: collapse_shape:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  result #0: device

// -----

func.func @test_slice_backward_infer(%arg0: tensor<10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32>) {
  %cst_i32 = stablehlo.constant {tag = "constant"} dense<1> : tensor<1xi32>
  %offset = stablehlo.reshape %cst_i32 {tag= "reshape"} : (tensor<1xi32>) -> tensor<i32>
  %0 = "stablehlo.dynamic_slice"(%arg0, %offset) {
    slice_sizes = array<i64: 1>
  } : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
  %1 = stablehlo.add %cst_i32, %arg1 {tag = "add1"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2 = stablehlo.add %0, %arg2 {tag = "add2"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %1, %2 : tensor<1xi32>, tensor<1xf32>
}

// CHECK-LABEL: func test_slice_backward_infer:
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT:  arg #1: device
// CHECK-NEXT:  arg #2: device
// CHECK-NEXT: test_tag: constant:
// CHECK-NEXT:  result #0: both
// CHECK-NEXT: test_tag: reshape:
// CHECK-NEXT:  operand #0: both
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: add1:
// CHECK-NEXT:  operand #0: both
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT:  result #0: device
// CHECK-NEXT: test_tag: add2:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT:  result #0: device

// -----

func.func @test_slice_backward_infer2(%arg0: tensor<10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32>) {
  %cst_i32 = stablehlo.constant {tag = "constant"} dense<1> : tensor<1xi32>
  %offset = stablehlo.reshape %arg1 {tag= "reshape"} : (tensor<1xi32>) -> tensor<i32>
  %0 = "stablehlo.dynamic_slice"(%arg0, %offset) {
    slice_sizes = array<i64: 1>
  } : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
  %1 = stablehlo.add %cst_i32, %arg1 {tag = "add1"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2 = stablehlo.add %0, %arg2 {tag = "add2"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %1, %2 : tensor<1xi32>, tensor<1xf32>
}

// CHECK-LABEL: func test_slice_backward_infer2:
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT:  arg #1: both
// CHECK-NEXT:  arg #2: device
// CHECK-NEXT: test_tag: constant:
// CHECK-NEXT:  result #0: device
// CHECK-NEXT: test_tag: reshape:
// CHECK-NEXT:  operand #0: both
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: add1:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: both
// CHECK-NEXT:  result #0: device
// CHECK-NEXT: test_tag: add2:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT:  result #0: device

// -----

func.func @test_max_size_limit(%arg0: tensor<10xf32>, %arg1: tensor<i32>) -> f32 {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1) {
    slice_sizes = array<i64: 1>,
    tag = "dynamic_slice"
  } : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
  %c0 = arith.constant 0 : index
  %1 = tensor.extract %0[%c0] {tag = "extract"} : tensor<1xf32>
  return %1 : f32
}

// CHECK-LABEL: func test_max_size_limit:
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT:  arg #1: both
// CHECK-NEXT: test_tag: dynamic_slice:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: both
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: extract:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: <<uninitialized>>

// -----

func.func @scf_while_loop() -> (tensor<i32> {tensorrt.host_tensor}, tensor<1xf32>) {
  %cst_i32 = stablehlo.constant dense<1> : tensor<i32>
  %cst_i32_0 = stablehlo.constant dense<0> : tensor<i32>
  %cst_i32_1 = stablehlo.constant dense<10> : tensor<1xi32>
  %cst_f32 = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
  %cst_f32_1 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
  %c10 = arith.constant 10 : i32
  %0:2 = scf.while (%arg1 = %cst_i32_0, %arg2 = %cst_f32) : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>) {
    %1 = tensor.extract %arg1[]:tensor<i32>
    %2 = arith.cmpi slt, %1, %c10 : i32
    scf.condition(%2) {tag = "condition"} %arg1, %arg2 : tensor<i32>, tensor<1xf32>
  } do {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<1xf32>):
    %1 = stablehlo.add %arg1, %cst_i32 {tag = "add"} : tensor<i32>
    %2 = stablehlo.add %arg2, %cst_f32_1 : tensor<1xf32>
    scf.yield {tag = "yield"} %1, %2 : tensor<i32>, tensor<1xf32>
  } attributes {tag = "while"}
  return {tag = "return"} %0#0, %0#1 : tensor<i32>, tensor<1xf32>
}

// CHECK-LABEL: func scf_while_loop:
// CHECK-NEXT: test_tag: while:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT:  result #0: host
// CHECK-NEXT:  result #1: device
// CHECK-NEXT:   Region #0:
// CHECK-NEXT:     arg #0: host
// CHECK-NEXT:     arg #1: device
// CHECK-NEXT:   Region #1:
// CHECK-NEXT:     arg #0: host
// CHECK-NEXT:     arg #1: device
// CHECK-NEXT: test_tag: condition:
// CHECK-NEXT:  operand #0: <<uninitialized>>
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  operand #2: device
// CHECK-NEXT: test_tag: add:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: yield:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT: test_tag: return:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: device

// -----

func.func @scf_while_loop_reduce_large(%arg0: tensor<100xi32>) -> (tensor<i32> {tensorrt.host_tensor}) {
  %cst_i32 = stablehlo.constant dense<1> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0:2 = scf.while (%arg1 = %cst_i32, %arg2 = %arg0) : (tensor<i32>, tensor<100xi32>) -> (tensor<i32>, tensor<100xi32>) {
    %1 = tensor.extract %arg1[]:tensor<i32>
    %2 = arith.trunci %1 : i32 to i1
    scf.condition(%2) {tag = "condition"} %arg1, %arg2 : tensor<i32>, tensor<100xi32>
  } do {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<100xi32>):
    %reduce = stablehlo.reduce(%arg2 init: %c0) across dimensions = [0] {tag = "reduce"}
      : (tensor<100xi32>, tensor<i32>) -> tensor<i32>
    reducer(%arg3: tensor<i32>, %arg4: tensor<i32>)  {
      %a = stablehlo.or %arg3, %arg4 : tensor<i32>
      stablehlo.return %a : tensor<i32>
    }
    scf.yield {tag = "yield"} %reduce, %arg2 : tensor<i32>, tensor<100xi32>
  } attributes {tag = "while"}
  return {tag = "return"} %0#0 : tensor<i32>
}

// CHECK-LABEL: func scf_while_loop_reduce_large:
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT: test_tag: while:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT:  result #0: host
// CHECK-NEXT:  result #1: <<uninitialized>>
// CHECK-NEXT:   Region #0:
// CHECK-NEXT:     arg #0: host
// CHECK-NEXT:     arg #1: device
// CHECK-NEXT:   Region #1:
// CHECK-NEXT:     arg #0: <<uninitialized>>
// CHECK-NEXT:     arg #1: device
// CHECK-NEXT: test_tag: condition:
// CHECK-NEXT:  operand #0: <<uninitialized>>
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  operand #2: device
// CHECK-NEXT: test_tag: reduce:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT:  result #0: host
// CHECK-NEXT:   Region #0:
// CHECK-NEXT:     arg #0: <<uninitialized>>
// CHECK-NEXT:     arg #1: <<uninitialized>>
// CHECK-NEXT: test_tag: yield:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT: test_tag: return:
// CHECK-NEXT:  operand #0: host

// -----

func.func @test_dynamic_reshape(%arg0: tensor<?xf32>, %arg1: tensor<2xi32> {tensorrt.host_tensor}) -> tensor<?x?xf32> {
  %0 = "stablehlo.dynamic_reshape"(%arg0, %arg1) {tag = "dynamic_reshape"} : (tensor<?xf32>, tensor<2xi32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func test_dynamic_reshape:
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT:  arg #1: host
// CHECK-NEXT: test_tag: dynamic_reshape:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: device
