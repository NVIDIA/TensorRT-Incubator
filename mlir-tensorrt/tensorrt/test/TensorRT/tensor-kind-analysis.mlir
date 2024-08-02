// RUN: tensorrt-opt -split-input-file -test-tensor-kind-analysis %s 2>&1 | FileCheck %s

func.func @test_ewise(%arg0: tensor<1xi32>) -> tensor<i1> attributes {tensorrt.engine = dense<0> : vector<10xi8>} {
  %cst_i32 = tensorrt.constant dense<10> : tensor<1xi32>
  %0 = tensorrt.element_wise {tag = "ewise"} <kLESS>(%arg0, %cst_i32 : tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  %1 = tensorrt.collapse_rank {tag = "collapse_shape"} %0 : tensor<1xi1> to tensor<i1>
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

func.func @test_slice_backward_infer(%arg0: tensor<10xf32>, %arg1: tensor<1xi32> {tensorrt.host_tensor}, %arg2: tensor<1xf32>)
     -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xf32>) attributes {tensorrt.engine = dense<0> : vector<10xi8>} {
  %cst_i32 = tensorrt.constant dense<1> : tensor<1xi32>
  %0 = tensorrt.slice %arg0[%arg1: tensor<1xi32>][1][1] {
    tag = "slice"
  } : tensor<10xf32> to tensor<1xf32>
  %1 = tensorrt.element_wise <kSUM>(%cst_i32, %arg1 : tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2 = tensorrt.element_wise <kSUM>(%0, %arg2 : tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %1, %2 : tensor<1xi32>, tensor<1xf32>
}

// CHECK-LABEL: func test_slice_backward_infer
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT:  arg #1: host
// CHECK-NEXT:  arg #2: device
// CHECK-NEXT: test_tag: slice:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: device

// -----

func.func @test_slice_backward_infer2(%arg0: tensor<10xf32>, %arg1: tensor<1xf32> {tensorrt.host_tensor}, %arg2: tensor<1xf32>)
      -> (tensor<1xi32> {tensorrt.host_tensor}, tensor<1xf32>) attributes {tensorrt.engine = dense<0> : vector<10xi8>} {
  %cst_i32 = tensorrt.constant dense<1> : tensor<1xi32>
  %a = tensorrt.identity {tag = "iden"} %arg1: tensor<1xf32> to tensor<1xi32>
  %0 = tensorrt.slice %arg0[%a: tensor<1xi32>][1][1] {
    tag = "slice"
  } : tensor<10xf32> to tensor<1xf32>
  %1 = tensorrt.element_wise <kSUM>(%cst_i32, %a : tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2 = tensorrt.element_wise <kSUM>(%0, %arg2 : tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return {tag = "return"} %1, %2  : tensor<1xi32>, tensor<1xf32>
}

// CHECK-LABEL: func test_slice_backward_infer2
// CHECK-NEXT:  arg #0: device
// CHECK-NEXT:  arg #1: host
// CHECK-NEXT:  arg #2: device
// CHECK-NEXT: test_tag: iden:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: slice:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: device
// CHECK-NEXT: test_tag: return:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: device

// -----

func.func @scf_while_loop(%count_init: tensor<1xi32>, %arg0: tensor<10xf32>) -> tensor<1xf32> {
  %cst_i32 = tensorrt.constant dense<1> : tensor<1xi32>
  %cst_i32_0 = tensorrt.constant dense<0> : tensor<1xi32>
  %cst_i32_1 = tensorrt.constant dense<10> : tensor<1xi32>
  %cst_f32 = tensorrt.constant dense<0.000000e+00> : tensor<1xf32>
  %0:2 = scf.while (%arg1 = %count_init, %arg2 = %cst_f32) : (tensor<1xi32>, tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32>) {
    %1 = tensorrt.element_wise  {tag = "cond_ewise"} <kLESS>(%arg1, %cst_i32_1: tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %2 = tensorrt.reshape %1 {tag = "reshape"} : tensor<1xi1> to tensor<i1>
    %extracted = tensor.extract %2[] : tensor<i1>
    scf.condition(%extracted) %arg1, %arg2 : tensor<1xi32>, tensor<1xf32>
  } do {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<1xf32>):
    %1 = tensorrt.slice %arg0[%arg1: tensor<1xi32>][1][1] {tag = "slice"} : tensor<10xf32> to tensor<1xf32>
    %2 = tensorrt.element_wise {tag = "body_ewise1"} <kSUM>(%cst_i32, %arg1 : tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %3 = tensorrt.element_wise {tag = "body_ewise2"} <kSUM>(%1, %arg2 : tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    scf.yield %2, %3 : tensor<1xi32>, tensor<1xf32>
  } attributes {tag = "scf_while"}
  return %0#1 : tensor<1xf32>
}

// CHECK-LABEL: func scf_while_loop:
// CHECK-NEXT:  arg #0: host
// CHECK-NEXT:  arg #1: device
// CHECK-NEXT: test_tag: scf_while:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT:  result #0: <<uninitialized>>
// CHECK-NEXT:  result #1: device
// CHECK-NEXT:   Region #0:
// CHECK-NEXT:     arg #0: host
// CHECK-NEXT:     arg #1: device
// CHECK-NEXT:   Region #1:
// CHECK-NEXT:     arg #0: host
// CHECK-NEXT:     arg #1: device
// CHECK-NEXT: test_tag: cond_ewise:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: reshape:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: slice:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: device
// CHECK-NEXT: test_tag: body_ewise1:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: body_ewise2:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT:  result #0: device

// -----

func.func @trt_while_loop(%iter_init: tensor<1xi32>, %arg0: tensor<10xf32>) -> tensor<1xf32> {
  %one = tensorrt.constant dense<1> : tensor<1xi32>
  %limit = tensorrt.constant dense<10> : tensor<1xi32>
  %res_init = tensorrt.constant dense<0.0> : tensor<1xf32>
  %0, %1 = tensorrt.while {
    tag = "while"
  } (%iter_init, %res_init : tensor<1xi32>, tensor<1xf32>) -> tensor<1xi32>, tensor<1xf32> {
  ^bb0(%iter: tensor<1xi32>, %result: tensor<1xf32>):
    %cond = tensorrt.element_wise {tag = "cond_ewise"} <kLESS>(%iter, %limit : tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %cond1 = tensorrt.reshape %cond {tag = "reshape"} : tensor<1xi1> to tensor<i1>
    tensorrt.condition(%cond1 : tensor<i1>) {tag = "condition"} %iter, %result : tensor<1xi32>,tensor<1xf32>
  }, {
  ^bb1(%iter: tensor<1xi32>, %red: tensor<1xf32>):
    %slice = tensorrt.slice %arg0[%iter: tensor<1xi32>][1][1] {tag = "slice"} : tensor<10xf32> to tensor<1xf32>
    %new_iter = tensorrt.element_wise {tag = "body_ewise1"} <kSUM> (%one, %iter : tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %new_red =  tensorrt.element_wise {tag = "body_ewise2"} <kSUM> (%slice, %red : tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    tensorrt.yield %new_iter, %new_red: tensor<1xi32>, tensor<1xf32>
  }
  return %1: tensor<1xf32>
}

// CHECK-LABEL: func trt_while_loop:
// CHECK-NEXT:  arg #0: host
// CHECK-NEXT:  arg #1: device
// CHECK-NEXT: test_tag: while:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT:  result #0: <<uninitialized>>
// CHECK-NEXT:  result #1: device
// CHECK-NEXT:   Region #0:
// CHECK-NEXT:     arg #0: host
// CHECK-NEXT:     arg #1: device
// CHECK-NEXT:   Region #1:
// CHECK-NEXT:     arg #0: host
// CHECK-NEXT:     arg #1: device
// CHECK-NEXT: test_tag: cond_ewise:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: reshape:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: condition:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  operand #2: device
// CHECK-NEXT: test_tag: slice:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: device
// CHECK-NEXT: test_tag: body_ewise1:
// CHECK-NEXT:  operand #0: host
// CHECK-NEXT:  operand #1: host
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: body_ewise2:
// CHECK-NEXT:  operand #0: device
// CHECK-NEXT:  operand #1: device
// CHECK-NEXT:  result #0: device
