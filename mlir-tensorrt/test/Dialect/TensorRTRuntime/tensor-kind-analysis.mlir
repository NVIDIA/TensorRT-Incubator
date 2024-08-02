// RUN: mlir-tensorrt-opt %s -split-input-file -test-tensor-kind-analysis 2>&1 | FileCheck %s

func.func @enqueue_host_tensors_space_check(
    %ctx: !trtrt.context, %stream: !cuda.stream,
    %arg0: tensor<4xi32>,
    %arg1: tensor<4xi32>,
    %arg2: tensor<128xf32>) -> tensor<128xf32> {
  %empty = tensor.empty() {tag = "empty0"} : tensor<4xi32>
  %empty1 = tensor.empty() {tag = "empty1"} : tensor<4xi32>
  %3 = trtrt.enqueue %ctx
    stream(%stream) host_tensor_args [0, 1]
    (%empty, %arg0, %arg1, %empty1)
    outs(%arg2)
    {tag = "enqueue"} : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<128xf32>
  return %3 : tensor<128xf32>
}

// CHECK-LABEL: func enqueue_host_tensors_space_check:
// CHECK-NEXT:  arg #0: <<uninitialized>>
// CHECK-NEXT:  arg #1: <<uninitialized>>
// CHECK-NEXT:  arg #2: both
// CHECK-NEXT:  arg #3: device
// CHECK-NEXT:  arg #4: device
// CHECK-NEXT: test_tag: empty0:
// CHECK-NEXT:  result #0: host
// CHECK-NEXT: test_tag: empty1:
// CHECK-NEXT:  result #0: device
// CHECK-NEXT: test_tag: enqueue:
// CHECK-NEXT:  operand #0: <<uninitialized>>
// CHECK-NEXT:  operand #1: <<uninitialized>>
// CHECK-NEXT:  operand #2: host
// CHECK-NEXT:  operand #3: both
// CHECK-NEXT:  operand #4: device
// CHECK-NEXT:  operand #5: device
// CHECK-NEXT:  operand #6: device
// CHECK-NEXT:  result #0: device
