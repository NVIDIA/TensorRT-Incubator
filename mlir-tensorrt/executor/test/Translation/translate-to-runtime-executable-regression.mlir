// Regression test for bug where FunctionSignature was serialized in wrong order.

// REQUIRES: host-has-at-least-1-gpus
// RUN: executor-translate -mlir-to-runtime-executable %s | \
// RUN: executor-runner -dump-function-signature -input-type=rtexe | FileCheck %s
executor.data_segment @__constant_4xf32 align 16
    address_space <device> dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>

func.func @test_all_reduce(%arg0: !executor.table<!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>) attributes {
    executor.function_metadata = #executor.func_meta<[memref<4xf32, #executor.memory_type<device>>],
                                                     [],
                                                     num_output_args = 0>
} {
  return
}


func.func @main() -> i32 attributes {
    executor.function_metadata = #executor.func_meta<[i32], [i32], num_output_args = 0>
} {
  %0 = executor.constant 0 : i32
  return %0 : i32
}

// CHECK: Function<test_all_reduce, Signature<args=[MemRef<4xf32, strides=[1], device>], results=[], num_output_args=0, arg_bounds=[UNK], result_bounds=[], cconv=unpacked>>
// CHECK: Function<main, Signature<args=[i32], results=[i32], num_output_args=0, arg_bounds=[UNK], result_bounds=[UNK], cconv=unpacked>>
