// RUN:  executor-translate -mlir-to-runtime-executable %s | executor-runner -dump-function-signature -input-type=rtexe | FileCheck %s

func.func @arg_abi_input_byval_memref(
    %arg0: !executor.ptr<host>
      {executor.abi = #executor.arg<byval, memref<?xi32, #executor.memory_type<host>>>,
       executor.shape_profile = #executor.dim_bounds<min = [1], max = [10]>},
    %arg1: i32
      {executor.value_bounds = #executor.value_bounds<min = dense<1> : tensor<i32>, max = dense<10> : tensor<i32>>},
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<?xi32, #executor.memory_type<device>>, undef>},
    %arg3: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<1xi32, #executor.memory_type<host_pinned>>>},
    %arg4: !executor.ptr<host> {executor.abi = #executor.arg<byref, i32>,
      executor.value_bounds = #executor.value_bounds<min = dense<2> : tensor<i32>, max = dense<11> : tensor<i32>>},
    %arg5: !executor.ptr<host> {executor.abi = #executor.arg<byref, complex<f32>, undef>})
      attributes {
        executor.func_abi = (memref<?xi32, #executor.memory_type<host>>, i32)
        -> (memref<?xi32, #executor.memory_type<device>>, memref<1xi32, #executor.memory_type<host_pinned>>, i32, complex<f32>)
      } {
  return
}

// CHECK-LABEL: Function<arg_abi_input_byval_memref
// CHECK-SAME:  Signature<args=[MemRef<?xi32, strides=[1], host>, i32],
// CHECK-SAME:  results=[MemRef<?xi32, strides=[1], device>, MemRef<1xi32, strides=[1], pinned_host>, i32, complex32],
// CHECK-SAME:  num_output_args=4,
// CHECK-SAME:  arg_bounds=[dim_bounds<min = [1], max = [10]>, UNK],
// CHECK-SAME:  result_bounds=[UNK, UNK, value_bounds<min = [2], max = [11]>, UNK],
// CHECK-SAME:  cconv=unpacked,
// CHECK-SAME:  undef=[1, 0, 0, 1],
// CHECK-SAME:  abi_version=1>>
