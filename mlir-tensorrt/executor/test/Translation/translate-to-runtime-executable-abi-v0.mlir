// REQUIRES: host-has-at-least-1-gpus
// RUN:  executor-opt %s -split-input-file -convert-executor-to-executor | \
// RUN:  executor-translate -mlir-to-runtime-executable | executor-runner -dump-function-signature -input-type=rtexe | FileCheck %s

// CHECK: Function<value_bounds,
// CHECK-SAME: Signature<args=[MemRef<1xi64, strides=[1], unknown>, MemRef<1xi64, strides=[1], unknown>],
// CHECK-SAME: results=[MemRef<1xi64, strides=[1], unknown>],
// CHECK-SAME: num_output_args=1,
// CHECK-SAME: arg_bounds=[value_bounds<min = [1], max = [6]>, value_bounds<min = [2], max = [4]>],
// CHECK-SAME: result_bounds=[value_bounds<min = [1], max = [6]>],
// CHECK-SAME: cconv=unpacked,
// CHECK-SAME: undef=[],
// CHECK-SAME: abi_version=0>>
func.func @value_bounds() attributes {
  executor.function_metadata = #executor.func_meta<
    [memref<1xi64> {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>},
     memref<1xi64> {#executor.value_bounds<min = dense<2> : vector<1xi64>, max = dense<4> : vector<1xi64>>}],
    [memref<1xi64> {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}],
     num_output_args = 1>} {
  return
}

// CHECK-NEXT: Function<dim_bounds,
// CHECK-SAME: Signature<args=[MemRef<?xf32, strides=[1], unknown>],
// CHECK-SAME: results=[MemRef<?xf32, strides=[1], unknown>, MemRef<?xf32, strides=[1], unknown>],
// CHECK-SAME: num_output_args=1,
// CHECK-SAME: arg_bounds=[dim_bounds<min = [1], max = [6]>],
// CHECK-SAME: result_bounds=[dim_bounds<min = [2], max = [8]>, dim_bounds<min = [3], max = [5]>],
// CHECK-SAME: cconv=unpacked,
// CHECK-SAME: undef=[],
// CHECK-SAME: abi_version=0>>
func.func @dim_bounds() attributes {
  executor.function_metadata = #executor.func_meta<
    [memref<?xf32> {#executor.dim_bounds<min = [1], max = [6]>}],
    [memref<?xf32> {#executor.dim_bounds<min = [2], max = [8]>},
     memref<?xf32> {#executor.dim_bounds<min = [3], max = [5]>}],
     num_output_args = 1>} {
  return
}

// CHECK-LABEL: Function<unit_attr,
// CHECK-SAME: Signature<args=[MemRef<1xi64, strides=[1], unknown>],
// CHECK-SAME: results=[],
// CHECK-SAME: num_output_args=0,
// CHECK-SAME: arg_bounds=[UNK],
// CHECK-SAME: result_bounds=[],
// CHECK-SAME: cconv=unpacked,
// CHECK-SAME: undef=[],
// CHECK-SAME: abi_version=0>>
func.func @unit_attr() attributes {
  executor.function_metadata = #executor.func_meta<[memref<1xi64> {unit}], [], num_output_args = 0>
} {
  return
}

// CHECK-LABEL: Function<scalar_value_bounds,
// CHECK-SAME: Signature<args=[i64],
// CHECK-SAME: results=[],
// CHECK-SAME: num_output_args=0,
// CHECK-SAME: arg_bounds=[value_bounds<min = [1], max = [6]>],
// CHECK-SAME: result_bounds=[],
// CHECK-SAME: cconv=unpacked,
// CHECK-SAME: undef=[],
// CHECK-SAME: abi_version=0>>
func.func @scalar_value_bounds() attributes {
    executor.function_metadata = #executor.func_meta<
      [i64 {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}],
      [], num_output_args = 0>
} {
  return
}

// CHECK-LABEL: Function<mixed_bounds,
// CHECK-SAME: Signature<args=[i64, MemRef<1xi64, strides=[1], unknown>, MemRef<1xi64, strides=[1], unknown>],
// CHECK-SAME: results=[MemRef<?xf32, strides=[1], unknown>, MemRef<1xf32, strides=[1], unknown>],
// CHECK-SAME: num_output_args=1,
// CHECK-SAME: arg_bounds=[value_bounds<min = [1], max = [6]>, UNK, value_bounds<min = [1], max = [6]>],
// CHECK-SAME: result_bounds=[dim_bounds<min = [2], max = [8]>, dim_bounds<min = [3], max = [5]>],
// CHECK-SAME: cconv=packed,
// CHECK-SAME: undef=[],
// CHECK-SAME: abi_version=0>>
func.func @mixed_bounds() attributes {
  executor.function_metadata = #executor.func_meta<
    [i64 {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>},
     memref<1xi64> {unit},
     memref<1xi64> {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}],
    [memref<?xf32> {#executor.dim_bounds<min = [2], max = [8]>},
     memref<1xf32> {#executor.dim_bounds<min = [3], max = [5]>}],
     num_output_args = 1,
     cconv=packed>} {
  return
}

// CHECK-LABEL: Function<scalar_only,
// CHECK-SAME: Signature<args=[i32, i32],
// CHECK-SAME: results=[i32],
// CHECK-SAME: num_output_args=0,
// CHECK-SAME: arg_bounds=[UNK, UNK],
// CHECK-SAME: result_bounds=[UNK],
// CHECK-SAME: cconv=unpacked,
// CHECK-SAME: undef=[],
// CHECK-SAME: abi_version=0>>
func.func @scalar_only(%arg0: i32, %arg1: i32) -> i32 attributes {
    executor.function_metadata = #executor.func_meta<
        [i32, i32],
        [i32],
        num_output_args = 0
    >
} {
  %1 = executor.addi %arg0, %arg1 : i32
  return %1 : i32
}
