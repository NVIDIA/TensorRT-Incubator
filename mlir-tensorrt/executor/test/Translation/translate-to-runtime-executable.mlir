// REQUIRES: host-has-at-least-1-gpus
// RUN:  executor-opt %s -split-input-file -convert-executor-to-executor | \
// RUN:  executor-translate -mlir-to-runtime-executable | executor-runner -dump-function-signature -input-type=rtexe | FileCheck %s

func.func public @value_bounds() attributes {executor.function_metadata = #executor.func_meta<[memref<1xi64> {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}, memref<1xi64> {#executor.value_bounds<min = dense<2> : vector<1xi64>, max = dense<4> : vector<1xi64>>}], [memref<1xi64> {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}], num_output_args = 1>} {
  return
}

func.func public @dim_bounds() attributes {executor.function_metadata = #executor.func_meta<[memref<?xf32> {#executor.dim_bounds<min = [1], max = [6]>}], [memref<?xf32> {#executor.dim_bounds<min = [2], max = [8]>}, memref<?xf32> {#executor.dim_bounds<min = [3], max = [5]>}], num_output_args = 1>} {
  return
}

func.func public @unit_attr() attributes {executor.function_metadata = #executor.func_meta<[memref<1xi64> {unit}], [], num_output_args = 0>} {
  return
}

func.func public @scalar_value_bounds() attributes {executor.function_metadata = #executor.func_meta<[i64 {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}], [], num_output_args = 0>} {
  return
}

func.func public @mixed_bounds() attributes {executor.function_metadata = #executor.func_meta<[i64 {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}, memref<1xi64> {unit}, memref<1xi64> {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}], [memref<?xf32> {#executor.dim_bounds<min = [2], max = [8]>}, memref<1xf32> {#executor.dim_bounds<min = [3], max = [5]>}], num_output_args = 1>} {
  return
}

//      CHECK: Signature<args=[MemRef<1xi64,1,unknown>, MemRef<1xi64,1,unknown>], results=[MemRef<1xi64,1,unknown>], num_output_args=1, arg_bounds=[value_bounds<min = [1], max = [6]>, value_bounds<min = [2], max = [4]>], result_bounds=[value_bounds<min = [1], max = [6]>]>
// CHECK-NEXT: Signature<args=[MemRef<?xf32,1,unknown>], results=[MemRef<?xf32,1,unknown>, MemRef<?xf32,1,unknown>], num_output_args=1, arg_bounds=[dim_bounds<min = [1], max = [6]>], result_bounds=[dim_bounds<min = [2], max = [8]>, dim_bounds<min = [3], max = [5]>]>
// CHECK-NEXT: Signature<args=[MemRef<1xi64,1,unknown>], results=[], num_output_args=0, arg_bounds=[UNK], result_bounds=[]>
// CHECK-NEXT: Signature<args=[i64], results=[], num_output_args=0, arg_bounds=[value_bounds<min = [1], max = [6]>], result_bounds=[]>
// CHECK-NEXT: Signature<args=[i64, MemRef<1xi64,1,unknown>, MemRef<1xi64,1,unknown>], results=[MemRef<?xf32,1,unknown>, MemRef<1xf32,1,unknown>], num_output_args=1, arg_bounds=[value_bounds<min = [1], max = [6]>, UNK, value_bounds<min = [1], max = [6]>], result_bounds=[dim_bounds<min = [2], max = [8]>, dim_bounds<min = [3], max = [5]>]>
