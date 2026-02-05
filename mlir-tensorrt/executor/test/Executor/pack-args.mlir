// RUN: executor-opt %s -executor-pack-arguments=max-arguments=2 -split-input-file | FileCheck %s

func.func @small_func(%arg0: i32, %arg1: i32) -> i32 attributes {
  executor.function_metadata = #executor.func_meta<
    [i32, i32], [i32], num_output_args=0>
} {
  return %arg0: i32
}

func.func @big_func(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 attributes {
  executor.function_metadata = #executor.func_meta<[i32, i32, i32], [i32], num_output_args=0>
} {
  return %arg0: i32
}

// CHECK-LABEL: func.func @small_func
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i32
//  CHECK-SAME:  executor.function_metadata = #executor.func_meta<[i32, i32], [i32], num_output_args = 0>
//       CHECK:     return %[[arg0]]

// CHECK-LABEL: func.func @big_func
//  CHECK-SAME: (%[[arg0:.+]]: !executor.table<i32, i32, i32>) -> i32
//  CHECK-SAME:    executor.function_metadata = #executor.func_meta<[i32, i32, i32], [i32], num_output_args = 0, cconv = packed>
//       CHECK:     %[[v0:.+]] = executor.table.get %[[arg0]][0]
//       CHECK:     return %[[v0]]

// -----


func.func @arg_abi_input_byval_memref(
    %arg0: !executor.ptr<host> {executor.abi = #executor.arg<byval, memref<10xi32>>},
    %arg1: i32,
    %arg2: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>},
    %arg3: !executor.ptr<host> {executor.abi = #executor.arg<byref, memref<10xi32>>})
      attributes {
        executor.func_abi = (memref<10xi32>, i32) -> (memref<10xi32>, memref<10xi32>)
      } {


  return
}

// CHECK-LABEL: func.func @arg_abi_input_byval_memref
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>)
//  CHECK-SAME:   executor.func_abi = (memref<10xi32>, i32) -> (memref<10xi32>, memref<10xi32>)
//  CHECK-SAME:   executor.func_abi_packed_args = [
//  CHECK-SAME:     {abi.attr = #executor.arg<byval, memref<10xi32>>, abi.type = !executor.ptr<host>},
//  CHECK-SAME:     {abi.type = i32},
//  CHECK-SAME:     {abi.attr = #executor.arg<byref, memref<10xi32>>, abi.type = !executor.ptr<host>},
//  CHECK-SAME:     {abi.attr = #executor.arg<byref, memref<10xi32>>, abi.type = !executor.ptr<host>}
//  CHECK-SAME:   ]
//       CHECK:     return
