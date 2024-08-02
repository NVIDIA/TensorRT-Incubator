// RUN: executor-opt %s -executor-pack-arguments=max-arguments=2 | FileCheck %s

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
