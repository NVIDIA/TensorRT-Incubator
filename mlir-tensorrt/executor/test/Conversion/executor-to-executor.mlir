// RUN: executor-opt %s -split-input-file -convert-executor-to-executor="use-packed-memref-cconv=false index-bitwidth=32" | FileCheck %s
// RUN: executor-opt %s -split-input-file -convert-executor-to-executor="index-bitwidth=32" | FileCheck %s --check-prefix=PACKED

executor.func private @my_func(index, memref<128xf32>) -> index

func.func @executor_func_call(%arg0: index, %arg1: memref<128xf32>) -> index {
  %0 = executor.call @my_func(%arg0, %arg1) : (index, memref<128xf32>) -> index
  return %0 : index
}

// CHECK-LABEL: executor.func private @my_func(i32, !executor.ptr<host>, !executor.ptr<host>, i32, i32, i32) -> i32
// CHECK-LABEL: @executor_func_call
//  CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: memref<128xf32>)
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : index to i32
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<128xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     %[[v2:.+]] = executor.table.get %[[v1]][0] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     %[[v3:.+]] = executor.table.get %[[v1]][1] : <!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       CHECK:     %[[c0_i32:.+]] = executor.constant 0 : i32
//       CHECK:     %[[c128_i32:.+]] = executor.constant 128 : i32
//       CHECK:     %[[c1_i32:.+]] = executor.constant 1 : i32
//       CHECK:     %[[v4:.+]] = executor.call @my_func(%[[v0]], %[[v2]], %[[v3]], %[[c0_i32]], %[[c128_i32]], %[[c1_i32]]) : (i32, !executor.ptr<host>, !executor.ptr<host>, i32, i32, i32) -> i32
//       CHECK:     %[[v5:.+]] = builtin.unrealized_conversion_cast %[[v4]] : i32 to index
//       CHECK:     return %[[v5]] : index

// PACKED-LABEL: executor.func private @my_func(i32, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>) -> i32
// PACKED-LABEL: @executor_func_call
//  PACKED-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: memref<128xf32>) -> index {
//   PACKED-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : index to i32
//   PACKED-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : memref<128xf32> to !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>
//       PACKED:     %[[v2:.+]] = executor.call @my_func(%[[v0]], %[[v1]]) : (i32, !executor.table<!executor.ptr<host>, !executor.ptr<host>, i32, i32, i32>) -> i32
//       PACKED:     %[[v3:.+]] = builtin.unrealized_conversion_cast %[[v2]] : i32 to index
//       PACKED:     return %[[v3]] : index

// -----

executor.global @global_with_index_type_conversion
    : index {
  %0 = arith.constant 0 : index
  executor.return %0 : index
}

// CHECK-LABEL:   executor.global @global_with_index_type_conversion : i32 {

// -----

func.func @exec_add_index(%arg0: index, %arg1: index) -> index {
  %0 = executor.addi %arg0, %arg1 : index
  return %0 : index
}

// CHECK-LABEL: @exec_add_index
//  CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: index) -> index {
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : index to i32
//   CHECK-DAG:     %[[v1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : index to i32
//       CHECK:     %[[v2:.+]] = executor.addi %[[v0]], %[[v1]] : i32
//       CHECK:     %[[v3:.+]] = builtin.unrealized_conversion_cast %[[v2]] : i32 to index
//       CHECK:     return %[[v3]] : index

// -----


func.func @memref_load(%arg0: !executor.ptr<host>, %arg1: i64) {
  %8 = executor.load %arg0 + %arg1 : (!executor.ptr<host>, i64) -> !executor.table<!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  %9 = executor.table.get %8[0] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
  executor.dealloc %9 : <host>
  return
}

//   CHECK-DAG:   executor.func private @_dealloc(...)
//   CHECK-DAG:   executor.func private @_load_i64(...) -> i64
//   CHECK-DAG:   executor.func private @_load_ptr_host(...) -> !executor.ptr<host>
// CHECK-LABEL: @memref_load
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: i64) {
//       CHECK:     %[[c0_i64:.+]] = executor.constant 0 : i64
//       CHECK:     %[[v0:.+]] = executor.addi %[[arg1]], %[[c0_i64]] : i64
//       CHECK:     %[[v1:.+]] = executor.call @_load_ptr_host(%[[arg0]], %[[v0]]) : (!executor.ptr<host>, i64) -> !executor.ptr<host>
//       CHECK:     %[[c8_i64:.+]] = executor.constant 8 : i64
//       CHECK:     %[[v2:.+]] = executor.addi %[[arg1]], %[[c8_i64]] : i64
//       CHECK:     %[[v3:.+]] = executor.call @_load_ptr_host(%[[arg0]], %[[v2]]) : (!executor.ptr<host>, i64) -> !executor.ptr<host>
//       CHECK:     %[[c16_i64:.+]] = executor.constant 16 : i64
//       CHECK:     %[[v4:.+]] = executor.addi %[[arg1]], %[[c16_i64]] : i64
//       CHECK:     %[[v5:.+]] = executor.call @_load_i64(%[[arg0]], %[[v4]]) : (!executor.ptr<host>, i64) -> i64
//       CHECK:     %[[c24_i64:.+]] = executor.constant 24 : i64
//       CHECK:     %[[v6:.+]] = executor.addi %[[arg1]], %[[c24_i64]] : i64
//       CHECK:     %[[v7:.+]] = executor.call @_load_i64(%[[arg0]], %[[v6]]) : (!executor.ptr<host>, i64) -> i64
//       CHECK:     %[[c32_i64:.+]] = executor.constant 32 : i64
//       CHECK:     %[[v8:.+]] = executor.addi %[[arg1]], %[[c32_i64]] : i64
//       CHECK:     %[[v9:.+]] = executor.call @_load_i64(%[[arg0]], %[[v8]]) : (!executor.ptr<host>, i64) -> i64
//       CHECK:     %[[v10:.+]] = executor.table.create(%[[v1]], %[[v3]], %[[v5]], %[[v7]], %[[v9]] :
//       CHECK:     %[[v11:.+]] = executor.table.get %[[v10]][0] : <!executor.ptr<host>, !executor.ptr<host>, i64, i64, i64>
//       CHECK:     executor.call @_dealloc(%[[v11]]) :

// -----

func.func @metadata_conversion(%arg0: i32)  attributes {
  executor.function_metadata = #executor.func_meta<[
      index
    ], [], num_output_args=0>
} {
  return
}

// CHECK-LABEL: @metadata_conversion
//  CHECK-SAME:  attributes {executor.function_metadata = #executor.func_meta<[i32], [], num_output_args = 0>}
