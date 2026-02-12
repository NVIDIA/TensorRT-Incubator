// RUN: executor-opt %s -split-input-file -convert-executor-to-executor -cse | FileCheck %s

!memref_type = memref<?xf32, strided<[?], offset: ?>, #executor.memory_type<device>>

executor.plugin @my_plugin {
  plugin_name = "test_plugin",
  function_name = "my_func",
  config = {},
  ffi_backend = #executor.ffi_backend<tvm_ffi>
} : (!memref_type, !memref_type) -> (!memref_type)

func.func @test_call_plugin(%arg0: !memref_type) {
  executor.call_plugin @my_plugin
    ins(%arg0, %arg0: !memref_type, !memref_type)
    outs(%arg0 : !memref_type)
    {arg_spec = ["args.0", "args.1"]}
  return
}

//   CHECK-DAG:   executor.data_segment @str_literal1 constant align 4 "my_func\00"
//   CHECK-DAG:   executor.data_segment @str_literal0 constant align 4 "test_plugin\00"
//       CHECK:     %[[v0:.+]] = executor.load_data_segment @str_literal0 : !executor.ptr<host>
//       CHECK:     %[[v1:.+]] = executor.load_data_segment @str_literal1 : !executor.ptr<host>
//       CHECK:     %[[v2:.+]] = executor.call @_create_plugin_callable_tvm_ffi(%[[v0]], %[[v1]])
//       CHECK:     executor.return %[[v2]] : !executor.ptr<host>
// CHECK-LABEL: func.func @test_call_plugin
//  CHECK-SAME: (%[[arg0:.+]]: memref<
//   CHECK-DAG:     %[[v0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] :
//   CHECK-DAG:     %[[v1:.+]] = executor.call @__spmd_global_rank() : () -> i32
//   CHECK-DAG:     %[[c0_i64:.+]] = executor.constant 0 : i64
//   CHECK-DAG:     %[[v2:.+]] = executor.inttoptr %[[c0_i64]] : (i64) -> !executor.ptr<host>
//   CHECK-DAG:     %[[data_ptr:.+]] = executor.table.get %[[v0]][1] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK-DAG:     %[[c2_i32:.+]] = executor.constant 2 : i32
//   CHECK-DAG:     %[[v4:.+]] = executor.table.create(%[[c2_i32]], %[[v1]] : i32, i32) : <i32, i32>
//   CHECK-DAG:     %[[c1_i32:.+]] = executor.constant 1 : i32
//   CHECK-DAG:     %[[c2_i8:.+]] = executor.constant 2 : i8
//   CHECK-DAG:     %[[c32_i8:.+]] = executor.constant 32 : i8
//   CHECK-DAG:     %[[c1_i16:.+]] = executor.constant 1 : i16
//   CHECK-DAG:     %[[dl_data_type:.+]] = executor.table.create(%[[c2_i8]], %[[c32_i8]], %[[c1_i16]] : i8, i8, i16) : <i8, i8, i16>
//   CHECK-DAG:     %[[dim0:.+]] = executor.table.get %[[v0]][3] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK-DAG:     %[[stride0:.+]] = executor.table.get %[[v0]][4] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK-DAG:     %[[shape_array:.+]] = executor.table.create(%[[dim0]] : i64) : <i64>
//   CHECK-DAG:     %[[c1_i64:.+]] = executor.constant 1 : i64
//   CHECK-DAG:     %[[shape_array_ptr:.+]] = executor.alloca %[[c1_i64]] x !executor.table<i64> : (i64) -> !executor.ptr<host>
//   CHECK-DAG:     executor.store %[[shape_array]] to %[[shape_array_ptr]] + %[[c0_i64]] :
//   CHECK-DAG:     %[[strides_array:.+]] = executor.table.create(%[[stride0]] : i64) : <i64>
//   CHECK-DAG:     %[[strides_array_ptr:.+]] = executor.alloca %[[c1_i64]] x !executor.table<i64> : (i64) -> !executor.ptr<host>
//   CHECK-DAG:     executor.store %[[strides_array]] to %[[strides_array_ptr]] + %[[c0_i64]] : !executor.table<i64>, !executor.ptr<host>, i64
//   CHECK-DAG:     %[[el_offset:.+]] = executor.table.get %[[v0]][2] : <!executor.ptr<device>, !executor.ptr<device>, i64, i64, i64>
//   CHECK-DAG:     %[[bytes_offset:.+]] = executor.getoffset[%[[el_offset]]] : (i64) -> i64, f32
//   CHECK-DAG:     %[[dltensor:.+]] = executor.table.create(%[[data_ptr]], %[[v4]], %[[c1_i32]], %[[dl_data_type]], %[[shape_array_ptr]], %[[strides_array_ptr]], %[[bytes_offset]] : !executor.ptr<device>, !executor.table<i32, i32>, i32, !executor.table<i8, i8, i16>, !executor.ptr<host>, !executor.ptr<host>, i64) : <!executor.ptr<device>, !executor.table<i32, i32>, i32, !executor.table<i8, i8, i16>, !executor.ptr<host>, !executor.ptr<host>, i64>
//   CHECK-DAG:     %[[dltensor_ptr:.+]] = executor.alloca %[[c1_i64]] x !executor.table<!executor.ptr<device>, !executor.table<i32, i32>, i32, !executor.table<i8, i8, i16>, !executor.ptr<host>, !executor.ptr<host>, i64> : (i64) -> !executor.ptr<host>
//   CHECK-DAG:     executor.store %[[dltensor]] to %[[dltensor_ptr]] + %[[c0_i64]] : !executor.table<!executor.ptr<device>, !executor.table<i32, i32>, i32, !executor.table<i8, i8, i16>, !executor.ptr<host>, !executor.ptr<host>, i64>, !executor.ptr<host>, i64
//   CHECK-DAG:     %[[c7_i32:.+]] = executor.constant 7 : i32
//   CHECK-DAG:     %[[c0_i32:.+]] = executor.constant 0 : i32
//   CHECK-DAG:     %[[v16:.+]] = executor.ptrtoint %[[dltensor_ptr]] : (!executor.ptr<host>) -> i64
//   CHECK-DAG:     %[[any_val:.+]] = executor.table.create(%[[c7_i32]], %[[c0_i32]], %[[v16]] :
//   CHECK-DAG:     %[[any_array:.+]] = executor.table.create(%[[any_val]], %[[any_val]] :
//   CHECK-DAG:     %[[any_array_ptr:.+]] = executor.alloca
//   CHECK-DAG:     executor.store %[[any_array]] to %[[any_array_ptr]] + %[[c0_i64]] :
//   CHECK-DAG:     %[[plugin_handle:.+]] = executor.get_global @my_plugin : !executor.ptr<host>
//   CHECK-DAG:     executor.call @_call_plugin_tvm_ffi(%[[plugin_handle]], %[[v2]], %[[any_array_ptr]], %[[c2_i32]]) :
//       CHECK:     return
