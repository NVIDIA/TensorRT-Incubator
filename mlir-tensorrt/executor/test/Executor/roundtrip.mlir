// RUN: executor-opt %s -split-input-file | executor-opt -split-input-file | FileCheck %s

func.func @executor_attributes() attributes {
  executor.some_property = #executor.memory_type<host>
} {
  return
}

// CHECK-LABEL: @executor_attributes
//  CHECK-SAME: () attributes {executor.some_property = #executor.memory_type<host>} {

//-----

func.func @addi(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.addi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: @addi(
//  CHECK-SAME:  %[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i32 {
//       CHECK:     %[[v0:.+]] = executor.addi %[[arg0]], %[[arg1]] : i32
//       CHECK:  return %[[v0]] : i32

// -----

func.func @addf(%arg0: f32, %arg1: f32) -> f32 {
  %0 = executor.addf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: @addf(
//  CHECK-SAME:  %[[arg0:.+]]: f32, %[[arg1:.+]]: f32) -> f32 {
//       CHECK:     %[[v0:.+]] = executor.addf %[[arg0]], %[[arg1]] : f32
//       CHECK:  return %[[v0]] : f32

// -----

func.func @subi(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.subi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: @subi(
//   CHECK-SAME:  %[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i32 {
//       CHECK:  %[[v0:.+]] = executor.subi %[[arg0]], %[[arg1]] : i32
//       CHECK:  return %[[v0]] : i32

// -----

func.func @subf(%arg0: f32, %arg1: f32) -> f32 {
  %0 = executor.subf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: @subf(
//   CHECK-SAME:  %[[arg0:.+]]: f32, %[[arg1:.+]]: f32) -> f32 {
//       CHECK:  %[[v0:.+]] = executor.subf %[[arg0]], %[[arg1]] : f32
//       CHECK:  return %[[v0]] : f32

// -----

func.func @sdivi(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.sdivi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: @sdivi(
//   CHECK-SAME:  %[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i32 {
//       CHECK:  %[[v0:.+]] = executor.sdivi %[[arg0]], %[[arg1]] : i32
//       CHECK:  return %[[v0]] : i32

// -----

func.func @divf(%arg0: f32, %arg1: f32) -> f32 {
  %0 = executor.divf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: @divf(
//   CHECK-SAME:  %[[arg0:.+]]: f32, %[[arg1:.+]]: f32) -> f32 {
//       CHECK:  %[[v0:.+]] = executor.divf %[[arg0]], %[[arg1]] : f32
//       CHECK:  return %[[v0]] : f32

// -----

func.func @muli(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.muli %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: @muli(
//   CHECK-SAME:  %[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i32 {
//       CHECK:  %[[v0:.+]] = executor.muli %[[arg0]], %[[arg1]] : i32
//       CHECK:  return %[[v0]] : i32

// -----

func.func @mulf(%arg0: f32, %arg1: f32) -> f32 {
  %0 = executor.mulf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: @mulf(
//   CHECK-SAME:  %[[arg0:.+]]: f32, %[[arg1:.+]]: f32) -> f32 {
//       CHECK:  %[[v0:.+]] = executor.mulf %[[arg0]], %[[arg1]] : f32
//       CHECK:  return %[[v0]] : f32

// -----

func.func @sfloordivi(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.sfloor_divi %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: sfloordivi
//       CHECK: %[[v_0:.+]] = executor.sfloor_divi %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @sremi(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.sremi %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: sremi
//       CHECK: %[[v_0:.+]] = executor.sremi %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @shift_left(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.shift_lefti %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: shift_left
//       CHECK: %[[v_0:.+]] = executor.shift_lefti %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @shift_right_arithmetic(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.shift_right_arithmetici %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: shift_right_arithmetic
//       CHECK: %[[v_0:.+]] = executor.shift_right_arithmetici %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @shift_right_logical(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.shift_right_logicali %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: shift_right_logical
//       CHECK: %[[v_0:.+]] = executor.shift_right_logicali %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @bitwise_and(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.bitwise_andi %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: bitwise_and
//       CHECK: %[[v_0:.+]] = executor.bitwise_andi %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @bitwise_or(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.bitwise_ori %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: bitwise_or
//       CHECK: %[[v_0:.+]] = executor.bitwise_ori %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @bitwise_xor(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.bitwise_xori %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: bitwise_xor
//       CHECK: %[[v_0:.+]] = executor.bitwise_xori %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @maxi(%arg0: i32, %arg1: i32) -> i32 {
  %0 = executor.smax %arg0, %arg1 : i32
  return %0 : i32
}
// CHECK-LABEL: maxi
//       CHECK: %[[v_0:.+]] = executor.smax %[[arg0:.+]], %[[arg1:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @maxf(%arg0: f32, %arg1: f32) -> f32 {
  %0 = executor.fmax %arg0, %arg1 : f32
  return %0 : f32
}
// CHECK-LABEL: maxf
//       CHECK: %[[v_0:.+]] = executor.fmax %[[arg0:.+]], %[[arg1:.+]] : f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @bitcast(%arg0: f32) -> i32 {
  %0 = executor.bitcast %arg0 : f32 to i32
  return %0 : i32
}
// CHECK-LABEL: bitcast
//       CHECK: %[[v_0:.+]] = executor.bitcast %[[arg0:.+]] : f32 to i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @bitcast_same(%arg0: i32) -> i32 {
  %0 = executor.bitcast %arg0 : i32 to i32
  return %0 : i32
}
// CHECK-LABEL: bitcast_same
//       CHECK: %[[v_0:.+]] = executor.bitcast %[[arg0:.+]] : i32 to i32
//       CHECK: return %[[v_0]] : i32
// -----

func.func @select(%arg0: i1, %arg1: i32, %arg2 : i32) -> i32 {
    %0 = executor.select %arg0, %arg1, %arg2 : i32
    return %0: i32
}
// CHECK-LABEL: select
//       CHECK: %[[v_0:.+]] = executor.select %[[arg0:.+]], %[[arg1:.+]], %[[arg2:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @negf(%arg0: f32) -> f32 {
    %0 = executor.negf %arg0 : f32
    return %0 : f32
}
// CHECK-LABEL:  negf
//       CHECK: %[[v_0:.+]] = executor.negf %[[arg0:.+]] : f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @absi(%arg0: i32) -> i32 {
    %0 = executor.absi %arg0 : i32
    return %0 : i32
}
// CHECK-LABEL: absi
//       CHECK: %[[v_0:.+]] = executor.absi %[[arg0:.+]] : i32
//       CHECK: return %[[v_0]] : i32

// -----

func.func @absf(%arg0: f32) -> f32 {
    %0 = executor.absf %arg0 : f32
    return %0 : f32
}
// CHECK-LABEL: absf
//       CHECK: %[[v_0:.+]] = executor.absf %[[arg0:.+]] : f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @sqrt(%arg0: f32) -> f32 {
    %0 = executor.sqrt %arg0 : f32
    return %0 : f32
}
// CHECK-LABEL: sqrt
//       CHECK: %[[v_0:.+]] = executor.sqrt %[[arg0:.+]] : f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @log1p(%arg0: f32) -> f32 {
    %0 = executor.log1p %arg0 : f32
    return %0 : f32
}
// CHECK-LABEL: log1p
//       CHECK: %[[v_0:.+]] = executor.log1p %[[arg0:.+]] : f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @cast_si_to_fp(%arg0: i16) -> f32 {
  %0 = executor.sitofp %arg0 : i16 to f32
  return %0 : f32
}
// CHECK-LABEL: cast_si_to_fp
//       CHECK: %[[v_0:.+]] = executor.sitofp %[[arg0:.+]] : i16 to f32
//       CHECK: return %[[v_0]] : f32

// -----

func.func @cast_fp_to_si(%arg0: f16) -> i32 {
  %0 = executor.fptosi %arg0 : f16 to i32
  return %0 : i32
}
// CHECK-LABEL: cast_fp_to_si
//       CHECK: %[[v_0:.+]] = executor.fptosi %[[arg0:.+]] : f16 to i32
//       CHECK: return %[[v_0]] : i32

// -----

!descriptor2d = !executor.table<
    i64,
    i64,
    i32,
    i32, i32,
    i32, i32>

func.func @executor_aggregates(%arg0: i64, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) -> (!descriptor2d, !descriptor2d, i32, !descriptor2d) {
  %0 = executor.table.create : !descriptor2d
  %1 = executor.table.create (%arg0, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : i64, i64, i32, i32, i32, i32, i32): !descriptor2d
  %2 = executor.table.get %1[2] : !descriptor2d
  %c0 = executor.constant 0 : i32
  %3 = executor.table.set %c0 into %1[2]  : i32, !descriptor2d
  return %0, %1, %2, %3 : !descriptor2d, !descriptor2d, i32, !descriptor2d
}

// CHECK-LABEL: @executor_aggregates
//  CHECK-SAME: (%[[arg0:.+]]: i64, %[[arg1:.+]]: i32, %[[arg2:.+]]: i32, %[[arg3:.+]]: i32, %[[arg4:.+]]: i32, %[[arg5:.+]]: i32)
//       CHECK:     %[[v0:.+]] = executor.table.create : <i64, i64, i32, i32, i32, i32, i32>
//       CHECK:     %[[v1:.+]] = executor.table.create(%[[arg0]], %[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]], %[[arg4]], %[[arg5]] : i64, i64, i32, i32, i32, i32, i32) : <i64, i64, i32, i32, i32, i32, i32>
//       CHECK:     %[[v2:.+]] = executor.table.get %[[v1]][2] : <i64, i64, i32, i32, i32, i32, i32>
//       CHECK:     %[[c0_i32:.+]] = executor.constant 0 : i32
//       CHECK:     %[[v3:.+]] = executor.table.set %[[c0_i32]] into %[[v1]][2] : i32, <i64, i64, i32, i32, i32, i32, i32>
//       CHECK:     return %[[v0]], %[[v1]], %[[v2]], %[[v3]] : !executor.table<i64, i64, i32, i32, i32, i32, i32>, !executor.table<i64, i64, i32, i32, i32, i32, i32>, i32, !executor.table<i64, i64, i32, i32, i32, i32, i32>

// -----

func.func @icmp(%arg0: i32, %arg1: i32) -> i1 {
  %0 = executor.icmp <eq> %arg0, %arg1 : i32
  return %0 : i1
}

// CHECK-LABEL: @icmp
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i1 {
//       CHECK:     %[[v0:.+]] = executor.icmp <eq> %[[arg0]], %[[arg1]] : i32
//       CHECK:     return %[[v0]] : i1

// -----

func.func @alloc(%arg0: i32, %alignment: i32) -> !executor.ptr<host> {
  %0 = executor.alloc %arg0 bytes align(%alignment) : (i32, i32) -> !executor.ptr<host>
  return %0 : !executor.ptr<host>
}

// CHECK-LABEL: @alloc
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[align:.+]]: i32) -> !executor.ptr<host> {
//       CHECK:     %[[v0:.+]] = executor.alloc %[[arg0]] bytes align(%[[align]]) : (i32, i32) -> !executor.ptr<host>
//       CHECK:     return %[[v0]] : !executor.ptr<host>

// -----

func.func @load(%arg0: !executor.ptr<host>, %arg1: i32) -> f32 {
  %0 = executor.load %arg0 + %arg1 : (!executor.ptr<host>, i32) -> f32
  return %0 : f32
}

// CHECK-LABEL: @load
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: i32) -> f32 {
//       CHECK:     %[[v0:.+]] = executor.load %[[arg0]] + %[[arg1]] : (!executor.ptr<host>, i32) -> f32
//       CHECK:     return %[[v0]] : f32

// -----

func.func @store(%arg0: !executor.ptr<host>, %arg1: i32, %arg2: f32) {
  executor.store %arg2 to %arg0 + %arg1 : f32, !executor.ptr<host>, i32
  return
}

// CHECK-LABEL: @store
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: i32, %[[arg2:.+]]: f32) {
//       CHECK:     executor.store %[[arg2]] to %[[arg0]] + %[[arg1]] : f32, !executor.ptr<host>, i32
//       CHECK:     return

// -----

func.func @memcpy(%arg0: !executor.ptr<host>, %arg1: !executor.ptr<host>, %arg2: i32, %arg3: i32) {
  executor.memcpy %arg0 + %arg2 to %arg1 + %arg2 size %arg3 : !executor.ptr<host>, i32, !executor.ptr<host>, i32, i32
  return
}

// CHECK-LABEL: @memcpy
//  CHECK-SAME: (%[[arg0:.+]]: !executor.ptr<host>, %[[arg1:.+]]: !executor.ptr<host>, %[[arg2:.+]]: i32, %[[arg3:.+]]: i32) {
//       CHECK:     executor.memcpy %[[arg0]] + %[[arg2]] to %[[arg1]] + %[[arg2]] size %[[arg3]] : !executor.ptr<host>, i32, !executor.ptr<host>, i32, i32

// -----

executor.global @global1 constant : vector<10xf32> attributes {
  initial_value = dense<0.0> : vector<10xf32>
}

executor.global @global2 : vector<10xf32> attributes {
  initial_value = dense<0.0> : vector<10xf32>
}

executor.global @global3 constant : vector<10xf32> {
  %0 = arith.constant dense<0.0> : vector<10xf32>
  executor.return %0 : vector<10xf32>
}

func.func @global_get() {
  %0 = executor.get_global @global1 : vector<10xf32>
  executor.set_global %0, @global2 : vector<10xf32>
  return
}

//       CHECK:   executor.global @global1 constant : vector<10xf32> attributes {initial_value = dense<0.000000e+00> : vector<10xf32>}
//       CHECK:   executor.global @global2 : vector<10xf32> attributes {initial_value = dense<0.000000e+00> : vector<10xf32>}
//       CHECK:   executor.global @global3 constant : vector<10xf32> {
//       CHECK:     %[[cst:.+]] = arith.constant dense<0.000000e+00> : vector<10xf32>
//       CHECK:     executor.return %[[cst]] : vector<10xf32>
// CHECK-LABEL: @global_get
//       CHECK:     %[[v0:.+]] = executor.get_global @global1 : vector<10xf32>
//       CHECK:     executor.set_global %[[v0]], @global2 : vector<10xf32>
//       CHECK:     return

// -----

// Constant globals are allowed to be set within the global initialzer func.
builtin.module @module attributes { executor.global_init_func = @global_init } {
  executor.global @global1 constant : vector<10xf32>

  func.func @global_init() {
    %0 = arith.constant dense<0.0> : vector<10xf32>
    executor.set_global %0, @global1 : vector<10xf32>
    return
  }
}

// CHECK-LABEL: @global_init
//       CHECK:     %[[cst:.+]] = arith.constant
//       CHECK:     executor.set_global %[[cst]], @global1 : vector<10xf32>

// -----

func.func @const_literal() -> !executor.str_literal{
  %0 = executor.str_literal "function_1"
  return %0 : !executor.str_literal
}

// CHECK-LABEL: @const_literal
//       CHECK: %{{.+}} = executor.str_literal "function_1"

// -----

executor.func private @enqueue_variadic(i32, ...)
executor.func private @enqueue_non_variadic(i32, !executor.ptr<device>)

func.func @main(%arg0: i32, %arg1: !executor.ptr<device>) {
  executor.call @enqueue_variadic(%arg0, %arg1) : (i32, !executor.ptr<device>) -> ()
  executor.call @enqueue_non_variadic(%arg0, %arg1) : (i32, !executor.ptr<device>) -> ()
  return
}

// CHECK-LABEL: executor.func private @enqueue_variadic(i32, ...)
//       CHECK: executor.func private @enqueue_non_variadic(i32, !executor.ptr<device>)
//       CHECK: func.func @main
//  CHECK-SAME: (%[[arg0:.+]]: i32, %[[arg1:.+]]: !executor.ptr<device>) {
//       CHECK:   executor.call @enqueue_variadic(%[[arg0]], %[[arg1]]) : (i32, !executor.ptr<device>) -> ()
//       CHECK:   executor.call @enqueue_non_variadic(%[[arg0]], %[[arg1]]) : (i32, !executor.ptr<device>) -> ()

// -----

module @func_type attributes {
  executor.func_type = !executor.func<(i32, ...)->()>,
  executor.func_type2 = !executor.func<(i32)->(i32)>,
  executor.func_type3 = !executor.func<(...)->(i32, i32)>,
  executor.func_type4 = !executor.func<()->()>
} {

}

// CHECK-LABEL: module @func_type
//  CHECK-SAME: executor.func_type = !executor.func<(i32, ...) -> ()>
//  CHECK-SAME: executor.func_type2 = !executor.func<(i32) -> (i32)>
//  CHECK-SAME: executor.func_type3 = !executor.func<(...) -> (i32, i32)>
//  CHECK-SAME: executor.func_type4 = !executor.func<() -> ()>

// -----


func.func @siext(%arg0: i8) -> i32 {
  %0 = executor.siext %arg0 : i8 to i32
  return %0 : i32
}

// CHECK-LABEL: @siext
//       CHECK:     %[[v0:.+]] = executor.siext %{{.+}} : i8 to i32

// -----

func.func @zext(%arg0: i8) -> i32 {
  %0 = executor.zext %arg0 : i8 to i32
  return %0 : i32
}

// CHECK-LABEL: @zext
//       CHECK:  executor.zext %{{.+}} : i8 to i32

// -----

func.func @func_metadata0(%arg0: i32, %arg1: i32) -> i32 attributes {
    executor.function_metadata = #executor.func_meta<[i32, i32], [i32], num_output_args=0>
} {
  return %arg0: i32
}

// CHECK-LABEL: @func_metadata0
//  CHECK-SAME: #executor.func_meta<[i32, i32], [i32], num_output_args = 0>

// -----

func.func @func_metadata1(%arg0: i32, %arg1: i32) -> i32 attributes {
    executor.function_metadata = #executor.func_meta<[i32, i32], [i32], num_output_args=0, cconv=unpacked>
} {
  return %arg0: i32
}

// CHECK-LABEL: @func_metadata1
//  CHECK-SAME: #executor.func_meta<[i32, i32], [i32], num_output_args = 0>

// -----

func.func @func_metadata2(
  %arg0: !executor.ptr<device>, %arg1: !executor.ptr<device>, %arg2: i64, %arg3: i64, %arg4: i64,
  %arg5: !executor.ptr<device>, %arg6: !executor.ptr<device>, %arg7: i64, %arg8: i64, %arg9: i64) -> i32 attributes {
    executor.function_metadata = #executor.func_meta<[memref<?xf32> {#executor.dim_bounds<min=[1], max=[3]>},
                                                      memref<?xf32> {#executor.dim_bounds<min=[1], max=[3]>}], [i32],
                                                      num_output_args=1, cconv=unpacked, shape_func = @func_metadata2_shape>
} {
  %c0 = executor.constant 0 : i32
  return %c0  : i32
}

// CHECK-LABEL: @func_metadata2
//  CHECK-SAME:  #executor.func_meta<[memref<?xf32> {#executor.dim_bounds<min = [1], max = [3]>}, memref<?xf32> {#executor.dim_bounds<min = [1], max = [3]>}], [i32], num_output_args = 1, shape_func = @func_metadata2_shape>


// -----

func.func @func_metadata3(%arg0: !executor.table<i32, i32>) -> i32 attributes {
    executor.function_metadata = #executor.func_meta<[i32, i32], [i32], num_output_args=0, cconv=packed>
} {
  %0 = executor.table.get %arg0[0] : !executor.table<i32, i32>
  return %0: i32
}

// CHECK-LABEL: @func_metadata3
//  CHECK-SAME: #executor.func_meta<[i32, i32], [i32], num_output_args = 0, cconv = packed>

// -----

func.func @func_metadata4(%arg0: i32, %arg1: i32) -> i32 attributes {
    executor.function_metadata = #executor.func_meta<[i32, i32], [i32], num_output_args=1>
} {
  return %arg0: i32
}

// CHECK-LABEL: @func_metadata4
//  CHECK-SAME: #executor.func_meta<[i32, i32], [i32], num_output_args = 1>

// -----

func.func @getoffset() -> i64 {
  %0 = executor.getoffset [0] : () -> i64, f32
  return %0 : i64
}

// CHECK-LABEL: func.func @getoffset
//       CHECK:     %[[v0:.+]] = executor.getoffset[0] : () -> i64, f32
//       CHECK:     return %[[v0]] : i64

// -----

func.func @getoffset_dynamic(%arg1: i64) -> i64 {
  %0 = executor.getoffset [%arg1] : (i64) -> i64, f32
  return %0 : i64
}

// CHECK-LABEL: func.func @getoffset_dynamic
//  CHECK-SAME: (%[[arg0:.+]]: i64) -> i64 {
//       CHECK:     %[[v0:.+]] = executor.getoffset[%[[arg0]]] : (i64) -> i64, f32
//       CHECK:     return %[[v0]] : i64

// -----

func.func @getoffset_mixed(%arg0: i32) -> i64 {
  %0 = executor.getoffset [%arg0, 1] : (i32) -> i64, !executor.table<f32, f32>
  return %0 : i64
}

// CHECK-LABEL: func.func @getoffset_mixed
//  CHECK-SAME: (%[[arg0:.+]]: i32) -> i64 {
//       CHECK:     %[[v0:.+]] = executor.getoffset[%[[arg0]], 1] : (i32) -> i64, !executor.table<f32, f32>
//       CHECK:     return %[[v0]] : i64

// -----

func.func @table_extract(%arg0: !executor.table<f32, i32>) -> (f32, i32) {
  %0 = executor.table.get %arg0 [0] : !executor.table<f32, i32>
  %1 = executor.table.get %arg0 [1] : !executor.table<f32, i32>
  return %0, %1 : f32, i32
}

// CHECK-LABEL: func.func @table_extract(
//  CHECK-SAME: %[[arg0:.+]]: !executor.table<f32, i32>) -> (f32, i32) {
//       CHECK:     %[[v0:.+]] = executor.table.get %[[arg0]][0] : <f32, i32>
//       CHECK:     %[[v1:.+]] = executor.table.get %[[arg0]][1] : <f32, i32>
//       CHECK:     return %[[v0]], %[[v1]] : f32, i32

// -----

func.func @table_extract_table(%arg0: !executor.table<i32, !executor.table<f32, f32>>) -> !executor.table<f32, f32> {
  %0 = executor.table.get %arg0 [1] : !executor.table<i32, !executor.table<f32, f32>>
  return %0 : !executor.table<f32, f32>
}

// CHECK-LABEL: func.func @table_extract_table(
//  CHECK-SAME: %[[arg0:.+]]: !executor.table<i32, !executor.table<f32, f32>>)
//       CHECK:     %[[v0:.+]] = executor.table.get %[[arg0]][1] : <i32, !executor.table<f32, f32>>
//       CHECK:     return %[[v0]] : !executor.table<f32, f32>

// -----

func.func @my_coro(%arg0: index, %arg1: i32) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %arg1) -> i32 {
    %next = arith.addi %acc, %c1_i32 : i32
    executor.coro_yield %next : i32
    scf.yield %next : i32
  }
  return %0 : i32
}

// CHECK-LABEL: func.func @my_coro
//  CHECK-SAME: (%[[arg0:.+]]: index, %[[arg1:.+]]: i32) -> i32 {
//       CHECK:     %[[c1_i32:.+]] = arith.constant 1 : i32
//       CHECK:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK:     %[[v0:.+]] = scf.for %[[arg2]] = %[[c0]] to %[[arg0]] step %[[c1]] iter_args(%[[arg3]] = %[[arg1]]) -> (i32)
//       CHECK:       %[[v1:.+]] = arith.addi %[[arg3]], %[[c1_i32]] : i32
//       CHECK:       executor.coro_yield %[[v1]] : i32
//       CHECK:       scf.yield %[[v1]] : i32
//       CHECK:     return %[[v0]] : i32


// -----

func.func @coro(%arg0: f32, %arg1: i32) -> i32 {
  %c2_i32 = arith.constant 2 : i32
  executor.coro_yield %c2_i32 : i32
  return %c2_i32 : i32
}

func.func @coro_await() -> (i32) {
  %c0 = arith.constant 0 : i32
  %c0_f32 = arith.constant 0.0 : f32
  %coro = executor.coro_create @coro : (f32, i32) -> i32
  %0:2 = executor.coro_await %coro (%c0_f32, %c0 : f32, i32) : (f32, i32) -> i32
  %1:2 = executor.coro_await %coro () : (f32, i32) -> i32
  return %1#1 : i32
}

// CHECK-LABEL: func.func @coro
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: i32) -> i32 {
//       CHECK:     %[[c2_i32:.+]] = arith.constant 2 : i32
//       CHECK:     executor.coro_yield %[[c2_i32]] : i32
//       CHECK:     return %[[c2_i32]] : i32
// CHECK-LABEL: func.func @coro_await
//       CHECK:     %[[c0_i32:.+]] = arith.constant 0 : i32
//       CHECK:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:     %[[v0:.+]] = executor.coro_create @coro : (f32, i32) -> i32
//       CHECK:     %[[status:status.*]], %[[results:.+]] = executor.coro_await %[[v0]](%[[cst]], %[[c0_i32]] : f32, i32) : (f32, i32) -> i32
//       CHECK:     %[[status_0:.+]], %[[results_1:.+]] = executor.coro_await %[[v0]]() : (f32, i32) -> i32
//       CHECK:     return %[[results_1]] : i32