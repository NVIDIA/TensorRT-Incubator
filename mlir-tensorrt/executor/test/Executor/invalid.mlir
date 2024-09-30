// RUN: executor-opt -split-input-file --verify-diagnostics %s

func.func @executor_alloc(%arg0: i32) -> !executor.ptr<host> {
  %align = executor.constant 5 : i32
  // expected-error @below {{'executor.alloc' op alignment must be a power of 2, but got 5}}
  %0 = executor.alloc %arg0 bytes align(%align) : (i32, i32) -> !executor.ptr<host>
  return %0 : !executor.ptr<host>
}

// -----

// expected-error @below {{'executor.global' op expected initial_value to have type 'vector<10xf32>'}}
executor.global @global1 constant : vector<10xf32> attributes {
  initial_value = dense<0.0> : vector<11xf32>
}

// -----

executor.global @global1 constant : vector<10xf32> attributes {
  initial_value = dense<0.0> : vector<10xf32>
}

executor.global @global2 constant : vector<10xf32> attributes {
  initial_value = dense<0.0> : vector<10xf32>
}

func.func @global_get() {
  %0 = executor.get_global @global1 : vector<10xf32>
  // expected-error @below {{'executor.set_global' op trying to set a global marked as constant}}
  executor.set_global %0, @global2 : vector<10xf32>
  return
}

// -----

// expected-error @below {{'executor.global' op expected initialization region to return one value of type 'vector<10xf32>'}}
executor.global @global3 constant : vector<10xf32> {
  %0 = arith.constant dense<0.0> : vector<11xf32>
  executor.return %0 : vector<11xf32>
}

// -----

// expected-error @below {{'executor.global' op expected either initialization region or initial_value but not both}}
executor.global @global constant : vector<10xf32> attributes {
  initial_value = dense<0.0> : vector<10xf32>
} {
  %0 = arith.constant dense<0.0> : vector<10xf32>
  executor.return %0 : vector<10xf32>
}

// -----

executor.global @global1 constant : vector<10xf32> attributes {
  initial_value = dense<0.0> : vector<10xf32>
}

executor.global @global2 : vector<11xf32> attributes {
  initial_value = dense<0.0> : vector<11xf32>
}

func.func @global_get() {
  %0 = executor.get_global @global1 : vector<10xf32>
  // expected-error @below {{'executor.set_global' op global has type 'vector<11xf32>' vs value of type 'vector<10xf32>'}}
  executor.set_global %0, @global2 : vector<10xf32>
  return
}

// -----

func.func @memcpy(%arg0: !executor.ptr<device>, %arg1: !executor.ptr<host>, %arg2: i32, %arg3: i32) {
  // expected-error @below {{'executor.memcpy' op operand #0 must be pointer of type host_pinned or pointer of type host, but got '!executor.ptr<device>'}}
  executor.memcpy %arg0 + %arg2 to %arg1 + %arg2 size %arg3 : !executor.ptr<device>, i32, !executor.ptr<host>, i32, i32
  return
}

// -----

executor.func private @enqueue_non_variadic(i32, !executor.ptr<device>)

func.func @main(%arg0: i64, %arg1: !executor.ptr<device>) {
  // expected-error @below {{'executor.call' op call signature is not compatible with the callee signature '!executor.func<(i32, !executor.ptr<device>) -> ()>'}}
  executor.call @enqueue_non_variadic(%arg0, %arg1, %arg1) : (i64, !executor.ptr<device>, !executor.ptr<device>) -> ()
  return
}

// -----

executor.func private @enqueue_variadic(i32, ...)

func.func @main(%arg0: i32, %arg1: !executor.ptr<device>) {
  // expected-error @below {{'executor.call' op call signature is not compatible with the callee signature '!executor.func<(i32, ...) -> ()>'}}
  executor.call @enqueue_variadic() : () -> ()
  return
}

// -----

executor.func private @myfunc(i32)->(i32, i32)

func.func @main(%arg0: i32, %arg1: !executor.ptr<device>) -> (i32, i64) {
  // expected-error @below {{'executor.call' op call signature is not compatible with the callee signature '!executor.func<(i32) -> (i32, i32)>'}}
  %0, %1 = executor.call @myfunc(%arg0) : (i32) -> (i32, i64)
  return %0, %1 : i32, i64
}

// -----

executor.func private @myfunc(i32)->(i32, i64)

func.func @main(%arg0: i64, %arg1: !executor.ptr<device>) -> (i32, i64) {
  // expected-error @below {{'executor.call' op call signature is not compatible with the callee signature '!executor.func<(i32) -> (i32, i64)>'}}
  %0, %1 = executor.call @myfunc(%arg0) : (i64) -> (i32, i64)
  return %0, %1 : i32, i64
}

// -----

func.func @bitcast(%arg0: i64) -> i32 {
  // expected-error @below {{'executor.bitcast' op failed to verify that all of {input, result} have same bit width}}
  %0 = executor.bitcast %arg0 : i64 to i32
  return %0 : i32
}

// -----

// expected-note @below {{prior use here}}
func.func @select(%arg0: i1, %arg1: i32, %arg2 : f32) -> f32 {
    // expected-error @below {{use of value '%arg1' expects different type than prior uses: 'f32' vs 'i32'}}
    %0 = executor.select %arg0, %arg1, %arg2 : f32
    return %0: f32
}

// -----

func.func @invalid_index_type(%arg0: i16, %arg1: i16) ->!executor.ptr<host> {
  // expected-error @below {{'executor.alloc' op operand #0 must be 64-bit signless integer or 32-bit signless integer, but got 'i16'}}
  %0 = executor.alloc %arg0 bytes align(%arg1) : (i16, i16) -> !executor.ptr<host>
  return %0 : !executor.ptr<host>
}

// -----

func.func @siext(%arg0: i32, %arg1: i8) -> i8 {
  // expected-error @below {{'executor.siext' op result type should be have a larger bitwidth than input type}}
  %0 = executor.siext %arg0 : i32 to i8
  return %0 : i8
}

// -----

func.func @siext(%arg0: i32, %arg1: i8) -> i8 {
  // expected-error @below {{'executor.zext' op result type should be have a larger bitwidth than input type}}
  %0 = executor.zext %arg0 : i32 to i8
  return %0 : i8
}

// -----

// expected-error @below {{ValueBoundsAttr 'min/max' and corresponding memref type must have matching element types; found min/max type: 'i64', memref type: 'f32'}}
// expected-error @below {{Unsupported ValueBoundsAttr}}
func.func public @main(%arg0: memref<1xf32>) attributes { executor.function_metadata = #executor.func_meta<[memref<1xf32> {#executor.value_bounds<min = dense<2> : vector<1xi64>, max = dense<6> : vector<1xi64>>}], [], num_output_args = 1>} {
  return
}

// -----

// expected-error @below {{min[0] : 2 must be less than equal to max[0] : 1}}
func.func public @main(%arg0: memref<1xi64>) attributes {executor.function_metadata = #executor.func_meta<[memref<1xi64> {#executor.value_bounds<min = dense<2> : vector<1xi64>, max = dense<1> : vector<1xi64>>}], [], num_output_args = 1>} {
  return
}

// -----
// expected-error @below {{ValueBoundsAttr 'min' and 'max' must have matching types; found min type: 'vector<2xi64>', max type: 'vector<1xi64>'}}
func.func public @main(%arg0: memref<1xi64>) attributes {executor.function_metadata = #executor.func_meta<[memref<1xi64> {#executor.value_bounds<min = dense<1> : vector<2xi64>, max = dense<6> : vector<1xi64>>}], [], num_output_args = 1>} {
  return
}

// -----

// expected-error @below {{ValueBoundsAttr is only for memref, index, int, or float type}}
// expected-error @below {{Unsupported ValueBoundsAttr}}
func.func public @main(%arg0: memref<1xi64>) attributes {executor.function_metadata = #executor.func_meta<[tensor<i32> {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}], [], num_output_args = 1>} {
  return
}

// -----

// expected-error @below {{ValueBoundsAttr must not be present for dynamic memref}}
// expected-error @below {{Unsupported ValueBoundsAttr}}
func.func public @main(%arg0: memref<1xi64>) attributes {executor.function_metadata = #executor.func_meta<[memref<?xi64> {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}], [], num_output_args = 1>} {
  return
}

// -----

// expected-error @below {{ValueBoundsAttr 'min/max' and corresponding memref type must have matching element types; found min/max type: 'i64', memref type: 'i32'}}
// expected-error @below {{Unsupported ValueBoundsAttr}}
func.func public @main(%arg0: memref<1xi64>) attributes {executor.function_metadata = #executor.func_meta<[memref<1xi32> {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}], [], num_output_args = 1>} {
  return
}

// -----

// expected-error @below {{ValueBoundsAttr 'min/max' and corresponding memref type must have matching shapes; found min/max shape: 1, memref shape: 2}}
// expected-error @below {{Unsupported ValueBoundsAttr}}
func.func public @main(%arg0: memref<1xi64>) attributes {executor.function_metadata = #executor.func_meta<[memref<2xi64> {#executor.value_bounds<min = dense<1> : vector<1xi64>, max = dense<6> : vector<1xi64>>}], [], num_output_args = 1>} {
  return
}

// -----

// expected-error @below {{DimensionBoundsAttr is only for a memref type}}
// expected-error @below {{Unsupported DimensionBoundsAttr}}
func.func public @main(%arg0: memref<?xf32>) attributes {executor.function_metadata = #executor.func_meta<[i32 {#executor.dim_bounds<min = [2], max = [6]>}], [], num_output_args = 1>} {
  return
}

// -----

// expected-error @below {{DimensionBoundsAttr 'min' and 'max' must have the same size; found min size: 2, max size: 1}}
func.func public @main(%arg0: memref<?xf32>) attributes {executor.function_metadata = #executor.func_meta<[memref<?xf32> {#executor.dim_bounds<min = [2, 6], max = [6]>}], [], num_output_args = 1>} {
  return
}

// -----

// expected-error @below {{DimensionBoundsAttr min[0] : -1 must be greater than or equal to 0}}
func.func public @main(%arg0: memref<?xf32>) attributes {executor.function_metadata = #executor.func_meta<[memref<?xf32> {#executor.dim_bounds<min = [-1], max = [1]>}], [], num_output_args = 1>} {
  return
}

// -----

// expected-error @below {{DimensionBoundsAttr min[0] : 4 must be less than equal to max[0] : 1}}
func.func public @main(%arg0: memref<?xf32>) attributes {executor.function_metadata = #executor.func_meta<[memref<?xf32> {#executor.dim_bounds<min = [4], max = [1]>}], [], num_output_args = 1>} {
  return
}

// -----

func.func @getoffset_invalid_table_offset(%arg0: !executor.ptr<host>, %arg1: i32, %arg2: i32) -> i64 {
  // expected-error @below {{'executor.getoffset' op expected index 1 indexing a struct to be constant}}
  %0 = executor.getoffset [%arg1, %arg2] : (i32, i32) -> i64, !executor.table<f32, f32>
  return %0 : i64
}

// -----

func.func @getoffset_invalid_table_offset(%arg0: !executor.ptr<host>, %arg1: i32, %arg2: i32) -> i64 {
  // expected-error @below {{'executor.getoffset' op type 'f32' cannot be indexed (index #2)}}
  %0 = executor.getoffset [%arg1, 1, 0] : (i32) -> i64, !executor.table<f32, f32>
  return %0 : i64
}

// -----

func.func @getoffset_invalid_table_offset(%arg0: !executor.ptr<host>, %arg1: i32, %arg2: i32) -> i64 {
  // expected-error @below {{'executor.getoffset' op type 'f32' cannot be indexed (index #2)}}
  %0 = executor.getoffset [1, 0, 0, 0] : () -> i64, !executor.table<f32, f32>
  return %0 : i64
}

// -----

builtin.module {
  // expected-error @below {{'executor.coro_yield' op must have FunctionOpInterface parent}}
  executor.coro_yield
}

// -----

func.func @coro(%arg0: i32, %arg1: i32) -> i32 {
  // expected-error @below {{'executor.coro_yield' op operand types yielded from coroutine must match the parent function result types}}
  executor.coro_yield %arg1, %arg0 : i32, i32
  return %arg0 : i32
}

// -----

func.func @coro(%arg0: f32, %arg1: i32) -> i32 {
  %c2_i32 = arith.constant 2 : i32
  executor.coro_yield %c2_i32 : i32
  return %c2_i32 : i32
}

func.func @coro_create() -> ((i32) -> i32) {
  %c0 = arith.constant 0 : i32
  %c0_f32 = arith.constant 0.0 : f32
  // expected-error @below {{'executor.coro_create' op reference to function with mismatched type}}
  %coro = executor.coro_create @coro : (i32) -> i32
  return %coro : (i32) -> i32
}

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
  // expected-error @below {{'executor.coro_await' op callee operands must either be empty or their types must match the callee function input types}}
  %0:2 = executor.coro_await %coro (%c0, %c0_f32 : i32, f32) : (f32, i32) -> i32
  return %0#1 : i32
}
