// RUN: mlir-tensorrt-opt -split-input-file -wrap-module-in-emitc-class %s | FileCheck %s

// CHECK-LABEL: emitc.class @module_with_callsProgram
module @module_with_calls {
  emitc.global @gv1 : !emitc.array<1xf32>

  // CHECK: func.func @callee(%[[ARG0:.+]]: f32) -> f32
  func.func @callee(%arg0: f32) -> f32 {
    return %arg0 : f32
  }

  // CHECK: func.func @caller(%[[ARG1:.+]]: f32) -> f32
  func.func @caller(%arg0: f32) -> f32 {
    // CHECK: %[[RES:.+]] = call @callee(%[[ARG1]]) : (f32) -> f32
    %0 = func.call @callee(%arg0) : (f32) -> f32
    // CHECK: return %[[RES]]
    return %0 : f32
  }
}

// -----

// Test that it works even if the module is unnamed.
// CHECK-LABEL: emitc.class @unnamed_moduleProgram
module {
  emitc.global @gv1 : !emitc.array<1xf32>

  func.func @callee2(%arg0: f32) -> f32 {
    return %arg0 : f32
  }
  // CHECK-LABEL: func.func @caller2
  func.func @caller2(%arg0: f32) -> f32 {
    // CHECK: call @callee2
    %0 = func.call @callee2(%arg0) : (f32) -> f32
    return %0 : f32
  }
}

// -----

// Test aggregation of per-resource lifecycle helpers into `initialize()` /
// `destroy()` and the CUDA stream null-initialization behavior.
//
// NOTE: The pass only inlines helper methods that have no args and no results.
// Helpers with args/results must remain as explicit methods.
//
// CHECK-LABEL: module @agg_lifecycle
// CHECK: emitc.class @agg_lifecycleProgram
module @agg_lifecycle {
  emitc.global @agg_lifecycle_cuda_stream : !emitc.ptr<!emitc.opaque<"CUstream">>

  // CHECK-NOT: emitc.func @resA_initialize
  emitc.func @resA_initialize() {
    %c111 = "emitc.constant"() <{value = 111 : i32}> : () -> i32
    emitc.return
  }

  // CHECK-NOT: emitc.func @resB_initialize
  emitc.func @resB_initialize() {
    %c222 = "emitc.constant"() <{value = 222 : i32}> : () -> i32
    emitc.return
  }

  // CHECK-NOT: emitc.func @resA_destroy
  emitc.func @resA_destroy() {
    %c333 = "emitc.constant"() <{value = 333 : i32}> : () -> i32
    emitc.return
  }

  // CHECK-NOT: emitc.func @resB_destroy
  emitc.func @resB_destroy() {
    %c444 = "emitc.constant"() <{value = 444 : i32}> : () -> i32
    emitc.return
  }

  // CHECK-NOT: @resC_initialize
  func.func @resC_initialize() -> i32 {
    %c0 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    return %c0 : i32
  }

  // This should not be auto-inlined (has args).
  // CHECK: emitc.func @skipped_initialize
  emitc.func @skipped_initialize(%arg0: i32) {
    emitc.return
  }
}

// CHECK: emitc.func @initialize
// CHECK: get_field @agg_lifecycle_cuda_stream
// CHECK: #emitc.opaque<"nullptr">
// CHECK: value = 111 : i32
// CHECK: value = 222 : i32
// CHECK: %[[C0:.+]] = "emitc.constant"{{.*}}{value = 0 : i32}
// CHECK{LITERAL}: if ({} != 0) {{ return {}; }
// CHECK-SAME: args %[[C0]], %[[C0]]

// CHECK: emitc.func @destroy
// CHECK: value = 444 : i32
// CHECK: value = 333 : i32

// -----

// Test rewriting global accesses:
// - array globals should become `emitc.get_field` without a subscript
// - scalar globals should become `get_field` of `array<1xT>` then `subscript[0]`
//
// CHECK-LABEL: module @global_access
// CHECK: emitc.class @global_accessProgram
module @global_access {
  emitc.global @arr : !emitc.array<4xi32>
  emitc.global @ptrg : !emitc.ptr<!emitc.opaque<"void">>

  // CHECK-LABEL: func.func @use_arr
  func.func @use_arr() {
    // CHECK-NOT: emitc.get_global @arr
    // CHECK: emitc.get_field @arr
    %0 = emitc.get_global @arr : !emitc.array<4xi32>
    return
  }

  // CHECK-LABEL: func.func @use_ptr
  func.func @use_ptr() {
    // CHECK-NOT: emitc.get_global @ptrg
    // CHECK: emitc.get_field @ptrg
    // CHECK: emitc.subscript
    %0 = emitc.get_global @ptrg : !emitc.lvalue<!emitc.ptr<!emitc.opaque<"void">>>
    %1 = emitc.load %0 : !emitc.lvalue<!emitc.ptr<!emitc.opaque<"void">>>
    return
  }
}
