// RUN: mlir-tensorrt-opt %s -cuda-insert-host-sync -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @insert_host_wait_before_load
func.func @insert_host_wait_before_load(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>>) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst:.+]] :
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  %c0 = arith.constant 0 : index
  // CHECK: memref.load %[[dst]]
  %v = memref.load %dst[%c0] : memref<4xf32, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Insert host wait before loop that uses the copied buffer

// CHECK-LABEL: func.func @insert_host_wait_before_loop_first_use
func.func @insert_host_wait_before_loop_first_use(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>>) {
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  // CHECK: %[[e:.+]] = cuda.event.create_on_stream %[[stream]]

  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  // Sync is inserted BEFORE the loop (not inside it)

  // CHECK: scf.for
  scf.for %i = %c0 to %c4 step %c1 {
    // CHECK: cuda.event.sync %[[e]] : !cuda.event
    // CHECK: memref.load
    %v = memref.load %dst[%i] : memref<4xf32, #plan.memory_space<host>>
    // CHECK: cuda.event.release %[[e]] : !cuda.event
  }
  return
}

// -----
// Test subview alias: load from subview of the D2H target should trigger sync

// CHECK-LABEL: func.func @subview_alias_triggers_sync
func.func @subview_alias_triggers_sync(
    %stream: !cuda.stream,
    %src: memref<8xf32, #plan.memory_space<device>>,
    %dst: memref<8xf32, #plan.memory_space<host>>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index

  // Create a subview BEFORE the copy
  // CHECK: %[[subview:.+]] = memref.subview
  %subview = memref.subview %dst[0][4][1] : memref<8xf32, #plan.memory_space<host>> to memref<4xf32, strided<[1]>, #plan.memory_space<host>>

  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src, %dst : memref<8xf32, #plan.memory_space<device>> to memref<8xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  // Load from subview should trigger sync because subview aliases dst
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK: %{{.+}} = memref.load %[[subview]]
  %v = memref.load %subview[%c0] : memref<4xf32, strided<[1]>, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test subview created after copy: still aliases the original buffer

// CHECK-LABEL: func.func @subview_after_copy_triggers_sync
func.func @subview_after_copy_triggers_sync(
    %stream: !cuda.stream,
    %src: memref<8xf32, #plan.memory_space<device>>,
    %dst: memref<8xf32, #plan.memory_space<host>>) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst:.+]] :
  cuda.copy_d2h stream(%stream) %src, %dst : memref<8xf32, #plan.memory_space<device>> to memref<8xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index

  // Subview created after copy still aliases dst
  // CHECK: %[[subview:.+]] = memref.subview %[[dst]]
  %subview = memref.subview %dst[2][4][1] : memref<8xf32, #plan.memory_space<host>> to memref<4xf32, strided<[1], offset: 2>, #plan.memory_space<host>>

  // Load from subview should trigger sync
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: %{{.+}} = memref.load %[[subview]]
  %v = memref.load %subview[%c0] : memref<4xf32, strided<[1], offset: 2>, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test memref.cast alias

// CHECK-LABEL: func.func @cast_alias_triggers_sync
func.func @cast_alias_triggers_sync(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>>) {
  // Cast to dynamic shape
  // CHECK: %[[cast:.+]] = memref.cast
  %cast = memref.cast %dst : memref<4xf32, #plan.memory_space<host>> to memref<?xf32, #plan.memory_space<host>>

  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index
  // Load from cast should trigger sync
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: %{{.+}} = memref.load %[[cast]]
  %v = memref.load %cast[%c0] : memref<?xf32, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test multiple aliases: subview of subview

// CHECK-LABEL: func.func @nested_subview_alias
func.func @nested_subview_alias(
    %stream: !cuda.stream,
    %src: memref<16xf32, #plan.memory_space<device>>,
    %dst: memref<16xf32, #plan.memory_space<host>>) {
  // CHECK: %[[sv1:.+]] = memref.subview
  %sv1 = memref.subview %dst[0][8][1] : memref<16xf32, #plan.memory_space<host>> to memref<8xf32, strided<[1]>, #plan.memory_space<host>>
  // CHECK: %[[sv2:.+]] = memref.subview %[[sv1]]
  %sv2 = memref.subview %sv1[0][4][1] : memref<8xf32, strided<[1]>, #plan.memory_space<host>> to memref<4xf32, strided<[1]>, #plan.memory_space<host>>

  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src, %dst : memref<16xf32, #plan.memory_space<device>> to memref<16xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index
  // Load from nested subview should trigger sync
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: %{{.+}} = memref.load %[[sv2]]
  %v = memref.load %sv2[%c0] : memref<4xf32, strided<[1]>, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test expand_shape alias

// CHECK-LABEL: func.func @expand_shape_alias_triggers_sync
func.func @expand_shape_alias_triggers_sync(
    %stream: !cuda.stream,
    %src: memref<8xf32, #plan.memory_space<device>>,
    %dst: memref<8xf32, #plan.memory_space<host>>) {
  // CHECK: %[[expanded:.+]] = memref.expand_shape
  %expanded = memref.expand_shape %dst [[0, 1]] output_shape [2, 4]
      : memref<8xf32, #plan.memory_space<host>> into memref<2x4xf32, #plan.memory_space<host>>

  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src, %dst : memref<8xf32, #plan.memory_space<device>> to memref<8xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream
  %c0 = arith.constant 0 : index
  // Load from expanded shape should trigger sync
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: %{{.+}} = memref.load %[[expanded]]
  %v = memref.load %expanded[%c0, %c0] : memref<2x4xf32, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test collapse_shape alias

// CHECK-LABEL: func.func @collapse_shape_alias_triggers_sync
func.func @collapse_shape_alias_triggers_sync(
    %stream: !cuda.stream,
    %src: memref<2x4xf32, #plan.memory_space<device>>,
    %dst: memref<2x4xf32, #plan.memory_space<host>>) {
  // CHECK: %[[collapsed:.+]] = memref.collapse_shape
  %collapsed = memref.collapse_shape %dst [[0, 1]]
      : memref<2x4xf32, #plan.memory_space<host>> into memref<8xf32, #plan.memory_space<host>>

  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src, %dst : memref<2x4xf32, #plan.memory_space<device>> to memref<2x4xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream
  %c0 = arith.constant 0 : index
  // Load from collapsed shape should trigger sync
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: %{{.+}} = memref.load %[[collapsed]]
  %v = memref.load %collapsed[%c0] : memref<8xf32, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test: function args may alias, so sync is inserted conservatively

// CHECK-LABEL: func.func @conservative_sync_for_func_args
func.func @conservative_sync_for_func_args(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>>,
    %other: memref<4xf32, #plan.memory_space<host>>) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst:.+]] :
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index
  // Function arguments may alias, so conservative alias analysis inserts sync
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: %{{.+}} = memref.load %[[other:.+]][%{{.+}}]
  %v = memref.load %other[%c0] : memref<4xf32, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test control flow: sync should be inserted before branch that uses buffer

// CHECK-LABEL: func.func @sync_before_branch_with_use
func.func @sync_before_branch_with_use(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>>,
    %cond: i1) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream
  %c0 = arith.constant 0 : index
  // CHECK: scf.if
  scf.if %cond {
    // CHECK-NEXT: cuda.event.sync %[[e]] : !cuda.event
    // CHECK-NEXT: memref.load
    %v = memref.load %dst[%c0] : memref<4xf32, #plan.memory_space<host>>
    // CHECK: cuda.event.release %[[e]] : !cuda.event
  }
  return
}

// -----
// Test control flow: sync in both branches

// CHECK-LABEL: func.func @sync_in_if_else
func.func @sync_in_if_else(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>>,
    %cond: i1) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Sync should be inserted once before the if (not in each branch)
  // CHECK: scf.if
  scf.if %cond {
    // CHECK-NEXT: cuda.event.sync %[[e]] : !cuda.event
    // CHECK: memref.load
    %v = memref.load %dst[%c0] : memref<4xf32, #plan.memory_space<host>>
    // CHECK: else
  } else {
    // CHECK-NEXT: cuda.event.sync %[[e]] : !cuda.event
    // CHECK: memref.load
    %v = memref.load %dst[%c1] : memref<4xf32, #plan.memory_space<host>>
    // CHECK: cuda.event.release %[[e]] : !cuda.event
  }
  return
}

// -----
// Test multiple D2H copies to same buffer: only one sync needed

// CHECK-LABEL: func.func @multiple_copies_single_sync
func.func @multiple_copies_single_sync(
    %stream: !cuda.stream,
    %src1: memref<4xf32, #plan.memory_space<device>>,
    %src2: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>>) {
  // First copy
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst:.+]] :
  cuda.copy_d2h stream(%stream) %src1, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>

  // Second copy overwrites the same buffer
  // CHECK: cuda.copy_d2h stream(%[[stream]]) %{{.+}}, %[[dst]] :
  cuda.copy_d2h stream(%stream) %src2, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e2:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index
  // Only the latest event needs to be synced
  // CHECK: cuda.event.sync %[[e2]] : !cuda.event
  // CHECK-NEXT: %{{.+}} = memref.load %[[dst]]
  %v = memref.load %dst[%c0] : memref<4xf32, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e2]] : !cuda.event
  return
}

// -----
// Test memref.copy (host-side copy) triggers sync for source

// CHECK-LABEL: func.func @memref_copy_triggers_sync
func.func @memref_copy_triggers_sync(
    %stream: !cuda.stream,
    %src_dev: memref<4xf32, #plan.memory_space<device>>,
    %dst_host: memref<4xf32, #plan.memory_space<host>>,
    %other_host: memref<4xf32, #plan.memory_space<host>>) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst:.+]] :
  cuda.copy_d2h stream(%stream) %src_dev, %dst_host : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  // Host-side memref.copy reading from dst_host should trigger sync
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: memref.copy
  memref.copy %dst_host, %other_host : memref<4xf32, #plan.memory_space<host>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test memref.store triggers sync

// CHECK-LABEL: func.func @store_triggers_sync
func.func @store_triggers_sync(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>>,
    %val: f32) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst:.+]] :
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index
  // Store should trigger sync (writing to buffer being copied to)
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: memref.store
  memref.store %val, %dst[%c0] : memref<4xf32, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test store to subview alias triggers sync

// CHECK-LABEL: func.func @store_to_subview_triggers_sync
func.func @store_to_subview_triggers_sync(
    %stream: !cuda.stream,
    %src: memref<8xf32, #plan.memory_space<device>>,
    %dst: memref<8xf32, #plan.memory_space<host>>,
    %val: f32) {
  // CHECK: %[[subview:.+]] = memref.subview
  %subview = memref.subview %dst[0][4][1] : memref<8xf32, #plan.memory_space<host>> to memref<4xf32, strided<[1]>, #plan.memory_space<host>>

  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src, %dst : memref<8xf32, #plan.memory_space<device>> to memref<8xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index
  // Store to subview should trigger sync
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: memref.store %{{.+}}, %[[subview]]
  memref.store %val, %subview[%c0] : memref<4xf32, strided<[1]>, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test nested control flow with aliases

// CHECK-LABEL: func.func @nested_cf_with_alias
func.func @nested_cf_with_alias(
    %stream: !cuda.stream,
    %src: memref<8xf32, #plan.memory_space<device>>,
    %dst: memref<8xf32, #plan.memory_space<host>>,
    %cond1: i1, %cond2: i1) {
  // CHECK: %[[subview:.+]] = memref.subview
  %subview = memref.subview %dst[0][4][1] : memref<8xf32, #plan.memory_space<host>> to memref<4xf32, strided<[1]>, #plan.memory_space<host>>

  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src, %dst : memref<8xf32, #plan.memory_space<device>> to memref<8xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index

  // Outer if does not use the buffer directly, but inner if does

  // CHECK: scf.if
  scf.if %cond1 {
    // CHECK: scf.if
    scf.if %cond2 {
      // CHECK: cuda.event.sync %[[e]] : !cuda.event
      // CHECK: memref.load %[[subview]]
      %v = memref.load %subview[%c0] : memref<4xf32, strided<[1]>, #plan.memory_space<host>>
      // CHECK: cuda.event.release %[[e]] : !cuda.event
    }
  }
  return
}

// -----
// Test scf.while with alias usage

// CHECK-LABEL: func.func @while_loop_with_alias
func.func @while_loop_with_alias(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>>,
    %limit: index) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst:.+]] :
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Sync before while loop that accesses the buffer

  // CHECK: scf.while
  %result = scf.while (%i = %c0) : (index) -> index {
    %cond = arith.cmpi slt, %i, %limit : index
    scf.condition(%cond) %i : index
  // CHECK } do {
  } do {
  ^bb0(%i: index):
    // CHECK: cuda.event.sync %[[e]] : !cuda.event
    // CHECK-NEXT: memref.load
    %v = memref.load %dst[%i] : memref<4xf32, #plan.memory_space<host>>
    // CHECK: cuda.event.release %[[e]] : !cuda.event
    %next = arith.addi %i, %c1 : index
    scf.yield %next : index
  }
  return
}

// -----
// Test: reinterpret_cast creates alias

// CHECK-LABEL: func.func @reinterpret_cast_alias
func.func @reinterpret_cast_alias(
    %stream: !cuda.stream,
    %src: memref<8xf32, #plan.memory_space<device>>,
    %dst: memref<8xf32, #plan.memory_space<host>>) {
  // CHECK: %[[reinterpreted:.+]] = memref.reinterpret_cast
  %reinterpreted = memref.reinterpret_cast %dst to
      offset: [0], sizes: [2, 4], strides: [4, 1]
      : memref<8xf32, #plan.memory_space<host>> to memref<2x4xf32, strided<[4, 1]>, #plan.memory_space<host>>

  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src, %dst : memref<8xf32, #plan.memory_space<device>> to memref<8xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index
  // Load from reinterpreted view should trigger sync
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: %{{.+}} = memref.load %[[reinterpreted]]
  %v = memref.load %reinterpreted[%c0, %c0] : memref<2x4xf32, strided<[4, 1]>, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test: no spurious sync for device-side CUDA ops

// CHECK-LABEL: func.func @no_sync_for_cuda_ops
func.func @no_sync_for_cuda_ops(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>>,
    %other_dev: memref<4xf32, #plan.memory_space<device>>) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst:.+]] :
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK-NOT: cuda.event.sync
  // CHECK: cuda.copy_h2d
  cuda.copy_h2d stream(%stream) %dst, %other_dev : memref<4xf32, #plan.memory_space<host>> to memref<4xf32, #plan.memory_space<device>>
  // CHECK-NOT: cuda.event.sync
  // CHECK: return
  return
}

// -----
// Test: multiple independent buffers with aliases - conservative alias analysis
// syncs both events before first load due to potential aliasing of function args

// CHECK-LABEL: func.func @multiple_buffers_with_aliases
func.func @multiple_buffers_with_aliases(
    %stream: !cuda.stream,
    %src1: memref<8xf32, #plan.memory_space<device>>,
    %src2: memref<8xf32, #plan.memory_space<device>>,
    %dst1: memref<8xf32, #plan.memory_space<host>>,
    %dst2: memref<8xf32, #plan.memory_space<host>>) {
  // Create subviews
  // CHECK: %[[sv1:.+]] = memref.subview
  %sv1 = memref.subview %dst1[0][4][1] : memref<8xf32, #plan.memory_space<host>> to memref<4xf32, strided<[1]>, #plan.memory_space<host>>
  // CHECK: %[[sv2:.+]] = memref.subview
  %sv2 = memref.subview %dst2[4][4][1] : memref<8xf32, #plan.memory_space<host>> to memref<4xf32, strided<[1], offset: 4>, #plan.memory_space<host>>

  // Copy to first buffer
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src1, %dst1 : memref<8xf32, #plan.memory_space<device>> to memref<8xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e1:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  // Copy to second buffer
  // CHECK: cuda.copy_d2h stream(%[[stream]])
  cuda.copy_d2h stream(%stream) %src2, %dst2 : memref<8xf32, #plan.memory_space<device>> to memref<8xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e2:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index

  // Conservative alias analysis syncs both events before first load
  // (function args may alias). Order may vary based on iteration order.
  // CHECK-DAG: cuda.event.sync %[[e1]] : !cuda.event
  // CHECK-DAG: cuda.event.sync %[[e2]] : !cuda.event
  // CHECK: %{{.+}} = memref.load %[[sv1]]
  %v1 = memref.load %sv1[%c0] : memref<4xf32, strided<[1]>, #plan.memory_space<host>>

  // Second load doesn't need sync (already cleared)
  // CHECK-NOT: cuda.event.sync
  // CHECK: %{{.+}} = memref.load %[[sv2]]
  %v2 = memref.load %sv2[%c0] : memref<4xf32, strided<[1], offset: 4>, #plan.memory_space<host>>
  // CHECK-DAG: cuda.event.release %[[e1]] : !cuda.event
  // CHECK-DAG: cuda.event.release %[[e2]] : !cuda.event
  return
}

// -----
// Test: executor.restrict attribute makes function arguments non-aliasing
// With restrict, loading from %other should NOT trigger sync for %dst

// CHECK-LABEL: func.func @restrict_attr_no_sync
func.func @restrict_attr_no_sync(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>> {executor.restrict},
    %other: memref<4xf32, #plan.memory_space<host>> {executor.restrict}) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst:.+]] :
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>

  %c0 = arith.constant 0 : index
  // With executor.restrict, %other is known not to alias %dst, so no sync needed
  // CHECK-NOT: cuda.event.sync
  // CHECK: %{{.+}} = memref.load %[[other:.+]][%{{.+}}]
  %v = memref.load %other[%c0] : memref<4xf32, #plan.memory_space<host>>
  return
}

// -----
// Test: executor.restrict on one arg is sufficient for non-aliasing

// CHECK-LABEL: func.func @restrict_attr_one_arg
func.func @restrict_attr_one_arg(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>> {executor.restrict},
    %other: memref<4xf32, #plan.memory_space<host>>) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst:.+]] :
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>

  %c0 = arith.constant 0 : index
  // %dst has restrict, so %other is known not to alias it
  // CHECK-NOT: cuda.event.sync
  // CHECK: %{{.+}} = memref.load %[[other:.+]][%{{.+}}]
  %v = memref.load %other[%c0] : memref<4xf32, #plan.memory_space<host>>
  return
}

// -----
// Test: executor.restrict does NOT prevent sync when loading from the same buffer

// CHECK-LABEL: func.func @restrict_same_buffer_still_syncs
func.func @restrict_same_buffer_still_syncs(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>> {executor.restrict}) {
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst:.+]] :
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index
  // Loading from the same buffer still needs sync
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: %{{.+}} = memref.load %[[dst]]
  %v = memref.load %dst[%c0] : memref<4xf32, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test: restrict with subview - subview of restricted buffer still aliases itself

// CHECK-LABEL: func.func @restrict_with_subview
func.func @restrict_with_subview(
    %stream: !cuda.stream,
    %src: memref<8xf32, #plan.memory_space<device>>,
    %dst: memref<8xf32, #plan.memory_space<host>> {executor.restrict},
    %other: memref<8xf32, #plan.memory_space<host>> {executor.restrict}) {
  // CHECK: %[[subview:.+]] = memref.subview %[[dst:.+]][0]
  %subview = memref.subview %dst[0][4][1] : memref<8xf32, #plan.memory_space<host>> to memref<4xf32, strided<[1]>, #plan.memory_space<host>>

  // CHECK: cuda.copy_d2h stream(%[[stream:.+]]) %{{.+}}, %[[dst]] :
  cuda.copy_d2h stream(%stream) %src, %dst : memref<8xf32, #plan.memory_space<device>> to memref<8xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  %c0 = arith.constant 0 : index
  // Subview of %dst still aliases %dst, so sync is needed
  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: %{{.+}} = memref.load %[[subview]]
  %v = memref.load %subview[%c0] : memref<4xf32, strided<[1]>, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

// -----
// Test D2H inside nested control flow (scf.if branches)
// Both branches contain D2H, and the load after the if needs sync
// Since D2H is in nested regions, we use stream sync instead of events

// CHECK-LABEL: func.func @test_host_sync_nested
func.func @test_host_sync_nested(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>,
    %dst: memref<4xf32, #plan.memory_space<host>>,
    %cond: i1) -> f32 {
  // CHECK: scf.if
  scf.if %cond {
    // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
    cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  } else {
    // CHECK: cuda.copy_d2h stream(%[[stream]])
    cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  }

  %c0 = arith.constant 0 : index
  // D2H in nested region requires stream sync (events can't cross regions)
  // CHECK: cuda.stream.sync %[[stream]] : !cuda.stream
  // CHECK-NEXT: memref.load
  %v = memref.load %dst[%c0] : memref<4xf32, #plan.memory_space<host>>
  return %v : f32
}

// -----

// CHECK-LABEL: func.func @test_dealloc_syncs_with_d2h
func.func @test_dealloc_syncs_with_d2h(
    %stream: !cuda.stream,
    %src: memref<4xf32, #plan.memory_space<device>>) {
  // Allocate a host buffer
  // CHECK: %[[dst:.+]] = memref.alloc() : memref<4xf32, #plan.memory_space<host>>
  %dst = memref.alloc() : memref<4xf32, #plan.memory_space<host>>

  // CHECK: cuda.copy_d2h stream(%{{.+}}) %{{.+}}, %[[dst]] :
  cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>
  // CHECK-NEXT: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream

  // CHECK: cuda.event.sync %[[e]] : !cuda.event
  // CHECK-NEXT: memref.dealloc %[[dst]]
  memref.dealloc %dst : memref<4xf32, #plan.memory_space<host>>
  // CHECK: cuda.event.release %[[e]] : !cuda.event
  return
}

#dev = #plan.memory_space<device>
#host = #plan.memory_space<host>

// CHECK-LABEL: func.func @write_before_and_in_loop(
func.func @write_before_and_in_loop(
    %stream: !cuda.stream,
    %srcs: memref<?x4xf32, #dev>,
    %dst: memref<4xf32, #host>,
    %lb: index,
    %ub: index,
    %step: index) -> f32{

  %src0 = memref.subview %srcs[0, 0][1, 4][1, 1] : memref<?x4xf32, #dev> to memref<4xf32, #dev>
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src0, %dst : memref<4xf32, #dev> to memref<4xf32, #host>
  // CHECK: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream
  %c0 = arith.constant 0.0 : f32
  // CHECK: scf.for
  %1 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %c0) -> f32 {
    // CHECK: cuda.event.sync %[[e]]
    // CHECK: cuda.stream.sync %[[stream]]
    // CHECK: memref.load
    %v = memref.load %dst[%i] : memref<4xf32, #host>
    %src1 = memref.subview %srcs[%i, 0][1, 4][1, 1] : memref<?x4xf32, #dev> to memref<4xf32, strided<[1], offset: ?>, #dev>
    // CHECK: cuda.copy_d2h stream(%[[stream]])
    cuda.copy_d2h stream(%stream) %src1, %dst : memref<4xf32, strided<[1], offset: ?>, #dev> to memref<4xf32, #host>
    %1 = arith.addf %arg0, %v : f32
    // CHECK: yield
    scf.yield %1 : f32
  }

  // CHECK: cuda.event.release %[[e]]
  // CHECK: return
  return %1 : f32
}

// CHECK-LABEL: func.func @write_before_and_in_loop_with_new_stream(
func.func @write_before_and_in_loop_with_new_stream(
    %device: i32,
    %stream: !cuda.stream,
    %srcs: memref<?x4xf32, #dev>,
    %dst: memref<4xf32, #host>,
    %lb: index,
    %ub: index,
    %step: index) -> f32{

  %src0 = memref.subview %srcs[0, 0][1, 4][1, 1] : memref<?x4xf32, #dev> to memref<4xf32, #dev>
  // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
  cuda.copy_d2h stream(%stream) %src0, %dst : memref<4xf32, #dev> to memref<4xf32, #host>
  // CHECK: %[[e:.+]] = cuda.event.create_on_stream %[[stream]] : !cuda.stream
  %c0 = arith.constant 0.0 : f32
  %1 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %c0) -> f32 {
    // CHECK: cuda.event.sync %[[e]]
    // CHECK: memref.load
    %v = memref.load %dst[%i] : memref<4xf32, #host>
    %src1 = memref.subview %srcs[%i, 0][1, 4][1, 1] : memref<?x4xf32, #dev> to memref<4xf32, strided<[1], offset: ?>, #dev>

    // CHECK: %[[temp_stream:.+]] = cuda.get_global_stream
    %temp_stream = cuda.get_global_stream device(%device) [1]
    // CHECK: cuda.copy_d2h stream(%[[temp_stream]])
    cuda.copy_d2h stream(%temp_stream) %src1, %dst : memref<4xf32, strided<[1], offset: ?>, #dev> to memref<4xf32, #host>
    // CHECK: cuda.stream.sync %[[temp_stream]]
    %1 = arith.addf %arg0, %v : f32
    // CHECK: yield
    scf.yield %1 : f32
  }

  // CHECK: cuda.event.release %[[e]]
  return %1 : f32
}

// CHECK-LABEL: func.func @sync_before_and_after_region(
func.func @sync_before_and_after_region(
    %device: i32,
    %stream: !cuda.stream,
    %srcs: memref<?x4xf32, #dev>,
    %dst: memref<4xf32, #host>,
    %lb: index,
    %ub: index,
    %step: index) -> f32{

  %src0 = memref.subview %srcs[0, 0][1, 4][1, 1] : memref<?x4xf32, #dev> to memref<4xf32, #dev>
  // CHECK: cuda.copy_d2h stream(%[[stream]])
  cuda.copy_d2h stream(%stream) %src0, %dst : memref<4xf32, #dev> to memref<4xf32, #host>
  %c0_index = arith.constant 0 : index
  %c0 = arith.constant 0.0 : f32
  // CHECK: cuda.event.sync %[[e]]
  // CHECK: memref.load
  // CHECK-NOT: cuda.event.release
  %v0 = memref.load %dst[%c0_index] : memref<4xf32, #host>
  executor.print "v0: %f\n"(%v0 : f32)
  // CHECK: scf.for
  %1 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %c0) -> f32 {
    scf.yield %arg0 : f32
  }
  // CHECK-NOT: cuda.event
  // CHECK: memref.load
  %v1 = memref.load %dst[%c0_index] : memref<4xf32, #host>
  executor.print "v1: %f\n"(%v1 : f32)
  // CHECK: cuda.event.release %[[e]]
  return %1 : f32
}

// -----

// CHECK-LABEL: func.func @test_while_loop
func.func @test_while_loop(%stream: !cuda.stream, %src: memref<4xf32, #plan.memory_space<device>>, %dst: memref<4xf32, #plan.memory_space<host>>, %limit: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: scf.while
  %result = scf.while (%i = %c0) : (index) -> index {
    // CHECK: cuda.copy_d2h stream(%[[stream:.+]])
    cuda.copy_d2h stream(%stream) %src, %dst : memref<4xf32, #plan.memory_space<device>> to memref<4xf32, #plan.memory_space<host>>

    %cond = arith.cmpi slt, %i, %limit : index
    scf.condition(%cond) %i : index
  // CHECK } do {
  } do {
  ^bb0(%i: index):
    // CHECK: cuda.stream.sync %[[stream]] : !cuda.stream
    // CHECK-NEXT: memref.load
    %v = memref.load %dst[%i] : memref<4xf32, #plan.memory_space<host>>
    %next = arith.addi %i, %c1 : index
    scf.yield %next : index
  }
  return
}
