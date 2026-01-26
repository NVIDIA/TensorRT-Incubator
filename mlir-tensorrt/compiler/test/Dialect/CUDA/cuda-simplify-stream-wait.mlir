// RUN: mlir-tensorrt-opt %s -split-input-file -cuda-simplify-stream-wait | FileCheck %s

func.func @redundant_wait_in_block() {
  %c0_i32 = arith.constant 0 : i32
  %dev = cuda.get_program_device %c0_i32 : i32
  %s0 = cuda.get_global_stream device(%dev) [0]
  %e0 = cuda.event.create device(%dev)
  cuda.stream.wait_event %s0, %e0
  cuda.stream.wait_event %s0, %e0
  cuda.event.release %e0 : !cuda.event
  return
}

// CHECK-LABEL: func.func @redundant_wait_in_block
// CHECK-COUNT-1: cuda.stream.wait_event
// CHECK-NOT: cuda.stream.wait_event
// CHECK: cuda.event.release

// -----

// Control flow: waits in different branches are not redundant (no dominance).

func.func @waits_in_branches_not_redundant(%cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %dev = cuda.get_program_device %c0_i32 : i32
  %s0 = cuda.get_global_stream device(%dev) [0]
  %e0 = cuda.event.create device(%dev)
  scf.if %cond {
    cuda.stream.wait_event %s0, %e0
  } else {
    cuda.stream.wait_event %s0, %e0
  }
  cuda.event.release %e0 : !cuda.event
  return
}

// CHECK-LABEL: func.func @waits_in_branches_not_redundant
// CHECK-COUNT-2: cuda.stream.wait_event

// -----

func.func @not_redundant_after_if(%cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %dev = cuda.get_program_device %c0_i32 : i32
  %s0 = cuda.get_global_stream device(%dev) [0]
  %e0 = cuda.event.create device(%dev)
  scf.if %cond {
  } else {
    cuda.stream.wait_event %s0, %e0
  }
  cuda.stream.wait_event %s0, %e0
  cuda.event.release %e0 : !cuda.event
  return
}

// CHECK-LABEL: func.func @not_redundant_after_if
// CHECK-COUNT-2: cuda.stream.wait_event

// -----

func.func @ordered_event_waits_elide_first() {
  %c0_i32 = arith.constant 0 : i32
  %dev = cuda.get_program_device %c0_i32 : i32
  %record_stream = cuda.get_global_stream device(%dev) [0]
  %wait_stream = cuda.get_global_stream device(%dev) [1]
  %e1 = cuda.event.create_on_stream %record_stream : !cuda.stream
  %e2 = cuda.event.create_on_stream %record_stream : !cuda.stream
  cuda.stream.wait_event %wait_stream, %e1
  cuda.stream.wait_event %wait_stream, %e2
  cuda.event.release %e1 : !cuda.event
  cuda.event.release %e2 : !cuda.event
  return
}

// CHECK-LABEL: func.func @ordered_event_waits_elide_first
// CHECK: %[[E2:.+]] = cuda.event.create_on_stream
// CHECK-NOT: cuda.event.create_on_stream
// CHECK: cuda.stream.wait_event %[[WAIT:.+]], %[[E2]]
// CHECK-NOT: cuda.stream.wait_event
// CHECK: cuda.event.release %[[E2]]
// CHECK: return

// -----

func.func @different_record_streams_not_simplified() {
  %c0_i32 = arith.constant 0 : i32
  %dev = cuda.get_program_device %c0_i32 : i32
  %s0 = cuda.get_global_stream device(%dev) [0]
  %s1 = cuda.get_global_stream device(%dev) [1]
  %wait_stream = cuda.get_global_stream device(%dev) [2]
  %e1 = cuda.event.create_on_stream %s0 : !cuda.stream
  %e2 = cuda.event.create_on_stream %s1 : !cuda.stream
  cuda.stream.wait_event %wait_stream, %e1
  cuda.stream.wait_event %wait_stream, %e2
  cuda.event.release %e1 : !cuda.event
  cuda.event.release %e2 : !cuda.event
  return
}

// CHECK-LABEL: func.func @different_record_streams_not_simplified
// CHECK: cuda.stream.wait_event
// CHECK: cuda.stream.wait_event
// CHECK-NOT: cuda.stream.wait_event
// CHECK: return

// -----

func.func @stream_use_between_waits_not_simplified() {
  %c0_i32 = arith.constant 0 : i32
  %dev = cuda.get_program_device %c0_i32 : i32
  %record_stream = cuda.get_global_stream device(%dev) [0]
  %wait_stream = cuda.get_global_stream device(%dev) [1]
  %e1 = cuda.event.create_on_stream %record_stream : !cuda.stream
  %e2 = cuda.event.create_on_stream %record_stream : !cuda.stream
  cuda.stream.wait_event %wait_stream, %e1
  cuda.stream.sync %wait_stream : !cuda.stream
  cuda.stream.wait_event %wait_stream, %e2
  cuda.event.release %e1 : !cuda.event
  cuda.event.release %e2 : !cuda.event
  return
}

// CHECK-LABEL: func.func @stream_use_between_waits_not_simplified
// CHECK: cuda.stream.wait_event
// CHECK: cuda.stream.wait_event
// CHECK-NOT: cuda.stream.wait_event
// CHECK: return

// -----

func.func @event_used_elsewhere_not_simplified() {
  %c0_i32 = arith.constant 0 : i32
  %dev = cuda.get_program_device %c0_i32 : i32
  %record_stream = cuda.get_global_stream device(%dev) [0]
  %wait_stream = cuda.get_global_stream device(%dev) [1]
  %e1 = cuda.event.create_on_stream %record_stream : !cuda.stream
  %e2 = cuda.event.create_on_stream %record_stream : !cuda.stream
  cuda.stream.wait_event %wait_stream, %e1
  cuda.stream.wait_event %wait_stream, %e2
  cuda.event.sync %e1 : !cuda.event
  cuda.event.release %e1 : !cuda.event
  cuda.event.release %e2 : !cuda.event
  return
}

// CHECK-LABEL: func.func @event_used_elsewhere_not_simplified
// CHECK: cuda.stream.wait_event
// CHECK: cuda.stream.wait_event
// CHECK-NOT: cuda.stream.wait_event
// CHECK: return
